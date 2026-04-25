# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.24.0"]
# ///

"""Run DispatchR training from an exact GitHub commit on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import HfApi


def _current_job_id() -> str:
    """Return the HF Jobs identifier exposed inside the container."""
    return os.environ.get("JOB_ID", "local")


def _parse_github_repo(repo_url: str) -> tuple[str, str]:
    """Extract the owner/repo slug from a supported GitHub remote URL."""
    cleaned = repo_url.rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]

    if cleaned.startswith("https://github.com/"):
        slug = cleaned.removeprefix("https://github.com/")
    elif cleaned.startswith("git@github.com:"):
        slug = cleaned.removeprefix("git@github.com:")
    else:
        raise ValueError(f"Unsupported repo URL: {repo_url}")

    owner, repo = slug.split("/", 1)
    if not owner or not repo:
        raise ValueError(f"Could not parse owner/repo from {repo_url}")
    return owner, repo


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """Extract a tarball while preventing path traversal."""
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise RuntimeError(f"Refusing to extract outside destination: {member.name}")
        archive.extractall(destination)


def _download_repo_snapshot(repo_url: str, git_ref: str, workspace: Path) -> Path:
    """Download and extract a GitHub tarball for the exact requested ref."""
    owner, repo = _parse_github_repo(repo_url)
    archive_url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{git_ref}"
    archive_path = workspace / "repo.tar.gz"

    print(f"Downloading snapshot {archive_url}")
    request = Request(archive_url, headers={"User-Agent": "dispatchr-hf-job"})
    with urlopen(request, timeout=60) as response, archive_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    _safe_extract_tar(archive_path, workspace)
    extracted = [path for path in workspace.iterdir() if path.is_dir() and path.name.startswith(f"{repo}-")]
    if len(extracted) != 1:
        raise RuntimeError(f"Expected one extracted checkout for {repo}, found {len(extracted)}")
    return extracted[0]


def _run_checked(command: list[str], cwd: Path) -> None:
    """Run one command and fail fast if it exits non-zero."""
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def _get_flag_value(args: list[str], flag: str) -> str | None:
    """Return the value for a simple `--flag value` CLI argument."""
    for idx, token in enumerate(args[:-1]):
        if token == flag:
            return args[idx + 1]
    return None


def _has_flag(args: list[str], flag: str) -> bool:
    """Return whether an exact CLI flag is already present."""
    return flag in args


def _append_arg(args: list[str], flag: str, value: str) -> None:
    """Append a value flag only if the caller did not already provide it."""
    if _get_flag_value(args, flag) is None:
        args.extend([flag, value])


def _append_bool_flag(args: list[str], flag: str) -> None:
    """Append a boolean flag only if it is not already present."""
    if not _has_flag(args, flag):
        args.append(flag)


def _normalize_train_args(
    run_kind: str,
    hub_model_id: str,
    extra_args: list[str],
) -> list[str]:
    """Apply smoke/full defaults while allowing explicit overrides."""
    train_args = list(extra_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    defaults = {
        "smoke": {
            "--model": "unsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
            "--episodes": "16",
            "--batch-size": "8",
            "--max-completion-length": "512",
            "--save-every": "8",
            "--output-dir": "./outputs/hf_smoke",
            "--trajectory-file": "./outputs/hf_smoke/trajectory.jsonl",
        },
        "full": {
            "--model": "unsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
            "--episodes": "200",
            "--batch-size": "8",
            "--max-completion-length": "1536",
            "--save-every": "25",
            "--output-dir": "./outputs/hf_full",
            "--trajectory-file": "./outputs/hf_full/trajectory.jsonl",
        },
        "custom": {
            "--output-dir": "./outputs/hf_custom",
            "--trajectory-file": "./outputs/hf_custom/trajectory.jsonl",
        },
    }

    for flag, value in defaults[run_kind].items():
        _append_arg(train_args, flag, value)

    _append_bool_flag(train_args, "--curriculum")
    _append_bool_flag(train_args, "--use-4bit")
    _append_bool_flag(train_args, "--push-to-hub")
    _append_bool_flag(train_args, "--hub-private")
    _append_arg(train_args, "--hub-model-id", hub_model_id)
    return train_args


def _resolve_output_path(checkout_dir: Path, cli_value: str | None) -> Path | None:
    """Resolve a possibly-relative output path against the checkout root."""
    if cli_value is None:
        return None
    path = Path(cli_value)
    if path.is_absolute():
        return path
    return checkout_dir / path


def _upload_artifacts(
    api: HfApi,
    repo_id: str,
    output_dir: Path | None,
    trajectory_file: Path | None,
    run_kind: str,
) -> None:
    """Upload metrics and plots into `artifacts/<job_id>/`."""
    if output_dir is None:
        return

    staging_dir = output_dir / ".hf-job-artifacts"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    artifact_names = [
        "metrics.csv",
        "metrics.json",
        "training_progress.png",
        "training_curves.png",
    ]
    for name in artifact_names:
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, staging_dir / name)

    if trajectory_file is not None and trajectory_file.exists():
        shutil.copy2(trajectory_file, staging_dir / trajectory_file.name)

    if not list(staging_dir.iterdir()):
        shutil.rmtree(staging_dir)
        return

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(staging_dir),
        path_in_repo=f"artifacts/{_current_job_id()}",
        commit_message=f"Upload {run_kind} artifacts for job {_current_job_id()}",
    )
    shutil.rmtree(staging_dir)


def main() -> int:
    """Download the requested repo snapshot, run training, and upload artifacts."""
    parser = argparse.ArgumentParser(description="DispatchR HF Jobs wrapper")
    parser.add_argument("--repo-url", required=True, help="GitHub repo URL to fetch")
    parser.add_argument("--git-ref", required=True, help="Exact Git commit SHA or ref")
    parser.add_argument("--hub-model-id", required=True, help="HF model repo for checkpoints and artifacts")
    parser.add_argument(
        "--run-kind",
        default="custom",
        choices=["smoke", "full", "custom"],
        help="Which rollout defaults to apply before pass-through training args",
    )
    args, extra_args = parser.parse_known_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN must be available inside the job to push checkpoints.")

    train_args = _normalize_train_args(args.run_kind, args.hub_model_id, extra_args)

    with tempfile.TemporaryDirectory(prefix="dispatchr-job-") as temp_dir:
        workspace = Path(temp_dir)
        checkout_dir = _download_repo_snapshot(args.repo_url, args.git_ref, workspace)

        api = HfApi(token=hf_token)
        api.create_repo(
            repo_id=args.hub_model_id,
            repo_type="model",
            private=True,
            exist_ok=True,
        )

        _run_checked(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "-e",
                ".[train]",
                "unsloth",
                "huggingface_hub",
            ],
            cwd=checkout_dir,
        )

        train_command = [sys.executable, "train_unsloth_grpo.py", *train_args]
        print("$ " + " ".join(train_command))
        result = subprocess.run(train_command, cwd=str(checkout_dir), check=False)

        output_dir = _resolve_output_path(checkout_dir, _get_flag_value(train_args, "--output-dir"))
        trajectory_file = _resolve_output_path(checkout_dir, _get_flag_value(train_args, "--trajectory-file"))
        _upload_artifacts(
            api=api,
            repo_id=args.hub_model_id,
            output_dir=output_dir,
            trajectory_file=trajectory_file,
            run_kind=args.run_kind,
        )
        return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
