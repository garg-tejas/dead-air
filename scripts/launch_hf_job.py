#!/usr/bin/env python3
"""Launch DispatchR GRPO training on Hugging Face Hub Jobs.

Usage (PowerShell):
    $Env:HF_TOKEN = "hf_your_token"
    python scripts/launch_hf_job.py --flavor l4x1 --episodes 200

Prerequisites:
    1. Install HF CLI: curl -LsSf https://hf.co/cli/install.sh | bash
    2. Login: hf auth login
    3. Have HF credits: https://huggingface.co/pricing
"""

import argparse
import os
import subprocess
import sys


def run_command(cmd: list[str]) -> None:
    """Run a shell command, streaming output, then exit with its return code."""
    print(f"Running: {' '.join(cmd)}")
    print("")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def get_hf_token() -> str:
    """Read HF_TOKEN from environment variable only."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("ERROR: HF_TOKEN not found. Set it as an environment variable:")
        print("")
        print("  PowerShell:")
        print("    $Env:HF_TOKEN = 'hf_your_token'")
        print("")
        print("  Bash:")
        print("    export HF_TOKEN=hf_your_token")
        print("")
        print("  Windows CMD:")
        print("    set HF_TOKEN=hf_your_token")
        print("")
        print("Get your token: https://huggingface.co/settings/tokens")
        sys.exit(1)
    return token


def build_job_cmd(args) -> str:
    """Return the bash command that runs inside the HF Job container."""

    # ── training command ───────────────────────────────────────────────
    train_args = [
        f"--model {args.model}",
        f"--episodes {args.episodes}",
        f"--batch-size {args.batch_size}",
        "--output-dir /data/outputs",
        "--trajectory-file /data/outputs/trajectory.jsonl",
        "--push-to-hub",
        f"--hub-model-id {args.hub_model_id}",
    ]
    if not args.no_curriculum:
        train_args.append("--curriculum")
    if args.resume:
        train_args.append("--resume-from-hub")

    train_cmd = "python train_unsloth_grpo.py " + " ".join(train_args)

    # ── before/after inference for comparison ─────────────────────────
    if args.before_after:
        before_cmd = (
            "python inference.py "
            f"--model-path {args.model} "
            "--use-unsloth --load-in-4bit "
            "--episodes 3 "
            "--difficulty learning "
            "--output /data/outputs/before_inference.json "
            "--trajectory-file /data/outputs/before_trajectory.jsonl"
        )
        after_cmd = (
            "python inference.py "
            "--model-path /data/outputs/final "
            "--use-unsloth --load-in-4bit "
            "--episodes 3 "
            "--difficulty learning "
            "--output /data/outputs/after_inference.json "
            "--trajectory-file /data/outputs/after_trajectory.jsonl"
        )
    else:
        before_cmd = ""
        after_cmd = ""

    # ── artifact upload to HF dataset ─────────────────────────────────
    dataset_repo = f"{args.hub_model_id}-runs"
    upload_script = (
        "python -c \""
        "from huggingface_hub import HfApi; "
        "import os; "
        "api = HfApi(); "
        f"repo = '{dataset_repo}'; "
        "api.create_repo(repo_id=repo, repo_type='dataset', exist_ok=True); "
        "files = ["
        "  '/data/outputs/metrics.json',"
        "  '/data/outputs/trajectory.jsonl',"
        "  '/data/outputs/before_trajectory.jsonl',"
        "  '/data/outputs/after_trajectory.jsonl',"
        "  '/data/outputs/before_inference.json',"
        "  '/data/outputs/after_inference.json',"
        "]; "
        "[ api.upload_file("
        "    path_or_fileobj=f, "
        "    path_in_repo=os.path.basename(f), "
        "    repo_id=repo, repo_type='dataset'"
        "  ) or print(f'Uploaded {f}')"
        "  for f in files if os.path.exists(f) "
        "]\""
    )

    # ── assemble full pipeline ─────────────────────────────────────────
    parts = [f"git clone {args.github_repo} /workspace", "cd /workspace"]
    if before_cmd:
        parts.append(f"echo '=== Before Inference ===' && {before_cmd}")
    parts.append(f"echo '=== Training ===' && {train_cmd}")
    if after_cmd:
        parts.append(f"echo '=== After Inference ===' && {after_cmd}")
    parts.append(f"echo '=== Uploading artifacts ===' && {upload_script}")

    return " && ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Launch DispatchR GRPO training on HF Jobs")
    parser.add_argument("--flavor", default="l4x1",
                        help="GPU flavor (l4x1=$0.80/hr, a10g-large=$1.50/hr, a100-large=$2.50/hr)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", default="8h", help="Job timeout (e.g. 8h, 4h, 30m)")
    parser.add_argument("--model", default="unsloth/Qwen3-4B-Thinking-2507-bnb-4bit")
    parser.add_argument("--hub-model-id", default="ggtejas/dispatchr-grpo",
                        help="HF Hub model ID for checkpoint push (username/repo)")
    parser.add_argument("--github-repo", default="https://github.com/garg-tejas/dispatchR.git")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument(
        "--before-after", action="store_true",
        help="Run 3-episode inference before AND after training for comparison.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint pushed to --hub-model-id. "
             "The job downloads training_state.json from the hub, restores batch index "
             "and reward history, and continues from the next batch.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command without submitting")
    args = parser.parse_args()

    job_cmd = build_job_cmd(args)
    hf_token = get_hf_token()

    cmd = [
        "hf", "jobs", "uv", "run",
        "--flavor", args.flavor,
        "--timeout", args.timeout,
        # dependencies
        "--with", "unsloth",
        "--with", "transformers",
        "--with", "accelerate",
        "--with", "datasets",
        "--with", "trl",
        "--with", "numpy",
        "--with", "networkx",
        "--with", "matplotlib",
        "--with", "huggingface-hub",
        "--with", "jmespath",
        "--with", "openenv-core[core]",
        # pass token as env var
        "--env", f"HF_TOKEN={hf_token}",
        "--label", "project=dispatchr",
        "--label", f"model={args.model.split('/')[-1]}",
        "--", "bash", "-c", job_cmd,
    ]

    print("=" * 60)
    print("DispatchR HF Job Launcher")
    print("=" * 60)
    print(f"Flavor:       {args.flavor}")
    print(f"Episodes:     {args.episodes}")
    print(f"Batch:        {args.batch_size}")
    print(f"Timeout:      {args.timeout}")
    print(f"Model:        {args.model}")
    print(f"Hub ID:       {args.hub_model_id}")
    print(f"Before/After: {'yes' if args.before_after else 'no'}")
    print(f"Resume:       {'yes — continuing from last checkpoint' if args.resume else 'no'}")
    print(f"Runs dataset: https://huggingface.co/datasets/{args.hub_model_id}-runs")
    print("")

    if args.dry_run:
        print("[DRY RUN] hf CLI command:")
        redacted = [f"HF_TOKEN=***" if v.startswith("HF_TOKEN=hf_") else v for v in cmd]
        print(" \\\n  ".join(redacted))
        print("\n[DRY RUN] Job pipeline:")
        for i, part in enumerate(job_cmd.split(" && "), 1):
            print(f"  {i}. {part[:120]}{'...' if len(part) > 120 else ''}")
        return

    run_command(cmd)


if __name__ == "__main__":
    main()
