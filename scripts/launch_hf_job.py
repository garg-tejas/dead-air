#!/usr/bin/env python3
"""Launch DispatchR GRPO training on Hugging Face Jobs.

Supports both the hand-rolled Unsloth script (L4) and the new
TRL-native script (A100 / L40S).

Usage:
    # L40S — TRL script (best availability, 48GB, fast)
    export HF_TOKEN=hf_...
    python scripts/launch_hf_job.py \\
        --flavor l40s \\
        --script trl \\
        --episodes 500 \\
        --hub-model-id username/dispatchr-grpo

    # A100 Large — TRL script (fastest, 80GB, often queued)
    python scripts/launch_hf_job.py \\
        --flavor a100-large \\
        --script trl \\
        --episodes 500 \\
        --hub-model-id username/dispatchr-grpo

    # L4 — TRL script (main training path)
    python scripts/launch_hf_job.py \\
        --flavor l4x1 \\
        --script trl \\
        --episodes 200 \\
        --hub-model-id username/dispatchr-grpo
"""

import argparse
import os
import subprocess
import sys


# ── GPU flavors and their VRAM ─────────────────────────────────────────────
FLAVORS = {
    "l4x1":       {"vram_gb": 24,  "price": "$0.80/hr",  "recommended_script": "trl"},
    "a10g-large": {"vram_gb": 24,  "price": "$1.50/hr",  "recommended_script": "trl"},
    "l40sx1":       {"vram_gb": 48,  "price": "$1.80/hr",  "recommended_script": "trl"},
    "a100-large": {"vram_gb": 80,  "price": "$2.50/hr",  "recommended_script": "trl"},
}

# ── Dependencies per script ────────────────────────────────────────────────
DEPS_UNSLOTH = [
    "unsloth",
    "transformers",
    "accelerate",
    "datasets",
    "trl",
    "numpy",
    "networkx",
    "matplotlib",
    "huggingface-hub",
    "jmespath",
    "openenv-core[core]",
    "peft",
]

DEPS_TRL = [
    "torch>=2.3",
    "transformers>=4.45",
    "accelerate>=0.34",
    "datasets>=2.20",
    "trl>=0.15",          # GRPOTrainer with use_vllm support
    "vllm>=0.11,<0.19",   # TRL currently supports vLLM 0.11.x to 0.18.x
    "peft>=0.12",
    "numpy",
    "networkx",
    "huggingface-hub",
    "openenv-core[core]",
    "wandb",              # experiment tracking
]

DEPS_GRPO = [
    "torch>=2.3",
    "transformers>=4.45",
    "accelerate>=0.34",
    "datasets>=2.20",
    "peft>=0.12",
    "numpy",
    "networkx",
    "huggingface-hub",
    "openenv-core[core]",
    "wandb",              # experiment tracking
]


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("ERROR: HF_TOKEN not set.\n  export HF_TOKEN=hf_your_token")
        sys.exit(1)
    return token


def build_unsloth_cmd(args) -> str:
    """Shell command for the hand-rolled Unsloth training script."""
    train_args = [
        f"--model {args.model_unsloth}",
        f"--episodes {args.episodes}",
        f"--batch-size {args.batch_size}",
        f"--max-completion-length {args.max_completion_length}",
        "--output-dir /data/outputs",
        "--trajectory-file /data/outputs/trajectory.jsonl",
        "--push-to-hub",
        f"--hub-model-id {args.hub_model_id}",
    ]
    if not args.no_curriculum:
        train_args.append("--curriculum")

    train_cmd = "python train_unsloth_grpo.py " + " ".join(train_args)
    return _wrap_pipeline(args, train_cmd, "train_unsloth_grpo.py")


def build_trl_cmd(args) -> str:
    """Shell command for the TRL-native training script (L40S/A100)."""
    train_args = [
        f"--model {args.model_trl}",
        f"--episodes {args.episodes}",
        f"--batch-size {args.batch_size}",
        f"--n-seeds {args.n_seeds}",
        f"--max-completion-length {args.max_completion_length}",
        f"--grad-accum {args.grad_accum}",
        f"--num-generations {args.num_generations}",
        f"--cache-workers {args.cache_workers}",
        f"--steps-per-seed {args.steps_per_seed}",
        "--output-dir /data/outputs",
        "--trajectory-file /data/outputs/trajectory.jsonl",
        "--push-to-hub",
        f"--hub-model-id {args.hub_model_id}",
    ]
    if not args.no_curriculum:
        train_args.append("--curriculum")
        train_args.append(f"--curriculum-threshold {args.curriculum_threshold}")
    if args.wandb_project:
        train_args.append(f"--wandb-project {args.wandb_project}")
        if args.wandb_entity:
            train_args.append(f"--wandb-entity {args.wandb_entity}")
        if args.wandb_run_name:
            train_args.append(f"--wandb-run-name {args.wandb_run_name}")
    if args.tensorboard:
        train_args.append("--tensorboard")

    train_cmd = "python train_trl_grpo.py " + " ".join(train_args)
    return _wrap_pipeline(args, train_cmd, "train_trl_grpo.py")


def build_grpo_cmd(args) -> str:
    """Shell command for the hand-rolled GRPO script (L40S / A100)."""
    train_args = [
        f"--model {args.model_grpo}",
        f"--episodes {args.episodes}",
        f"--batch-size {args.batch_size}",
        f"--max-completion-length {args.max_completion_length}",
        "--output-dir /data/outputs",
        "--trajectory-file /data/outputs/trajectory.jsonl",
        "--push-to-hub",
        f"--hub-model-id {args.hub_model_id}",
    ]
    if not args.no_curriculum:
        train_args.append("--curriculum")

    train_cmd = "python train_grpo.py " + " ".join(train_args)
    return _wrap_pipeline(args, train_cmd, "train_grpo.py")


def _wrap_pipeline(args, train_cmd: str, script_name: str) -> str:
    """Wrap train command with git clone, before/after inference, and artifact upload."""
    parts = [
        f"git clone {args.github_repo} /workspace",
        "cd /workspace",
        # Install deps
        "pip install -q 'trl>=0.15' 'vllm>=0.11,<0.19' 'peft>=0.12' 'openenv-core[core]' wandb 2>&1 | tail -5",
    ]
    # Pass WANDB_API_KEY if available
    wandb_key = os.environ.get("WANDB_API_KEY", "").strip()
    if wandb_key and args.wandb_project:
        parts.append(f"export WANDB_API_KEY={wandb_key}")

    if args.before_after:
        model_for_inference = {
            "trl": args.model_trl,
            "unsloth": args.model_unsloth,
            "grpo": args.model_grpo,
        }[args.script]
        before_cmd = (
            f"python inference.py "
            f"--model-path {model_for_inference} "
            f"--max-new-tokens {args.max_completion_length} "
            "--episodes 3 --difficulty learning "
            "--output /data/outputs/before_inference.json "
            "--trajectory-file /data/outputs/before_trajectory.jsonl"
        )
        parts.append(f"echo '=== Before Inference ===' && {before_cmd}")

    parts.append(f"echo '=== Training ({script_name}) ===' && {train_cmd}")

    if args.before_after:
        after_cmd = (
            "python inference.py "
            "--model-path /data/outputs/final "
            f"--max-new-tokens {args.max_completion_length} "
            "--episodes 3 --difficulty learning "
            "--output /data/outputs/after_inference.json "
            "--trajectory-file /data/outputs/after_trajectory.jsonl"
        )
        parts.append(f"echo '=== After Inference ===' && {after_cmd}")

    # Upload all artifacts to a companion dataset repo
    dataset_repo = f"{args.hub_model_id}-runs"
    upload_script = (
        "python -c \""
        "from huggingface_hub import HfApi; import os; "
        "api = HfApi(); "
        f"repo = '{dataset_repo}'; "
        "api.create_repo(repo_id=repo, repo_type='dataset', exist_ok=True); "
        "files = ["
        "  '/data/outputs/metrics.json',"
        "  '/data/outputs/trajectory.jsonl',"
        "  '/data/outputs/before_inference.json',"
        "  '/data/outputs/after_inference.json',"
        "]; "
        "[api.upload_file("
        "    path_or_fileobj=f, path_in_repo=os.path.basename(f),"
        "    repo_id=repo, repo_type='dataset'"
        "  ) or print(f'Uploaded {f}')"
        "  for f in files if os.path.exists(f)"
        "]\""
    )
    parts.append(f"echo '=== Uploading artifacts ===' && {upload_script}")

    return " && ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Launch DispatchR GRPO training on HF Jobs"
    )

    # ── Job config ────────────────────────────────────────────────────
    parser.add_argument(
        "--flavor",
        default="l40sx1",
        choices=list(FLAVORS.keys()),
        help="GPU flavor. l40sx1 (48GB, $1.80/hr) is best availability and sufficient for Qwen3-4B.",
    )
    parser.add_argument(
        "--script",
        default="trl",
        choices=["trl", "unsloth", "grpo"],
        help=(
            "'trl' = TRL-native with trajectory cache (L40S/A100, vLLM, recommended). "
            "'unsloth' = hand-rolled loop (L4, BNB quantization). "
            "'grpo' = custom GRPO loop (L40S/A100, no vLLM, slower but trains all steps)."
        ),
    )
    parser.add_argument("--timeout", default="8h")
    parser.add_argument("--dry-run", action="store_true")

    # ── Training config ───────────────────────────────────────────────
    parser.add_argument("--episodes",    type=int,   default=500)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--grad-accum",  type=int,   default=2,
                        help="Gradient accumulation steps (TRL script only).")
    parser.add_argument("--n-seeds",     type=int,   default=1000,
                        help="Dataset manifest size (TRL script only).")
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="GRPO group size. 8 is optimal for L40S 48GB with 512-token completions.",
    )
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=8,
        help="Parallel workers for trajectory cache. Match vCPU count (8 on L40S).",
    )
    parser.add_argument(
        "--steps-per-seed",
        type=int,
        default=7,
        help="Random steps sampled per episode into dataset. 7 x 1000 seeds = 7000 rows.",
    )
    parser.add_argument(
        "--model-trl",
        default="Qwen/Qwen3-4B",
        help="Model for TRL script. Must be BF16-compatible (no BNB).",
    )
    parser.add_argument(
        "--model-unsloth",
        default="unsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
        help="Model for Unsloth script.",
    )
    parser.add_argument(
        "--model-grpo",
        default="Qwen/Qwen3-4B",
        help="Model for custom GRPO script. BF16 on L40S/A100.",
    )
    parser.add_argument("--no-curriculum",  action="store_true")
    parser.add_argument("--curriculum-threshold", type=float, default=0.65)
    parser.add_argument("--before-after",   action="store_true",
                        help="Run inference before and after training for comparison.")

    # ── Experiment tracking ───────────────────────────────────────────
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dispatchr-grpo",
        help="Weights & Biases project name. Set to '' to disable.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (username or team).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom wandb run name.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging.",
    )

    # ── Hub config ────────────────────────────────────────────────────
    parser.add_argument(
        "--hub-model-id",
        default="ggtejas/dispatchr-grpo",
        help="HF Hub repo to push checkpoints to.",
    )
    parser.add_argument(
        "--github-repo",
        default="https://github.com/garg-tejas/dispatchR.git",
    )
    args = parser.parse_args()

    # ── Build job command ─────────────────────────────────────────────
    if args.script == "trl":
        job_cmd = build_trl_cmd(args)
        deps    = DEPS_TRL
        model   = args.model_trl
    elif args.script == "grpo":
        job_cmd = build_grpo_cmd(args)
        deps    = DEPS_GRPO
        model   = args.model_grpo
    else:
        job_cmd = build_unsloth_cmd(args)
        deps    = DEPS_UNSLOTH
        model   = args.model_unsloth

    hf_token = get_hf_token()
    flavor_info = FLAVORS[args.flavor]

    # ── Warn on mismatched flavor / script ────────────────────────────
    recommended = flavor_info["recommended_script"]
    if recommended != args.script:
        print(
            f"WARN: --flavor {args.flavor} is optimised for '{recommended}' "
            f"but you chose --script {args.script}.\n"
            f"      This will work but may be slower or hit OOM.\n"
        )

    # ── Assemble hf CLI command ───────────────────────────────────────
    cmd = [
        "hf", "jobs", "uv", "run",
        "--flavor", args.flavor,
        "--timeout", args.timeout,
    ]
    for dep in deps:
        cmd += ["--with", dep]
    cmd += [
        "--env", f"HF_TOKEN={hf_token}",
        "--label", "project=dispatchr",
        "--label", f"script={args.script}",
        "--label", f"model={model.split('/')[-1]}",
        "--", "bash", "-c", job_cmd,
    ]

    # ── Print summary ─────────────────────────────────────────────────
    print("=" * 60)
    print("DispatchR HF Job Launcher")
    print("=" * 60)
    print(f"  Flavor:       {args.flavor}  ({flavor_info['vram_gb']}GB VRAM, {flavor_info['price']})")
    print(f"  Script:       {args.script}")
    print(f"  Model:        {model}")
    print(f"  Episodes:     {args.episodes}")
    print(f"  Batch size:   {args.batch_size}")
    if args.script == "trl":
        print(f"  Grad accum:   {args.grad_accum}  (eff. batch = {args.batch_size * args.grad_accum})")
        print(f"  Seeds:        {args.n_seeds}")
    print(f"  Max tokens:   {args.max_completion_length}")
    print(f"  Timeout:      {args.timeout}")
    print(f"  Hub ID:       {args.hub_model_id}")
    print(f"  Runs dataset: https://huggingface.co/datasets/{args.hub_model_id}-runs")
    print(f"  Curriculum:   {'yes' if not args.no_curriculum else 'no'}")
    print(f"  Before/After: {'yes' if args.before_after else 'no'}")
    print()

    if args.dry_run:
        print("[DRY RUN] hf CLI command:")
        redacted = [f"HF_TOKEN=***" if v.startswith("HF_TOKEN=hf_") else v for v in cmd]
        print(" \\\n  ".join(redacted))
        print("\n[DRY RUN] Job pipeline:")
        for i, part in enumerate(job_cmd.split(" && "), 1):
            print(f"  {i}. {part[:120]}{'...' if len(part) > 120 else ''}")
        return

    print(f"Running: {' '.join(cmd[:8])} ...")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
