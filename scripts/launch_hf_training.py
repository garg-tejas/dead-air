#!/usr/bin/env python3
"""Launch Dead Air GRPO training on Hugging Face Spaces GPU.

Usage:
    export HF_TOKEN=hf_...
    python scripts/launch_hf_training.py --model unsloth/Qwen3-14B-unsloth-bnb-4bit --episodes 200 --hub-model-id yourname/dead-air-grpo

This script:
1. Authenticates with Hugging Face
2. Pre-downloads the model to local cache
3. Launches training with nohup so it survives disconnects
4. Streams logs to a file
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Launch Dead Air training on HF Spaces")
    parser.add_argument("--model", type=str, default="unsloth/Qwen3-14B-unsloth-bnb-4bit",
                        help="Unsloth model to train (default: 14B)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hub-model-id", type=str, default=None,
                        help="HF Hub model ID to push checkpoints to")
    parser.add_argument("--hub-private", action="store_true",
                        help="Make HF Hub repo private")
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--curriculum", action="store_true", default=True,
                        help="Enable curriculum learning")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without running")
    args = parser.parse_args()

    # Check HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[ERROR] HF_TOKEN environment variable not set.")
        print("        Get one at: https://huggingface.co/settings/tokens")
        print("        Then run: export HF_TOKEN=hf_...")
        sys.exit(1)

    print("=" * 60)
    print("Dead Air HF Spaces Training Launcher")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hub Model ID: {args.hub_model_id or '(not pushing)'}")
    print("")

    # Authenticate with HF
    print("Authenticating with Hugging Face...")
    subprocess.run([sys.executable, "-m", "huggingface_hub", "login", "--token", hf_token],
                   check=True, capture_output=True)
    print("Authenticated successfully.")
    print("")

    # Pre-download model
    print(f"Pre-downloading model to cache: {args.model}")
    cache_script = f"""
import unsloth
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{args.model}")
model, _ = FastLanguageModel.from_pretrained(
    model_name="{args.model}",
    load_in_4bit=True,
    max_seq_length=4096,
)
print("Model cached successfully!")
"""
    result = subprocess.run([sys.executable, "-c", cache_script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Model download failed:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout.strip())
    print("")

    # Build command
    cmd = [
        sys.executable, "train_unsloth_grpo.py",
        "--model", args.model,
        "--episodes", str(args.episodes),
        "--batch-size", str(args.batch_size),
        "--max-completion-length", str(args.max_completion_length),
        "--curriculum",
        "--save-every", "25",
        "--output-dir", "./outputs/unsloth_grpo",
        "--trajectory-file", "./outputs/unsloth_grpo/trajectory.jsonl",
    ]

    if args.hub_model_id:
        cmd.extend(["--push-to-hub", "--hub-model-id", args.hub_model_id])
        if args.hub_private:
            cmd.append("--hub-private")

    # Create log file
    os.makedirs("./logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./logs/training_{timestamp}.log"

    print("Command:")
    print(" ".join(cmd))
    print("")

    if args.dry_run:
        print("[DRY RUN] Not starting training.")
        return

    # Launch with nohup
    print(f"Starting training... (log: {log_file})")
    print(f"Monitor with: tail -f {log_file}")
    print("")

    with open(log_file, "w") as log:
        process = subprocess.Popen(
            ["nohup"] + cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid = process.pid
    with open("./training.pid", "w") as f:
        f.write(str(pid))

    print("=" * 60)
    print("Training launched in background!")
    print(f"PID: {pid}")
    print(f"Log: tail -f {log_file}")
    print(f"Kill: kill {pid}")
    print("=" * 60)


if __name__ == "__main__":
    main()
