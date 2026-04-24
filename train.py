"""Rollout collection script for Dead Air.

Collects episodes by running the LLM against the environment.
Useful for warm-start data generation or debugging before GRPO training.

Usage:
    uv sync --extra train
    python train.py --model Qwen/Qwen3-1.7B --episodes 50 --output-dir ./outputs/rollouts
"""

import argparse
import json
import os

from server.dispatcher_environment import DispatcherEnvironment
from server.rollout_utils import collect_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="./outputs/rollouts")
    parser.add_argument("--difficulty", type=str, default="learning")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.device == "cpu":
        model = model.to("cpu")

    print(f"Model loaded. Collecting {args.episodes} rollouts...")
    env = DispatcherEnvironment(seed=42)
    os.makedirs(args.output_dir, exist_ok=True)

    all_rollouts = []
    for ep in range(args.episodes):
        if args.difficulty == "curriculum":
            diff = env.curriculum.phase
        else:
            diff = args.difficulty
        rollout = collect_rollout(env, model, tokenizer, device=args.device)
        rollout["episode"] = ep + 1
        rollout["difficulty"] = diff
        all_rollouts.append(rollout)
        if (ep + 1) % 10 == 0:
            print(
                f"Episode {ep + 1}/{args.episodes}: reward={rollout['reward']:.3f}, steps={rollout['steps']}"
            )

    # Save rollouts
    with open(os.path.join(args.output_dir, "rollouts.json"), "w") as f:
        json.dump(all_rollouts, f, indent=2)

    rewards = [r["reward"] for r in all_rollouts]
    print(f"\nMean reward: {sum(rewards) / len(rewards):.3f}")
    print(f"Rollouts saved to {args.output_dir}/rollouts.json")


if __name__ == "__main__":
    main()
