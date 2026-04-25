#!/usr/bin/env python3
"""Measure actual completion lengths from the model to pick max_completion_length.

Usage:
    python scripts/measure_completion_lengths.py --model unsloth/Qwen3-4B-Thinking-2507-bnb-4bit --episodes 10

This runs inference with a very large max_new_tokens and records the actual
length of every completion. The stats tell you what max_completion_length
to use for training (recommendation: p95 + 20% headroom).
"""

import argparse
import json
import statistics
import time

import torch
from transformers import AutoTokenizer

from server.grpo_env_wrapper import DispatchRGRPOEnv

# Re-use inference helpers
from inference import build_chat_prompt, format_observation, generate_action, SYSTEM_PROMPT


def measure(
    model,
    tokenizer,
    device: str,
    episodes: int = 10,
    difficulty: str = "learning",
    max_new_tokens: int = 2048,  # generous so we don't truncate
):
    lengths = []
    rewards = []

    for ep in range(episodes):
        env = DispatchRGRPOEnv(seed=42 + ep, difficulty=difficulty)
        env.reset()

        for step_idx in range(80):
            if env._obs and env._obs.get("done"):
                break

            obs_text = format_observation(env._obs)
            prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, obs_text)

            # Tokenize prompt to count its length
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
            prompt_len = prompt_ids.shape[1]

            # Generate with very large max_new_tokens
            completion = generate_action(model, tokenizer, prompt, max_new_tokens, device)

            # Count completion tokens
            comp_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids
            comp_len = comp_ids.shape[1]

            lengths.append(comp_len)
            env.step(completion)

        rewards.append(env.reward if env.reward is not None else 0.0)

    return lengths, rewards


def print_stats(lengths: list[int], rewards: list[float]):
    if not lengths:
        print("No data collected.")
        return

    lengths.sort()
    n = len(lengths)

    def percentile(p: float) -> int:
        idx = int(n * p / 100)
        return lengths[min(idx, n - 1)]

    print("\n" + "=" * 60)
    print("COMPLETION LENGTH STATS")
    print("=" * 60)
    print(f"  Samples:     {n}")
    print(f"  Min:         {lengths[0]}")
    print(f"  Mean:        {statistics.mean(lengths):.1f}")
    print(f"  Median:      {statistics.median(lengths):.1f}")
    print(f"  Std:         {statistics.stdev(lengths) if n > 1 else 0:.1f}")
    print(f"  P90:         {percentile(90)}")
    print(f"  P95:         {percentile(95)}")
    print(f"  P99:         {percentile(99)}")
    print(f"  Max:         {lengths[-1]}")
    print("")

    p95 = percentile(95)
    suggested = int(p95 * 1.2)  # 20% headroom above P95
    print(f"  SUGGESTED max_completion_length: {suggested}")
    print(f"    (P95 = {p95}, +20% headroom = {suggested})")
    print("")

    print("  Reward stats:")
    print(f"    Mean:      {statistics.mean(rewards):.3f}")
    print(f"    Min/Max:   {min(rewards):.3f} / {max(rewards):.3f}")
    print("=" * 60)

    return suggested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3-4B-Thinking-2507-bnb-4bit")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--difficulty", default="learning")
    parser.add_argument("--use-unsloth", action="store_true", default=True)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    if args.use_unsloth:
        import unsloth
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            load_in_4bit=args.load_in_4bit,
            max_seq_length=4096,
            dtype=torch.bfloat16,
        )
        FastLanguageModel.for_inference(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = next(model.parameters()).device

    print(f"Running {args.episodes} episodes with max_new_tokens=2048 (no truncation)...")
    start = time.time()
    lengths, rewards = measure(model, tokenizer, device, args.episodes, args.difficulty)
    elapsed = time.time() - start

    suggested = print_stats(lengths, rewards)
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/args.episodes:.1f}s per episode)")

    # Save raw data for later analysis
    with open("completion_lengths.json", "w") as f:
        json.dump({
            "lengths": lengths,
            "rewards": rewards,
            "suggested_max_completion_length": suggested,
            "model": args.model,
            "episodes": args.episodes,
        }, f, indent=2)
    print("Saved raw data to completion_lengths.json")


if __name__ == "__main__":
    main()
