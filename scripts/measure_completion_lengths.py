#!/usr/bin/env python3
"""Measure actual completion lengths from the model to pick max_completion_length.

Uses vLLM for fast inference when available (no training needed).
Falls back to standard transformers if vLLM is not installed.

Usage:
    python scripts/measure_completion_lengths.py --model unsloth/Qwen3-4B-Thinking-2507-bnb-4bit --episodes 10
    python scripts/measure_completion_lengths.py --model Qwen/Qwen3-4B --episodes 10 --use-vllm
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import statistics
import time
from typing import List

from server.grpo_env_wrapper import DispatchRGRPOEnv
from server.prompt_utils import SYSTEM_PROMPT, build_chat_prompt, format_observation


def generate_vllm(llm, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Generate one completion using vLLM."""
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text


def generate_transformers(model, tokenizer, prompt: str, max_new_tokens: int, device: str):
    """Generate one completion using standard transformers."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def measure(
    generate_fn,
    tokenizer,
    episodes: int = 10,
    difficulty: str = "learning",
    max_new_tokens: int = 2048,
) -> tuple[List[int], List[float]]:
    lengths = []
    rewards = []

    for ep in range(episodes):
        env = DispatchRGRPOEnv(seed=42 + ep, difficulty=difficulty)
        env.reset()

        for _ in range(80):
            if env._obs and env._obs.get("done"):
                break

            obs_text = format_observation(env._obs)
            prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, obs_text)

            completion = generate_fn(prompt, max_new_tokens)

            # Count completion tokens
            comp_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids
            comp_len = comp_ids.shape[1]
            lengths.append(comp_len)

            env.step(completion)

        rewards.append(env.reward if env.reward is not None else 0.0)

    return lengths, rewards


def print_stats(lengths: list[int], rewards: list[float]) -> int:
    if not lengths:
        print("No data collected.")
        return 256

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
    suggested = int(p95 * 1.2)
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
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for faster inference")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    generate_fn = None
    tokenizer = None

    if args.use_vllm:
        try:
            from vllm import LLM
            print(f"Loading model with vLLM: {args.model}")
            llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096)
            tokenizer = llm.get_tokenizer()
            generate_fn = lambda prompt, max_tok: generate_vllm(llm, tokenizer, prompt, max_tok)
            print("vLLM loaded successfully.")
        except ImportError:
            print("vLLM not installed. Falling back to standard transformers.")
            args.use_vllm = False

    if not args.use_vllm:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model with transformers: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto" if args.device == "cuda" else None,
            trust_remote_code=True,
        )
        if args.device == "cpu":
            model = model.to("cpu")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        device = next(model.parameters()).device
        generate_fn = lambda prompt, max_tok: generate_transformers(model, tokenizer, prompt, max_tok, device)

    print(f"Running {args.episodes} episodes with max_new_tokens=2048 (no truncation)...")
    start = time.time()
    lengths, rewards = measure(generate_fn, tokenizer, args.episodes, args.difficulty)
    elapsed = time.time() - start

    suggested = print_stats(lengths, rewards)
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/args.episodes:.1f}s per episode)")

    with open("completion_lengths.json", "w") as f:
        json.dump({
            "lengths": lengths,
            "rewards": rewards,
            "suggested_max_completion_length": suggested,
            "model": args.model,
            "episodes": args.episodes,
            "used_vllm": args.use_vllm,
        }, f, indent=2)
    print("Saved raw data to completion_lengths.json")


if __name__ == "__main__":
    main()
