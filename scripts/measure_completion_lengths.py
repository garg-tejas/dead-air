#!/usr/bin/env python3
"""Measure actual completion lengths from the model to pick max_completion_length.

Uses vLLM for fast batched inference when available (no training needed).
Falls back to standard transformers if vLLM is not installed.

IMPORTANT: vLLM does NOT support BNB (bitsandbytes) quantized models well.
Use a plain FP16/BF16 model (e.g. Qwen/Qwen3-4B) or AWQ/GPTQ with vLLM.

Usage:
    # Fast batched inference with vLLM (recommended)
    python scripts/measure_completion_lengths.py --model Qwen/Qwen3-4B --episodes 10 --use-vllm

    # Fallback to transformers (single-batch, slower)
    python scripts/measure_completion_lengths.py --model Qwen/Qwen3-4B --episodes 10
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


def measure_batched(
    generate_fn,
    tokenizer,
    episodes: int = 10,
    difficulty: str = "learning",
    max_new_tokens: int = 512,
) -> tuple[List[int], List[float]]:
    """Run episodes with batched inference across all active envs per step."""
    envs = [DispatchRGRPOEnv(seed=42 + i, difficulty=difficulty) for i in range(episodes)]
    for env in envs:
        env.reset()

    lengths = []
    rewards = []
    active = list(range(episodes))

    for step_idx in range(80):  # MAX_STEPS
        if not active:
            break

        # Build one batch of all active prompts
        prompts = []
        for idx in active:
            obs_text = format_observation(envs[idx]._obs)
            prompts.append(build_chat_prompt(tokenizer, SYSTEM_PROMPT, obs_text))

        # Single batched inference call for all active envs
        completions = generate_fn(prompts, max_new_tokens)

        for i, idx in enumerate(active):
            completion = completions[i]
            comp_ids = tokenizer(
                completion, return_tensors="pt", add_special_tokens=False
            ).input_ids
            lengths.append(comp_ids.shape[1])
            envs[idx].step(completion)

        active = [idx for idx in active if not envs[idx]._obs.get("done")]

    rewards = [env.reward if env.reward is not None else 0.0 for env in envs]
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
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        help="Model ID. Use plain FP16/BF16 for vLLM (BNB models may not work).",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--difficulty", default="learning")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for faster batched inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens per completion. 512 is plenty for DispatchR JSON actions.",
    )
    args = parser.parse_args()

    generate_fn = None
    tokenizer = None

    if args.use_vllm:
        if "bnb" in args.model.lower() or "4bit" in args.model.lower() or "8bit" in args.model.lower():
            print(
                "WARNING: BNB-quantized models are not well supported by vLLM.\n"
                "         Consider using a plain FP16/BF16 model or AWQ/GPTQ variant.\n"
                "         Falling back to standard transformers."
            )
            args.use_vllm = False

    if args.use_vllm:
        try:
            from vllm import LLM, SamplingParams

            print(f"Loading model with vLLM: {args.model}")
            llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096)
            tokenizer = llm.get_tokenizer()
            sampling_params = SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )

            def _generate(prompts: list, _max_tok: int) -> list[str]:
                outputs = llm.generate(prompts, sampling_params)
                return [out.outputs[0].text for out in outputs]

            generate_fn = _generate
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

        def _generate(prompts: list, max_tok: int) -> list[str]:
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    max_length=None,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_lens = inputs.input_ids.shape[1]
            completions = []
            for i in range(len(prompts)):
                gen_ids = outputs[i][prompt_lens:]
                completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
            return completions

        generate_fn = _generate

    print(f"Running {args.episodes} episodes, max_new_tokens={args.max_new_tokens}...")
    start = time.time()
    lengths, rewards = measure_batched(
        generate_fn, tokenizer, args.episodes, args.difficulty, args.max_new_tokens
    )
    elapsed = time.time() - start

    suggested = print_stats(lengths, rewards)
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / args.episodes:.1f}s per episode)")

    with open("completion_lengths.json", "w") as f:
        json.dump(
            {
                "lengths": lengths,
                "rewards": rewards,
                "suggested_max_completion_length": suggested,
                "model": args.model,
                "episodes": args.episodes,
                "used_vllm": args.use_vllm,
            },
            f,
            indent=2,
        )
    print("Saved raw data to completion_lengths.json")


if __name__ == "__main__":
    main()
