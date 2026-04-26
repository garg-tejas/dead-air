#!/usr/bin/env python3
"""Baseline evaluation for DispatchR using vLLM fast inference.

Runs the base model (no training) through N episodes to establish
a reward baseline. Uses vLLM for fast batched generation.

Usage:
    python baseline_vllm.py \
        --model Qwen/Qwen3-4B \
        --episodes 50 \
        --batch-size 8 \
        --max-completion-length 1536 \
        --output baseline_results.json

Output:
    baseline_results.json with:
    - mean_reward, std_reward, min/max reward
    - action distribution (hold %, dispatch %, etc.)
    - per-episode breakdown
"""

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from server.constants import MAX_STEPS
from server.grpo_env_wrapper import DispatchRGRPOEnv
from server.prompt_utils import SYSTEM_PROMPT, build_chat_prompt, format_observation


def run_baseline_batched(
    llm,
    tokenizer,
    episodes: int,
    batch_size: int,
    max_completion_length: int,
    difficulty: str,
    seed: int = 42,
) -> Dict:
    """Run batched baseline episodes using vLLM.

    Returns dict with rewards, action stats, and per-episode details.
    """
    rewards = []
    action_counts = {"hold": 0, "dispatch": 0, "stage": 0, "reroute": 0,
                     "verify": 0, "divert": 0, "request_mutual_aid": 0, "log": 0}
    episode_details = []
    total_actions = 0

    sampling_params = SamplingParams(
        max_tokens=max_completion_length,
        temperature=0.7,
        top_p=0.9,
    )

    num_batches = (episodes + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_ep = batch_idx * batch_size
        end_ep = min(start_ep + batch_size, episodes)
        actual_batch = end_ep - start_ep

        # Initialize environments
        envs = []
        for i in range(actual_batch):
            env = DispatchRGRPOEnv(seed=seed + start_ep + i, difficulty=difficulty)
            env.reset(seed=seed + start_ep + i, difficulty=difficulty)
            envs.append(env)

        # Track which envs are still active
        active = list(range(actual_batch))
        batch_episodes = [[] for _ in range(actual_batch)]

        print(f"\n--- Batch {batch_idx + 1}/{num_batches} (episodes {start_ep + 1}-{end_ep}) ---")

        step_idx = 0
        while active and step_idx < MAX_STEPS:
            # Collect prompts from active envs
            prompts = []
            for idx in active:
                obs_text = format_observation(envs[idx]._obs)
                prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, obs_text)
                prompts.append(prompt)

            # Batched generation via vLLM
            outputs = llm.generate(prompts, sampling_params)

            # Step each env and update active list
            new_active = []
            for i, idx in enumerate(active):
                completion = outputs[i].outputs[0].text
                parsed = envs[idx]._parse_action(completion)
                envs[idx].step(completion)

                atype = parsed.get("action_type", "hold")
                action_counts[atype] = action_counts.get(atype, 0) + 1
                total_actions += 1

                batch_episodes[idx].append({
                    "step": step_idx,
                    "action": atype,
                    "completion": completion[:200],  # truncated
                })

                if not envs[idx]._obs.get("done"):
                    new_active.append(idx)

            active = new_active
            step_idx += 1

            if step_idx % 20 == 0:
                print(f"  Step {step_idx}, active envs: {len(active)}")

        # Collect rewards
        for i in range(actual_batch):
            reward = envs[i].reward if envs[i].reward is not None else 0.0
            rewards.append(reward)
            episode_details.append({
                "episode": start_ep + i + 1,
                "seed": seed + start_ep + i,
                "reward": float(reward),
                "steps": len(batch_episodes[i]),
                "parse_failures": envs[i].parse_failures,
            })
            print(f"  Episode {start_ep + i + 1}: reward={reward:.3f} | steps={len(batch_episodes[i])}")

    # Compute stats
    rewards_arr = np.array(rewards)
    action_dist = {k: v / max(1, total_actions) for k, v in action_counts.items()}

    return {
        "config": {
            "model": llm.llm_engine.model_config.model,
            "episodes": episodes,
            "batch_size": batch_size,
            "max_completion_length": max_completion_length,
            "difficulty": difficulty,
            "seed": seed,
        },
        "summary": {
            "mean_reward": float(rewards_arr.mean()),
            "std_reward": float(rewards_arr.std()),
            "min_reward": float(rewards_arr.min()),
            "max_reward": float(rewards_arr.max()),
            "median_reward": float(np.median(rewards_arr)),
            "total_actions": total_actions,
        },
        "action_distribution": action_dist,
        "episodes": episode_details,
    }


def main():
    parser = argparse.ArgumentParser(description="DispatchR baseline with vLLM")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Episodes to run in parallel")
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--difficulty", default="learning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="baseline_results.json")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="vLLM GPU memory fraction (0.9 for inference-only)")
    args = parser.parse_args()

    if LLM is None or SamplingParams is None:
        print("[ERROR] vLLM not installed. Install: pip install vllm")
        return

    print("=" * 60)
    print("DispatchR Baseline Evaluation (vLLM)")
    print(f"  Model: {args.model}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max completion length: {args.max_completion_length}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load vLLM
    print("\nLoading vLLM...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        trust_remote_code=True,
    )

    # Run baseline
    t0 = time.time()
    results = run_baseline_batched(
        llm=llm,
        tokenizer=tokenizer,
        episodes=args.episodes,
        batch_size=args.batch_size,
        max_completion_length=args.max_completion_length,
        difficulty=args.difficulty,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    s = results["summary"]
    print(f"  Mean reward:   {s['mean_reward']:.4f}")
    print(f"  Std reward:    {s['std_reward']:.4f}")
    print(f"  Min/Max:       {s['min_reward']:.4f} / {s['max_reward']:.4f}")
    print(f"  Median:        {s['median_reward']:.4f}")
    print(f"  Total actions: {s['total_actions']}")
    print(f"  Time:          {elapsed:.1f}s ({elapsed / args.episodes:.1f}s/episode)")
    print("\n  Action Distribution:")
    for action, pct in results["action_distribution"].items():
        bar = "█" * int(pct * 40)
        print(f"    {action:20s} {pct:6.2%} {bar}")
    print("=" * 60)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
