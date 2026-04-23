"""Unsloth-based GRPO training for Dead Air.

Usage (on Lightning AI L4):
    uv pip install unsloth
    uv run python train_unsloth_grpo.py \
        --model unsloth/Qwen3.5-2B \
        --episodes 200 \
        --batch-size 8

This script replaces ``train_grpo.py`` when you want 2-5x faster
rollouts via Unsloth's optimized 4-bit kernels.  The old TRL-based
script is kept as a fallback.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from server.grpo_env_wrapper import DeadAirGRPOEnv
from server.unsloth_grpo_utils import (
    compute_episode_reward,
    compute_grpo_loss,
    generate_episode_step,
    should_enable_arrival_bonus,
)


def main():
    parser = argparse.ArgumentParser(
        description="Dead Air GRPO training with Unsloth"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Qwen3.5-2B",
        help="HF model id (default: unsloth/Qwen3.5-2B)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Total training episodes (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Parallel envs per batch (default: 8).  Reduce if OOM.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens per action completion (default: 512)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/unsloth_grpo",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="AdamW learning rate (default: 5e-6)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="learning",
        help="Difficulty phase (default: learning)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("Dead Air Unsloth GRPO Training")
    print(f"Model          : {args.model}")
    print(f"Episodes       : {args.episodes}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Max new tokens : {args.max_new_tokens}")
    print(f"Output dir     : {args.output_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1.  Load model via Unsloth
    # ------------------------------------------------------------------
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        print("\nERROR: Unsloth is not installed.")
        print("Install it with:")
        print("  uv pip install unsloth")
        print("\nOr fall back to the TRL trainer:")
        print("  uv run accelerate launch train_grpo.py --no-vllm ...")
        raise SystemExit(1) from exc

    print("\nLoading model (4-bit via Unsloth)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        load_in_4bit=True,
        max_seq_length=4096,
        dtype=torch.bfloat16,
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_rslora=True,          # rank-stabilized LoRA (good for RL)
        bias="none",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate
    )

    # Auto-detect bf16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("BF16 mixed precision: enabled")
    else:
        print("BF16 mixed precision: not available")

    # ------------------------------------------------------------------
    # 2.  Training loop
    # ------------------------------------------------------------------
    num_batches = max(1, args.episodes // args.batch_size)
    reward_history = []
    loss_history = []

    for batch_idx in range(num_batches):
        # ---- a)  Reset envs ----
        envs = [
            DeadAirGRPOEnv(
                seed=args.seed + batch_idx * args.batch_size + i,
                difficulty=args.difficulty,
            )
            for i in range(args.batch_size)
        ]
        for env in envs:
            env.reset()

        # ---- b)  Roll out episodes ----
        episodes_data = [[] for _ in range(args.batch_size)]

        for step in range(25):          # MAX_STEPS from constants
            active = [
                i
                for i, env in enumerate(envs)
                if env._obs is not None and not env._obs.get("done")
            ]
            if not active:
                break

            prompts = [
                envs[i]._format_prompt(envs[i]._obs) for i in active
            ]

            action_texts, old_log_probs, comp_ids_list, p_lens = (
                generate_episode_step(
                    model,
                    tokenizer,
                    prompts,
                    max_new_tokens=args.max_new_tokens,
                )
            )

            for idx, env_idx in enumerate(active):
                envs[env_idx].step(action_texts[idx])
                episodes_data[env_idx].append(
                    (
                        prompts[idx],
                        comp_ids_list[idx],
                        p_lens[idx],
                        old_log_probs[idx],
                    )
                )

        # ---- c)  Compute rewards ----
        enable_bonus = should_enable_arrival_bonus(reward_history)
        rewards = [
            compute_episode_reward(env, enable_bonus) for env in envs
        ]
        reward_history.extend(rewards)

        # ---- d)  GRPO update ----
        if any(episodes_data):
            loss = compute_grpo_loss(
                model, tokenizer, episodes_data, rewards
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_history.append(loss.item())
        else:
            loss_history.append(0.0)

        # ---- e)  Logging ----
        mean_reward = float(np.mean(rewards))
        print(
            f"Batch {batch_idx + 1:3d}/{num_batches} | "
            f"Mean Reward: {mean_reward:.4f} | "
            f"Loss: {loss_history[-1]:.4f} | "
            f"Arrival Bonus: {enable_bonus}"
        )

        # ---- f)  Checkpointing ----
        episodes_done = (batch_idx + 1) * args.batch_size
        if episodes_done % args.save_every == 0:
            save_path = os.path.join(
                args.output_dir, f"checkpoint-{episodes_done}"
            )
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  -> Saved checkpoint to {save_path}")

    # ------------------------------------------------------------------
    # 3.  Final save + metrics
    # ------------------------------------------------------------------
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")

    metrics = {
        "rewards": reward_history,
        "losses": loss_history,
        "config": vars(args),
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Print summary stats
    if reward_history:
        print(f"\nReward Statistics:")
        print(f"  Mean : {np.mean(reward_history):.4f}")
        print(f"  Std  : {np.std(reward_history):.4f}")
        print(f"  Min  : {np.min(reward_history):.4f}")
        print(f"  Max  : {np.max(reward_history):.4f}")


if __name__ == "__main__":
    main()
