"""GRPO training with TRL for Dead Air environment.

Usage (on Lightning AI L4 or H100):
    uv sync --extra train
    accelerate launch train_grpo.py --model Qwen/Qwen3-1.7B --episodes 200

Requirements:
    - torch>=2.2.0
    - trl>=1.0.0
    - transformers>=5.2.0  (required for environment_factory)
    - accelerate>=0.30.0
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset


def make_reward_func(difficulty: str = "learning"):
    """Create a reward function that reads episode reward from environment state."""

    def reward_func(environments: List[Any], **kwargs: Any) -> List[float]:
        """Return episode reward for each environment instance."""
        rewards = []
        for env in environments:
            r = env.reward
            if r is None:
                # Episode hasn't ended yet — penalize incomplete episodes
                rewards.append(0.0)
            else:
                rewards.append(float(r))
        return rewards

    return reward_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--difficulty", type=str, default="curriculum")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Dead Air GRPO Training")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    # Imports here so script can be parsed even without TRL installed
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise ImportError(
            "TRL is required for training. Install with: uv sync --extra train"
        ) from e

    # Create dummy dataset with repeated prompts.
    # The actual state is injected by environment.reset() each episode.
    system_prompt = (
        "You are an emergency dispatch commander for a 20-node city. "
        "You have 6 units and must respond to emergency calls. "
        "Use the available tools to dispatch units, stage them, verify suspicious calls, "
        "request mutual aid, or hold. Minimize fatalities and response time. "
        "Think step by step before acting."
    )
    dataset = Dataset.from_dict({
        "prompt": [system_prompt] * args.episodes,
        "difficulty": [args.difficulty] * args.episodes,
    })

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=args.save_every,
        # L4 24GB VRAM settings
        use_vllm=True,
        vllm_gpu_memory_utilization=0.7,
        # Reward scaling: batch-level std reduces difficulty bias
        scale_rewards="batch",
        # Disable KL penalty (standard practice in modern GRPO)
        beta=0.0,
    )

    # Import wrapper inside main to avoid import errors when TRL not installed
    from dead_air.server.grpo_env_wrapper import DeadAirGRPOEnv

    reward_func = make_reward_func(difficulty=args.difficulty)

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=DeadAirGRPOEnv,
    )

    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    trainer.save_model(final_path)
    print(f"\nFinal model saved to {final_path}")


if __name__ == "__main__":
    main()
