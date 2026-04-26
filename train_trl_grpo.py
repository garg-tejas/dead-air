#!/usr/bin/env python3
"""TRL-native GRPO training for DispatchR — with trajectory cache.

Uses TRL's GRPOTrainer with vLLM colocation, but trains on *random steps*
from pre-computed greedy trajectories instead of only step 0.

Architecture:
  1. Build (or load) a trajectory cache: run greedy episodes, save actions.
  2. Sample random steps from cache as dataset rows.
  3. TRL generates one action for the sampled observation.
  4. Reward fn replays cached actions up to step t, injects model action,
     then continues with greedy reference policy to episode end.
  5. Episode reward is returned — variance comes from model vs greedy.

This creates non-zero reward variance within GRPO groups, enabling
actual learning. The model learns to map ANY observation to a good
action, not just the empty-city step-0 state.

Usage (HF Jobs, L40S):
    python train_trl_grpo.py \\
        --model Qwen/Qwen3-4B \\
        --episodes 500 \\
        --batch-size 8 \\
        --output-dir /data/outputs \\
        --push-to-hub \\
        --hub-model-id username/dispatchr-grpo

Usage (local, debug):
    python train_trl_grpo.py \\
        --model Qwen/Qwen3-4B \\
        --episodes 50 \\
        --batch-size 4 \\
        --no-vllm \\
        --output-dir ./outputs/debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import create_repo
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from server.constants import CURRICULUM_PHASES, MAX_STEPS
from server.grpo_env_wrapper import DispatchRGRPOEnv
from server.prompt_utils import SYSTEM_PROMPT, build_chat_prompt, format_observation
from server.greedy_policy import greedy_action
from server.trajectory_cache import (
    ensure_cache_exists,
    get_cache_path,
    replay_steps,
    iter_trajectory_cache,
)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════


def build_seed_dataset(
    n_seeds: int,
    difficulty: str,
    tokenizer=None,
    cache_dir: str = "./trajectory_cache",
    cache_workers: int = 8,
    steps_per_seed: int = 5,
    base_seed: int = 0,
) -> Dataset:
    """Build a dataset of random-step observations from greedy trajectories.

    1. Ensure trajectory cache exists (builds it in parallel if missing).
    2. Sample ``steps_per_seed`` random steps from each cached episode.
    3. Replay each episode to the sampled step, capture observation.
    4. Format observation as chat prompt.

    Each row represents a single decision point in a specific episode:
      - "prompt": the observation at step ``step_idx``
      - "seed": episode seed (for reproducible replay)
      - "step_idx": which step this row corresponds to
      - "difficulty": curriculum phase

    Args:
        n_seeds:         Number of episode seeds in cache.
        difficulty:      Curriculum phase.
        tokenizer:       HF tokenizer for chat template formatting.
        cache_dir:       Where to store/load trajectory cache.
        cache_workers:   Parallel workers for cache building.
        steps_per_seed:  How many random steps to sample per episode.
        base_seed:       Seed offset.

    Returns:
        HuggingFace Dataset with columns: seed, difficulty, step_idx, prompt.
    """
    import numpy as np

    cache_path = ensure_cache_exists(
        cache_dir=cache_dir,
        n_seeds=n_seeds,
        difficulty=difficulty,
        base_seed=base_seed,
        num_workers=cache_workers,
    )

    rows = []
    for episode in iter_trajectory_cache(cache_path):
        seed = episode["seed"]
        diff = episode["difficulty"]
        cached_steps = episode["steps"]
        n_steps = len(cached_steps)

        if n_steps <= 1:
            continue

        upper = min(79, n_steps - 1)
        if upper < 1:
            continue

        # Vectorized sampling: choose K random step indices
        k = min(steps_per_seed, upper)
        sampled_indices = np.random.choice(
            np.arange(1, upper + 1),
            size=k,
            replace=False,
        )

        # Replay episode once, capturing observations at sampled steps
        env = DispatchRGRPOEnv(seed=seed, difficulty=diff)
        env.reset(seed=seed, difficulty=diff)

        for step_idx in range(n_steps):
            obs = env._obs
            if obs is None or obs.get("done"):
                break

            if step_idx in sampled_indices:
                obs_text = format_observation(obs)
                if tokenizer is not None:
                    prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, obs_text)
                else:
                    prompt = obs_text
                rows.append(
                    {
                        "seed": seed,
                        "difficulty": diff,
                        "step_idx": int(step_idx),
                        "prompt": prompt,
                    }
                )

            # Advance env using cached action
            if step_idx < len(cached_steps):
                action = cached_steps[step_idx]["action"]
                env.step(json.dumps(action))

    return Dataset.from_list(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


def make_reward_fn(
    tokenizer,
    cache_path: str,
    max_steps: int = MAX_STEPS,
    max_new_tokens: int = 512,
    trajectory_path: str | None = None,
):
    """Factory that returns the reward function GRPOTrainer expects.

    For each dataset row (which represents a random step in an episode):
      1. Reset env with the row's seed.
      2. Replay cached greedy actions for steps 0 .. step_idx-1.
      3. Apply the model's completion (TRL-generated) at step_idx.
      4. Continue steps step_idx+1 .. end with greedy reference policy.
      5. Return episode reward.

    The reward variance comes from: model action vs greedy action at step_idx,
    propagated through the rest of the episode.

    Args:
        tokenizer:        The model tokenizer.
        cache_path:       Path to trajectory cache JSONL.
        max_steps:        Episode length (default MAX_STEPS = 80).
        max_new_tokens:   Token budget per action (default 512).
        trajectory_path:  Optional JSONL path to log full episode traces.
    """
    # Load cache into memory for fast lookup
    cache_by_seed: dict[int, list[dict]] = {}
    for episode in iter_trajectory_cache(cache_path):
        cache_by_seed[int(episode["seed"])] = episode["steps"]

    # Trajectory writer
    traj_file = None
    if trajectory_path:
        os.makedirs(os.path.dirname(trajectory_path) or ".", exist_ok=True)
        traj_file = open(trajectory_path, "a", encoding="utf-8")

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        model=None,
        **kwargs,
    ) -> list[float]:
        seeds = kwargs.get("seed", [42] * len(prompts))
        difficulty = kwargs.get("difficulty", ["learning"] * len(prompts))
        step_indices = kwargs.get("step_idx", [0] * len(prompts))

        rewards = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            seed = int(seeds[i]) if not isinstance(seeds[i], int) else seeds[i]
            diff = difficulty[i] if isinstance(difficulty, list) else difficulty
            step_idx = int(step_indices[i]) if isinstance(step_indices, list) else int(step_indices)

            cached_steps = cache_by_seed.get(seed, [])

            env = DispatchRGRPOEnv(seed=seed, difficulty=diff)
            env.reset(seed=seed, difficulty=diff)

            # Episode header
            print(
                f"\n📋 Episode {i+1}/{len(prompts)} | seed={seed} | "
                f"diff={diff} | step_idx={step_idx}"
            )

            # ── Replay cached greedy actions up to step_idx ───────────
            replay_steps(env, cached_steps, up_to_step=step_idx)

            # If episode already ended during replay (cached episode was shorter),
            # skip injecting model action
            if env._obs and env._obs.get("done"):
                episode_reward = env.reward if env.reward is not None else 0.0
                rewards.append(float(episode_reward))
                print(f"  ⚠️ Episode ended early at step {step_idx}, skipping model action")
                continue

            # ── Inject model's action at step_idx ─────────────────────
            parsed = env._parse_action(completion)
            env.step(completion)
            print(f"  [Step {step_idx}] MODEL: {env._format_step_log(parsed, completion)}")

            # ── Continue to episode end with greedy reference ─────────
            for cont_step in range(step_idx + 1, max_steps):
                if env._obs and env._obs.get("done"):
                    break
                action = greedy_action(env._obs)
                env.step(json.dumps(action))
                # Only print first and last few continuation steps
                if cont_step <= step_idx + 3 or cont_step >= max_steps - 3:
                    print(f"  [Step {cont_step}] GREEDY: {env._format_step_log(action)}")
                elif cont_step == step_idx + 4:
                    print(f"  ... ({max_steps - step_idx - 8} greedy steps omitted) ...")

            episode_reward = env.reward if env.reward is not None else 0.0
            rewards.append(float(episode_reward))

            print(
                f"  ✅ DONE | reward={episode_reward:.3f} | "
                f"parse_fails={env.parse_failures}"
            )

            # Write trajectory
            if traj_file:
                traj_file.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "difficulty": diff,
                            "step_idx": step_idx,
                            "reward": episode_reward,
                            "model_action": completion,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                traj_file.flush()

        return rewards

    return reward_fn


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CURRICULUM CALLBACK
# ══════════════════════════════════════════════════════════════════════════════


class CurriculumCallback:
    """Update dataset difficulty mid-training based on reward threshold.

    GRPOTrainer doesn't natively support mid-training dataset swaps, so
    we track reward history ourselves and swap the trainer's dataset when
    the escalation condition is met.
    """

    PHASES = ["warmup", "learning", "advanced", "expert"]

    def __init__(
        self,
        trainer: GRPOTrainer,
        tokenizer,
        n_seeds: int,
        phases: list[str] | None = None,
        threshold: float = 0.65,
        min_steps: int = 30,
        window: int = 3,
        cache_dir: str = "./trajectory_cache",
        cache_workers: int = 8,
        steps_per_seed: int = 5,
    ):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.n_seeds = n_seeds
        self.phases = phases or self.PHASES
        self.threshold = threshold
        self.min_steps = min_steps
        self.window = window
        self.cache_dir = cache_dir
        self.cache_workers = cache_workers
        self.steps_per_seed = steps_per_seed

        self.phase_idx = 0
        self.steps_in_phase = 0
        self.reward_buf: list[float] = []

    @property
    def current_phase(self) -> str:
        return self.phases[self.phase_idx]

    def on_batch_end(self, mean_reward: float) -> bool:
        """Call after each batch. Returns True if phase escalated."""
        self.steps_in_phase += 1
        self.reward_buf.append(mean_reward)
        if len(self.reward_buf) > self.window:
            self.reward_buf.pop(0)

        if (
            self.phase_idx < len(self.phases) - 1
            and self.steps_in_phase >= self.min_steps
            and len(self.reward_buf) >= self.window
            and (sum(self.reward_buf) / len(self.reward_buf))
            >= self.threshold
        ):
            self.phase_idx += 1
            self.steps_in_phase = 0
            self.reward_buf = []

            new_phase = self.phases[self.phase_idx]
            print(
                f"\n🎓 CURRICULUM → {new_phase} "
                f"(reward avg >= {self.threshold})"
            )

            # Rebuild dataset with new difficulty and swap it in
            new_dataset = build_seed_dataset(
                n_seeds=self.n_seeds,
                difficulty=new_phase,
                tokenizer=self.tokenizer,
                cache_dir=self.cache_dir,
                cache_workers=self.cache_workers,
                steps_per_seed=self.steps_per_seed,
            )
            self.trainer.train_dataset = new_dataset
            return True

        return False


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="DispatchR GRPO training with TRL GRPOTrainer (A100-optimised)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        help=(
            "HF model ID. Use plain BF16 models with vLLM (e.g. Qwen/Qwen3-4B). "
            "BNB-quantized (bnb-4bit) models are NOT compatible with vLLM. "
            "On A100 80GB you have enough VRAM for 4B in BF16."
        ),
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Episodes per gradient step (num_generations in TRL).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Gradient accumulation steps. Effective batch = batch_size × grad_accum.",
    )
    parser.add_argument(
        "--difficulty", type=str, default="learning"
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1000,
        help="Number of episode seeds in the dataset manifest.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/trl_grpo"
    )
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=512,
        help="Max tokens per action. 512 is safe for 48GB GPUs. Use 1024+ only on 80GB A100.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vLLM (use HF generate instead). Slower but works anywhere.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning (auto-escalate difficulty on reward threshold).",
    )
    parser.add_argument(
        "--curriculum-threshold",
        type=float,
        default=0.65,
        help="Mean reward threshold for curriculum escalation.",
    )
    parser.add_argument(
        "--trajectory-file",
        type=str,
        default=None,
        help="JSONL file to log full episode traces.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print metrics every N training steps.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Save model checkpoint every N training steps.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/data/trajectory_cache",
        help="Directory to store/load trajectory cache. Use /data on HF Jobs for persistence.",
    )
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=8,
        help="Parallel workers for building trajectory cache. Match vCPU count (8 on L40S).",
    )
    parser.add_argument(
        "--steps-per-seed",
        type=int,
        default=7,
        help="How many random steps to sample per episode into the dataset. 7 x 1000 seeds = 7000 rows.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="GRPO group size (num completions per prompt). 8 is optimal for L40S 48GB with 512 tokens.",
    )
    # ── Experiment tracking ───────────────────────────────────────────
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dispatchr-grpo",
        help="Weights & Biases project name. Set to '' to disable wandb.",
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
        help="Custom wandb run name. Default: auto-generated from config.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (always saved to output_dir/runs).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Experiment tracking ───────────────────────────────────────────
    report_to = []
    if args.tensorboard:
        report_to.append("tensorboard")
    if args.wandb_project:
        try:
            import wandb
            run_name = args.wandb_run_name or (
                f"dispatchr-{args.difficulty}-"
                f"bs{args.batch_size}-ng{args.num_generations}-"
                f"lr{args.learning_rate}"
            )
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
            )
            report_to.append("wandb")
            print(f"📊 W&B: https://wandb.ai/{args.wandb_entity or '~'}/{args.wandb_project}/runs/{wandb.run.id}")
        except ImportError:
            print("[WARN] wandb not installed. Install: pip install wandb")
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")

    # ── GPU check + dynamic vLLM memory split ────────────────────────
    vllm_mem_util = 0.4  # default for A100 80GB
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram_gb:.0f} GB)")
        if vram_gb < 70:
            # L40S (48GB) — vLLM needs ~6GB KV cache for 40K context window;
            # model weights ~8GB, so vLLM needs at least 14GB total.
            vllm_mem_util = 0.35
            print(f"  Detected mid-range GPU — vLLM memory util = {vllm_mem_util}")
        if vram_gb < 30 and not args.no_vllm:
            print(
                "WARN: <30GB VRAM detected. vLLM colocate may OOM. "
                "Consider --no-vllm or a larger GPU."
            )
    else:
        print("No CUDA GPU found — training on CPU (very slow).")

    # Warn if token budget is too aggressive for the GPU
    if vram_gb < 70 and args.max_completion_length > 512:
        print(
            f"WARN: max_completion_length={args.max_completion_length} on {vram_gb:.0f}GB GPU "
            f"may OOM. Consider --max-completion-length 512 or --batch-size 4."
        )

    # ── Tokenizer ─────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────────
    print(
        f"Building dataset from trajectory cache: {args.n_seeds} seeds, "
        f"difficulty={args.difficulty}, steps_per_seed={args.steps_per_seed}"
    )
    t0 = time.time()
    dataset = build_seed_dataset(
        n_seeds=args.n_seeds,
        difficulty=args.difficulty,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        cache_workers=args.cache_workers,
        steps_per_seed=args.steps_per_seed,
    )
    print(f"Dataset ready in {time.time() - t0:.1f}s  ({len(dataset)} rows)")
    print(f"Sample row keys: {list(dataset[0].keys())}")
    print(
        f"Sample prompt (first 200 chars):\n  "
        f"{dataset[0]['prompt'][:200]}...\n"
    )

    # ── Reward function ───────────────────────────────────────────────
    cache_path = get_cache_path(args.cache_dir, args.difficulty, args.n_seeds)
    reward_fn = make_reward_fn(
        tokenizer=tokenizer,
        cache_path=cache_path,
        max_steps=MAX_STEPS,
        max_new_tokens=args.max_completion_length,
        trajectory_path=args.trajectory_file,
    )

    # ── Architecture note ─────────────────────────────────────────────
    # DispatchR has 80 steps with a NEW observation each step (calls appear,
    # units move, traffic changes). TRL's GRPOTrainer generates one completion
    # per dataset row and trains only on that completion's tokens.
    #
    # Our fix: each dataset row is a RANDOM STEP from a greedy trajectory.
    #   1. Build cache: run greedy episodes, save action sequences.
    #   2. Sample random steps: replay episode to step t, capture observation.
    #   3. TRL generates action for that observation (fast, batched via vLLM).
    #   4. reward_fn replays cached steps 0..t-1, injects model action at t,
    #      continues t+1..end with greedy reference. Returns episode reward.
    #
    # The model learns to map ANY observation to a good action, not just
    # the empty-city step-0 state. Reward variance comes from model vs greedy.

    # ── GRPOConfig ────────────────────────────────────────────────────
    # num_train_epochs is computed from episodes and batch_size.
    # TRL steps = ceil(len(dataset) / batch_size) * epochs.
    # We want approximately args.episodes gradient steps total.
    # Since dataset cycles, set epochs so steps ≈ episodes.
    steps_per_epoch = max(1, len(dataset) // args.batch_size)
    target_epochs = max(1, args.episodes // steps_per_epoch)

    grpo_config = GRPOConfig(
        # ── model / output ─────────────────────────────────────────
        output_dir=args.output_dir,
        run_name=f"dispatchr-{args.difficulty}",
        # ── generation ─────────────────────────────────────────────
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,  # GRPO group size (8 optimal for L40S 48GB)
        generation_batch_size=args.batch_size,  # completions vLLM generates per fwd pass
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",  # vLLM shares GPU with trainer (no extra server)
        vllm_gpu_memory_utilization=vllm_mem_util,  # dynamic: 0.35 on 48GB, 0.40 on 80GB
        temperature=0.7,
        top_p=0.9,
        # ── training ───────────────────────────────────────────────
        num_train_epochs=target_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=True,  # A100 supports BF16 natively
        fp16=False,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,  # required to fit colocated vLLM + training on 1 GPU
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # ── logging / saving ───────────────────────────────────────
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=report_to if report_to else "none",
        # ── GRPO-specific ──────────────────────────────────────────
        epsilon=0.2,  # PPO clip range
        beta=0.01,  # KL penalty weight
        # ── misc ───────────────────────────────────────────────────
        seed=args.seed,
        dataloader_num_workers=0,  # env objects aren't picklable
        remove_unused_columns=False,  # keep seed + difficulty columns
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private,
    )

    # ── LoRA config ───────────────────────────────────────────────────
    peft_config = None
    if args.lora_r > 0:
        try:
            from peft import LoraConfig, TaskType

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        except ImportError:
            print("WARN: peft not installed — training full model weights.")

    # ── Trainer ───────────────────────────────────────────────────────
    print(f"\nInitialising GRPOTrainer...")
    print(f"  use_vllm:    {not args.no_vllm}")
    print(f"  batch_size:  {args.batch_size}  (num_generations)")
    print(f"  grad_accum:  {args.grad_accum}")
    print(
        f"  epochs:      {target_epochs}  "
        f"(~{target_epochs * steps_per_epoch} steps)"
    )
    print(f"  bf16:        True")
    print()

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # ── Auto-create HF Hub repo if needed ─────────────────────────────
    if args.hub_model_id:
        try:
            create_repo(args.hub_model_id, repo_type="model", exist_ok=True)
            print(f"📦 HF Hub repo ready: https://huggingface.co/{args.hub_model_id}")
        except Exception as e:
            print(f"[WARN] Could not create HF repo: {e}")

    # ── Metrics logging + curriculum callback wiring ──────────────────
    curriculum = None
    _batch_counter = 0
    _metrics_history = []

    if args.curriculum:
        curriculum = CurriculumCallback(
            trainer=trainer,
            tokenizer=tokenizer,
            n_seeds=args.n_seeds,
            threshold=args.curriculum_threshold,
            cache_dir=args.cache_dir,
            cache_workers=args.cache_workers,
            steps_per_seed=args.steps_per_seed,
        )
        print(
            f"🎓 Curriculum enabled (threshold={args.curriculum_threshold})"
        )

    # Monkey-patch the log method to intercept metrics
    _orig_log = trainer.log

    def _log_with_metrics(logs, *_pos_args, **kw):
        _orig_log(logs, *_pos_args, **kw)
        nonlocal _batch_counter
        _batch_counter += 1

        mean_reward = logs.get("reward", logs.get("train/reward", None))
        loss = logs.get("loss", logs.get("train/loss", None))

        # Print metrics
        if _batch_counter % args.log_every == 0:
            r_str = f"{mean_reward:.4f}" if mean_reward is not None else "N/A"
            l_str = f"{loss:.4f}" if loss is not None else "N/A"
            print(
                f"\n📊 BATCH {_batch_counter} | reward={r_str} | loss={l_str}"
            )

        # Save metrics to JSONL
        metric_row = {
            "batch": _batch_counter,
            "reward": float(mean_reward) if mean_reward is not None else None,
            "loss": float(loss) if loss is not None else None,
        }
        _metrics_history.append(metric_row)

        metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_row) + "\n")

        # Curriculum escalation check
        if mean_reward is not None and curriculum is not None:
            curriculum.on_batch_end(float(mean_reward))

        # Periodic model checkpoint save + HF Hub push
        if _batch_counter % args.save_every == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint-{_batch_counter}"
            )
            trainer.save_model(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"💾 Saved checkpoint to {ckpt_path}")

            if args.hub_model_id:
                try:
                    trainer.push_to_hub(
                        commit_message=f"Checkpoint {_batch_counter}"
                    )
                    print(f"🚀 Pushed checkpoint-{_batch_counter} to HF Hub")
                except Exception as e:
                    print(f"[WARN] HF push failed: {e}")

    trainer.log = _log_with_metrics

    # ── Train ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("DispatchR TRL GRPO Trainer")
    print("  Version: 2026-04-26-random-step-cache")
    print("  Features: random-step-dataset | trajectory-cache | greedy-reference")
    print("=" * 60)
    trainer.train()

    # ── Final save ────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved final model to {final_path}")

    if args.hub_model_id:
        try:
            trainer.push_to_hub(
                commit_message="Final checkpoint — training complete"
            )
            print(
                f"🚀 Pushed final model to https://huggingface.co/{args.hub_model_id}"
            )
        except Exception as e:
            print(f"[WARN] Final HF push failed: {e}")


if __name__ == "__main__":
    main()
