#!/usr/bin/env python3
"""TRL-native GRPO training for DispatchR — optimised for A100 80GB.

Replaces the hand-rolled loop with TRL's GRPOTrainer, which handles:
  - vLLM rollout generation (use_vllm=True) — single biggest speedup
  - Gradient accumulation, mixed precision, distributed training
  - Proper GRPO loss with PPO clipping, batch normalisation
  - Logging to wandb / tensorboard out of the box

Key design decisions vs the hand-rolled loop:
  - No custom micro-batching — GRPOTrainer handles this via
    per_device_train_batch_size + gradient_accumulation_steps
  - No manual log-prob loop — TRL computes these internally,
    correctly vectorised
  - Dataset is a list of seed dicts — see THE DATASET PROBLEM below

THE DATASET PROBLEM (why GRPOTrainer needs a dataset at all)
─────────────────────────────────────────────────────────────
TRL's GRPOTrainer inherits from Trainer, which is fundamentally
dataset-driven. It iterates over a DataLoader, draws a batch of rows,
and for each row calls the model to generate completions, then scores
them with your reward function.

For supervised learning this is obvious. For RL it feels wrong — why
do you need a dataset when the environment generates experience on the
fly?

The answer is that the "dataset" here is not training data in the
traditional sense. It is a *seed manifest*: a list of episode
configurations (seed, difficulty) that the trainer cycles through to
generate rollouts. Each "row" is just a specification of which episode
to run next.

Concretely, each row contains:
  - "prompt": the *initial* observation formatted as a chat string.
    GRPOTrainer feeds this to the model (via vLLM) to get the first
    action. For a multi-step env like DispatchR this is only step 0;
    subsequent steps are handled inside the reward function callback.
  - "seed": passed into the reward function so it can reconstruct the
    exact episode the prompt came from.
  - "difficulty": curriculum phase label.

The reward function then runs the *full* episode (all 80 steps) using
the seed to reset the env, collects the model's actions by calling
generate() itself (or using the completions TRL passes in for step 0),
and returns the episode-level scalar reward.

This means TRL effectively sees each episode as a single-turn
interaction: prompt → completion → reward. The multi-step loop lives
inside the reward function, not in TRL's training loop.

Why not use TRL's environment_factory / tool-calling mode instead?
That mode works but requires the model to emit structured tool calls
(XML or JSON schema), and the env steps happen *inside TRL's generation
loop*. It is harder to debug, harder to mix with vLLM, and loses the
batched-episode parallelism we want. The single-turn reward-function
approach is simpler and runs faster on A100 with vLLM.

Dataset sizing: you want enough seeds that the model sees varied
episodes, but the dataset is cycled (repeat=True), so even 500-1000
seeds is fine. We generate them procedurally — no download needed.

Usage (HF Jobs, A100 Large):
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
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from server.constants import CURRICULUM_PHASES, MAX_STEPS
from server.grpo_env_wrapper import DispatchRGRPOEnv
from server.prompt_utils import SYSTEM_PROMPT, build_chat_prompt, format_observation


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════


def build_seed_dataset(
    n_seeds: int,
    difficulty: str,
    base_seed: int = 0,
    tokenizer=None,
) -> Dataset:
    """Build a seed manifest dataset for GRPOTrainer.

    Each row is one episode specification. GRPOTrainer will draw rows in
    order (cycling when exhausted), pass the "prompt" to the model for
    generation, and hand the completion + row metadata to the reward fn.

    Args:
        n_seeds:    Number of distinct episode seeds. 500-1000 is a good
                    range — enough variety to prevent overfitting to a
                    handful of city configurations.
        difficulty: Starting curriculum phase ("warmup", "learning",
                    "advanced", "expert"). The reward function will use
                    this; you can also update it mid-training via a
                    callback (see CurriculumCallback below).
        base_seed:  Offset so resumed runs don't repeat seeds 0..n.
        tokenizer:  If provided, the initial prompt is pre-formatted with
                    the chat template. Otherwise a raw string is stored
                    and formatted at reward-function time.

    Returns:
        HuggingFace Dataset with columns: seed, difficulty, prompt.
    """
    rows = []
    for i in range(n_seeds):
        seed = base_seed + i

        # Spin up a throwaway env just to get the initial observation.
        # This is cheap — no model involved yet.
        env = DispatchRGRPOEnv(seed=seed, difficulty=difficulty)
        env.reset()
        obs_text = format_observation(env._obs)

        if tokenizer is not None:
            prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, obs_text)
        else:
            prompt = obs_text  # reward fn will format with tokenizer

        rows.append(
            {
                "seed": seed,
                "difficulty": difficulty,
                "prompt": prompt,
            }
        )

    return Dataset.from_list(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


def make_reward_fn(
    tokenizer,
    max_steps: int = MAX_STEPS,
    max_new_tokens: int = 512,
    trajectory_path: str | None = None,
):
    """Factory that returns the reward function GRPOTrainer expects.

    GRPOTrainer calls reward_fn(prompts, completions, **kwargs) where:
      - prompts:     list[str]  — the "prompt" column from the dataset batch
      - completions: list[str]  — model generations for step 0
      - kwargs:      the rest of the dataset columns (seed, difficulty, etc.)

    The reward function must return list[float], one per item in the batch.

    For DispatchR this function:
      1. Resets the env with the row's seed (reproducing the exact episode
         the prompt came from — this is why we store the seed in the dataset).
      2. Uses the step-0 completion TRL already generated.
      3. Runs steps 1..max_steps by calling generate() via the policy model
         handle TRL passes in via the `model` kwarg.
      4. Returns the episode reward scalar.

    NOTE: steps 1..N are generated inside this function, not by TRL.
    TRL only generates step 0. This means vLLM's continuous batching
    is used for step 0 (where TRL controls generation) but NOT for
    steps 1..N (where we call sequentially). For DispatchR's short JSON
    completions this is acceptable — each step generates ~30-50 tokens.
    The main speedup from vLLM is still in TRL's outer generation loop
    across the batch.

    Args:
        tokenizer:        The model tokenizer.
        max_steps:        Episode length (default MAX_STEPS = 80).
        max_new_tokens:   Token budget per action (default 512).
        trajectory_path:  Optional JSONL path to log full episode traces.
    """
    # Trajectory writer (append mode so crashes don't lose data)
    traj_file = None
    if trajectory_path:
        os.makedirs(os.path.dirname(trajectory_path) or ".", exist_ok=True)
        traj_file = open(trajectory_path, "a", encoding="utf-8")

    def _generate_action(model_ref, obs_text: str, device: str) -> str:
        """Generate a single action using the reference model (steps 1..N)."""
        prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, obs_text)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)
        with torch.no_grad():
            out = model_ref.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0, inputs.input_ids.shape[1] :]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        model=None,  # GRPOTrainer passes the current policy model
        **kwargs,
    ) -> list[float]:
        seeds = kwargs.get("seed", [42] * len(prompts))
        difficulty = kwargs.get("difficulty", ["learning"] * len(prompts))
        device = (
            next(model.parameters()).device if model is not None else "cuda"
        )

        rewards = []

        for i, (prompt, step0_completion) in enumerate(zip(prompts, completions)):
            seed = int(seeds[i]) if not isinstance(seeds[i], int) else seeds[i]
            diff = difficulty[i] if isinstance(difficulty, list) else difficulty

            env = DispatchRGRPOEnv(seed=seed, difficulty=diff)
            # env.reset() was already called during dataset construction
            # but we need a fresh env here — reset re-seeds internally
            env.reset()

            episode_trace = []

            # ── Step 0: use TRL's generation ──────────────────────────
            env.step(step0_completion)
            episode_trace.append(
                {
                    "step": 0,
                    "prompt": prompt,
                    "completion": step0_completion,
                }
            )

            # ── Steps 1..N: generate ourselves ────────────────────────
            for step_idx in range(1, max_steps):
                if env._obs and env._obs.get("done"):
                    break

                obs_text = format_observation(env._obs)

                if model is not None:
                    completion = _generate_action(
                        model, obs_text, str(device)
                    )
                else:
                    # Fallback: greedy hold if no model reference
                    completion = '{"action_type":"hold"}'

                env.step(completion)
                episode_trace.append(
                    {
                        "step": step_idx,
                        "prompt": obs_text,
                        "completion": completion,
                    }
                )

            episode_reward = (
                env.reward if env.reward is not None else 0.0
            )
            rewards.append(float(episode_reward))

            # Write trajectory
            if traj_file:
                traj_file.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "difficulty": diff,
                            "reward": episode_reward,
                            "steps": episode_trace,
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

    Usage:
        callback = CurriculumCallback(trainer, phases=[...], threshold=0.65)
        # call callback.on_batch_end(mean_reward) after each logged step
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
    ):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.n_seeds = n_seeds
        self.phases = phases or self.PHASES
        self.threshold = threshold
        self.min_steps = min_steps
        self.window = window

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
        "--max-completion-length", type=int, default=512
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── GPU check + dynamic vLLM memory split ────────────────────────
    vllm_mem_util = 0.4  # default for A100 80GB
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram_gb:.0f} GB)")
        if vram_gb < 70:
            # L40S (48GB) or smaller — give vLLM more room since model is smaller
            vllm_mem_util = 0.55
            print(f"  Detected mid-range GPU — vLLM memory util = {vllm_mem_util}")
        if vram_gb < 30 and not args.no_vllm:
            print(
                "WARN: <30GB VRAM detected. vLLM colocate may OOM. "
                "Consider --no-vllm or a larger GPU."
            )
    else:
        print("No CUDA GPU found — training on CPU (very slow).")

    # ── Tokenizer ─────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────────
    print(
        f"Building seed dataset: {args.n_seeds} seeds, "
        f"difficulty={args.difficulty}"
    )
    t0 = time.time()
    dataset = build_seed_dataset(
        n_seeds=args.n_seeds,
        difficulty=args.difficulty,
        tokenizer=tokenizer,
    )
    print(f"Dataset ready in {time.time() - t0:.1f}s  ({len(dataset)} rows)")
    print(f"Sample row keys: {list(dataset[0].keys())}")
    print(
        f"Sample prompt (first 200 chars):\n  "
        f"{dataset[0]['prompt'][:200]}...\n"
    )

    # ── Reward function ───────────────────────────────────────────────
    reward_fn = make_reward_fn(
        tokenizer=tokenizer,
        max_steps=MAX_STEPS,
        max_new_tokens=args.max_completion_length,
        trajectory_path=args.trajectory_file,
    )

    # ── Rollout function (multi-step via vLLM for all steps) ──────────
    # Unlike the reward_fn approach (which only uses vLLM for step 0),
    # rollout_func routes ALL steps through vLLM via generate_rollout_completions.
    # This is what the kube-sre-gym hackathon winner used and is the correct
    # pattern for multi-step environments in TRL >= 0.27.
    #
    # IMPORTANT: generate_rollout_completions is only available in
    # trl>=0.27 with experimental openenv support. If not available we
    # fall back to the reward_fn approach (steps 1..N via HF generate).
    try:
        from trl.experimental.openenv import generate_rollout_completions

        _has_openenv = True
    except ImportError:
        _has_openenv = False
        print(
            "WARN: trl.experimental.openenv not available — "
            "falling back to reward_fn approach."
        )
        print(
            "      Install trl>=0.27 for full multi-step vLLM rollout support."
        )

    # Counter for deterministic seeds when rollout_func can't access dataset rows
    _rollout_counter = 0

    def rollout_func(prompts: list, trainer) -> dict:
        """Run full DispatchR episodes with all steps going through vLLM.

        Each call processes one batch of episodes (len(prompts) == batch_size).
        Steps 1..N use generate_rollout_completions which talks to the
        colocated vLLM engine.

        NOTE: seeds are drawn from a global counter because rollout_func
        receives prompt strings, not the raw dataset rows.  Episode
        variety comes from the counter, not from the dataset seeds.
        """
        global _rollout_counter
        keys = ["prompt_ids", "completion_ids", "logprobs", "reward"]
        results = {k: [] for k in keys}

        for i, _prompt in enumerate(prompts):
            seed = args.seed + _rollout_counter
            diff = args.difficulty
            _rollout_counter += 1

            env = DispatchRGRPOEnv(seed=seed, difficulty=diff)
            env.reset()

            all_prompt_ids = []
            all_completion_ids = []
            all_logprobs = []

            for step_idx in range(MAX_STEPS):
                if env._obs and env._obs.get("done"):
                    break

                obs_text = format_observation(env._obs)
                step_prompt = build_chat_prompt(
                    tokenizer, SYSTEM_PROMPT, obs_text
                )

                # Single-step vLLM generation (batched across
                # rollout_func calls when trainer calls with a batch)
                if _has_openenv:
                    step_outputs = generate_rollout_completions(
                        trainer, [step_prompt]
                    )
                    out = step_outputs[0]
                    completion_text = tokenizer.decode(
                        out["completion_ids"], skip_special_tokens=True
                    )
                    all_prompt_ids.append(out["prompt_ids"])
                    all_completion_ids.append(out["completion_ids"])
                    all_logprobs.append(out.get("logprobs", []))
                else:
                    # Fallback: use trainer's underlying model directly
                    device = next(trainer.model.parameters()).device
                    inputs = tokenizer(
                        step_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048,
                    ).to(device)
                    with torch.no_grad():
                        out_ids = trainer.model.generate(
                            **inputs,
                            max_new_tokens=args.max_completion_length,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    gen = out_ids[0, inputs.input_ids.shape[1] :]
                    completion_text = tokenizer.decode(
                        gen, skip_special_tokens=True
                    )
                    all_prompt_ids.append(inputs.input_ids[0])
                    all_completion_ids.append(gen)
                    all_logprobs.append([])

                env.step(completion_text)

            episode_reward = (
                float(env.reward) if env.reward is not None else 0.0
            )

            # TRL expects one (prompt_ids, completion_ids) pair per sample.
            # For multi-step we return the step-0 prompt and the concatenation
            # of all completions so the policy is updated on the full action
            # sequence.
            results["prompt_ids"].append(
                all_prompt_ids[0] if all_prompt_ids else torch.tensor([])
            )
            results["completion_ids"].append(
                torch.cat(all_completion_ids)
                if all_completion_ids
                else torch.tensor([])
            )
            results["logprobs"].append(
                all_logprobs[0] if all_logprobs else []
            )
            results["reward"].append(episode_reward)

        return results

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
        num_generations=args.batch_size,  # rollouts per GRPO group
        generation_batch_size=args.batch_size,  # completions vLLM generates per fwd pass
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",  # vLLM shares GPU with trainer (no extra server)
        vllm_gpu_memory_utilization=vllm_mem_util,  # dynamic: 0.55 on 48GB, 0.4 on 80GB
        vllm_max_model_len=4096,  # prompt + completion context window
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
        report_to="none",  # set "wandb" if you want W&B
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
        rollout_func=rollout_func if _has_openenv else None,
    )

    # ── Curriculum callback wiring ────────────────────────────────────
    curriculum = None
    if args.curriculum:
        curriculum = CurriculumCallback(
            trainer=trainer,
            tokenizer=tokenizer,
            n_seeds=args.n_seeds,
            threshold=args.curriculum_threshold,
        )
        print(
            f"🎓 Curriculum enabled (threshold={args.curriculum_threshold})"
        )

        # Monkey-patch the log method to intercept reward metrics
        _orig_log = trainer.log

        def _log_with_curriculum(logs: dict[str, Any], **kw):
            _orig_log(logs, **kw)
            mean_reward = logs.get(
                "reward", logs.get("train/reward", None)
            )
            if mean_reward is not None and curriculum is not None:
                curriculum.on_batch_end(float(mean_reward))

        trainer.log = _log_with_curriculum

    # ── Train ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("Starting training")
    print("=" * 60)
    trainer.train()

    # ── Final save ────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved final model to {final_path}")

    if args.push_to_hub and args.hub_model_id:
        trainer.push_to_hub(
            commit_message="Final checkpoint — training complete"
        )
        print(
            f"Pushed to https://huggingface.co/{args.hub_model_id}"
        )


if __name__ == "__main__":
    main()
