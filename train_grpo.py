"""GRPO training with batched generation for DispatchR environment.

Compatible with TRL >= 0.15. Uses standard transformers + PEFT.
No vLLM required — batched transformers generation is fast enough for
24 GB L4 and avoids the vLLM/training memory conflict entirely.

Usage:
    python train_grpo.py --model Qwen/Qwen3.5-2B --episodes 200 --batch-size 8
    python train_grpo.py --model Qwen/Qwen3.5-2B --episodes 200 --batch-size 8 --use-4bit
"""

import argparse
import gc
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from server.constants import MAX_STEPS
from server.grpo_env_wrapper import DispatchRGRPOEnv
from server.prompt_utils import SYSTEM_PROMPT, build_chat_prompt, format_observation


def greedy_action(obs: Dict) -> Dict:
    """Greedy dispatch action: send closest idle unit to highest-priority call."""
    active_calls = obs.get("active_calls", [])
    unit_statuses = obs.get("unit_statuses", [])
    if not active_calls:
        return {"action_type": "hold"}
    priority = {"cardiac": 3, "trauma": 2, "fire": 1, "false_alarm": 0}
    sorted_calls = sorted(
        active_calls,
        key=lambda c: priority.get(c.get("reported_type", "trauma"), 1),
        reverse=True,
    )
    for call in sorted_calls:
        best_unit = None
        best_dist = float("inf")
        for u in unit_statuses:
            if u.get("last_known_status") == "idle":
                dist = abs(u.get("last_known_location", 0) - call["location"])
                if dist < best_dist:
                    best_dist = dist
                    best_unit = u["unit_id"]
        if best_unit is not None:
            return {
                "action_type": "dispatch",
                "unit_id": best_unit,
                "call_id": call["call_id"],
            }
    return {"action_type": "hold"}


def run_episodes_batched(
    model,
    tokenizer,
    envs: List[DispatchRGRPOEnv],
    max_steps: int,
    max_new_tokens: int,
    device: str,
    epsilon: float = 0.0,
    trajectory_records: List[Dict] = None,
    batch_offset: int = 0,
):
    """Run a batch of episodes with batched generation.

    Returns (episodes, rewards) where episodes is a list of step-lists.
    Each step dict contains:
        - prompt_ids: torch.LongTensor (on CPU)
        - completion_ids: torch.LongTensor (on CPU)
        - old_log_prob: torch.Tensor (scalar, detached)
    """
    batch_size = len(envs)
    for env in envs:
        env.reset()

    active = list(range(batch_size))
    all_episodes = [[] for _ in range(batch_size)]

    model.eval()
    for step_idx in range(max_steps):
        if not active:
            break

        # Collect prompts from still-active episodes
        prompts = []
        active_indices = []
        for idx in active:
            if envs[idx]._obs and envs[idx]._obs.get("done"):
                continue
            obs_text = format_observation(envs[idx]._obs)
            prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, obs_text)
            prompts.append(prompt)
            active_indices.append(idx)

        if not prompts:
            break

        # --- Batched tokenization & generation ---
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        prompt_lens = inputs.attention_mask.sum(dim=1).tolist()

        with torch.no_grad():
            output_sequences = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                max_length=None,  # Prevent conflict with model generation_config
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Extract per-sequence completions (truncate at first EOS)
        completions = []
        completion_ids_list = []
        for i in range(len(prompts)):
            p_len = prompt_lens[i]
            gen_ids = output_sequences[i, p_len:]
            eos_positions = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                gen_ids = gen_ids[: eos_positions[0].item() + 1]
            completion_ids_list.append(gen_ids)
            completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

        # --- Epsilon-greedy: replace some completions with greedy actions ---
        for i, idx in enumerate(active_indices):
            if (
                epsilon > 0
                and np.random.rand() < epsilon
                and envs[idx]._obs.get("active_calls")
            ):
                greedy_act = greedy_action(envs[idx]._obs)
                completions[i] = json.dumps(greedy_act)
                comp_ids = tokenizer(
                    completions[i],
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids[0].to(device)
                completion_ids_list[i] = comp_ids

        # --- Compute old log-probs in one batched forward pass ---
        prompt_ids_list = [
            inputs.input_ids[i, : prompt_lens[i]] for i in range(len(prompts))
        ]
        full_ids_list = [
            torch.cat([prompt_ids_list[i], completion_ids_list[i]])
            for i in range(len(prompts))
        ]

        max_full_len = max(t.shape[0] for t in full_ids_list)
        full_ids_batch = torch.full(
            (len(prompts), max_full_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(prompts), max_full_len),
            dtype=torch.long,
            device=device,
        )
        for i, ids in enumerate(full_ids_list):
            full_ids_batch[i, : ids.shape[0]] = ids
            attention_mask[i, : ids.shape[0]] = 1

        with torch.no_grad():
            model_outputs = model(
                input_ids=full_ids_batch, attention_mask=attention_mask
            )

        logits = model_outputs.logits[:, :-1, :]
        log_probs_all = F.log_softmax(logits, dim=-1)

        old_log_probs = []
        for i in range(len(prompts)):
            p_len = prompt_ids_list[i].shape[0]
            c_len = completion_ids_list[i].shape[0]
            if c_len == 0:
                old_log_probs.append(torch.tensor(0.0, device=device))
                continue

            token_lps = []
            for k in range(c_len):
                tok_id = full_ids_batch[i, p_len + k]
                # logits[:, t, :] predicts token at position t+1
                lp = log_probs_all[i, p_len + k - 1, tok_id]
                token_lps.append(lp)
            old_log_probs.append(torch.stack(token_lps).sum())

        # --- Step environments & store trajectory ---
        for i, idx in enumerate(active_indices):
            envs[idx].step(completions[i])

            if trajectory_records is not None:
                parsed = envs[idx]._parse_action(completions[i])
                action_type = (
                    parsed.get("action_type", "unknown")
                    if isinstance(parsed, dict)
                    else "unknown"
                )
                trajectory_records.append(
                    {
                        "episode": batch_offset + idx,
                        "step": step_idx,
                        "prompt": prompts[i],
                        "completion": completions[i],
                        "parsed_action": parsed,
                        "action_type": action_type,
                        "log_prob": float(old_log_probs[i].cpu()),
                        "episode_reward": None,  # back-filled later
                    }
                )

            # Move token IDs to CPU to free GPU memory
            all_episodes[idx].append(
                {
                    "prompt_ids": prompt_ids_list[i].cpu(),
                    "completion_ids": completion_ids_list[i].cpu(),
                    "old_log_prob": old_log_probs[i],
                }
            )

        # Update active list
        active = [idx for idx in active if not envs[idx]._obs.get("done")]

        # Aggressive cleanup between steps
        del (
            inputs,
            output_sequences,
            model_outputs,
            logits,
            log_probs_all,
            full_ids_batch,
            attention_mask,
            old_log_probs,
        )
        torch.cuda.empty_cache()

    rewards = [env.reward if env.reward is not None else 0.0 for env in envs]
    return all_episodes, rewards


def compute_grpo_loss(
    model,
    tokenizer,
    episodes: List[List[Dict]],
    rewards: List[float],
    epsilon: float = 0.2,
    micro_batch_size: int = 1,
):
    """Compute GRPO clipped surrogate loss using pre-tokenized trajectories.

    Calls backward() internally on each micro-batch. Returns detached scalar.
    """
    if not episodes or not rewards:
        return 0.0

    # Normalize rewards
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=model.device)
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    # Flatten all steps
    all_steps = []
    for ep_idx, steps in enumerate(episodes):
        for step in steps:
            all_steps.append(
                {
                    "prompt_ids": step["prompt_ids"].to(model.device),
                    "completion_ids": step["completion_ids"].to(model.device),
                    "old_log_prob": step["old_log_prob"],
                    "advantage": advantages[ep_idx].item(),
                }
            )

    total_steps = len(all_steps)
    loss_sum = 0.0

    model.train()
    for i in range(0, total_steps, micro_batch_size):
        batch_steps = all_steps[i : i + micro_batch_size]

        # Build padded batch of (prompt + completion)
        full_ids_list = [
            torch.cat([s["prompt_ids"], s["completion_ids"]])
            for s in batch_steps
        ]
        max_len = max(t.shape[0] for t in full_ids_list)
        full_ids = torch.full(
            (len(batch_steps), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=model.device,
        )
        attention_mask = torch.zeros(
            (len(batch_steps), max_len),
            dtype=torch.long,
            device=model.device,
        )
        prompt_lens = []
        for j, ids in enumerate(full_ids_list):
            full_ids[j, : ids.shape[0]] = ids
            attention_mask[j, : ids.shape[0]] = 1
            prompt_lens.append(batch_steps[j]["prompt_ids"].shape[0])

        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        log_probs_all = F.log_softmax(logits, dim=-1)

        # Gather new log-probs for completion tokens only
        new_log_probs = []
        for j in range(len(batch_steps)):
            p_len = prompt_lens[j]
            c_len = batch_steps[j]["completion_ids"].shape[0]
            if c_len == 0:
                new_log_probs.append(torch.tensor(0.0, device=model.device))
                continue

            token_lps = []
            for k in range(c_len):
                tok_id = full_ids[j, p_len + k]
                lp = log_probs_all[j, p_len + k - 1, tok_id]
                token_lps.append(lp)
            new_log_probs.append(torch.stack(token_lps).sum())

        new_log_probs = torch.stack(new_log_probs)
        old_log_probs = torch.stack(
            [s["old_log_prob"] for s in batch_steps]
        ).to(model.device)
        step_advantages = torch.tensor(
            [s["advantage"] for s in batch_steps], device=model.device
        )

        # Clipped surrogate loss
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        batch_loss = -torch.min(
            ratio * step_advantages, clipped * step_advantages
        ).mean()

        # Scale so gradients sum correctly across all steps
        scaled_loss = batch_loss / max(1, total_steps)
        scaled_loss.backward()

        loss_sum += batch_loss.item()

        # Free memory
        del (
            full_ids,
            attention_mask,
            outputs,
            logits,
            log_probs_all,
            new_log_probs,
            ratio,
            clipped,
            batch_loss,
            scaled_loss,
        )
        torch.cuda.empty_cache()

    return loss_sum / max(1, total_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--difficulty", type=str, default="learning")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora-r", type=int, default=16, help="LoRA rank (0 = disable LoRA)"
    )
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Episodes per batch"
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Steps per micro-batch during update (reduce if OOM)",
    )
    parser.add_argument(
        "--trajectory-file",
        type=str,
        default=None,
        help="If set, saves every prompt, raw completion, and parsed action to this JSON file. "
        "Useful for auditing model behavior post-training.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable performance-gated curriculum learning. Difficulty auto-escalates "
             "when mean batch reward stays above threshold.",
    )
    parser.add_argument(
        "--curriculum-phases",
        type=str,
        default="warmup,learning,advanced,expert",
        help="Comma-separated difficulty phases (default: warmup,learning,advanced,expert).",
    )
    parser.add_argument(
        "--curriculum-min-episodes",
        type=int,
        default=30,
        help="Minimum episodes per phase before escalation (default: 30).",
    )
    parser.add_argument(
        "--curriculum-escalate-threshold",
        type=float,
        default=0.65,
        help="Mean reward threshold to escalate (default: 0.65).",
    )
    parser.add_argument(
        "--curriculum-window",
        type=int,
        default=3,
        help="Number of recent batches to average for escalation check (default: 3).",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Load model in 4-bit quantization via bitsandbytes. Saves ~75% VRAM.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial epsilon for greedy warmup",
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=0.2, help="Final epsilon after decay"
    )
    parser.add_argument(
        "--epsilon-decay-batches",
        type=int,
        default=50,
        help="Number of batches to decay epsilon",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.trajectory_file:
        os.makedirs(os.path.dirname(args.trajectory_file) or ".", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("DispatchR GRPO Training (Batched Generation)")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Micro-batch size: {args.micro_batch_size}")
    print("=" * 60)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"BF16 supported: {bf16}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_4bit:
        print("[4-bit] Loading model with BitsAndBytes quantization...")
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print("[4-bit] Model loaded (~75% VRAM saved)")
        except ImportError:
            print(
                "[WARN] bitsandbytes not installed. Install with: pip install bitsandbytes"
            )
            print("[WARN] Falling back to standard bfloat16/float32 load")
            args.use_4bit = False

    if not args.use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    # Add LoRA
    if args.lora_r > 0:
        print("Adding LoRA adapters...")
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    else:
        print("LoRA disabled (r=0)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    device = next(model.parameters()).device

    # Sanity check: greedy baseline
    print("\n--- Sanity Check: Greedy Baseline ---")
    from server.dispatcher_environment import DispatcherEnvironment

    env_check = DispatcherEnvironment(seed=args.seed)
    greedy_rewards = []
    for _ in range(5):
        obs = env_check.reset(difficulty=args.difficulty)
        env_check.radio_buffer.delay_prob = 0.0
        for u in env_check.units:
            env_check._last_known_statuses[u.unit_id] = u.get_observable_status()
        done = False
        step_count = 0
        while not done and step_count < 100:
            active_calls = obs.get("active_calls", [])
            unit_statuses = obs.get("unit_statuses", [])
            action = {"action_type": "hold"}
            if active_calls:
                priority = {"cardiac": 3, "trauma": 2, "fire": 1, "false_alarm": 0}
                sorted_calls = sorted(
                    active_calls,
                    key=lambda c: priority.get(c.get("reported_type", "trauma"), 1),
                    reverse=True,
                )
                for call in sorted_calls:
                    best_unit = None
                    best_dist = float("inf")
                    for u in unit_statuses:
                        if u.get("last_known_status") == "idle":
                            dist = abs(
                                u.get("last_known_location", 0) - call["location"]
                            )
                            if dist < best_dist:
                                best_dist = dist
                                best_unit = u["unit_id"]
                    if best_unit is not None:
                        action = {
                            "action_type": "dispatch",
                            "unit_id": best_unit,
                            "call_id": call["call_id"],
                        }
                        break
            obs = env_check.step(action)
            done = obs.get("done", False)
            step_count += 1
        greedy_rewards.append(obs.get("reward", 0.0) or 0.0)
    print(f"Greedy baseline rewards: {[round(r, 3) for r in greedy_rewards]}")
    print(f"Mean greedy reward: {np.mean(greedy_rewards):.3f}")
    if np.mean(greedy_rewards) <= 0:
        print(
            "[WARN] Greedy baseline reward is <= 0. Environment may not produce learning signal."
        )

    # Training loop
    num_batches = max(1, args.episodes // args.batch_size)
    reward_history = []

    # Curriculum state
    if args.curriculum:
        curriculum_phases = args.curriculum_phases.split(",")
        current_phase_idx = 0
        episodes_in_phase = 0
        phase_reward_buffer = []
        current_difficulty = curriculum_phases[0]
        print(f"🎓 Curriculum enabled: {curriculum_phases}")
        print(f"   Starting at: {current_difficulty}")
    else:
        current_difficulty = args.difficulty

    trajectory_records = [] if args.trajectory_file else None

    try:
        for batch_idx in range(num_batches):
            batch_start = time.time()
            print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")

            # Decay epsilon
            if batch_idx < args.epsilon_decay_batches:
                epsilon = args.epsilon_start - (args.epsilon_start - args.epsilon_end) * (
                    batch_idx / args.epsilon_decay_batches
                )
            else:
                epsilon = args.epsilon_end
            print(f"Epsilon: {epsilon:.2f}")

            # Collect episodes (batched)
            envs = [
                DispatchRGRPOEnv(
                    seed=args.seed + batch_idx * args.batch_size + i,
                    difficulty=current_difficulty,
                )
                for i in range(args.batch_size)
            ]

            episodes, rewards = run_episodes_batched(
                model,
                tokenizer,
                envs,
                max_steps=MAX_STEPS,
                max_new_tokens=args.max_completion_length,
                device=device,
                epsilon=epsilon,
                trajectory_records=trajectory_records,
                batch_offset=batch_idx * args.batch_size,
            )

            mean_reward = np.mean(rewards)
            non_zero = sum(1 for r in rewards if r > 0)
            total_steps = sum(len(ep) for ep in episodes)
            print(
                f"Mean reward: {mean_reward:.4f} | Non-zero: {non_zero}/{len(rewards)} | "
                f"Total steps: {total_steps}"
            )

            # Curriculum: update phase based on recent performance
            if args.curriculum:
                episodes_in_phase += len(rewards)
                phase_reward_buffer.append(mean_reward)
                if len(phase_reward_buffer) > args.curriculum_window:
                    phase_reward_buffer.pop(0)

                print(
                    f"Curriculum: phase={current_difficulty} | "
                    f"episodes_in_phase={episodes_in_phase} | "
                    f"window_reward_avg={sum(phase_reward_buffer)/len(phase_reward_buffer):.4f}"
                )

                if (
                    episodes_in_phase >= args.curriculum_min_episodes
                    and len(phase_reward_buffer) >= args.curriculum_window
                    and (sum(phase_reward_buffer) / len(phase_reward_buffer)) >= args.curriculum_escalate_threshold
                    and current_phase_idx < len(curriculum_phases) - 1
                ):
                    current_phase_idx += 1
                    current_difficulty = curriculum_phases[current_phase_idx]
                    episodes_in_phase = 0
                    phase_reward_buffer = []
                    print(
                        f"🎓 CURRICULUM ESCALATED to {current_difficulty}! "
                        f"({args.curriculum_min_episodes}+ episodes, reward >= {args.curriculum_escalate_threshold})"
                    )

            # Skip update if all rewards are 0
            if all(r == 0 for r in rewards):
                print("All rewards zero — skipping update")
                if args.trajectory_file and trajectory_records:
                    # Back-fill rewards for episodes seen so far
                    for rec in trajectory_records:
                        ep = rec["episode"]
                        if ep < len(reward_history):
                            rec["episode_reward"] = reward_history[ep]
                    with open(args.trajectory_file, "w") as f:
                        json.dump(trajectory_records, f, indent=2)
                    print(f"Saved trajectory snapshot to {args.trajectory_file}")
                continue

            # GRPO update
            torch.cuda.empty_cache()
            gc.collect()
            optimizer.zero_grad()
            loss = compute_grpo_loss(
                model,
                tokenizer,
                episodes,
                rewards,
                micro_batch_size=args.micro_batch_size,
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(f"Loss: {loss:.4f}")

            batch_time = time.time() - batch_start
            print(f"Batch time: {batch_time:.1f}s")

            reward_history.extend(rewards)

            # Save checkpoint
            episodes_done = (batch_idx + 1) * args.batch_size
            if episodes_done % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{episodes_done}")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Saving emergency checkpoint...")
        interrupt_path = os.path.join(args.output_dir, "interrupted")
        os.makedirs(interrupt_path, exist_ok=True)
        model.save_pretrained(interrupt_path)
        tokenizer.save_pretrained(interrupt_path)
        print(f"Saved emergency checkpoint to {interrupt_path}")

    # Final save
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save metrics
    metrics = {
        "rewards": [float(r) for r in reward_history],
        "mean": float(np.mean(reward_history)),
        "std": float(np.std(reward_history)),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save trajectory audit log
    if args.trajectory_file and trajectory_records:
        for rec in trajectory_records:
            ep = rec["episode"]
            if ep < len(reward_history):
                rec["episode_reward"] = reward_history[ep]
        with open(args.trajectory_file, "w") as f:
            json.dump(trajectory_records, f, indent=2)
        print(f"Saved trajectory audit to {args.trajectory_file} "
              f"({len(trajectory_records)} records)")

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"Mean reward: {metrics['mean']:.4f}")
    print(f"Std reward:  {metrics['std']:.4f}")
    print(f"Final model: {final_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
