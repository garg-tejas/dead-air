"""Unsloth-based GRPO training for Dead Air.

Uses Unsloth's FastLanguageModel for optimized 4-bit kernels.
The training loop mirrors train_grpo.py but swaps in Unsloth model loading.

Usage (on Lightning AI L4):
    python train_unsloth_grpo.py \
        --model unsloth/Qwen3-4B-Thinking-2507-bnb-4bit \
        --episodes 200 \
        --batch-size 8 \
        --use-4bit

    # Or with a standard HF model (Unsloth will still optimize it):
    python train_unsloth_grpo.py \
        --model Qwen/Qwen3.5-2B-Instruct \
        --episodes 200 \
        --batch-size 8
"""

import unsloth  # Must be first, before transformers

import argparse
import gc
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from server.constants import MAX_STEPS
from server.grpo_env_wrapper import DeadAirGRPOEnv
from server.unsloth_grpo_utils import compute_grpo_loss

SYSTEM_PROMPT = (
    "You are an emergency dispatch AI managing 6 ambulance units in a 20-node city. "
    "Every step, output exactly one JSON object on the VERY LAST LINE.\n\n"
    "RULES:\n"
    "- If Active Calls is empty or says '(none)', output: {\"action_type\":\"hold\"}\n"
    "- If there are active calls, dispatch the closest idle unit to the most urgent call.\n"
    "- Keep reasoning to 1-2 sentences. Do not overthink.\n"
    "- The JSON must be the very last thing you output. No markdown, no extra text after it.\n\n"
    "ACTIONS:\n"
    '{"action_type":"dispatch","unit_id":0,"call_id":1}\n'
    '{"action_type":"hold"}\n'
    '{"action_type":"verify","call_id":1}\n\n'
    "Example (no calls): All units idle, no active calls. {\"action_type\":\"hold\"}\n"
    "Example (with calls): Call 2 is cardiac (most urgent). Unit 1 is idle and closest. {\"action_type\":\"dispatch\",\"unit_id\":1,\"call_id\":2}"
)


def format_observation(obs: Dict) -> str:
    """Format env observation into prompt text."""
    lines = [
        "# Emergency Dispatch Commander",
        f"Step {obs['step_number']}/{obs['max_steps']}",
        "",
        "## Units",
    ]
    for u in obs.get("unit_statuses", []):
        call_info = f" -> Call {u['current_call']}" if u.get("current_call") else ""
        lines.append(
            f"- Unit {u['unit_id']}: {u['last_known_status']} at Node {u['last_known_location']}{call_info}"
        )
    lines.append("")
    lines.append("## Active Calls")
    active_calls = obs.get("active_calls", [])
    if active_calls:
        for c in active_calls:
            assigned = f" (Unit {c['assigned_unit']})" if c.get("assigned_unit") else ""
            lines.append(
                f"- Call {c['call_id']}: {c['reported_type']} at Node {c['location']} ({c['caller_tone']}) elapsed={c['time_elapsed']}min{assigned}"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Traffic & Hospitals")
    for alert in obs.get("traffic_alerts", []):
        lines.append(f"- {alert}")
    for h in obs.get("hospital_statuses", []):
        lines.append(f"- Hospital {h['hospital_id']}: {h['reported_status']}")
    lines.append("")
    lines.append(f"Mutual aid remaining: {obs['mutual_aid_remaining']}")
    lines.append("")
    lines.append("Choose your next action.")
    return "\n".join(lines)


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    """Build a chat-formatted prompt using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": True},
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return f"{system}\n\n{user}\n\nAssistant:"


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
    envs: List[DeadAirGRPOEnv],
    max_steps: int,
    max_new_tokens: int,
    device: str,
    epsilon: float = 0.0,
    batch_offset: int = 0,
):
    """Run a batch of episodes with batched generation.

    Returns (episodes, rewards, trajectory) where episodes is a list of step-lists
    and trajectory is a list of per-step dicts with human-readable prompt + completion.
    Each step dict in episodes contains:
        - prompt_ids: torch.LongTensor (on CPU)
        - completion_ids: torch.LongTensor (on CPU)
        - old_log_prob: torch.Tensor (scalar, detached)
    Each step dict in trajectory contains:
        - prompt: full prompt text
        - completion: generated completion text
        - action_text: parsed action text (what env.step received)
    """
    batch_size = len(envs)
    for env in envs:
        env.reset()

    active = list(range(batch_size))
    all_episodes = [[] for _ in range(batch_size)]
    trajectory = [[] for _ in range(batch_size)]  # human-readable

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
                lp = log_probs_all[i, p_len + k - 1, tok_id]
                token_lps.append(lp)
            old_log_probs.append(torch.stack(token_lps).sum())

        # --- Step environments & store trajectory ---
        for i, idx in enumerate(active_indices):
            envs[idx].step(completions[i])

            # Move token IDs to CPU to free GPU memory
            all_episodes[idx].append(
                {
                    "prompt_ids": prompt_ids_list[i].cpu(),
                    "completion_ids": completion_ids_list[i].cpu(),
                    "old_log_prob": old_log_probs[i],
                }
            )

            # Store human-readable trajectory
            trajectory[idx].append(
                {
                    "prompt": prompts[i],
                    "completion": completions[i],
                    "action_text": completions[i],
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
    return all_episodes, rewards, trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Dead Air GRPO training with Unsloth"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
        help="Unsloth model id (default: unsloth/Qwen3-4B-Thinking-2507-bnb-4bit). "
             "Use a thinking-enabled model so RL can optimize reasoning traces. "
             "The -bnb-4bit variant loads instantly (pre-quantized).",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--difficulty", type=str, default="learning")
    parser.add_argument("--output-dir", type=str, default="./outputs/unsloth_grpo")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--max-completion-length", type=int, default=512)
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
        "--use-4bit",
        action="store_true",
        help="Load model in 4-bit quantization. Unsloth does this by default.",
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
    parser.add_argument(
        "--trajectory-file",
        type=str,
        default=None,
        help="JSONL file to save full prompt + generated text for every step. "
             "Useful for auditing model behaviour.",
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
        "--push-to-hub",
        action="store_true",
        help="Push checkpoints to Hugging Face Hub (requires HF token in env).",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="HF Hub model ID to push checkpoints to (e.g., 'username/dead-air-grpo').",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Make HF Hub model private (default: public).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("Dead Air Unsloth GRPO Training")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Micro-batch size: {args.micro_batch_size}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1.  Load model via Unsloth
    # ------------------------------------------------------------------
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        print("\nERROR: Unsloth is not installed.")
        print("Install it with: pip install unsloth")
        print("\nOr fall back to train_grpo.py")
        raise SystemExit(1) from exc

    print("\nLoading model via Unsloth...")
    # Unsloth handles 4-bit automatically; pass load_in_4bit for explicit control
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        load_in_4bit=args.use_4bit,
        max_seq_length=4096,
        dtype=torch.bfloat16,
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_rslora=True,
        bias="none",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    device = next(model.parameters()).device

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

    trajectory_writer = None
    if args.trajectory_file:
        os.makedirs(os.path.dirname(args.trajectory_file) or ".", exist_ok=True)
        trajectory_writer = open(args.trajectory_file, "a", encoding="utf-8")

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
                DeadAirGRPOEnv(
                    seed=args.seed + batch_idx * args.batch_size + i,
                    difficulty=current_difficulty,
                )
                for i in range(args.batch_size)
            ]

            episodes, rewards, trajectory = run_episodes_batched(
                model,
                tokenizer,
                envs,
                max_steps=MAX_STEPS,
                max_new_tokens=args.max_completion_length,
                device=device,
                epsilon=epsilon,
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

            # Save trajectory for auditing
            if trajectory_writer:
                for ep_idx, (ep_steps, ep_reward) in enumerate(zip(trajectory, rewards)):
                    for step_idx, step in enumerate(ep_steps):
                        trajectory_writer.write(
                            json.dumps(
                                {
                                    "batch": batch_idx,
                                    "episode": batch_idx * args.batch_size + ep_idx,
                                    "step": step_idx,
                                    "reward": ep_reward,
                                    "prompt": step["prompt"],
                                    "completion": step["completion"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                trajectory_writer.flush()

            # Skip update if all rewards are 0
            if all(r == 0 for r in rewards):
                print("All rewards zero — skipping update")
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
            optimizer.step()
            print(f"Loss: {loss:.4f}")

            batch_time = time.time() - batch_start
            print(f"Batch time: {batch_time:.1f}s")

            reward_history.extend(rewards)
            loss_history.append(loss)

            # Save checkpoint
            episodes_done = (batch_idx + 1) * args.batch_size
            if episodes_done % args.save_every == 0:
                save_path = os.path.join(
                    args.output_dir, f"checkpoint-{episodes_done}"
                )
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

                # Push to HF Hub if requested
                if args.push_to_hub and args.hub_model_id:
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi()
                        api.upload_folder(
                            folder_path=save_path,
                            repo_id=args.hub_model_id,
                            repo_type="model",
                            private=args.hub_private,
                            commit_message=f"Checkpoint after {episodes_done} episodes (reward={mean_reward:.3f})",
                        )
                        print(f"Pushed checkpoint to https://huggingface.co/{args.hub_model_id}")
                    except Exception as e:
                        print(f"[WARN] HF Hub push failed: {e}")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Saving emergency checkpoint...")
        interrupt_path = os.path.join(args.output_dir, "interrupted")
        os.makedirs(interrupt_path, exist_ok=True)
        model.save_pretrained(interrupt_path)
        tokenizer.save_pretrained(interrupt_path)
        print(f"Saved emergency checkpoint to {interrupt_path}")
        if args.push_to_hub and args.hub_model_id:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_folder(
                    folder_path=interrupt_path,
                    repo_id=args.hub_model_id,
                    repo_type="model",
                    private=args.hub_private,
                    commit_message="Emergency interrupt checkpoint",
                )
                print(f"Pushed emergency checkpoint to https://huggingface.co/{args.hub_model_id}")
            except Exception as e:
                print(f"[WARN] HF Hub push failed: {e}")
    finally:
        if trajectory_writer:
            trajectory_writer.close()

    # Final save
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved final checkpoint to {final_path}")
    if args.push_to_hub and args.hub_model_id:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=final_path,
                repo_id=args.hub_model_id,
                repo_type="model",
                private=args.hub_private,
                commit_message="Final checkpoint",
            )
            print(f"Pushed final checkpoint to https://huggingface.co/{args.hub_model_id}")
        except Exception as e:
            print(f"[WARN] HF Hub push failed: {e}")

    # Save metrics
    metrics = {
        "rewards": [float(r) for r in reward_history],
        "losses": [float(l) for l in loss_history],
        "mean": float(np.mean(reward_history)) if reward_history else 0.0,
        "std": float(np.std(reward_history)) if reward_history else 0.0,
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"Mean reward: {metrics['mean']:.4f}")
    print(f"Std reward:  {metrics['std']:.4f}")
    print(f"Final model: {final_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
