"""Unsloth-based GRPO training for DispatchR.

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

from unsloth import FastLanguageModel

import argparse
import copy
import gc
import json
import os
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Suppress noisy transformers deprecation warnings during training
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from server.constants import MAX_STEPS
from server.grpo_env_wrapper import DispatchRGRPOEnv
from server.city_graph import CityGraph
from server.prompt_utils import SYSTEM_PROMPT, build_chat_prompt, format_observation

_CITY_GRAPH = CityGraph()  # static singleton for greedy distance lookups
from server.training_tracker import ConsoleReporter, TrainingPlotter, TrainingTracker
from server.unsloth_grpo_utils import compute_grpo_loss


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
                dist = _CITY_GRAPH.travel_time(
                    u.get("last_known_location", 0), call["location"]
                )
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

            # Vectorized gather: avoid Python loop over tokens
            comp_ids = full_ids_batch[i, p_len : p_len + c_len]
            positions = torch.arange(p_len - 1, p_len + c_len - 1, device=device)
            token_lps = log_probs_all[i, positions, comp_ids]
            old_log_probs.append(token_lps.sum())

        # --- Step environments & store trajectory ---
        for i, idx in enumerate(active_indices):
            # Capture obs before stepping so the trajectory reflects what the
            # model was observing when it produced each completion.
            obs_snapshot = copy.deepcopy(envs[idx]._obs) if envs[idx]._obs else {}

            envs[idx].step(completions[i])

            # Move token IDs to CPU to free GPU memory
            all_episodes[idx].append(
                {
                    "prompt_ids": prompt_ids_list[i].cpu(),
                    "completion_ids": completion_ids_list[i].cpu(),
                    "old_log_prob": old_log_probs[i],
                }
            )

            # Store human-readable trajectory (obs enables city animation)
            trajectory[idx].append(
                {
                    "prompt": prompts[i],
                    "completion": completions[i],
                    "action_text": completions[i],
                    "obs": obs_snapshot,
                }
            )

        # Update active list
        active = [idx for idx in active if not envs[idx]._obs.get("done")]

    # Aggressive cleanup once per batch (not per step) to avoid CUDA sync stalls
    torch.cuda.empty_cache()

    rewards = [env.reward if env.reward is not None else 0.0 for env in envs]
    return all_episodes, rewards, trajectory


def main():
    parser = argparse.ArgumentParser(
        description="DispatchR GRPO training with Unsloth"
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
        default=8,
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
        help="HF Hub model ID to push checkpoints to (e.g., 'username/dispatchr-grpo').",
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
    print("DispatchR Unsloth GRPO Training")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Micro-batch size: {args.micro_batch_size}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1.  Hub setup (create repo so push never hits a 404)
    # ------------------------------------------------------------------
    if args.push_to_hub and args.hub_model_id:
        try:
            from huggingface_hub import HfApi
            _hub_api = HfApi()
            _hub_api.create_repo(
                repo_id=args.hub_model_id,
                repo_type="model",
                exist_ok=True,
                private=args.hub_private,
            )
            print(f"Model repo ready: https://huggingface.co/{args.hub_model_id}")
        except Exception as _e:
            print(f"[WARN] Could not create/verify model repo: {_e}")

    # ------------------------------------------------------------------
    # 2.  Load model and LoRA adapters
    # ------------------------------------------------------------------
    print("\nLoading base model via Unsloth...")
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
    # 3.  Training loop
    # ------------------------------------------------------------------
    num_batches = max(1, args.episodes // args.batch_size)

    # Restore history from a previous run (empty lists on fresh start)
    reward_history: list = list(resume_state.get("reward_history", []))
    loss_history: list   = list(resume_state.get("loss_history", []))
    start_batch: int     = resume_state.get("batch_idx_done", -1) + 1

    if start_batch >= num_batches:
        print(f"\nAll {num_batches} batches already completed in the previous run. Nothing to do.")
        return

    if start_batch > 0:
        print(f"\nResuming: starting at batch {start_batch + 1}/{num_batches}  "
              f"({len(reward_history)} episodes already logged)")

    # Initialize progress tracker
    tracker = TrainingTracker(output_dir=args.output_dir)
    reporter = ConsoleReporter()
    plotter = TrainingPlotter(tracker)
    reporter.print_header()

    # Curriculum state — restore if resuming, otherwise start from scratch
    if args.curriculum:
        curriculum_phases = args.curriculum_phases.split(",")
        current_phase_idx  = resume_state.get("current_phase_idx",  0)
        episodes_in_phase  = resume_state.get("episodes_in_phase",  0)
        phase_reward_buffer = list(resume_state.get("phase_reward_buffer", []))
        current_difficulty = resume_state.get(
            "current_difficulty", curriculum_phases[0]
        )
        print(f"🎓 Curriculum enabled: {curriculum_phases}")
        print(f"   Current phase:  {current_difficulty}  (idx {current_phase_idx})")
    else:
        current_difficulty = args.difficulty

    trajectory_writer = None
    if args.trajectory_file:
        os.makedirs(os.path.dirname(args.trajectory_file) or ".", exist_ok=True)
        trajectory_writer = open(args.trajectory_file, "a", encoding="utf-8")

    try:
        for batch_idx in range(start_batch, num_batches):
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
            # Disable env internal curriculum when training script manages it
            if args.curriculum:
                for env in envs:
                    env._env._use_internal_curriculum = False

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

            # Collect environment metrics from each episode
            env_metrics_list = []
            for env in envs:
                m = env.metrics
                if m is not None:
                    env_metrics_list.append(m)

            # Curriculum: update phase based on recent performance
            mean_reward = np.mean(rewards)
            if args.curriculum:
                episodes_in_phase += len(rewards)
                phase_reward_buffer.append(mean_reward)
                if len(phase_reward_buffer) > args.curriculum_window:
                    phase_reward_buffer.pop(0)

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
                                    "obs": step.get("obs", {}),
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
            model.train()  # Unsloth needs train mode for backward
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
            model.eval()  # Switch back for next batch's generation
            print(f"Loss: {loss:.4f}")

            batch_time = time.time() - batch_start

            reward_history.extend(rewards)
            loss_history.append(loss)

            # Log to tracker and print progress report
            record = tracker.log_batch(
                batch_idx=batch_idx,
                rewards=rewards,
                loss=float(loss) if loss is not None else None,
                epsilon=epsilon,
                difficulty=current_difficulty,
                batch_time=batch_time,
                env_metrics_list=env_metrics_list,
            )
            reporter.print_batch_report(record, tracker.get_summary(), num_batches)

            # Incremental metrics.json — overwrites each batch so data survives a crash
            _partial = {
                "episodes_done": (batch_idx + 1) * args.batch_size,
                "batches_done": batch_idx + 1,
                "rewards": [float(r) for r in reward_history],
                "losses": [float(l) for l in loss_history],
                "batch_records": tracker.records,
                "summary": tracker.get_summary(),
                "config": vars(args),
            }
            with open(os.path.join(args.output_dir, "metrics.json"), "w") as _mf:
                json.dump(_partial, _mf)

            # Save checkpoint
            episodes_done = (batch_idx + 1) * args.batch_size
            if episodes_done % args.save_every == 0:
                save_path = os.path.join(
                    args.output_dir, f"checkpoint-{episodes_done}"
                )
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

                # Save training state for resumability
                _ts = {
                    "batch_idx_done": batch_idx,
                    "reward_history": [float(r) for r in reward_history],
                    "loss_history":   [float(l) for l in loss_history],
                    "current_difficulty": current_difficulty,
                    "current_phase_idx":  current_phase_idx if args.curriculum else 0,
                    "episodes_in_phase":  episodes_in_phase if args.curriculum else 0,
                    "phase_reward_buffer": (
                        [float(x) for x in phase_reward_buffer]
                        if args.curriculum else []
                    ),
                }
                _ts_path = os.path.join(save_path, "training_state.json")
                with open(_ts_path, "w") as _tsf:
                    json.dump(_ts, _tsf)
                print(f"Saved checkpoint to {save_path}")

                # Push to HF Hub if requested
                if args.push_to_hub and args.hub_model_id:
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi()
                        # Push model weights
                        api.upload_folder(
                            folder_path=save_path,
                            repo_id=args.hub_model_id,
                            repo_type="model",
                            private=args.hub_private,
                            commit_message=(
                                f"Checkpoint after {episodes_done} episodes "
                                f"(reward={mean_reward:.3f})"
                            ),
                        )
                        # Also push training_state.json to the repo root so that
                        # --resume-from-hub can find it via hf_hub_download()
                        api.upload_file(
                            path_or_fileobj=_ts_path,
                            path_in_repo="training_state.json",
                            repo_id=args.hub_model_id,
                            repo_type="model",
                            commit_message="Update training state (for resume)",
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
        # Save training state so the run is resumable
        _last_batch = start_batch + len(tracker.records) - 1
        _ts_int = {
            "batch_idx_done": _last_batch,
            "reward_history": [float(r) for r in reward_history],
            "loss_history":   [float(l) for l in loss_history],
            "current_difficulty": current_difficulty,
            "current_phase_idx":  current_phase_idx if args.curriculum else 0,
            "episodes_in_phase":  episodes_in_phase if args.curriculum else 0,
            "phase_reward_buffer": (
                [float(x) for x in phase_reward_buffer] if args.curriculum else []
            ),
        }
        _ts_int_path = os.path.join(interrupt_path, "training_state.json")
        with open(_ts_int_path, "w") as _f:
            json.dump(_ts_int, _f)
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
                api.upload_file(
                    path_or_fileobj=_ts_int_path,
                    path_in_repo="training_state.json",
                    repo_id=args.hub_model_id,
                    repo_type="model",
                    commit_message="Update training state (interrupted)",
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
    # Mark training as complete in the state file so resume doesn't re-run
    _ts_final = {
        "batch_idx_done": num_batches - 1,
        "training_complete": True,
        "reward_history": [float(r) for r in reward_history],
        "loss_history":   [float(l) for l in loss_history],
        "current_difficulty": current_difficulty,
        "current_phase_idx":  current_phase_idx if args.curriculum else 0,
        "episodes_in_phase":  episodes_in_phase if args.curriculum else 0,
        "phase_reward_buffer": (
            [float(x) for x in phase_reward_buffer] if args.curriculum else []
        ),
    }
    _ts_final_path = os.path.join(final_path, "training_state.json")
    with open(_ts_final_path, "w") as _f:
        json.dump(_ts_final, _f)
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
                commit_message="Final checkpoint — training complete",
            )
            api.upload_file(
                path_or_fileobj=_ts_final_path,
                path_in_repo="training_state.json",
                repo_id=args.hub_model_id,
                repo_type="model",
                commit_message="Final training state",
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
        "batch_records": tracker.records,
        "summary": tracker.get_summary(),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate training plots
    plotter.generate()
    plotter.generate_simple_plot()

    # Final summary
    summary = tracker.get_summary()
    reporter.print_final_summary(summary)


if __name__ == "__main__":
    main()
