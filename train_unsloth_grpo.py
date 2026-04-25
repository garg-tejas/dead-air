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

import unsloth  # Must be first, before transformers

import argparse
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from server.constants import MAX_STEPS
from server.grpo_env_wrapper import DispatchRGRPOEnv
from server.training_tracker import ConsoleReporter, TrainingPlotter, TrainingTracker
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


def _current_job_id() -> str:
    """Return the HF Job ID when running on Jobs, else a stable local label."""
    return os.environ.get("JOB_ID", "local")


def _read_json(path: Path) -> Dict[str, Any]:
    """Read one JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write one JSON object to disk with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_resume_state(resume_from: str) -> Dict[str, Any]:
    """Load resume metadata from a local checkpoint folder or Hub repo."""
    resume_path = Path(resume_from)
    if resume_path.is_dir():
        state_path = resume_path / "resume_state.json"
    elif resume_path.is_file():
        state_path = resume_path
    else:
        state_path = Path(
            hf_hub_download(
                repo_id=resume_from,
                filename="resume_state.json",
                repo_type="model",
            )
        )

    if not state_path.exists():
        raise FileNotFoundError(f"Could not find resume_state.json at {state_path}")
    return _read_json(state_path)


def _build_hub_api(args: argparse.Namespace) -> Optional[HfApi]:
    """Create an authenticated Hub client and ensure the target repo exists."""
    if not args.push_to_hub or not args.hub_model_id:
        return None

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required when --push-to-hub is enabled.")

    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=args.hub_model_id,
        repo_type="model",
        private=args.hub_private,
        exist_ok=True,
    )
    return api


def _upload_folder_with_retry(
    api: HfApi,
    repo_id: str,
    folder_path: Path,
    commit_message: str,
    path_in_repo: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    attempts: int = 3,
) -> None:
    """Upload a folder to the Hub with bounded retries."""
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(folder_path),
                path_in_repo=path_in_repo,
                commit_message=commit_message,
                allow_patterns=allow_patterns,
            )
            return
        except Exception as exc:  # pragma: no cover - exercised in live jobs
            last_error = exc
            if attempt == attempts:
                break
            print(f"[WARN] Hub upload attempt {attempt}/{attempts} failed: {exc}")
            time.sleep(min(5, attempt * 2))

    raise RuntimeError(
        f"Failed to upload '{folder_path}' to '{repo_id}' after {attempts} attempts."
    ) from last_error


def _save_checkpoint_bundle(
    model,
    tokenizer,
    output_dir: Path,
    resume_state: Dict[str, Any],
) -> None:
    """Persist one resumable checkpoint bundle to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _write_json(output_dir / "resume_state.json", resume_state)
    print(f"Saved checkpoint to {output_dir}")


def _snapshot_resume_state(
    args: argparse.Namespace,
    tracker: TrainingTracker,
    reward_history: List[float],
    loss_history: List[float],
    episodes_done: int,
    completed_batches: int,
    current_difficulty: str,
    current_phase_idx: int,
    episodes_in_phase: int,
    phase_reward_buffer: List[float],
    epsilon: float,
    checkpoint_name: str,
) -> Dict[str, Any]:
    """Capture all state needed to restart training safely."""
    return {
        "version": 1,
        "job_id": _current_job_id(),
        "checkpoint_name": checkpoint_name,
        "episodes_done": episodes_done,
        "completed_batches": completed_batches,
        "target_episodes": args.episodes,
        "batch_size": args.batch_size,
        "current_difficulty": current_difficulty,
        "curriculum_state": {
            "enabled": args.curriculum,
            "current_phase_idx": current_phase_idx,
            "episodes_in_phase": episodes_in_phase,
            "phase_reward_buffer": [float(x) for x in phase_reward_buffer],
        },
        "epsilon_state": {
            "start": args.epsilon_start,
            "end": args.epsilon_end,
            "decay_batches": args.epsilon_decay_batches,
            "last_epsilon": float(epsilon),
        },
        "reward_history": [float(x) for x in reward_history],
        "loss_history": [float(x) for x in loss_history],
        "tracker_state": tracker.snapshot(),
        "model_source": args.model,
        "resume_source": args.resume_from,
        "hub_model_id": args.hub_model_id,
        "updated_at": int(time.time()),
    }


def _write_metrics_bundle(
    output_dir: Path,
    args: argparse.Namespace,
    reward_history: List[float],
    loss_history: List[float],
    tracker: TrainingTracker,
) -> None:
    """Persist training metrics for later artifact upload."""
    metrics = {
        "rewards": [float(r) for r in reward_history],
        "losses": [float(l) for l in loss_history],
        "mean": float(np.mean(reward_history)) if reward_history else 0.0,
        "std": float(np.std(reward_history)) if reward_history else 0.0,
        "config": vars(args),
        "batch_records": tracker.records,
        "summary": tracker.get_summary(),
    }
    _write_json(output_dir / "metrics.json", metrics)


def _upload_run_artifacts(
    api: Optional[HfApi],
    repo_id: Optional[str],
    output_dir: Path,
    trajectory_file: Optional[str],
    run_label: str,
) -> None:
    """Upload metrics and plots into the job-specific artifacts prefix."""
    if api is None or not repo_id:
        return

    staging_dir = output_dir / ".hf-artifacts"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    artifact_names = [
        "metrics.csv",
        "metrics.json",
        "training_progress.png",
        "training_curves.png",
    ]
    for name in artifact_names:
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, staging_dir / name)

    if trajectory_file:
        trajectory_path = Path(trajectory_file)
        if trajectory_path.exists():
            shutil.copy2(trajectory_path, staging_dir / trajectory_path.name)

    staged_items = list(staging_dir.iterdir())
    if not staged_items:
        shutil.rmtree(staging_dir)
        return

    _upload_folder_with_retry(
        api=api,
        repo_id=repo_id,
        folder_path=staging_dir,
        path_in_repo=f"artifacts/{_current_job_id()}",
        commit_message=f"Upload {run_label} artifacts for job {_current_job_id()}",
    )
    shutil.rmtree(staging_dir)


def _calculate_epsilon(args: argparse.Namespace, batch_idx: int) -> float:
    """Return the epsilon-greedy rate for the current batch."""
    if args.epsilon_decay_batches <= 0:
        return args.epsilon_end
    if batch_idx < args.epsilon_decay_batches:
        return args.epsilon_start - (args.epsilon_start - args.epsilon_end) * (
            batch_idx / args.epsilon_decay_batches
        )
    return args.epsilon_end


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

        # --- Batched tokenization ---
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        completions: List[Optional[str]] = [None] * len(prompts)
        completion_ids_list: List[Optional[torch.Tensor]] = [None] * len(prompts)
        generation_slots = []

        # --- Epsilon-greedy: choose greedy actions before generation ---
        for i, idx in enumerate(active_indices):
            if (
                epsilon > 0
                and np.random.rand() < epsilon
            ):
                greedy_act = greedy_action(envs[idx]._obs)
                completions[i] = json.dumps(greedy_act)
                comp_ids = tokenizer(
                    completions[i],
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids[0].to(device)
                completion_ids_list[i] = comp_ids
            else:
                generation_slots.append(i)

        missing_completion_ids = sum(
            completion_ids is None for completion_ids in completion_ids_list
        )
        if missing_completion_ids != len(generation_slots):
            raise RuntimeError("Internal epsilon-greedy slot accounting failed.")

        skipped_generations = len(prompts) - len(generation_slots)
        if skipped_generations:
            print(
                f"Step {step_idx + 1}: skipped generation for "
                f"{skipped_generations}/{len(prompts)} epsilon-greedy actions",
                flush=True,
            )

        gen_inputs = None
        output_sequences = None
        if generation_slots:
            gen_prompts = [prompts[i] for i in generation_slots]
            gen_inputs = tokenizer(
                gen_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)

            with torch.no_grad():
                output_sequences = model.generate(
                    **gen_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Extract generated completions after the padded prompt length.
            generation_start = gen_inputs.input_ids.shape[1]
            for gen_idx, slot_idx in enumerate(generation_slots):
                gen_ids = output_sequences[gen_idx, generation_start:]
                eos_positions = (gen_ids == tokenizer.eos_token_id).nonzero(
                    as_tuple=True
                )[0]
                if len(eos_positions) > 0:
                    gen_ids = gen_ids[: eos_positions[0].item() + 1]
                completion_ids_list[slot_idx] = gen_ids
                completions[slot_idx] = tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                )

        if any(completion is None for completion in completions):
            raise RuntimeError("Internal error: missing generated or greedy completion.")
        if any(completion_ids is None for completion_ids in completion_ids_list):
            raise RuntimeError("Internal error: missing completion token ids.")

        # --- Compute old log-probs in one batched forward pass ---
        prompt_ids_list = [
            inputs.input_ids[i, inputs.attention_mask[i].bool()]
            for i in range(len(prompts))
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
            model_outputs,
            logits,
            log_probs_all,
            full_ids_batch,
            attention_mask,
            old_log_probs,
        )
        if gen_inputs is not None:
            del gen_inputs
        if output_sequences is not None:
            del output_sequences
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
        help="HF Hub model ID to push checkpoints to (e.g., 'username/dispatchr-grpo').",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Make HF Hub model private (default: public).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from a local checkpoint folder or HF model repo containing resume_state.json.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_state: Optional[Dict[str, Any]] = None
    if args.resume_from:
        resume_state = _load_resume_state(args.resume_from)
        if resume_state.get("curriculum_state", {}).get("enabled"):
            args.curriculum = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("DispatchR Unsloth GRPO Training")
    print(f"Model: {args.resume_from or args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Micro-batch size: {args.micro_batch_size}")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        print("\nERROR: Unsloth is not installed.")
        print("Install it with: pip install unsloth")
        print("\nOr fall back to train_grpo.py")
        raise SystemExit(1) from exc

    print("\nLoading model via Unsloth...")
    if args.resume_from:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.resume_from,
            load_in_4bit=args.use_4bit,
            max_seq_length=4096,
            dtype=torch.bfloat16,
        )
        print(f"Loaded resumable adapter from {args.resume_from}")
    else:
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
    tokenizer.padding_side = "left"

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

    total_batches = max(1, math.ceil(args.episodes / args.batch_size))
    reward_history: List[float] = []
    loss_history: List[float] = []
    tracker = TrainingTracker(output_dir=str(output_dir))
    reporter = ConsoleReporter()
    plotter = TrainingPlotter(tracker)
    reporter.print_header()

    curriculum_phases = args.curriculum_phases.split(",")
    current_phase_idx = 0
    episodes_in_phase = 0
    phase_reward_buffer: List[float] = []
    current_difficulty = curriculum_phases[0] if args.curriculum else args.difficulty
    episodes_done = 0
    completed_batches = 0
    last_epsilon = args.epsilon_start
    last_mean_reward = 0.0

    if resume_state:
        reward_history = [float(x) for x in resume_state.get("reward_history", [])]
        loss_history = [float(x) for x in resume_state.get("loss_history", [])]
        tracker.restore(resume_state.get("tracker_state"))
        episodes_done = int(resume_state.get("episodes_done", 0))
        completed_batches = int(resume_state.get("completed_batches", 0))
        current_difficulty = str(
            resume_state.get("current_difficulty", current_difficulty)
        )

        curriculum_state = resume_state.get("curriculum_state", {})
        if args.curriculum:
            current_phase_idx = min(
                max(0, int(curriculum_state.get("current_phase_idx", 0))),
                len(curriculum_phases) - 1,
            )
            current_difficulty = curriculum_phases[current_phase_idx]
            episodes_in_phase = int(curriculum_state.get("episodes_in_phase", 0))
            phase_reward_buffer = [
                float(x) for x in curriculum_state.get("phase_reward_buffer", [])
            ]

        last_epsilon = float(
            resume_state.get("epsilon_state", {}).get("last_epsilon", last_epsilon)
        )
        print(
            f"Resuming from episode {episodes_done}, batch {completed_batches}, "
            f"difficulty={current_difficulty}, epsilon={last_epsilon:.2f}"
        )
    elif args.curriculum:
        print(f"🎓 Curriculum enabled: {curriculum_phases}")
        print(f"   Starting at: {current_difficulty}")

    hub_api = _build_hub_api(args)

    trajectory_writer = None
    if args.trajectory_file:
        trajectory_path = Path(args.trajectory_file)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_writer = trajectory_path.open("a", encoding="utf-8")

    failure: Optional[BaseException] = None
    run_status = "completed"

    try:
        while episodes_done < args.episodes:
            batch_idx = completed_batches
            batch_size = min(args.batch_size, args.episodes - episodes_done)
            batch_start = time.time()

            print(f"\n--- Batch {batch_idx + 1}/{total_batches} ---")
            epsilon = _calculate_epsilon(args, batch_idx)
            last_epsilon = epsilon
            print(f"Epsilon: {epsilon:.2f}")

            envs = [
                DispatchRGRPOEnv(
                    seed=args.seed + episodes_done + i,
                    difficulty=current_difficulty,
                )
                for i in range(batch_size)
            ]

            episodes, rewards, trajectory = run_episodes_batched(
                model,
                tokenizer,
                envs,
                max_steps=MAX_STEPS,
                max_new_tokens=args.max_completion_length,
                device=device,
                epsilon=epsilon,
                batch_offset=episodes_done,
            )

            env_metrics_list = []
            for env in envs:
                if env.metrics is not None:
                    env_metrics_list.append(env.metrics)

            last_mean_reward = float(np.mean(rewards)) if rewards else 0.0
            if args.curriculum:
                episodes_in_phase += len(rewards)
                phase_reward_buffer.append(last_mean_reward)
                if len(phase_reward_buffer) > args.curriculum_window:
                    phase_reward_buffer.pop(0)

                if (
                    episodes_in_phase >= args.curriculum_min_episodes
                    and len(phase_reward_buffer) >= args.curriculum_window
                    and (
                        sum(phase_reward_buffer) / len(phase_reward_buffer)
                    ) >= args.curriculum_escalate_threshold
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

            if trajectory_writer:
                for ep_idx, (ep_steps, ep_reward) in enumerate(
                    zip(trajectory, rewards)
                ):
                    for step_idx, step in enumerate(ep_steps):
                        trajectory_writer.write(
                            json.dumps(
                                {
                                    "batch": batch_idx,
                                    "episode": episodes_done + ep_idx,
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

            loss: Optional[float] = None
            if all(r == 0 for r in rewards):
                print("All rewards zero — skipping update")
            else:
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                loss = float(
                    compute_grpo_loss(
                        model,
                        tokenizer,
                        episodes,
                        rewards,
                        micro_batch_size=args.micro_batch_size,
                    )
                )
                optimizer.step()
                loss_history.append(loss)
                print(f"Loss: {loss:.4f}")

            batch_time = time.time() - batch_start
            reward_history.extend(rewards)

            record = tracker.log_batch(
                batch_idx=batch_idx,
                rewards=rewards,
                loss=loss,
                epsilon=epsilon,
                difficulty=current_difficulty,
                batch_time=batch_time,
                env_metrics_list=env_metrics_list,
            )
            reporter.print_batch_report(record, tracker.get_summary(), total_batches)

            episodes_done += batch_size
            completed_batches += 1

            if args.save_every > 0 and episodes_done < args.episodes:
                if episodes_done % args.save_every == 0:
                    checkpoint_name = f"checkpoint-{episodes_done}"
                    resume_payload = _snapshot_resume_state(
                        args=args,
                        tracker=tracker,
                        reward_history=reward_history,
                        loss_history=loss_history,
                        episodes_done=episodes_done,
                        completed_batches=completed_batches,
                        current_difficulty=current_difficulty,
                        current_phase_idx=current_phase_idx,
                        episodes_in_phase=episodes_in_phase,
                        phase_reward_buffer=phase_reward_buffer,
                        epsilon=epsilon,
                        checkpoint_name=checkpoint_name,
                    )
                    checkpoint_dir = output_dir / checkpoint_name
                    _save_checkpoint_bundle(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=checkpoint_dir,
                        resume_state=resume_payload,
                    )
                    if hub_api is not None and args.hub_model_id:
                        _upload_folder_with_retry(
                            api=hub_api,
                            repo_id=args.hub_model_id,
                            folder_path=checkpoint_dir,
                            commit_message=(
                                f"Checkpoint after {episodes_done} episodes "
                                f"(reward={last_mean_reward:.3f})"
                            ),
                        )
                        print(
                            f"Pushed checkpoint to https://huggingface.co/{args.hub_model_id}"
                        )

    except KeyboardInterrupt as exc:
        run_status = "interrupted"
        failure = exc
        print("\n\n[INTERRUPTED] Saving emergency checkpoint...")
    except Exception as exc:  # pragma: no cover - exercised in live jobs
        run_status = "failed"
        failure = exc
        print(f"\n[ERROR] Training failed: {exc}")
    finally:
        if trajectory_writer:
            trajectory_writer.close()

        checkpoint_name = (
            "final"
            if run_status == "completed"
            else ("interrupted" if run_status == "interrupted" else "failed")
        )
        resume_payload = _snapshot_resume_state(
            args=args,
            tracker=tracker,
            reward_history=reward_history,
            loss_history=loss_history,
            episodes_done=episodes_done,
            completed_batches=completed_batches,
            current_difficulty=current_difficulty,
            current_phase_idx=current_phase_idx,
            episodes_in_phase=episodes_in_phase,
            phase_reward_buffer=phase_reward_buffer,
            epsilon=last_epsilon,
            checkpoint_name=checkpoint_name,
        )
        checkpoint_dir = output_dir / checkpoint_name
        _save_checkpoint_bundle(
            model=model,
            tokenizer=tokenizer,
            output_dir=checkpoint_dir,
            resume_state=resume_payload,
        )

        _write_metrics_bundle(
            output_dir=output_dir,
            args=args,
            reward_history=reward_history,
            loss_history=loss_history,
            tracker=tracker,
        )
        plotter.generate()
        plotter.generate_simple_plot()

        if hub_api is not None and args.hub_model_id:
            commit_message = (
                "Final checkpoint"
                if run_status == "completed"
                else f"{checkpoint_name.capitalize()} checkpoint"
            )
            _upload_folder_with_retry(
                api=hub_api,
                repo_id=args.hub_model_id,
                folder_path=checkpoint_dir,
                commit_message=commit_message,
            )
            print(f"Pushed checkpoint to https://huggingface.co/{args.hub_model_id}")
            _upload_run_artifacts(
                api=hub_api,
                repo_id=args.hub_model_id,
                output_dir=output_dir,
                trajectory_file=args.trajectory_file,
                run_label=checkpoint_name,
            )

        summary = tracker.get_summary()
        if run_status == "completed" and summary:
            reporter.print_final_summary(summary)
        elif summary:
            print(
                f"[{run_status.upper()}] Episodes={summary['total_episodes']} "
                f"Best={summary['best_reward']:+.4f} "
                f"Mean={summary['mean_reward']:+.4f}"
            )

    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
