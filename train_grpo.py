"""GRPO training with TRL for Dead Air environment (manual loop version).

Compatible with TRL >= 0.15 where environment_factory was removed.
Uses standard transformers + PEFT (no Unsloth required).

Usage:
    python train_grpo.py --model Qwen/Qwen3-1.7B --episodes 200 --no-vllm
    python train_grpo.py --model Qwen/Qwen3.5-2B --episodes 200 --no-vllm
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Env wrapper
try:
    from server.grpo_env_wrapper import DeadAirGRPOEnv
except ImportError:
    sys.path.insert(0, ".")
    from server.grpo_env_wrapper import DeadAirGRPOEnv


SYSTEM_PROMPT = (
    "You are an emergency dispatch commander for a 20-node city. "
    "You have 6 units and must respond to emergency calls. "
    "Minimize fatalities and response time.\n\n"
    "AVAILABLE ACTIONS:\n"
    "dispatch(unit_id=<int>, call_id=<int>)\n"
    "reroute(unit_id=<int>, call_id=<int>)\n"
    "stage(unit_id=<int>, location_node=<int>)\n"
    "divert(unit_id=<int>, hospital_id=<int>)\n"
    "verify(call_id=<int>)\n"
    "request_mutual_aid()\n"
    "log(note=\"<text>\")\n"
    "hold()\n\n"
    "Think step by step about which calls are most urgent and which units are closest. "
    "Then output the action on the LAST line and ONLY the action. No markdown, no quotes."
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
    for c in obs.get("active_calls", []):
        assigned = f" (Unit {c['assigned_unit']})" if c.get("assigned_unit") else ""
        lines.append(
            f"- Call {c['call_id']}: {c['reported_type']} at Node {c['location']} ({c['caller_tone']}) elapsed={c['time_elapsed']}min{assigned}"
        )
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


def generate_step(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> Tuple[str, torch.Tensor]:
    """Generate one completion and return (text, log_prob_sum)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute log-prob of generated tokens
    scores = torch.stack(outputs.scores, dim=1)  # [1, seq_len, vocab]
    log_probs = F.log_softmax(scores, dim=-1)
    token_log_probs = []
    for i, token_id in enumerate(generated_ids):
        token_log_probs.append(log_probs[0, i, token_id])
    log_prob_sum = torch.stack(token_log_probs).sum()

    return completion, log_prob_sum


def run_episode(model, tokenizer, env: DeadAirGRPOEnv, max_steps: int, max_new_tokens: int, device: str):
    """Run one episode and collect (prompt, completion, log_prob, reward)."""
    steps = []
    for step in range(max_steps):
        if env._obs and env._obs.get("done"):
            break

        prompt = f"{SYSTEM_PROMPT}\n\n{format_observation(env._obs)}\n\nAction:"
        completion, log_prob = generate_step(model, tokenizer, prompt, max_new_tokens, device)
        env.step(completion)

        steps.append({
            "prompt": prompt,
            "completion": completion,
            "log_prob": log_prob,
        })

    reward = env.reward if env.reward is not None else 0.0
    return steps, reward


def compute_grpo_loss(model, tokenizer, episodes, rewards, epsilon=0.2):
    """Compute GRPO clipped surrogate loss."""
    if not episodes or not rewards:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    # Normalize rewards
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=model.device)
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    # Flatten all steps
    all_prompts = []
    all_completions = []
    all_old_log_probs = []
    episode_indices = []

    for ep_idx, (steps, _) in enumerate(zip(episodes, rewards)):
        for step in steps:
            all_prompts.append(step["prompt"])
            all_completions.append(step["completion"])
            all_old_log_probs.append(step["log_prob"])
            episode_indices.append(ep_idx)

    # Tokenize prompts + completions
    full_texts = [f"{p}\n{c}" for p, c in zip(all_prompts, all_completions)]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)

    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]
    log_probs_all = F.log_softmax(logits, dim=-1)

    # Compute new log probs for completion tokens only
    new_log_probs = []
    for i in range(len(all_prompts)):
        prompt_ids = tokenizer(all_prompts[i], return_tensors="pt").input_ids[0]
        prompt_len = prompt_ids.shape[0]
        total_len = inputs.input_ids.shape[1]
        comp_len = total_len - prompt_len

        if comp_len <= 0:
            new_log_probs.append(torch.tensor(0.0, device=model.device))
            continue

        token_lps = []
        for j in range(comp_len - 1):  # -1 because logits are shifted
            tok_id = inputs.input_ids[i, prompt_len + j + 1]
            lp = log_probs_all[i, prompt_len + j, tok_id]
            token_lps.append(lp)

        new_log_probs.append(torch.stack(token_lps).sum() if token_lps else torch.tensor(0.0, device=model.device))

    new_log_probs = torch.stack(new_log_probs)
    old_log_probs = torch.stack(all_old_log_probs).to(model.device)

    # Match advantages to steps
    step_advantages = advantages[torch.tensor(episode_indices, device=model.device)]

    # Clipped surrogate loss
    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    loss = -torch.min(ratio * step_advantages, clipped * step_advantages).mean()

    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--difficulty", type=str, default="learning")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8, help="Episodes per batch")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for fast generation (requires vllm installed)")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.25)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Dead Air GRPO Training (Manual Loop)")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Use vLLM: {args.use_vllm}")
    print("=" * 60)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"BF16 supported: {bf16}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
    )

    # Add LoRA
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = next(model.parameters()).device

    # vLLM setup (optional)
    vllm_engine = None
    if args.use_vllm:
        try:
            from vllm import LLM, SamplingParams
            print("[vLLM] Initializing vLLM engine...")
            vllm_engine = LLM(
                model=args.model,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                dtype="bfloat16" if bf16 else "float16",
            )
            vllm_sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=args.max_completion_length,
            )
            print("[vLLM] Engine ready")
        except ImportError:
            print("[WARN] vLLM not available, falling back to model.generate()")
            args.use_vllm = False

    # Training loop
    num_batches = max(1, args.episodes // args.batch_size)
    reward_history = []

    for batch_idx in range(num_batches):
        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")

        # If using vLLM, merge LoRA and reload engine
        if args.use_vllm and vllm_engine is not None:
            print("[vLLM] Merging LoRA weights for generation...")
            merged_path = os.path.join(args.output_dir, "_temp_merged")
            os.makedirs(merged_path, exist_ok=True)
            model.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)
            # Note: vLLM 0.10.2 may not support dynamic LoRA loading.
            # For simplicity, we fall back to model.generate() when LoRA is active.
            print("[vLLM] Dynamic LoRA not supported in vLLM 0.10.2 — using model.generate()")
            args.use_vllm = False

        # Collect episodes
        episodes = []
        rewards = []
        for i in range(args.batch_size):
            env = DeadAirGRPOEnv(seed=args.seed + batch_idx * args.batch_size + i, difficulty=args.difficulty)
            env.reset()
            steps, reward = run_episode(model, tokenizer, env, max_steps=25, max_new_tokens=args.max_completion_length, device=device)
            episodes.append(steps)
            rewards.append(reward)
            reward_history.append(reward)

        mean_reward = np.mean(rewards)
        print(f"Mean reward: {mean_reward:.4f} | Non-zero: {sum(1 for r in rewards if r > 0)}/{len(rewards)}")

        # Skip update if all rewards are 0 (no learning signal)
        if all(r == 0 for r in rewards):
            print("All rewards zero — skipping update")
            continue

        # GRPO update
        loss = compute_grpo_loss(model, tokenizer, episodes, rewards)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Loss: {loss.item():.4f}")

        # Save checkpoint
        episodes_done = (batch_idx + 1) * args.batch_size
        if episodes_done % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{episodes_done}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved checkpoint to {save_path}")

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

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Mean reward: {metrics['mean']:.4f}")
    print(f"Std reward:  {metrics['std']:.4f}")
    print(f"Final model: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
