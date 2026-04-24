"""GRPO training with manual loop for Dead Air environment.

Compatible with TRL >= 0.15. Uses standard transformers + PEFT.
vLLM can be used for fast generation (disable LoRA when using vLLM).

Usage:
    python train_grpo.py --model Qwen/Qwen3.5-2B --episodes 200 --batch-size 8
    python train_grpo.py --model Qwen/Qwen3.5-2B --episodes 200 --batch-size 8 --use-vllm --lora-r 0
"""

import argparse
import json
import os
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
    "AVAILABLE ACTIONS (respond with exactly one JSON object):\n"
    '{"action_type":"dispatch","unit_id":0,"call_id":1}\n'
    '{"action_type":"reroute","unit_id":0,"call_id":1}\n'
    '{"action_type":"stage","unit_id":0,"location_node":5}\n'
    '{"action_type":"divert","unit_id":0,"hospital_id":1}\n'
    '{"action_type":"verify","call_id":1}\n'
    '{"action_type":"request_mutual_aid"}\n'
    '{"action_type":"log","note":"short plain text note"}\n'
    '{"action_type":"hold"}\n\n'
    "Think step by step about which calls are most urgent and which units are closest. "
    "Then output ONLY the JSON action on the LAST line. No markdown, no quotes around the JSON."
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


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    """Build a chat-formatted prompt using the tokenizer's chat template.

    Enables thinking mode for Qwen 3/3.5 models so the model reasons before
    outputting the final action.
    """
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
            # Older tokenizers don't accept chat_template_kwargs
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    # Fallback for models without chat template
    return f"{system}\n\n{user}\n\nAssistant:"


def generate_step(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    vllm_engine=None,
    vllm_params=None,
) -> Tuple[str, torch.Tensor]:
    """Generate one completion and return (text, log_prob_sum)."""
    # Use vLLM if available (much faster)
    if vllm_engine is not None and vllm_params is not None:
        outputs = vllm_engine.generate(prompt, vllm_params)
        output = outputs[0].outputs[0]
        completion = output.text
        # vLLM provides cumulative_logprob (sum of all token logprobs)
        log_prob_sum = torch.tensor(
            output.cumulative_logprob if output.cumulative_logprob is not None else 0.0,
            device=device,
        )
        return completion, log_prob_sum

    # Fallback to transformers generate
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)

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

    generated_ids = outputs.sequences[0, inputs.input_ids.shape[1] :]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute log-prob of generated tokens
    scores = torch.stack(outputs.scores, dim=1)  # [1, seq_len, vocab]
    log_probs = F.log_softmax(scores, dim=-1)
    token_log_probs = []
    for i, token_id in enumerate(generated_ids):
        token_log_probs.append(log_probs[0, i, token_id])
    log_prob_sum = torch.stack(token_log_probs).sum()

    return completion, log_prob_sum


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
            return {"action_type": "dispatch", "unit_id": best_unit, "call_id": call["call_id"]}
    return {"action_type": "hold"}


def run_episode(
    model,
    tokenizer,
    env: DeadAirGRPOEnv,
    max_steps: int,
    max_new_tokens: int,
    device: str,
    vllm_engine=None,
    vllm_params=None,
    debug_log: List[Dict] = None,
    episode_id: int = 0,
    epsilon: float = 0.0,
):
    """Run one episode and collect (prompt, completion, log_prob, reward)."""
    steps = []
    for step_idx in range(max_steps):
        if env._obs and env._obs.get("done"):
            break

        obs_text = format_observation(env._obs)
        # Epsilon-greedy: occasionally take greedy action for exploration
        if epsilon > 0 and np.random.rand() < epsilon and env._obs.get("active_calls"):
            greedy_act = greedy_action(env._obs)
            completion = json.dumps(greedy_act)
            prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, obs_text)
            # Compute actual log prob of greedy completion under current policy
            full_text = prompt + completion
            inputs_lp = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
            with torch.no_grad():
                outputs_lp = model(**inputs_lp)
            logits_lp = outputs_lp.logits[:, :-1, :]
            log_probs_all_lp = F.log_softmax(logits_lp, dim=-1)
            prompt_ids_lp = tokenizer(prompt, return_tensors="pt").input_ids[0]
            prompt_len_lp = prompt_ids_lp.shape[0]
            comp_len_lp = inputs_lp.input_ids.shape[1] - prompt_len_lp
            if comp_len_lp > 0:
                token_lps = []
                for j in range(comp_len_lp - 1):
                    tok_id = inputs_lp.input_ids[0, prompt_len_lp + j + 1]
                    token_lps.append(log_probs_all_lp[0, prompt_len_lp + j, tok_id])
                log_prob = torch.stack(token_lps).sum() if token_lps else torch.tensor(0.0, device=device)
            else:
                log_prob = torch.tensor(0.0, device=device)
        else:
            prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, obs_text)
            completion, log_prob = generate_step(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                device,
                vllm_engine=vllm_engine,
                vllm_params=vllm_params,
            )
        env.step(completion)

        # Debug: log what the model generated and what action was parsed
        if debug_log is not None:
            parsed = env._parse_action(completion)
            debug_log.append(
                {
                    "episode": episode_id,
                    "step": step_idx,
                    "prompt": prompt,
                    "completion": completion,
                    "parsed_action": parsed,
                    "log_prob": float(log_prob.cpu()) if log_prob.numel() == 1 else 0.0,
                }
            )

        steps.append(
            {
                "prompt": prompt,
                "completion": completion,
                "log_prob": log_prob,
            }
        )

    reward = env.reward if env.reward is not None else 0.0
    return steps, reward


def compute_grpo_loss(model, tokenizer, episodes, rewards, epsilon=0.2, micro_batch_size=1):
    """Compute GRPO clipped surrogate loss (memory-efficient with micro-batching).

    Calls backward() internally on each micro-batch to avoid accumulating
    computation graphs across all steps. Returns a detached scalar for logging.
    """
    if not episodes or not rewards:
        return 0.0

    # Normalize rewards
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=model.device)
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    # Flatten all steps
    all_steps = []
    for ep_idx, (steps, _) in enumerate(zip(episodes, rewards)):
        for step in steps:
            all_steps.append({
                "prompt": step["prompt"],
                "completion": step["completion"],
                "old_log_prob": step["log_prob"],
                "advantage": advantages[ep_idx].item(),
            })

    total_steps = len(all_steps)
    loss_sum = 0.0

    # Process in micro-batches; call backward() immediately to free graph
    for i in range(0, total_steps, micro_batch_size):
        batch_steps = all_steps[i:i + micro_batch_size]
        prompts = [s["prompt"] for s in batch_steps]
        completions = [s["completion"] for s in batch_steps]
        old_log_probs = torch.stack([s["old_log_prob"] for s in batch_steps]).to(model.device)
        step_advantages = torch.tensor([s["advantage"] for s in batch_steps], device=model.device)

        full_texts = [f"{p}{c}" for p, c in zip(prompts, completions)]
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        log_probs_all = F.log_softmax(logits, dim=-1)

        new_log_probs = []
        for j in range(len(prompts)):
            prompt_ids = tokenizer(prompts[j], return_tensors="pt").input_ids[0]
            prompt_len = prompt_ids.shape[0]
            total_len = inputs.input_ids.shape[1]
            comp_len = total_len - prompt_len

            if comp_len <= 0:
                new_log_probs.append(torch.tensor(0.0, device=model.device))
                continue

            token_lps = []
            for k in range(comp_len - 1):
                tok_id = inputs.input_ids[j, prompt_len + k + 1]
                lp = log_probs_all[j, prompt_len + k, tok_id]
                token_lps.append(lp)

            new_log_probs.append(
                torch.stack(token_lps).sum()
                if token_lps
                else torch.tensor(0.0, device=model.device)
            )

        new_log_probs = torch.stack(new_log_probs)

        # Clipped surrogate loss
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        batch_loss = -torch.min(ratio * step_advantages, clipped * step_advantages).mean()

        # Scale so gradients sum correctly across all steps, then backward immediately
        scaled_loss = batch_loss / max(1, total_steps)
        scaled_loss.backward()

        loss_sum += batch_loss.item()

        # Free memory
        del inputs, outputs, logits, log_probs_all, new_log_probs, ratio, clipped, batch_loss, scaled_loss
        torch.cuda.empty_cache()

    return loss_sum / max(1, total_steps)


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
    parser.add_argument(
        "--lora-r", type=int, default=16, help="LoRA rank (0 = disable LoRA)"
    )
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8, help="Episodes per batch")
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for fast generation. Automatically disables LoRA.",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--debug-file", type=str, default="./outputs/grpo/debug.json")
    parser.add_argument("--use-4bit", action="store_true", help="Load model in 4-bit quantization via bitsandbytes. Saves ~75% VRAM.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon for greedy warmup")
    parser.add_argument("--epsilon-end", type=float, default=0.2, help="Final epsilon after decay")
    parser.add_argument("--epsilon-decay-batches", type=int, default=50, help="Number of batches to decay epsilon")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.debug_file) or ".", exist_ok=True)
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
            print("[WARN] bitsandbytes not installed. Install with: pip install bitsandbytes")
            print("[WARN] Falling back to standard bfloat16/float32 load")
            args.use_4bit = False

    if not args.use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    # Add LoRA (unless disabled for vLLM)
    if args.lora_r > 0 and not args.use_vllm:
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
    elif args.use_vllm:
        print("[vLLM] LoRA disabled for vLLM compatibility")
    else:
        print("LoRA disabled (r=0)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    device = next(model.parameters()).device

    # vLLM setup (optional)
    vllm_engine = None
    vllm_sampling_params = None
    if args.use_vllm:
        try:
            from vllm import LLM, SamplingParams

            print("[vLLM] Initializing vLLM engine...")
            vllm_engine = LLM(
                model=args.model,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=2048,
                dtype="bfloat16" if bf16 else "float16",
                trust_remote_code=True,
            )
            vllm_sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=args.max_completion_length,
            )
            print("[vLLM] Engine ready — generation will be ~10x faster")
        except ImportError as e:
            print(f"[WARN] vLLM not available ({e}), falling back to model.generate()")
            args.use_vllm = False

    # Quick sanity check: run greedy baseline to confirm env rewards are non-zero
    print("\n--- Sanity Check: Greedy Baseline ---")
    from dead_air.server.dispatcher_environment import DispatcherEnvironment

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
                            dist = abs(u.get("last_known_location", 0) - call["location"])
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
    print(
        f"Greedy baseline rewards: {[round(r, 3) for r in greedy_rewards]}"
    )
    print(f"Mean greedy reward: {np.mean(greedy_rewards):.3f}")
    if np.mean(greedy_rewards) <= 0:
        print("[WARN] Greedy baseline reward is <= 0. Environment may not produce learning signal.")

    # Training loop
    num_batches = max(1, args.episodes // args.batch_size)
    reward_history = []
    debug_log = []

    for batch_idx in range(num_batches):
        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")

        # Decay epsilon
        if batch_idx < args.epsilon_decay_batches:
            epsilon = args.epsilon_start - (args.epsilon_start - args.epsilon_end) * (batch_idx / args.epsilon_decay_batches)
        else:
            epsilon = args.epsilon_end
        print(f"Epsilon: {epsilon:.2f}")

        # Collect episodes
        episodes = []
        rewards = []
        for i in range(args.batch_size):
            env = DeadAirGRPOEnv(
                seed=args.seed + batch_idx * args.batch_size + i,
                difficulty=args.difficulty,
            )
            env.reset()
            steps, reward = run_episode(
                model,
                tokenizer,
                env,
                max_steps=100,
                max_new_tokens=args.max_completion_length,
                device=device,
                vllm_engine=vllm_engine,
                vllm_params=vllm_sampling_params,
                debug_log=debug_log,
                episode_id=batch_idx * args.batch_size + i,
                epsilon=epsilon,
            )
            episodes.append(steps)
            rewards.append(reward)
            reward_history.append(reward)

        mean_reward = np.mean(rewards)
        non_zero = sum(1 for r in rewards if r > 0)
        print(
            f"Mean reward: {mean_reward:.4f} | Non-zero: {non_zero}/{len(rewards)}"
        )
        print(
            f"Completion lengths: "
            f"{[len(s['completion']) for ep in episodes for s in ep][:20]}..."
        )

        # Skip update if all rewards are 0 (no learning signal)
        if all(r == 0 for r in rewards):
            print("All rewards zero — skipping update")
            # Save debug log so we can inspect what went wrong
            with open(args.debug_file, "w") as f:
                json.dump(debug_log, f, indent=2)
            continue

        # GRPO update
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss = compute_grpo_loss(model, tokenizer, episodes, rewards)
        optimizer.step()
        print(f"Loss: {loss:.4f}")

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

    # Save debug log
    with open(args.debug_file, "w") as f:
        json.dump(debug_log, f, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Mean reward: {metrics['mean']:.4f}")
    print(f"Std reward:  {metrics['std']:.4f}")
    print(f"Final model: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
