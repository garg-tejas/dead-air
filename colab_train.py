"""Colab debug training script for Dead Air with Unsloth (or fallback).

Usage in Colab:
    !python colab_train.py --episodes 10 --batch-size 4 --debug-steps 3

This script is designed for DEBUGGING — it prints verbose output at every
step so you can trace exactly what the model is doing.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# ------------------------------------------------------------------
# 0.  Try Unsloth, fallback to standard transformers
# ------------------------------------------------------------------
USE_UNSLOTH = False
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
    print("[OK] Unsloth loaded — using fast 4-bit kernels")
except ImportError:
    print("[WARN] Unsloth not available — falling back to standard transformers")
    print("  Install with: !pip install unsloth")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------
# 1.  Environment wrapper (self-contained, no openenv import needed)
# ------------------------------------------------------------------
# We inline a minimal env wrapper so the script works even if the
# repo structure is not perfectly set up.

# Import from repo if available
try:
    from server.grpo_env_wrapper import DeadAirGRPOEnv
    print("[OK] Imported DeadAirGRPOEnv from repo")
except ImportError:
    sys.path.insert(0, ".")
    from server.grpo_env_wrapper import DeadAirGRPOEnv
    print("[OK] Imported DeadAirGRPOEnv from repo (after path fix)")


# ------------------------------------------------------------------
# 2.  Action parser (same as grpo_env_wrapper)
# ------------------------------------------------------------------
import re

def parse_action(text: str) -> Dict:
    """Parse completion text into action dict."""
    text = text.strip()
    if not text:
        return {"action_type": "hold"}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    action_line = lines[-1] if lines else text.strip()
    lower = action_line.lower()

    m = re.match(r"dispatch\s*\(\s*unit_id\s*=\s*(\d+)\s*,\s*call_id\s*=\s*(\d+)\s*\)", lower)
    if m:
        return {"action_type": "dispatch", "unit_id": int(m.group(1)), "call_id": int(m.group(2))}

    m = re.match(r"reroute\s*\(\s*unit_id\s*=\s*(\d+)\s*,\s*call_id\s*=\s*(\d+)\s*\)", lower)
    if m:
        return {"action_type": "reroute", "unit_id": int(m.group(1)), "call_id": int(m.group(2))}

    m = re.match(r"stage\s*\(\s*unit_id\s*=\s*(\d+)\s*,\s*location_node\s*=\s*(\d+)\s*\)", lower)
    if m:
        return {"action_type": "stage", "unit_id": int(m.group(1)), "location_node": int(m.group(2))}

    m = re.match(r"divert\s*\(\s*unit_id\s*=\s*(\d+)\s*,\s*hospital_id\s*=\s*(\d+)\s*\)", lower)
    if m:
        return {"action_type": "divert", "unit_id": int(m.group(1)), "hospital_id": int(m.group(2))}

    m = re.match(r"verify\s*\(\s*call_id\s*=\s*(\d+)\s*\)", lower)
    if m:
        return {"action_type": "verify", "call_id": int(m.group(1))}

    m = re.match(r'log\s*\(\s*note\s*=\s*"([^"]*)"\s*\)', lower)
    if m:
        return {"action_type": "log", "note": m.group(1)}

    if lower.startswith("hold") or lower.startswith("wait"):
        return {"action_type": "hold"}

    if lower.startswith("request_mutual_aid") or lower.startswith("mutual_aid"):
        return {"action_type": "request_mutual_aid"}

    return {"action_type": "hold"}


# ------------------------------------------------------------------
# 3.  Prompt formatter (same as grpo_env_wrapper)
# ------------------------------------------------------------------
def format_prompt(obs: Dict) -> str:
    """Format observation into a prompt for the LLM."""
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
    lines.append(
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
    return "\n".join(lines)


# ------------------------------------------------------------------
# 4.  Model loading
# ------------------------------------------------------------------
def load_model(model_name: str, use_unsloth: bool = True):
    """Load model and tokenizer."""
    if use_unsloth and USE_UNSLOTH:
        print(f"[LOAD] Loading {model_name} via Unsloth (4-bit)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
            max_seq_length=4096,
            dtype=torch.bfloat16,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            use_rslora=True,
            bias="none",
        )
        print("[LOAD] LoRA adapters added")
    else:
        print(f"[LOAD] Loading {model_name} via standard Transformers...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ------------------------------------------------------------------
# 5.  Generation
# ------------------------------------------------------------------
def generate_action(model, tokenizer, prompt: str, max_new_tokens: int = 512, device: str = "cuda") -> str:
    """Generate one action completion from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return completion


# ------------------------------------------------------------------
# 6.  Debug training loop
# ------------------------------------------------------------------
def run_debug_training(
    model,
    tokenizer,
    num_episodes: int = 10,
    max_steps: int = 25,
    debug_steps: int = 3,
    difficulty: str = "learning",
    seed: int = 42,
) -> Tuple[List[float], List[Dict]]:
    """Run training episodes with verbose debug output.

    Args:
        model: The LLM.
        tokenizer: Matching tokenizer.
        num_episodes: Number of episodes to run.
        max_steps: Max steps per episode.
        debug_steps: Print verbose output for first N steps of first N episodes.
        difficulty: Difficulty phase.
        seed: Random seed.

    Returns:
        rewards: List of episode rewards.
        logs: Detailed logs for analysis.
    """
    rewards = []
    logs = []
    device = next(model.parameters()).device

    for ep in range(num_episodes):
        env = DeadAirGRPOEnv(seed=seed + ep, difficulty=difficulty)
        env.reset()
        ep_reward = None
        ep_log = {"episode": ep, "steps": [], "reward": None}

        verbose = ep < debug_steps

        if verbose:
            print(f"\n{'='*60}")
            print(f"EPISODE {ep + 1}/{num_episodes}")
            print(f"{'='*60}")

        for step in range(max_steps):
            if env._obs and env._obs.get("done"):
                break

            prompt = format_prompt(env._obs)
            completion = generate_action(model, tokenizer, prompt, device=device)
            action = parse_action(completion)
            result = env.step(completion)  # env.step() parses internally

            if env._obs.get("reward") is not None:
                ep_reward = env._obs["reward"]

            step_log = {
                "step": step,
                "prompt_len": len(prompt),
                "completion": completion,
                "parsed_action": action,
                "result": result,
            }
            ep_log["steps"].append(step_log)

            if verbose and step < debug_steps:
                print(f"\n--- Step {step} ---")
                print(f"PROMPT (last 200 chars): ...{prompt[-200:]}")
                print(f"COMPLETION: {completion[:200]}{'...' if len(completion) > 200 else ''}")
                print(f"PARSED ACTION: {action}")
                print(f"RESULT: {result[:100]}{'...' if len(result) > 100 else ''}")
                print(f"Active calls: {len(env._obs.get('active_calls', []))}")
                print(f"Done: {env._obs.get('done')}")

        ep_log["reward"] = ep_reward
        rewards.append(ep_reward if ep_reward is not None else 0.0)
        logs.append(ep_log)

        if verbose:
            print(f"\nEPISODE {ep + 1} REWARD: {ep_reward}")

        # Print progress every 10 episodes
        if (ep + 1) % 10 == 0:
            recent = rewards[-10:]
            print(f"Progress: {ep + 1}/{num_episodes} | Recent mean reward: {np.mean(recent):.4f}")

    return rewards, logs


# ------------------------------------------------------------------
# 7.  Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Dead Air debug training on Colab")
    parser.add_argument("--model", type=str, default="unsloth/Qwen3.5-2B", help="HF model ID")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=25, help="Max steps per episode")
    parser.add_argument("--debug-steps", type=int, default=3, help="Verbose output for first N steps/episodes")
    parser.add_argument("--difficulty", type=str, default="learning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./outputs/colab_debug")
    parser.add_argument("--no-unsloth", action="store_true", help="Force standard transformers (no Unsloth)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Dead Air Colab Debug Training")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Unsloth: {not args.no_unsloth and USE_UNSLOTH}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(args.model, use_unsloth=not args.no_unsloth)

    # Run training
    rewards, logs = run_debug_training(
        model,
        tokenizer,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        debug_steps=args.debug_steps,
        difficulty=args.difficulty,
        seed=args.seed,
    )

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Mean reward: {np.mean(rewards):.4f}")
    print(f"Std reward:  {np.std(rewards):.4f}")
    print(f"Max reward:  {np.max(rewards):.4f}")
    print(f"Min reward:  {np.min(rewards):.4f}")
    print(f"Non-zero rewards: {sum(1 for r in rewards if r > 0)}/{len(rewards)}")

    # Save results
    results = {
        "config": vars(args),
        "rewards": [float(r) for r in rewards],
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "max": float(np.max(rewards)),
        "min": float(np.min(rewards)),
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Save detailed logs
    logs_path = os.path.join(args.output_dir, "logs.json")
    with open(logs_path, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"Saved detailed logs to {logs_path}")

    # Save model (LoRA adapters only)
    model_path = os.path.join(args.output_dir, "model")
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
