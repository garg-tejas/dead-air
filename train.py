"""Rollout collection script for Dead Air.

Collects episodes by running the LLM against the environment.
Useful for warm-start data generation or debugging before GRPO training.

Usage:
    uv sync --extra train
    python train.py --model Qwen/Qwen3-1.7B --episodes 50 --output-dir ./outputs/rollouts
"""

import argparse
import json
import os
from typing import Any, Dict, List

from dead_air.server.dispatcher_environment import DispatcherEnvironment


def format_prompt(obs: Dict[str, Any]) -> str:
    """Format observation into a text prompt for the LLM."""
    lines = [
        "# Emergency Dispatch Commander",
        f"Step {obs['step_number']}/{obs['max_steps']}",
        "",
        "## Units",
    ]
    for u in obs.get("unit_statuses", []):
        lines.append(f"- Unit {u['unit_id']}: {u['last_known_status']} at Node {u['last_known_location']}")
    lines.append("")
    lines.append("## Active Calls")
    for c in obs.get("active_calls", []):
        lines.append(f"- Call {c['call_id']}: {c['reported_type']} at Node {c['location']} ({c['caller_tone']}) elapsed={c['time_elapsed']}min")
    lines.append("")
    lines.append("## Recent Events")
    for e in obs.get("recent_events", [])[-5:]:
        lines.append(f"- {e}")
    lines.append("")
    lines.append("## Dispatch Log")
    lines.append(obs.get("dispatch_log", "(empty)")[-500:] if obs.get("dispatch_log") else "(empty)")
    lines.append("")
    lines.append("Choose action: dispatch, reroute, stage, request_mutual_aid, divert, hold, log, verify")
    lines.append("Respond with JSON: {\"action_type\": ..., \"unit_id\": ..., \"call_id\": ...}")
    return "\n".join(lines)


def parse_action(text: str) -> Dict[str, Any]:
    """Parse LLM output into structured action."""
    import re
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        try:
            action = json.loads(json_match.group())
            if "action_type" in action:
                return action
        except json.JSONDecodeError:
            pass
    return {"action_type": "hold"}


def collect_rollout(env: DispatcherEnvironment, model, tokenizer, device: str = "cuda") -> Dict[str, Any]:
    """Collect one episode rollout."""
    import torch
    obs = env.reset(difficulty="learning")
    done = False
    steps = 0
    total_reward = 0.0
    prompts: List[str] = []
    completions: List[str] = []

    while not done and steps < 100:
        prompt = format_prompt(obs)
        prompts.append(prompt)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completions.append(completion)

        action = parse_action(completion)
        obs = env.step(action)
        done = obs.get("done", False)
        if obs.get("reward") is not None:
            total_reward = obs["reward"]
        steps += 1

    return {
        "prompts": prompts,
        "completions": completions,
        "reward": total_reward,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="./outputs/rollouts")
    parser.add_argument("--difficulty", type=str, default="learning")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.device == "cpu":
        model = model.to("cpu")

    print(f"Model loaded. Collecting {args.episodes} rollouts...")
    env = DispatcherEnvironment(seed=42)
    os.makedirs(args.output_dir, exist_ok=True)

    all_rollouts = []
    for ep in range(args.episodes):
        if args.difficulty == "curriculum":
            diff = env.curriculum.phase
        else:
            diff = args.difficulty
        rollout = collect_rollout(env, model, tokenizer, device=args.device)
        rollout["episode"] = ep + 1
        rollout["difficulty"] = diff
        all_rollouts.append(rollout)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{args.episodes}: reward={rollout['reward']:.3f}, steps={rollout['steps']}")

    # Save rollouts
    with open(os.path.join(args.output_dir, "rollouts.json"), "w") as f:
        json.dump(all_rollouts, f, indent=2)

    rewards = [r["reward"] for r in all_rollouts]
    print(f"\nMean reward: {sum(rewards)/len(rewards):.3f}")
    print(f"Rollouts saved to {args.output_dir}/rollouts.json")


if __name__ == "__main__":
    main()
