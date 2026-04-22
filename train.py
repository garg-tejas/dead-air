"""GRPO training script for Dead Air using HF TRL."""

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    lines.append(obs.get("dispatch_log", "(empty)"))
    lines.append("")
    lines.append("Choose action: dispatch, reroute, stage, request_mutual_aid, divert, hold, log, verify")
    lines.append("Respond with JSON: {\"action_type\": ..., \"unit_id\": ..., \"call_id\": ...}")
    return "\n".join(lines)


def parse_action(text: str) -> Dict[str, Any]:
    """Parse LLM output into structured action."""
    import re
    # Try to find JSON in the output
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        try:
            action = json.loads(json_match.group())
            if "action_type" in action:
                return action
        except json.JSONDecodeError:
            pass
    # Fallback to hold
    return {"action_type": "hold"}


def collect_rollout(env: DispatcherEnvironment, model, tokenizer, device: str = "cuda") -> Dict[str, Any]:
    """Collect one episode rollout."""
    obs = env.reset(difficulty="learning")
    done = False
    steps = 0
    total_reward = 0.0
    prompts = []
    completions = []

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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Loading model {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.device == "cpu":
        model = model.to("cpu")

    print(f"Model loaded. Starting {args.episodes} episodes...")
    env = DispatcherEnvironment(seed=42)
    rewards = []

    for ep in range(args.episodes):
        rollout = collect_rollout(env, model, tokenizer, device=args.device)
        rewards.append(rollout["reward"])
        print(f"Episode {ep+1}/{args.episodes}: reward={rollout['reward']:.3f}, steps={rollout['steps']}")

    print(f"\nMean reward: {sum(rewards)/len(rewards):.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "rewards.json"), "w") as f:
        json.dump({"rewards": rewards}, f)

    if not args.dry_run:
        print(f"Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
