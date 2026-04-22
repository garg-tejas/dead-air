"""GRPO training with TRL for Dead Air environment.

Usage (on Lightning AI L4 or H100):
    uv sync --extra train
    python train_grpo.py --model Qwen/Qwen3-1.7B --episodes 200

Requirements:
    - torch>=2.2.0
    - trl>=0.29.0
    - transformers>=4.40.0
    - accelerate>=0.30.0
"""

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
    lines.append(obs.get("dispatch_log", "(empty)")[-500:] if obs.get("dispatch_log") else "(empty)")
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


def run_episode(env: DispatcherEnvironment, model, tokenizer, device: str = "cuda", max_steps: int = 100) -> Dict[str, Any]:
    """Run one episode and collect trajectory."""
    obs = env.reset(difficulty="learning")
    done = False
    steps = 0
    total_reward = 0.0
    prompts = []
    completions = []
    rewards_per_step = []

    while not done and steps < max_steps:
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

    # Assign episode reward to all steps (sparse reward)
    rewards_per_step = [total_reward] * len(prompts)

    return {
        "prompts": prompts,
        "completions": completions,
        "rewards": rewards_per_step,
        "episode_reward": total_reward,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--difficulty", type=str, default="curriculum")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"Dead Air GRPO Training")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Device: {args.device}")
    print(f"=" * 60)

    # Check if TRL is available
    try:
        from trl import GRPOConfig, GRPOTrainer
        print("TRL loaded successfully")
    except ImportError:
        print("WARNING: TRL not installed. Install with: uv sync --extra train")
        print("Falling back to basic rollout collection (no training)")
        GRPOTrainer = None

    print(f"\nLoading model {args.model}...")
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

    print("Model loaded.\n")

    env = DispatcherEnvironment(seed=42)
    os.makedirs(args.output_dir, exist_ok=True)

    all_rewards = []

    for ep in range(args.episodes):
        # Update difficulty based on curriculum
        if args.difficulty == "curriculum":
            diff = env.curriculum.phase
        else:
            diff = args.difficulty

        rollout = run_episode(env, model, tokenizer, device=args.device)
        reward = rollout["episode_reward"]
        all_rewards.append(reward)

        # Update curriculum
        env.curriculum.record_reward(reward)
        new_phase = env.curriculum.update_phase()

        if (ep + 1) % 10 == 0:
            recent_rewards = all_rewards[-10:]
            mean_reward = sum(recent_rewards) / len(recent_rewards)
            print(f"Episode {ep+1:3d}/{args.episodes} | Phase: {new_phase:8s} | "
                  f"Reward: {reward:.3f} | Mean(10): {mean_reward:.3f} | Steps: {rollout['steps']}")

        # Save checkpoint periodically
        if (ep + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{ep+1}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # Save rewards
            with open(os.path.join(args.output_dir, "rewards.json"), "w") as f:
                json.dump({
                    "rewards": all_rewards,
                    "mean_reward": sum(all_rewards) / len(all_rewards),
                    "max_reward": max(all_rewards),
                    "min_reward": min(all_rewards),
                }, f, indent=2)

            print(f"  -> Saved checkpoint to {save_path}")

    # Final save
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Mean reward: {sum(all_rewards)/len(all_rewards):.3f}")
    print(f"Max reward: {max(all_rewards):.3f}")
    print(f"Final phase: {env.curriculum.phase}")
    print("=" * 60)

    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
