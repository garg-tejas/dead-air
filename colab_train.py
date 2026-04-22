"""Colab training script: stripped-down GRPO training for T4 GPU."""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from dead_air.server.dispatcher_environment import DispatcherEnvironment


def greedy_action(obs: Dict) -> Dict:
    """Simple greedy baseline for warm-start."""
    calls = obs.get("active_calls", [])
    units = obs.get("unit_statuses", [])
    if not calls:
        return {"action_type": "hold"}

    # Find highest priority call
    priority = {"cardiac": 3, "trauma": 2, "fire": 1}
    sorted_calls = sorted(calls, key=lambda c: priority.get(c.get("reported_type"), 1), reverse=True)
    target_call = sorted_calls[0]

    # Find closest idle unit
    best_unit = None
    best_dist = float("inf")
    for u in units:
        if u.get("last_known_status") == "idle":
            dist = abs(u.get("last_known_location", 0) - target_call["location"])
            if dist < best_dist:
                best_dist = dist
                best_unit = u["unit_id"]

    if best_unit is not None:
        return {"action_type": "dispatch", "unit_id": best_unit, "call_id": target_call["call_id"]}
    return {"action_type": "hold"}


def run_episodes(env: DispatcherEnvironment, num_episodes: int = 50) -> List[float]:
    """Run episodes with greedy agent and collect rewards."""
    rewards = []
    for ep in range(num_episodes):
        obs = env.reset(difficulty="learning")
        done = False
        steps = 0
        while not done and steps < 100:
            action = greedy_action(obs)
            obs = env.step(action)
            done = obs.get("done", False)
            steps += 1
        rewards.append(obs.get("reward", 0.0) or 0.0)
    return rewards


def plot_rewards(rewards: List[float], output_path: str = "reward_curve.png"):
    """Plot and save reward curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.axhline(y=np.mean(rewards), color="r", linestyle="--", label=f"Mean: {np.mean(rewards):.3f}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Dead Air: Greedy Baseline Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()
    print(f"Saved reward curve to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="reward_curve.png")
    args = parser.parse_args()

    print(f"Running {args.episodes} episodes with greedy baseline...")
    env = DispatcherEnvironment(seed=args.seed)
    rewards = run_episodes(env, num_episodes=args.episodes)

    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Max reward: {np.max(rewards):.3f}")
    print(f"Min reward: {np.min(rewards):.3f}")

    plot_rewards(rewards, output_path=args.output)

    # Save results as JSON
    results = {
        "mean_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "rewards": [float(r) for r in rewards],
    }
    with open("colab_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to colab_results.json")


if __name__ == "__main__":
    main()
