"""Plot training reward curves."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(rewards: list, output: str = "training_reward.png", title: str = "Dead Air Training Reward"):
    """Plot reward curve with moving average."""
    plt.figure(figsize=(12, 5))

    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.5, label="Episode Reward")
    if len(rewards) >= 10:
        ma = np.convolve(rewards, np.ones(10)/10, mode="valid")
        plt.plot(range(9, len(rewards)), ma, color="red", label="10-ep Moving Avg")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=20, edgecolor="black")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Saved plot to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/rewards.json")
    parser.add_argument("--output", type=str, default="assets/training_reward.png")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    rewards = data.get("rewards", [])

    print(f"Loaded {len(rewards)} episodes")
    print(f"Mean: {np.mean(rewards):.3f}, Max: {np.max(rewards):.3f}, Min: {np.min(rewards):.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_rewards(rewards, output=args.output)


if __name__ == "__main__":
    main()
