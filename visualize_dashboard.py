"""Create project visualizations for DispatchR.

This script generates publication-ready static charts from:
- reward JSON files (e.g. colab_results.json, outputs/rewards.json)
- training metrics CSV (outputs/*/metrics.csv)
- exported episode traces (episode.json from export_episode.py)

Examples:
    python visualize_dashboard.py --rewards colab_results.json
    python visualize_dashboard.py --rewards outputs/rewards.json --metrics outputs/unsloth_grpo/metrics.csv
    python visualize_dashboard.py --episode episode.json --outdir assets/visualizations
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Keep colors readable on both light and dark backgrounds.
COLOR = {
    "primary": "#2563EB",
    "success": "#16A34A",
    "warn": "#EA580C",
    "danger": "#DC2626",
    "muted": "#64748B",
    "purple": "#7C3AED",
    "teal": "#0D9488",
}


def ensure_outdir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def moving_average(values: List[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return np.array([float(np.mean(arr))] * len(arr), dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    valid = np.convolve(arr, kernel, mode="valid")
    head = np.array([float(np.mean(arr[: i + 1])) for i in range(window - 1)], dtype=float)
    return np.concatenate([head, valid])


def load_rewards(path: str) -> List[float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [float(x) for x in data]
    if isinstance(data, dict):
        rewards = data.get("rewards", [])
        return [float(x) for x in rewards]
    return []


def plot_rewards_overview(rewards: List[float], output: Path) -> Optional[Path]:
    if not rewards:
        return None

    x = np.arange(1, len(rewards) + 1)
    ma10 = moving_average(rewards, 10)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("DispatchR Reward Overview", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(x, rewards, color=COLOR["primary"], alpha=0.55, linewidth=1.8, label="Reward")
    ax.plot(x, ma10, color=COLOR["danger"], linewidth=2.2, label="MA(10)")
    ax.set_title("Episode Reward Trend")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.hist(rewards, bins=min(20, max(5, len(rewards) // 2)), color=COLOR["teal"], alpha=0.8, edgecolor="white")
    ax.set_title("Reward Distribution")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.boxplot(rewards, vert=True, patch_artist=True, boxprops={"facecolor": "#DBEAFE", "edgecolor": COLOR["primary"]})
    ax.set_title("Reward Spread")
    ax.set_ylabel("Reward")
    ax.set_xticks([1])
    ax.set_xticklabels(["All Episodes"])
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(x, np.maximum.accumulate(np.asarray(rewards)), color=COLOR["success"], linewidth=2.2)
    ax.set_title("Best-So-Far Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Best Reward")
    ax.grid(alpha=0.25)

    stats = {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
    }
    fig.text(
        0.01,
        0.01,
        f"Episodes={len(rewards)}  Mean={stats['mean']:.3f}  Std={stats['std']:.3f}  Min={stats['min']:.3f}  Max={stats['max']:.3f}",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def load_metrics_csv(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    continue
            rows.append(parsed)
    return rows


def col(rows: List[Dict[str, float]], key: str) -> np.ndarray:
    vals = [r[key] for r in rows if key in r]
    return np.asarray(vals, dtype=float)


def plot_training_metrics(rows: List[Dict[str, float]], output: Path) -> Optional[Path]:
    if not rows:
        return None

    batch = col(rows, "batch")
    if len(batch) == 0:
        batch = np.arange(1, len(rows) + 1)
    else:
        batch = batch + 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("DispatchR Training Metrics", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    mean_reward = col(rows, "mean_reward")
    reward_ma_10 = col(rows, "reward_ma_10")
    if len(mean_reward):
        ax.plot(batch[: len(mean_reward)], mean_reward, color=COLOR["primary"], linewidth=2, label="Batch mean reward")
    if len(reward_ma_10):
        ax.plot(batch[: len(reward_ma_10)], reward_ma_10, color=COLOR["danger"], linewidth=2, label="Reward MA(10)")
    ax.set_title("Reward by Batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    loss = col(rows, "loss")
    loss_ma_10 = col(rows, "loss_ma_10")
    if len(loss):
        ax.plot(batch[: len(loss)], loss, color=COLOR["purple"], linewidth=1.8, label="Loss")
    if len(loss_ma_10):
        ax.plot(batch[: len(loss_ma_10)], loss_ma_10, color=COLOR["warn"], linewidth=2.2, label="Loss MA(10)")
    ax.set_title("Loss by Batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    valid = col(rows, "valid_action_rate")
    invalid = col(rows, "invalid_action_rate")
    hold = col(rows, "hold_rate")
    if len(valid):
        ax.plot(batch[: len(valid)], valid, color=COLOR["success"], linewidth=2, label="Valid")
    if len(invalid):
        ax.plot(batch[: len(invalid)], invalid, color=COLOR["danger"], linewidth=2, label="Invalid")
    if len(hold):
        ax.plot(batch[: len(hold)], hold, color=COLOR["muted"], linewidth=2, label="Hold")
    ax.set_title("Action Quality Rates")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    fatality = col(rows, "fatality_count")
    missed = col(rows, "calls_missed")
    response_time = col(rows, "avg_response_time")
    if len(fatality):
        ax.plot(batch[: len(fatality)], fatality, color=COLOR["danger"], linewidth=2, label="Fatalities")
    if len(missed):
        ax.plot(batch[: len(missed)], missed, color=COLOR["warn"], linewidth=2, label="Calls missed")
    if len(response_time):
        ax2 = ax.twinx()
        ax2.plot(batch[: len(response_time)], response_time, color=COLOR["teal"], linewidth=2, linestyle="--", label="Avg response time")
        ax2.set_ylabel("Response time (min)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax.legend()
    ax.set_title("Operational Outcomes")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def load_episode(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def plot_episode_timeline(episode: List[Dict], output: Path) -> Optional[Path]:
    if not episode:
        return None

    steps = [int(s.get("step", i + 1)) for i, s in enumerate(episode)]

    active_calls = []
    resolved_calls = []
    fatality_cumulative = []

    idle_units = []
    en_route_units = []
    on_scene_units = []
    unavailable_units = []

    running_fatalities = 0
    seen_fatal = set()

    for state in episode:
        calls = state.get("calls", [])
        units = state.get("units", [])

        active = sum(1 for c in calls if not c.get("resolved", False))
        resolved = sum(1 for c in calls if c.get("resolved", False))
        active_calls.append(active)
        resolved_calls.append(resolved)

        for c in calls:
            if c.get("fatality", False):
                cid = c.get("call_id")
                if cid not in seen_fatal:
                    seen_fatal.add(cid)
                    running_fatalities += 1
        fatality_cumulative.append(running_fatalities)

        idle_units.append(sum(1 for u in units if u.get("status") == "idle"))
        en_route_units.append(sum(1 for u in units if u.get("status") == "en_route"))
        on_scene_units.append(sum(1 for u in units if u.get("status") == "on_scene"))
        unavailable_units.append(sum(1 for u in units if u.get("status") in ("returning", "out_of_service")))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("DispatchR Episode Timeline", fontsize=16, fontweight="bold")

    ax = axes[0]
    ax.plot(steps, active_calls, color=COLOR["danger"], linewidth=2, label="Active calls")
    ax.plot(steps, resolved_calls, color=COLOR["success"], linewidth=2, label="Resolved calls")
    ax.plot(steps, fatality_cumulative, color="#111827", linewidth=2.2, linestyle="--", label="Fatalities (cum)")
    ax.set_ylabel("Calls")
    ax.set_title("Call Pressure and Outcomes")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.stackplot(
        steps,
        idle_units,
        en_route_units,
        on_scene_units,
        unavailable_units,
        labels=["Idle", "En route", "On scene", "Unavailable"],
        colors=["#93C5FD", "#BFDBFE", "#86EFAC", "#FDBA74"],
        alpha=0.9,
    )
    ax.set_title("Unit Utilization Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Units")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    final_reward = episode[-1].get("reward")
    if final_reward is not None:
        fig.text(0.01, 0.01, f"Final Reward: {float(final_reward):.3f}", fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DispatchR visualizations")
    parser.add_argument("--rewards", type=str, default="", help="Path to rewards JSON (e.g. colab_results.json)")
    parser.add_argument("--metrics", type=str, default="", help="Path to metrics.csv from training tracker")
    parser.add_argument("--episode", type=str, default="", help="Path to exported episode JSON")
    parser.add_argument("--outdir", type=str, default="assets/visualizations", help="Directory for generated images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_outdir(args.outdir)

    generated: List[Path] = []

    if args.rewards:
        rewards = load_rewards(args.rewards)
        out = plot_rewards_overview(rewards, outdir / "rewards_overview.png")
        if out:
            generated.append(out)

    if args.metrics:
        rows = load_metrics_csv(args.metrics)
        out = plot_training_metrics(rows, outdir / "training_metrics.png")
        if out:
            generated.append(out)

    if args.episode:
        episode = load_episode(args.episode)
        out = plot_episode_timeline(episode, outdir / "episode_timeline.png")
        if out:
            generated.append(out)

    if not generated:
        print("No visualizations generated. Provide at least one input via --rewards, --metrics, or --episode.")
        return

    print("Generated visualizations:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
