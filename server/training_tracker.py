"""Training progress tracking and visualization for DispatchR GRPO.

Provides real-time console reporting and post-training plots.
"""

import csv
import json
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np


class TrainingTracker:
    """Collects and stores training metrics per batch."""

    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Per-batch records
        self.records: List[Dict[str, Any]] = []

        # Per-episode rewards (flattened across all batches)
        self.episode_rewards: List[float] = []
        self.episode_indices: List[int] = []

        # Moving average windows
        self.reward_window = deque(maxlen=10)
        self.loss_window = deque(maxlen=10)

        # Training start time
        self.start_time = time.time()

        # CSV logger
        self.csv_path = os.path.join(output_dir, "metrics.csv")
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV with headers."""
        headers = [
            "batch", "total_episodes", "difficulty", "epsilon",
            "mean_reward", "std_reward", "min_reward", "max_reward", "median_reward",
            "loss", "batch_time_s", "total_time_s",
            "valid_action_rate", "invalid_action_rate", "hold_rate", "dispatch_rate",
            "avg_response_time", "fatality_count", "calls_missed",
            "reward_ma_5", "reward_ma_10", "loss_ma_5", "loss_ma_10",
            "best_reward", "episodes_since_improvement",
        ]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _write_record_row(self, record: Dict[str, Any]) -> None:
        """Append one normalized record row to the CSV log."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                record["batch"], record["total_episodes"], record["difficulty"], round(record["epsilon"], 3),
                round(record["mean_reward"], 4), round(record["std_reward"], 4),
                round(record["min_reward"], 4), round(record["max_reward"], 4), round(record["median_reward"], 4),
                round(record["loss"], 4), record["batch_time_s"], record["total_time_s"],
                round(record["valid_action_rate"], 3), round(record["invalid_action_rate"], 3),
                round(record["hold_rate"], 3), round(record["dispatch_rate"], 3),
                round(record["avg_response_time"], 2), record["fatality_count"], record["calls_missed"],
                round(record["reward_ma_5"], 4), round(record["reward_ma_10"], 4),
                round(record["loss_ma_5"], 4), round(record["loss_ma_10"], 4),
                round(record["best_reward"], 4), record["episodes_since_improvement"],
            ])

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot for resume workflows."""
        return {
            "records": self.records,
            "episode_rewards": self.episode_rewards,
            "episode_indices": self.episode_indices,
            "reward_window": list(self.reward_window),
            "loss_window": list(self.loss_window),
            "elapsed_time_s": round(time.time() - self.start_time, 2),
        }

    def restore(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore tracker state and rebuild the CSV log."""
        if not state:
            return

        self.records = list(state.get("records", []))
        self.episode_rewards = [float(x) for x in state.get("episode_rewards", [])]
        self.episode_indices = [int(x) for x in state.get("episode_indices", [])]
        self.reward_window = deque(
            [float(x) for x in state.get("reward_window", [])],
            maxlen=self.reward_window.maxlen,
        )
        self.loss_window = deque(
            [float(x) for x in state.get("loss_window", [])],
            maxlen=self.loss_window.maxlen,
        )

        elapsed_time = float(state.get("elapsed_time_s", 0.0))
        self.start_time = time.time() - max(0.0, elapsed_time)

        self._init_csv()
        for record in self.records:
            self._write_record_row(record)

    def log_batch(
        self,
        batch_idx: int,
        rewards: List[float],
        loss: Optional[float],
        epsilon: float,
        difficulty: str,
        batch_time: float,
        env_metrics_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Log metrics for one training batch.

        Args:
            batch_idx: Zero-based batch index.
            rewards: List of episode rewards for this batch.
            loss: GRPO loss value (None if skipped).
            epsilon: Current epsilon value.
            difficulty: Current curriculum difficulty.
            batch_time: Time spent on this batch in seconds.
            env_metrics_list: List of env.get_ground_truth() dicts per episode.

        Returns:
            The recorded metrics dict.
        """
        rewards_arr = np.array(rewards)
        mean_reward = float(np.mean(rewards_arr))
        std_reward = float(np.std(rewards_arr))
        min_reward = float(np.min(rewards_arr))
        max_reward = float(np.max(rewards_arr))
        median_reward = float(np.median(rewards_arr))

        # Update windows
        self.reward_window.append(mean_reward)
        if loss is not None:
            self.loss_window.append(loss)

        # Aggregate env metrics across batch
        def _avg(key: str) -> float:
            vals = [m.get(key, 0.0) for m in env_metrics_list if m.get(key) is not None]
            return float(np.mean(vals)) if vals else 0.0

        def _sum_int(key: str) -> int:
            vals = [m.get(key, 0) for m in env_metrics_list if m.get(key) is not None]
            return int(np.sum(vals)) if vals else 0

        total_episodes = len(self.episode_rewards) + len(rewards)

        # Append episode-level data
        start_ep_idx = len(self.episode_rewards)
        for i, r in enumerate(rewards):
            self.episode_rewards.append(float(r))
            self.episode_indices.append(start_ep_idx + i)

        # Moving averages
        reward_ma_5 = float(np.mean(list(self.reward_window)[-5:])) if len(self.reward_window) >= 5 else mean_reward
        reward_ma_10 = float(np.mean(self.reward_window)) if self.reward_window else mean_reward
        loss_ma_5 = float(np.mean(list(self.loss_window)[-5:])) if len(self.loss_window) >= 5 else (loss or 0.0)
        loss_ma_10 = float(np.mean(self.loss_window)) if self.loss_window else (loss or 0.0)

        # Best tracking
        best_so_far = max(self.episode_rewards) if self.episode_rewards else max_reward
        episodes_since_improvement = 0
        if self.episode_rewards:
            best_idx = self.episode_rewards.index(best_so_far)
            episodes_since_improvement = len(self.episode_rewards) - 1 - best_idx

        record = {
            "batch": batch_idx,
            "total_episodes": total_episodes,
            "difficulty": difficulty,
            "epsilon": epsilon,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "median_reward": median_reward,
            "loss": loss if loss is not None else 0.0,
            "batch_time_s": round(batch_time, 2),
            "total_time_s": round(time.time() - self.start_time, 2),
            "valid_action_rate": _avg("valid_action_rate"),
            "invalid_action_rate": _avg("invalid_action_rate"),
            "hold_rate": _avg("hold_rate"),
            "dispatch_rate": _avg("dispatch_rate"),
            "avg_response_time": _avg("avg_response_time"),
            "fatality_count": _sum_int("fatality_count"),
            "calls_missed": _sum_int("calls_missed"),
            "reward_ma_5": reward_ma_5,
            "reward_ma_10": reward_ma_10,
            "loss_ma_5": loss_ma_5,
            "loss_ma_10": loss_ma_10,
            "best_reward": best_so_far,
            "episodes_since_improvement": episodes_since_improvement,
        }
        self.records.append(record)

        self._write_record_row(record)

        return record

    def get_summary(self) -> Dict[str, Any]:
        """Return overall training summary."""
        if not self.episode_rewards:
            return {}
        rewards_arr = np.array(self.episode_rewards)
        return {
            "total_episodes": len(self.episode_rewards),
            "total_batches": len(self.records),
            "mean_reward": float(np.mean(rewards_arr)),
            "std_reward": float(np.std(rewards_arr)),
            "min_reward": float(np.min(rewards_arr)),
            "max_reward": float(np.max(rewards_arr)),
            "median_reward": float(np.median(rewards_arr)),
            "best_reward": float(np.max(rewards_arr)),
            "final_loss": self.records[-1]["loss"] if self.records else 0.0,
            "total_time_s": round(time.time() - self.start_time, 2),
        }


class ConsoleReporter:
    """Pretty-print training progress to the console."""

    @staticmethod
    def print_header():
        print("\n" + "=" * 100)
        print("DISPATCHR GRPO TRAINING")
        print("=" * 100)

    @staticmethod
    def print_batch_report(record: Dict[str, Any], summary: Dict[str, Any], num_batches: int):
        """Print a formatted progress line for one batch."""
        batch = record["batch"]
        total_eps = record["total_episodes"]
        difficulty = record["difficulty"]
        eps = record["epsilon"]
        mean_r = record["mean_reward"]
        std_r = record["std_reward"]
        loss = record["loss"]
        ma5 = record["reward_ma_5"]
        ma10 = record["reward_ma_10"]
        best = record["best_reward"]
        fatality = record["fatality_count"]
        valid_rate = record["valid_action_rate"]
        dispatch_rate = record["dispatch_rate"]
        batch_time = record["batch_time_s"]
        total_time = record["total_time_s"]
        since_improve = record["episodes_since_improvement"]

        # ETA
        if batch > 0:
            avg_batch_time = total_time / (batch + 1)
            remaining_batches = num_batches - (batch + 1)
            eta_s = avg_batch_time * remaining_batches
            eta_str = f"{int(eta_s // 60)}m{int(eta_s % 60)}s"
        else:
            eta_str = "---"

        print(f"\n  BATCH {batch + 1}/{num_batches}  |  Episodes: {total_eps}  |  Difficulty: {difficulty:>8}  |  eps={eps:.2f}  |  ETA: {eta_str}")
        print("  " + "-" * 96)
        print(f"  Reward    mean={mean_r:+.4f}  std={std_r:.4f}  min={record['min_reward']:+.4f}  max={record['max_reward']:+.4f}  median={record['median_reward']:+.4f}")
        print(f"  Moving    MA5={ma5:+.4f}  MA10={ma10:+.4f}  BEST={best:+.4f}  (last improved {since_improve} eps ago)")
        if loss > 0:
            print(f"  Loss      {loss:.4f}  (MA5={record['loss_ma_5']:.4f}  MA10={record['loss_ma_10']:.4f})")
        else:
            print(f"  Loss      --- (skipped)")
        print(f"  Actions   valid={valid_rate:.1%}  dispatch={dispatch_rate:.1%}  hold={record['hold_rate']:.1%}  invalid={record['invalid_action_rate']:.1%}")
        print(f"  Outcomes  fatalities={fatality}  missed={record['calls_missed']}  response_time={record['avg_response_time']:.1f}min")
        print(f"  Timing    batch={batch_time:.1f}s  total={total_time // 60:.0f}m{total_time % 60:.0f}s")

    @staticmethod
    def print_final_summary(summary: Dict[str, Any]):
        print("\n" + "=" * 100)
        print("TRAINING COMPLETE")
        print("=" * 100)
        print(f"  Total episodes:     {summary['total_episodes']}")
        print(f"  Total batches:      {summary['total_batches']}")
        print(f"  Mean reward:        {summary['mean_reward']:+.4f}")
        print(f"  Std reward:         {summary['std_reward']:.4f}")
        print(f"  Best reward:        {summary['best_reward']:+.4f}")
        print(f"  Median reward:      {summary['median_reward']:+.4f}")
        print(f"  Final loss:         {summary['final_loss']:.4f}")
        print(f"  Total time:         {summary['total_time_s'] // 60:.0f}m{summary['total_time_s'] % 60:.0f}s")
        print("=" * 100)


class TrainingPlotter:
    """Generate training progress plots."""

    def __init__(self, tracker: TrainingTracker):
        self.tracker = tracker

    def generate(self, output_path: Optional[str] = None) -> str:
        """Generate a 6-panel training progress figure.

        Returns:
            Path to the saved PNG file.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not available - skipping plot generation")
            return ""

        if not self.tracker.records:
            print("[WARN] No training records to plot")
            return ""

        records = self.tracker.records
        episode_rewards = self.tracker.episode_rewards
        episode_indices = self.tracker.episode_indices

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.patch.set_facecolor("#0F172A")
        for ax in axes.flat:
            ax.set_facecolor("#1E293B")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        # Color palette
        c_reward = "#22C55E"
        c_loss = "#EF4444"
        c_ma5 = "#F59E0B"
        c_ma10 = "#3B82F6"
        c_dispatch = "#06B6D4"
        c_fatality = "#DC2626"
        c_valid = "#10B981"

        batches = [r["batch"] + 1 for r in records]

        # ── Panel 1: Episode Rewards ──
        ax = axes[0, 0]
        if episode_rewards:
            ax.scatter(episode_indices, episode_rewards, s=8, alpha=0.4, color=c_reward, label="Episode reward")
            # Moving average line
            window = 10
            if len(episode_rewards) >= window:
                ma = []
                for i in range(len(episode_rewards)):
                    start = max(0, i - window + 1)
                    ma.append(np.mean(episode_rewards[start : i + 1]))
                ax.plot(episode_indices, ma, color=c_ma10, linewidth=1.5, label=f"MA{window}")
            ax.axhline(y=0.0, color="white", linestyle="--", alpha=0.3, linewidth=0.8)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.set_title("Episode Rewards")
            ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="white")

        # ── Panel 2: Batch Mean Reward + Loss ──
        ax = axes[0, 1]
        ax2 = ax.twinx()
        ax2.tick_params(colors="white")
        ax2.yaxis.label.set_color("white")

        mean_rewards = [r["mean_reward"] for r in records]
        losses = [r["loss"] for r in records if r["loss"] > 0]
        loss_batches = [batches[i] for i, r in enumerate(records) if r["loss"] > 0]

        ax.plot(batches, mean_rewards, color=c_reward, linewidth=1.5, label="Mean reward")
        if len(records) >= 5:
            ax.plot(batches, [r["reward_ma_5"] for r in records], color=c_ma5, linewidth=1.2, linestyle="--", label="MA5")
        ax.axhline(y=0.0, color="white", linestyle="--", alpha=0.3, linewidth=0.8)
        if losses:
            ax2.plot(loss_batches, losses, color=c_loss, linewidth=1.5, marker="o", markersize=3, label="Loss")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Reward", color=c_reward)
        ax2.set_ylabel("Loss", color=c_loss)
        ax.set_title("Batch Mean Reward & Loss")
        ax.legend(loc="upper left", facecolor="#1E293B", edgecolor="#334155", labelcolor="white")
        if losses:
            ax2.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="white")

        # ── Panel 3: Valid Action Rate ──
        ax = axes[1, 0]
        valid_rates = [r["valid_action_rate"] for r in records]
        ax.plot(batches, valid_rates, color=c_valid, linewidth=1.5)
        ax.fill_between(batches, valid_rates, alpha=0.2, color=c_valid)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Valid Action Rate")
        ax.set_title("Valid Action Rate")

        # ── Panel 4: Dispatch Rate ──
        ax = axes[1, 1]
        dispatch_rates = [r["dispatch_rate"] for r in records]
        hold_rates = [r["hold_rate"] for r in records]
        invalid_rates = [r["invalid_action_rate"] for r in records]
        ax.plot(batches, dispatch_rates, color=c_dispatch, linewidth=1.5, label="Dispatch")
        ax.plot(batches, hold_rates, color="#6B7280", linewidth=1.5, label="Hold")
        ax.plot(batches, invalid_rates, color=c_fatality, linewidth=1.5, label="Invalid")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Action Rate")
        ax.set_title("Action Type Distribution")
        ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="white")

        # ── Panel 5: Fatality Count per Batch ──
        ax = axes[2, 0]
        fatalities = [r["fatality_count"] for r in records]
        ax.bar(batches, fatalities, color=c_fatality, alpha=0.7, width=0.8)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Fatalities")
        ax.set_title("Fatalities per Batch")

        # ── Panel 6: Epsilon Decay & Difficulty ──
        ax = axes[2, 1]
        epsilons = [r["epsilon"] for r in records]
        ax.plot(batches, epsilons, color="#8B5CF6", linewidth=1.5, label="Epsilon")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Epsilon", color="#8B5CF6")
        ax.set_title("Exploration & Curriculum")

        # Difficulty as background regions
        difficulties = [r["difficulty"] for r in records]
        unique_diffs = []
        for d in difficulties:
            if not unique_diffs or unique_diffs[-1] != d:
                unique_diffs.append(d)
        diff_colors = {"warmup": "#10B981", "learning": "#3B82F6", "advanced": "#F59E0B", "expert": "#EF4444"}
        ax2_diff = ax.twinx()
        ax2_diff.tick_params(colors="white")
        ax2_diff.set_ylim(-0.5, len(unique_diffs) - 0.5)
        ax2_diff.set_yticks(range(len(unique_diffs)))
        ax2_diff.set_yticklabels(unique_diffs)
        ax2_diff.set_ylabel("Difficulty", color="white")

        # Shade difficulty regions
        current_diff = difficulties[0]
        start_batch = batches[0]
        for i, d in enumerate(difficulties):
            if d != current_diff or i == len(difficulties) - 1:
                end_batch = batches[i]
                color = diff_colors.get(current_diff, "#334155")
                ax.axvspan(start_batch - 0.5, end_batch + 0.5, alpha=0.1, color=color)
                current_diff = d
                start_batch = end_batch
        # Last region
        ax.axvspan(start_batch - 0.5, batches[-1] + 0.5, alpha=0.1, color=diff_colors.get(current_diff, "#334155"))

        ax.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="white")

        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(top=0.95)
        fig.suptitle("DispatchR GRPO Training Progress", color="white", fontsize=16, fontweight="bold")

        if output_path is None:
            output_path = os.path.join(self.tracker.output_dir, "training_progress.png")
        plt.savefig(output_path, dpi=150, facecolor="#0F172A", bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved training progress plot to {output_path}")
        return output_path

    def generate_simple_plot(self, output_path: Optional[str] = None) -> str:
        """Generate a minimal 2-panel plot (reward + loss only) for quick checks."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return ""

        if not self.tracker.records:
            return ""

        records = self.tracker.records
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("white")

        batches = [r["batch"] + 1 for r in records]

        # Reward
        ax = axes[0]
        ax.plot(batches, [r["mean_reward"] for r in records], color="#22C55E", linewidth=1.5)
        ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Reward Curve")

        # Loss
        ax = axes[1]
        losses = [(batches[i], r["loss"]) for i, r in enumerate(records) if r["loss"] > 0]
        if losses:
            xs, ys = zip(*losses)
            ax.plot(xs, ys, color="#EF4444", linewidth=1.5, marker="o", markersize=3)
        ax.set_xlabel("Batch")
        ax.set_ylabel("GRPO Loss")
        ax.set_title("Loss Curve")

        plt.tight_layout()
        if output_path is None:
            output_path = os.path.join(self.tracker.output_dir, "training_curves.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved curves to {output_path}")
        return output_path
