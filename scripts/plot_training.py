#!/usr/bin/env python3
"""Plot DispatchR training curves from a saved metrics.json.

Reads the metrics.json written by train_unsloth_grpo.py and generates a
multi-panel figure covering reward curves, loss, action rates, fatalities,
epsilon decay, and curriculum phases.

Usage:
    # Single run:
    python scripts/plot_training.py --metrics outputs/unsloth_grpo/metrics.json

    # Compare two runs (before vs after, or two different configs):
    python scripts/plot_training.py \\
        --metrics outputs/run_a/metrics.json \\
        --compare outputs/run_b/metrics.json \\
        --labels "Baseline" "With curriculum"

    # Quick reward-only view:
    python scripts/plot_training.py --metrics outputs/metrics.json --quick
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ── Style constants ────────────────────────────────────────────────────────────

BG_DARK    = "#0F172A"
BG_PANEL   = "#1E293B"
GRID_COLOR = "#334155"
TEXT_COLOR = "white"

C_REWARD   = "#22C55E"
C_MA5      = "#F59E0B"
C_MA10     = "#3B82F6"
C_LOSS     = "#EF4444"
C_DISPATCH = "#06B6D4"
C_HOLD     = "#6B7280"
C_INVALID  = "#DC2626"
C_VALID    = "#10B981"
C_EPSILON  = "#8B5CF6"
C_FATALITY = "#DC2626"

DIFF_COLORS = {
    "warmup":   "#10B981",
    "learning": "#3B82F6",
    "advanced": "#F59E0B",
    "expert":   "#EF4444",
}

COMPARE_COLORS = [
    ("#22C55E", "#3B82F6"),   # pair 0: green / blue
    ("#F59E0B", "#EC4899"),   # pair 1: amber / pink
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_metrics(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def _moving_avg(data: List[float], window: int) -> List[float]:
    ma = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        ma.append(float(np.mean(data[start:i + 1])))
    return ma


def _ax_style(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.4, alpha=0.6)


def _shade_curriculum(ax, records: List[Dict]) -> None:
    """Shade background by curriculum difficulty phase."""
    if not records:
        return
    difficulties = [r["difficulty"] for r in records]
    batches      = [r["batch"] + 1 for r in records]
    start_b, cur_d = batches[0], difficulties[0]
    for b, d in zip(batches[1:], difficulties[1:]):
        if d != cur_d:
            ax.axvspan(start_b - 0.5, b - 0.5,
                       alpha=0.08, color=DIFF_COLORS.get(cur_d, GRID_COLOR),
                       linewidth=0)
            cur_d, start_b = d, b
    ax.axvspan(start_b - 0.5, batches[-1] + 0.5,
               alpha=0.08, color=DIFF_COLORS.get(cur_d, GRID_COLOR), linewidth=0)


# ── Single-run full plot ───────────────────────────────────────────────────────

def plot_full(metrics: Dict, output: str, run_label: str = "") -> None:
    records = metrics.get("batch_records", [])
    ep_rewards = metrics.get("rewards", [])

    if not records:
        print("ERROR: metrics.json has no batch_records — training may not have run.")
        sys.exit(1)

    batches = [r["batch"] + 1 for r in records]

    fig = plt.figure(figsize=(18, 14), facecolor=BG_DARK)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]

    title = f"DispatchR GRPO — Training Progress"
    if run_label:
        title += f"  [{run_label}]"
    cfg = metrics.get("config", {})
    subtitle = (
        f"model: {cfg.get('model', '?').split('/')[-1]}  ·  "
        f"episodes: {cfg.get('episodes', '?')}  ·  "
        f"batch_size: {cfg.get('batch_size', '?')}  ·  "
        f"lr: {cfg.get('learning_rate', '?')}"
    )
    fig.suptitle(title, color=TEXT_COLOR, fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.955, subtitle, ha="center", color="#94A3B8", fontsize=9)

    # ── 1. Episode rewards (scatter + MA) ─────────────────────────────────────
    ax = axes[0][0]
    _ax_style(ax)
    if ep_rewards:
        ep_idx = list(range(len(ep_rewards)))
        ax.scatter(ep_idx, ep_rewards, s=6, alpha=0.35, color=C_REWARD, label="Episode reward")
        if len(ep_rewards) >= 10:
            ax.plot(ep_idx, _moving_avg(ep_rewards, 10),
                    color=C_MA10, lw=1.8, label="MA-10")
        if len(ep_rewards) >= 5:
            ax.plot(ep_idx, _moving_avg(ep_rewards, 5),
                    color=C_MA5, lw=1.2, linestyle="--", label="MA-5")
    ax.axhline(0, color=TEXT_COLOR, lw=0.5, alpha=0.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards")
    ax.legend(fontsize=7, facecolor=BG_PANEL, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)

    # ── 2. Batch mean reward + loss (dual axis) ────────────────────────────────
    ax = axes[0][1]
    _ax_style(ax)
    _shade_curriculum(ax, records)
    mean_r = [r["mean_reward"] for r in records]
    ax.plot(batches, mean_r, color=C_REWARD, lw=1.8, label="Mean reward")
    if len(records) >= 5:
        ax.plot(batches, [r["reward_ma_5"] for r in records],
                color=C_MA5, lw=1.2, ls="--", label="MA-5")
    ax.fill_between(batches, [r["min_reward"] for r in records],
                    [r["max_reward"] for r in records],
                    color=C_REWARD, alpha=0.08, label="min–max band")
    ax.axhline(0, color=TEXT_COLOR, lw=0.5, alpha=0.3)
    ax.set_ylabel("Reward", color=C_REWARD)
    ax.set_xlabel("Batch")
    ax.set_title("Batch Reward + Loss")

    ax2 = ax.twinx()
    ax2.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax2.yaxis.label.set_color(TEXT_COLOR)
    loss_pairs = [(batches[i], r["loss"]) for i, r in enumerate(records) if r["loss"] > 0]
    if loss_pairs:
        lx, ly = zip(*loss_pairs)
        ax2.plot(lx, ly, color=C_LOSS, lw=1.5, marker="o",
                 markersize=2.5, label="Loss", alpha=0.85)
        ax2.set_ylabel("Loss", color=C_LOSS)
        ax2.legend(loc="upper right", fontsize=7,
                   facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    ax.legend(loc="upper left", fontsize=7,
              facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # ── 3. Valid action rate ───────────────────────────────────────────────────
    ax = axes[1][0]
    _ax_style(ax)
    _shade_curriculum(ax, records)
    vr = [r["valid_action_rate"] for r in records]
    ax.plot(batches, vr, color=C_VALID, lw=1.8)
    ax.fill_between(batches, vr, alpha=0.15, color=C_VALID)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Valid Action Rate")
    ax.set_title("Valid Action Rate")

    # ── 4. Action type breakdown ───────────────────────────────────────────────
    ax = axes[1][1]
    _ax_style(ax)
    _shade_curriculum(ax, records)
    ax.plot(batches, [r["dispatch_rate"] for r in records],
            color=C_DISPATCH, lw=1.5, label="Dispatch")
    ax.plot(batches, [r["hold_rate"] for r in records],
            color=C_HOLD, lw=1.5, label="Hold")
    ax.plot(batches, [r["invalid_action_rate"] for r in records],
            color=C_INVALID, lw=1.5, label="Invalid")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Rate")
    ax.set_title("Action Type Distribution")
    ax.legend(fontsize=7, facecolor=BG_PANEL, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)

    # ── 5. Fatalities per batch ────────────────────────────────────────────────
    ax = axes[2][0]
    _ax_style(ax)
    _shade_curriculum(ax, records)
    fatalities = [r["fatality_count"] for r in records]
    ax.bar(batches, fatalities, color=C_FATALITY, alpha=0.7, width=0.8)
    if len(records) >= 5:
        ax.plot(batches, _moving_avg(fatalities, 5),
                color="#FBBF24", lw=1.5, label="MA-5")
        ax.legend(fontsize=7, facecolor=BG_PANEL, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Fatalities")
    ax.set_title("Fatalities per Batch")

    # ── 6. Epsilon decay + curriculum phases ──────────────────────────────────
    ax = axes[2][1]
    _ax_style(ax)
    epsilons = [r["epsilon"] for r in records]
    ax.plot(batches, epsilons, color=C_EPSILON, lw=1.8, label="Epsilon")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Epsilon", color=C_EPSILON)
    ax.set_title("Exploration & Curriculum Phases")

    # Curriculum legend patches
    seen_diffs = dict.fromkeys(r["difficulty"] for r in records)
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=DIFF_COLORS.get(d, GRID_COLOR), alpha=0.5, label=d.title())
        for d in seen_diffs
    ]
    if patches:
        leg = ax.legend(handles=patches, loc="upper right", fontsize=7,
                        facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                        title="Phase", title_fontsize=7)
        ax.add_artist(leg)
    ax.legend(handles=[
        plt.Line2D([0], [0], color=C_EPSILON, lw=1.5, label="Epsilon")
    ], loc="upper left", fontsize=7,
       facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    _shade_curriculum(ax, records)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, facecolor=BG_DARK, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output}")


# ── Comparison plot ────────────────────────────────────────────────────────────

def plot_compare(
    metrics_a: Dict,
    metrics_b: Dict,
    output: str,
    label_a: str = "Run A",
    label_b: str = "Run B",
) -> None:
    """Side-by-side comparison of two runs: reward curves, loss, action rates."""

    rec_a = metrics_a.get("batch_records", [])
    rec_b = metrics_b.get("batch_records", [])
    rew_a = metrics_a.get("rewards", [])
    rew_b = metrics_b.get("rewards", [])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG_DARK)
    fig.suptitle(
        f"Training Comparison  ·  {label_a}  vs  {label_b}",
        color=TEXT_COLOR, fontsize=14, fontweight="bold",
    )

    ca, cb = "#22C55E", "#3B82F6"  # green / blue

    # ── 1. Episode reward curves ──────────────────────────────────────────────
    ax = axes[0][0]
    _ax_style(ax)
    for rew, col, lbl in [(rew_a, ca, label_a), (rew_b, cb, label_b)]:
        if rew:
            idx = list(range(len(rew)))
            ax.scatter(idx, rew, s=5, alpha=0.25, color=col)
            if len(rew) >= 10:
                ax.plot(idx, _moving_avg(rew, 10), color=col, lw=2.0, label=f"{lbl} MA-10")
    ax.axhline(0, color=TEXT_COLOR, lw=0.4, alpha=0.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards")
    ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # ── 2. Batch mean reward ──────────────────────────────────────────────────
    ax = axes[0][1]
    _ax_style(ax)
    for rec, col, lbl in [(rec_a, ca, label_a), (rec_b, cb, label_b)]:
        if rec:
            bs = [r["batch"] + 1 for r in rec]
            ax.plot(bs, [r["mean_reward"] for r in rec], color=col, lw=1.8, label=lbl)
            ax.fill_between(
                bs,
                [r["min_reward"] for r in rec],
                [r["max_reward"] for r in rec],
                color=col, alpha=0.07,
            )
    ax.axhline(0, color=TEXT_COLOR, lw=0.4, alpha=0.3)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Batch Mean Reward")
    ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # ── 3. Loss ───────────────────────────────────────────────────────────────
    ax = axes[1][0]
    _ax_style(ax)
    for rec, col, lbl in [(rec_a, ca, label_a), (rec_b, cb, label_b)]:
        if rec:
            pairs = [(r["batch"] + 1, r["loss"]) for r in rec if r["loss"] > 0]
            if pairs:
                lx, ly = zip(*pairs)
                ax.plot(lx, ly, color=col, lw=1.5, marker="o",
                        markersize=2, alpha=0.9, label=lbl)
    ax.set_xlabel("Batch")
    ax.set_ylabel("GRPO Loss")
    ax.set_title("Policy Loss")
    ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # ── 4. Valid action rate + fatalities ────────────────────────────────────
    ax = axes[1][1]
    _ax_style(ax)
    ax2 = ax.twinx()
    ax2.tick_params(colors=TEXT_COLOR, labelsize=8)
    for rec, col, lbl in [(rec_a, ca, label_a), (rec_b, cb, label_b)]:
        if rec:
            bs = [r["batch"] + 1 for r in rec]
            ax.plot(bs, [r["valid_action_rate"] for r in rec],
                    color=col, lw=1.8, label=f"{lbl} valid")
            ax2.plot(bs, _moving_avg([r["fatality_count"] for r in rec], 5),
                     color=col, lw=1.2, ls="--", alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Valid Action Rate")
    ax2.set_ylabel("Fatalities (MA-5)", color="#94A3B8")
    ax.set_title("Valid Actions  &  Fatalities (dashed)")
    ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout(pad=2.0)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, facecolor=BG_DARK, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output}")


# ── Quick 2-panel view ─────────────────────────────────────────────────────────

def plot_quick(metrics: Dict, output: str, run_label: str = "") -> None:
    """Minimal reward + loss for a fast sanity check."""
    records = metrics.get("batch_records", [])
    if not records:
        print("ERROR: no batch_records in metrics.json")
        sys.exit(1)

    batches = [r["batch"] + 1 for r in records]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), facecolor=BG_DARK)

    for ax in [ax1, ax2]:
        _ax_style(ax)

    # Reward
    ax1.plot(batches, [r["mean_reward"] for r in records],
             color=C_REWARD, lw=2, label="Mean reward")
    ax1.fill_between(
        batches,
        [r["min_reward"] for r in records],
        [r["max_reward"] for r in records],
        color=C_REWARD, alpha=0.12, label="min–max"
    )
    if len(records) >= 5:
        ax1.plot(batches, [r["reward_ma_5"] for r in records],
                 color=C_MA5, lw=1.4, ls="--", label="MA-5")
    ax1.axhline(0, color=TEXT_COLOR, lw=0.5, alpha=0.3)
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Reward")
    ax1.set_title(f"Reward Curve{' — ' + run_label if run_label else ''}")
    ax1.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Loss
    loss_pairs = [(batches[i], r["loss"]) for i, r in enumerate(records) if r["loss"] > 0]
    if loss_pairs:
        lx, ly = zip(*loss_pairs)
        ax2.plot(lx, ly, color=C_LOSS, lw=1.8, marker="o", markersize=3)
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("GRPO Loss")
    ax2.set_title("Loss Curve")

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, facecolor=BG_DARK, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DispatchR training curves from metrics.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--metrics", required=True,
                        help="Path to metrics.json (primary run)")
    parser.add_argument("--compare", default=None,
                        help="Path to a second metrics.json to compare against")
    parser.add_argument("--labels", nargs=2, default=["Run A", "Run B"],
                        metavar=("LABEL_A", "LABEL_B"),
                        help="Labels for the two runs when using --compare")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: same dir as --metrics)")
    parser.add_argument("--quick", action="store_true",
                        help="2-panel quick view instead of full 6-panel plot")
    args = parser.parse_args()

    m = load_metrics(args.metrics)

    # Summary to stdout
    summary = m.get("summary", {})
    cfg     = m.get("config", {})
    print(f"\nRun summary")
    print(f"  model:    {cfg.get('model', '?').split('/')[-1]}")
    print(f"  episodes: {m.get('episodes_done', len(m.get('rewards', [])))}")
    print(f"  batches:  {m.get('batches_done', len(m.get('batch_records', [])))}")
    if summary:
        print(f"  mean_reward:  {summary.get('mean_reward', 0):.4f}")
        print(f"  best_reward:  {summary.get('best_reward', 0):.4f}")
        print(f"  final_loss:   {summary.get('final_loss', 0):.4f}")

    metrics_dir = Path(args.metrics).parent

    if args.compare:
        m2 = load_metrics(args.compare)
        out = args.output or str(metrics_dir / "comparison.png")
        plot_compare(m, m2, out, label_a=args.labels[0], label_b=args.labels[1])
    elif args.quick:
        out = args.output or str(metrics_dir / "training_curves_quick.png")
        label = Path(args.metrics).parent.name
        plot_quick(m, out, run_label=label)
    else:
        out = args.output or str(metrics_dir / "training_progress.png")
        label = Path(args.metrics).parent.name
        plot_full(m, out, run_label=label)


if __name__ == "__main__":
    main()
