# DispatchR Visualizations Quick Guide

## What You Can Generate

This project now supports three practical visualization groups:

1. Reward Overview

- Trend line
- Moving average
- Distribution histogram
- Best-so-far curve

2. Training Metrics Dashboard

- Reward by batch
- Loss by batch
- Action quality rates (valid, invalid, hold)
- Outcome metrics (fatalities, missed calls, response time)

3. Episode Timeline

- Active/resolved/fatality call progression
- Unit utilization over time (idle, en-route, on-scene, unavailable)

## Main Script

Use:

```bash
python visualize_dashboard.py --outdir assets/visualizations
```

Add one or more inputs:

```bash
python visualize_dashboard.py --rewards colab_results.json --outdir assets/visualizations
python visualize_dashboard.py --metrics outputs/unsloth_grpo/metrics.csv --outdir assets/visualizations
python visualize_dashboard.py --episode episode.json --outdir assets/visualizations
```

You can combine all three:

```bash
python visualize_dashboard.py \
  --rewards colab_results.json \
  --metrics outputs/unsloth_grpo/metrics.csv \
  --episode episode.json \
  --outdir assets/visualizations
```

## Generated Files

Depending on provided inputs, you will get:

- `assets/visualizations/rewards_overview.png`
- `assets/visualizations/training_metrics.png`
- `assets/visualizations/episode_timeline.png`

## Creating Episode Data for Timeline Visuals

If you do not have an episode export yet:

```bash
python export_episode.py --difficulty learning --output episode.json --seed 42
```

Then run:

```bash
python visualize_dashboard.py --episode episode.json --outdir assets/visualizations
```

## Existing Visualization Scripts

You can also still use existing scripts:

```bash
python plot_rewards.py --input colab_results.json --output assets/training_reward.png
python generate_assets.py
python visualize_env.py --input episode.json --output assets/dispatchr_demo.mp4
```

## Notes

- `visualize_env.py` creates an MP4 animation and requires FFmpeg available for matplotlib writer.
- The static dashboard plots only need matplotlib + numpy.
- If your reward JSON is a dictionary, it should include a `rewards` list.

## Suggested Workflow

1. Train/evaluate and save rewards/metrics.
2. Export one or more episodes.
3. Run `visualize_dashboard.py` to generate all static summary charts.
4. Use `visualize_env.py` for demo-ready animation clips.
