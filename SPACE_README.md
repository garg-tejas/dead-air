---
title: Dead Air Training
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - rl
  - grpo
---

# Dead Air GRPO Training

Emergency dispatch RL training environment for the Meta OpenEnv Hackathon.

## Usage

This Space runs the GRPO training loop. Access the terminal to launch training:

```bash
# Launch training
python scripts/launch_hf_training.py \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --episodes 200 \
  --batch-size 8 \
  --hub-model-id yourname/dead-air-grpo

# Monitor
python scripts/monitor_training.py
tail -f logs/training_*.log
```

## Checkpoints

Checkpoints are automatically pushed to the HF Hub model repo every 25 episodes.

## Hardware

Requires **Nvidia A100** (80GB VRAM) for 14B model training.
