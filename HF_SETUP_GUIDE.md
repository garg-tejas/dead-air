# Dead Air: HF Spaces Setup Guide

Complete step-by-step to deploy training on Hugging Face Spaces GPU.

---

## Phase 1: Account & Billing Setup (5 min)

### 1.1 Add Credits to Both Accounts

For **each** of your 2 HF accounts:

1. Go to https://huggingface.co/settings/billing
2. Click **"Add Credits"**
3. Enter your $30 gift code / payment method
4. Verify balance shows $30.00

> **Tip**: Use Account A for the main 200-episode run. Use Account B as backup if Account A runs out or for parallel experiments.

### 1.2 Create a User Access Token

For **each** account:

1. Go to https://huggingface.co/settings/tokens
2. Click **"New Token"**
3. Name: `dead-air-training`
4. Role: **Write** (needed to push checkpoints)
5. Copy the token (starts with `hf_...`)

---

## Phase 2: Create the Space (5 min)

### 2.1 Create Space from UI

1. Go to https://huggingface.co/new-space
2. **Space name**: `dead-air-training`
3. **SDK**: Select `Docker`
4. **Space hardware**: Start with `CPU (Free)` — we'll upgrade after setup
5. Click **"Create Space"**

### 2.2 Upgrade to GPU

1. In your Space, click **"Settings"** → **"Space Hardware"**
2. Select **"Nvidia A100"** ($2.50/hr, 80GB VRAM)
3. Click **"Upgrade"**

> ⚠️ **Billing starts NOW**. GPU billing begins immediately after selection, even if idle. Complete setup quickly.

---

## Phase 3: Deploy Code (10 min)

### 3.1 Clone Your Repo Locally (if not already)

```bash
git clone https://github.com/garg-tejas/dead-air.git
cd dead-air
```

### 3.2 Add HF Space as Remote

```bash
# Use Account A's token
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Login to HF CLI
huggingface-cli login --token $HF_TOKEN

# Add Space remote (replace USERNAME with your HF username)
git remote add hf https://huggingface.co/spaces/USERNAME/dead-air-training
```

### 3.3 Create Space Files

Create these files in your repo root:

**`Dockerfile`**:
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[train]" unsloth

# Copy code
COPY . .

# Default command keeps container alive
CMD ["sleep", "infinity"]
```

**`README.md`** (Space card, separate from your project README):
```markdown
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
---

# Dead Air GRPO Training

Emergency dispatch RL training environment.
```

**`.gitignore`** (add if not present):
```
outputs/
logs/
*.log
__pycache__/
*.pyc
.env
```

### 3.4 Push to Space

```bash
git add Dockerfile README.md
git commit -m "Add HF Spaces Docker config"
git push hf main
```

> The first push will take 5-10 minutes to build the Docker image.

---

## Phase 4: Access Space Terminal (2 min)

### 4.1 Open Terminal

1. Go to your Space: `https://huggingface.co/spaces/USERNAME/dead-air-training`
2. Click **"Files"** tab
3. Click **"Terminal"** (top right)

### 4.2 Verify GPU

In the terminal:
```bash
nvidia-smi
# Should show: NVIDIA A100-SXM4-80GB
```

---

## Phase 5: Launch Training (5 min)

### 5.1 Set Environment Variables

In the Space terminal:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
export CUDA_VISIBLE_DEVICES=0
```

### 5.2 Pre-download Model (Optional but Recommended)

```bash
# Cache the model to avoid timeout during training
python -c "
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
model_id = 'unsloth/Qwen3-14B-unsloth-bnb-4bit'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model, _ = FastLanguageModel.from_pretrained(model_id, load_in_4bit=True, max_seq_length=4096)
print('Model cached!')
"
```

### 5.3 Start Training

```bash
# Option A: Use the launcher script
python scripts/launch_hf_training.py \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --episodes 200 \
  --batch-size 8 \
  --hub-model-id USERNAME/dead-air-grpo-checkpoints

# Option B: Direct command (if launcher has issues)
nohup python train_unsloth_grpo.py \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --episodes 200 \
  --batch-size 8 \
  --max-completion-length 1536 \
  --curriculum \
  --save-every 25 \
  --push-to-hub \
  --hub-model-id USERNAME/dead-air-grpo-checkpoints \
  --output-dir ./outputs/unsloth_grpo \
  --trajectory-file ./outputs/unsloth_grpo/trajectory.jsonl \
  > training.log 2>&1 &
```

### 5.4 Verify Training Started

```bash
# Check process is running
ps aux | grep train_unsloth_grpo

# Check logs
tail -f training.log
```

You should see:
```
Loading model via Unsloth...
Adding LoRA adapters...
Dead Air Unsloth GRPO Training
Model: unsloth/Qwen3-14B-unsloth-bnb-4bit
...
--- Batch 1/25 ---
Epsilon: 1.00
```

---

## Phase 6: Monitor & Manage (Ongoing)

### 6.1 Check Progress

```bash
# Live log tail
tail -f training.log

# Or use monitor script
python scripts/monitor_training.py
```

### 6.2 Check Cost

1. Go to https://huggingface.co/settings/billing
2. View **"Usage"** tab
3. A100 costs $2.50/hr

**Budget math**:
- $30 credit = 12 hours on A100
- $60 credit (both accounts) = 24 hours on A100
- 200 episodes ≈ 8-10 hours → **well within budget**

### 6.3 Checkpoints on Hub

1. Go to `https://huggingface.co/USERNAME/dead-air-grpo-checkpoints`
2. Checkpoints auto-push every 25 episodes
3. Download any checkpoint: `git clone https://huggingface.co/USERNAME/dead-air-grpo-checkpoints`

### 6.4 Prevent Sleep (IMPORTANT)

HF Spaces may sleep after inactivity. To keep training alive:

```bash
# In a separate terminal, run a keepalive ping
while true; do
  curl -s https://USERNAME-dead-air-training.hf.space > /dev/null
  sleep 300  # every 5 minutes
done
```

Or use a background process:
```bash
nohup bash -c 'while true; do sleep 300; curl -s https://USERNAME-dead-air-training.hf.space > /dev/null; done' > /dev/null 2>&1 &
```

---

## Phase 7: Stop & Save (When Done)

### 7.1 Graceful Stop

```bash
# Find PID
ps aux | grep train_unsloth_grpo

# Kill process
kill -SIGINT <PID>  # Sends KeyboardInterrupt, saves emergency checkpoint
```

### 7.2 Downgrade Hardware (Stop Billing)

1. Go to Space Settings → Space Hardware
2. Select **"CPU (Free)"**
3. Click **"Downgrade"**

> ⚠️ **Always downgrade when not training!** A100 bills continuously.

### 7.3 Download Final Model

```bash
# From local machine
huggingface-cli download USERNAME/dead-air-grpo-checkpoints --local-dir ./final-model
```

---

## Two-Account Strategy

| Account | Purpose | Credit | Expected Usage |
|---------|---------|--------|----------------|
| **A** | Main 200-episode run | $30 | ~10 hrs = $25 |
| **B** | Backup / hyperparameter sweep / 8B comparison | $30 | ~5 hrs = $12.50 |

**If Account A runs out**:
1. Stop training on Account A (downgrade GPU)
2. Push code to Account B's Space
3. Resume from last checkpoint:
   ```bash
   python train_unsloth_grpo.py \
     --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
     --episodes 200 \
     --batch-size 8 \
     --output-dir ./outputs/unsloth_grpo \
     # Load from checkpoint-150 or similar
   ```

---

## Troubleshooting

### "No space left on device"
```bash
# Clear pip cache
pip cache purge
# Clear HuggingFace cache (be careful!)
rm -rf ~/.cache/huggingface/hub/models--*/snapshots/*
```

### "CUDA out of memory"
```bash
# Reduce batch size
python train_unsloth_grpo.py ... --batch-size 4
# Or reduce completion length
python train_unsloth_grpo.py ... --max-completion-length 1024
```

### "Model download timeout"
```bash
# Pre-download with huggingface-cli
huggingface-cli download unsloth/Qwen3-14B-unsloth-bnb-4bit
```

### "Training stopped after I closed the tab"
You didn't use `nohup`. Restart with the launcher script which handles this automatically.

### "Can't push to Hub"
1. Check token has **Write** access
2. Re-login: `huggingface-cli login --token hf_...`
3. Create repo manually: `huggingface-cli repo create dead-air-grpo-checkpoints --type model`

---

## Quick Reference Commands

```bash
# SSH into Space (alternative to web terminal)
# Install HF CLI first: pip install huggingface-hub
huggingface-cli space ssh USERNAME/dead-air-training

# Copy files to/from Space
huggingface-cli upload USERNAME/dead-air-training ./local-file.txt /app/

# Restart Space (if it crashes)
# Go to Space → Factory Reboot

# View GPU usage in real-time
watch -n 1 nvidia-smi
```

---

## Timeline for Hackathon

| Time | Action |
|------|--------|
| **T+0h** | Setup accounts, create Space, push code |
| **T+0.5h** | Launch training with 14B model |
| **T+3h** | Check progress, verify curriculum escalation |
| **T+8h** | First checkpoint pushed (episode 200 complete) |
| **T+10h** | Evaluate, run inference, generate demo |
| **T+12h** | Downgrade GPU, submit hackathon |

---

## Need Help?

- HF Spaces docs: https://huggingface.co/docs/hub/spaces
- GPU pricing: https://huggingface.co/docs/hub/spaces-gpus
- Unsloth issues: https://github.com/unslothai/unsloth/issues
