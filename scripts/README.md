# DispatchR Training Launchers

## HF Hub Jobs (Recommended)

Serverless GPU training — no Spaces needed.

```bash
# 1. Install HF CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# 2. Login
hf auth login

# 3. Launch training (L4 = $0.80/hr, 24GB)
python scripts/launch_hf_job.py --flavor l4x1 --episodes 200

# Or A100 for 14B model ($2.50/hr, 80GB)
python scripts/launch_hf_job.py \
  --flavor a100-large \
  --episodes 200 \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit

# 4. Watch logs
hf jobs logs <job-id>

# 5. Check status
hf jobs list
```

### Hardware Options

| Flavor | GPU | VRAM | $/hr | Best For |
|--------|-----|------|------|----------|
| `l4x1` | L4 | 24 GB | $0.80 | 4B model, budget runs |
| `a100-large` | A100 | 80 GB | $2.50 | 14B model, max performance |

### Python Launcher Args

```bash
python scripts/launch_hf_job.py \
  --flavor l4x1 \
  --episodes 200 \
  --batch-size 8 \
  --timeout 8h \
  --model unsloth/Qwen3-4B-Thinking-2507-bnb-4bit \
  --hub-model-id yourname/dispatchr-grpo
```
