# Dead Air HF Spaces Deployment

## Quick Start (A100 GPU)

```bash
# 1. Set your HF token
export HF_TOKEN=hf_...

# 2. Launch training with 14B model
python scripts/launch_hf_training.py \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --episodes 200 \
  --batch-size 8 \
  --hub-model-id yourname/dead-air-grpo

# 3. Monitor progress
python scripts/monitor_training.py
tail -f logs/training_*.log
```

## GPU Selection Guide

| GPU | VRAM | $/hr | $60 Gets You | Max Model | Recommendation |
|-----|------|------|--------------|-----------|----------------|
| T4 | 16GB | $0.40 | 150 hrs | ❌ Too small | Skip |
| **L4** | 24GB | $0.80 | 75 hrs | 4B-8B | Good for budget runs |
| **L40S** | 48GB | $1.80 | 33 hrs | 8B-14B | Balanced |
| **A10G** | 24GB | $1.00 | 60 hrs | 4B-8B | Same as L4 but pricier |
| **A100** | **80GB** | **$2.50** | **24 hrs** | **14B-30B** | **Best for hackathon** |

## Why A100 for This Hackathon

1. **14B model fits comfortably** (~30GB loaded, ~50GB training with batch=8)
2. **2.4× memory bandwidth** of L4 → faster generation and backward passes
3. **24 hours = $60 exactly** → perfect for the hackathon window
4. **HBM2e + NVLink** → much better training throughput than L40S

## Model Selection

| Model | Params | Type | A100 VRAM | Notes |
|-------|--------|------|-----------|-------|
| `unsloth/Qwen3-4B-Thinking-2507-bnb-4bit` | 4B | Dense | ~17GB | **Tested, working** |
| `unsloth/Qwen3-8B-unsloth-bnb-4bit` | 8B | Dense | ~24GB | Upgrade path |
| `unsloth/Qwen3-14B-unsloth-bnb-4bit` | 14B | Dense | ~30GB | **Recommended for A100** |
| `unsloth/Qwen3-30B-A3B-Instruct-2507` | 30B MoE | MoE | ~40GB | Experimental |

## Troubleshooting

### Training stops when I disconnect
Use `nohup` (already in launch script) or run inside `tmux`/`screen`:
```bash
tmux new -s training
python scripts/launch_hf_training.py ...
# Press Ctrl+B then D to detach
tmux attach -t training  # reconnect later
```

### Out of Memory
Reduce batch size: `--batch-size 4` or `--batch-size 2`
Reduce completion length: `--max-completion-length 1024`
Switch to 8B model instead of 14B

### Model download times out
The launch script pre-downloads the model. If it still fails:
```bash
huggingface-cli download unsloth/Qwen3-14B-unsloth-bnb-4bit
```

### Checkpoints not pushing to Hub
Ensure `HF_TOKEN` is set and has **write** access.
Check repo exists or the script will auto-create it.
