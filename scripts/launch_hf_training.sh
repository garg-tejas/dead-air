#!/bin/bash
# DispatchR Training Launch Script for HF Spaces
# Run this inside the Space terminal after build completes

set -e

MODEL=${1:-"unsloth/Qwen3-14B-unsloth-bnb-4bit"}
EPISODES=${2:-200}
BATCH_SIZE=${3:-8}
HUB_MODEL_ID=${4:-""}

echo "========================================"
echo "DispatchR HF Spaces Training Launcher"
echo "========================================"
echo "Model: $MODEL"
echo "Episodes: $EPISODES"
echo "Batch size: $BATCH_SIZE"
echo "Hub Model ID: ${HUB_MODEL_ID:-'(not pushing)'}"
echo ""

# Verify GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify HF token is set via Secrets
if [ -z "$HF_TOKEN" ]; then
    echo "[ERROR] HF_TOKEN not set."
    echo "        Add it in Space Settings -> Secrets and Variables -> Secrets"
    echo "        Name: HF_TOKEN, Value: your_hf_token"
    exit 1
fi

echo "HF_TOKEN is set (from Secrets)"

# Pre-download model to cache (saves time, avoids timeout)
echo ""
echo "Pre-downloading model: $MODEL"
python3 -c "
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
tokenizer = AutoTokenizer.from_pretrained('$MODEL')
model, _ = FastLanguageModel.from_pretrained(
    model_name='$MODEL',
    load_in_4bit=True,
    max_seq_length=4096,
)
print('Model cached successfully')
"

# Build command
CMD="python3 train_unsloth_grpo.py \\
  --model $MODEL \\
  --episodes $EPISODES \\
  --batch-size $BATCH_SIZE \\
  --max-completion-length 1536 \\
  --curriculum \\
  --save-every 25 \\
  --output-dir ./outputs/unsloth_grpo \\
  --trajectory-file ./outputs/unsloth_grpo/trajectory.jsonl"

if [ -n "$HUB_MODEL_ID" ]; then
    CMD="$CMD \\
  --push-to-hub \\
  --hub-model-id $HUB_MODEL_ID"
fi

echo ""
echo "Command:"
echo "$CMD"
echo ""

# Create log directory
mkdir -p ./logs

# Launch with nohup so it survives disconnect
LOG_FILE="./logs/training_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""
echo "Starting training..."
echo "Monitor with: tail -f $LOG_FILE"
echo ""

# shellcheck disable=SC2086
eval nohup $CMD > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training PID: $PID"
echo $PID > ./training.pid

echo ""
echo "========================================"
echo "Training launched in background!"
echo "PID: $PID"
echo "Log: tail -f $LOG_FILE"
echo "Kill: kill $PID"
echo "========================================"
