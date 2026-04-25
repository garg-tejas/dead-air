#!/bin/bash
# Launch Dead Air GRPO training on Hugging Face Spaces GPU
# Usage: bash scripts/launch_hf_training.sh [MODEL] [EPISODES] [BATCH_SIZE]

set -e

MODEL=${1:-"unsloth/Qwen3-14B-unsloth-bnb-4bit"}
EPISODES=${2:-200}
BATCH_SIZE=${3:-8}
HUB_MODEL_ID=${4:-""}

echo "========================================"
echo "Dead Air HF Spaces Training Launcher"
echo "========================================"
echo "Model: $MODEL"
echo "Episodes: $EPISODES"
echo "Batch size: $BATCH_SIZE"
echo "Hub Model ID: ${HUB_MODEL_ID:-'(not pushing)'}"
echo ""

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    echo "[WARN] HF_TOKEN not set. Set it with: export HF_TOKEN=hf_..."
    echo "       You can get one at https://huggingface.co/settings/tokens"
    echo ""
fi

# Install dependencies if needed
echo "Installing dependencies..."
pip install -q unsloth transformers torch trl accelerate numpy networkx huggingface_hub

# Pre-download model to cache (avoids timeout during training)
echo "Pre-downloading model: $MODEL"
python -c "
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
CMD="python train_unsloth_grpo.py \\
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
