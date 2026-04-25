FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/.cache/huggingface

RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[train]" unsloth huggingface-hub

# Copy all code
COPY . .

# Create directories for outputs
RUN mkdir -p /app/outputs /app/logs /data

# Keep container alive
CMD ["sleep", "infinity"]
