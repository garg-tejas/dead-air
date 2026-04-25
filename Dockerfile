FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl vim procps \
    && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000 (required by HF Spaces)
RUN useradd -m -u 1000 user

# Set up working directory and ensure local bin is on PATH for --user installs
RUN mkdir -p /app && chown 1000 /app
WORKDIR /app
USER 1000
ENV PATH="/home/user/.local/bin:${PATH}"

# Copy and install dependencies first (for layer caching)
COPY --chown=user pyproject.toml .
RUN pip install --no-cache-dir --user ".[train]" unsloth huggingface-hub

# Copy all code
COPY --chown=user . .

# Create output directories
RUN mkdir -p /app/outputs /app/logs

# Run minimal health-check server so HF Spaces stays alive
CMD ["python", "app.py"]
