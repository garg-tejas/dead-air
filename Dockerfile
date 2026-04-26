# Slim HF Spaces Dockerfile for DispatchR environment server.
# Only installs runtime deps — no training libraries.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Create user (HF Spaces convention)
RUN useradd -m -u 1000 user
WORKDIR /app
USER 1000
ENV PATH="/home/user/.local/bin:${PATH}"

# Copy and install ONLY runtime dependencies (no training libs)
COPY --chown=user pyproject.toml .
RUN pip install --no-cache-dir --user \
    "openenv-core[core]>=0.2.3" \
    "numpy>=1.24.0" \
    "networkx>=3.0" \
    "pydantic>=2.0" \
    "jmespath>=1.1.0" \
    "fastapi>=0.110.0" \
    "uvicorn>=0.27.0"

# Copy environment code
COPY --chown=user . .

# Expose the HF Spaces port
EXPOSE 7860

# Run the OpenEnv FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
