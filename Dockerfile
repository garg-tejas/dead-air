# DispatchR Environment Server — HF Spaces Dockerfile
# Combines the slim build from the root fix with the HEALTHCHECK and
# PYTHONPATH conventions from the OpenEnv template.
#
# Design choices:
#   - python:3.11-slim (not openenv-base) → smaller image, faster build
#   - pip (not uv) → works in standard HF Spaces build environment
#   - single-stage (not multi-stage) → simpler, no uv.lock dependency
#   - USER 1000 → required by HF Spaces
#   - port 7860 → HF Spaces proxy port
#   - only runtime deps → no training libraries

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
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

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

# Health check (from OpenEnv template)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose the HF Spaces port
EXPOSE 7860

# Run the OpenEnv FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
