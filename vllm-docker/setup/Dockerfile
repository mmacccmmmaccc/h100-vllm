# Start from the official vLLM OpenAI-compatible image
FROM vllm/vllm-openai:latest

# Switch to root to install OS-level packages
USER root

# 1. Install system audio libraries (fixes the PyAV/soundfile crash)
# Added `git` just in case you need it for fetching repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Reinstall vLLM with audio extension
RUN uv pip install vllm[audio] --pre

