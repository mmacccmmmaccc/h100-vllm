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

# 2. Uninstall the default vLLM version included in the base image
RUN pip uninstall -y vllm

# 3. Install the specific post-PR vLLM dev version WITH the audio ecosystem
# This combines your requirement for the specific version and the [audio] extra
RUN pip install --no-cache-dir "vllm[audio]" --pre
