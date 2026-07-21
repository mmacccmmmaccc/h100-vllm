docker run --runtime nvidia --gpus all \
    --env-file .env \
    --volume "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    --ipc=host \
    vllm-audio \
    cyankiwi/gemma-4-12B-it-AWQ-INT4 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --limit-mm-per-prompt.audio 1
