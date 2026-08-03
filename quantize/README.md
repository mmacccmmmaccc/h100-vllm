# CUDA quantization environment

This directory is a separate Python 3.10 environment because Fairseq 0.12.2
cannot be imported under Python 3.12. PyTorch is installed from the official
CUDA 12.8 wheel index. A sufficiently recent NVIDIA driver is required; a
CUDA 13-capable driver is backward-compatible with these CUDA 12.8 wheels.

## Create the environment

Run on the Linux CUDA host:

```bash
cd /workspace/h100-vllm/quantize
uv python install 3.10
uv sync
```

Confirm that Python, PyTorch, and the GPU are usable:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
uv run python -c "import fairseq; print(fairseq.__version__)"
```

## Run

```bash
uv run python quantize_gemma4_fp8.py --dry-run
uv run python quantize_typhoon2_audio_nvfp4.py --dry-run
```

Remove `--dry-run` to perform quantization. Authenticate with Hugging Face first
when accessing gated models (`hf auth login` or set `HF_TOKEN`). The H100 can
produce an NVFP4 checkpoint, but accelerated NVFP4 inference requires Blackwell.
