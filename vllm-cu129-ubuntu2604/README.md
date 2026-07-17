# vllm-cu129 on Ubuntu 26.04

This context runs the locked CUDA 12.9 builds of PyTorch and vLLM on Ubuntu 26.04 x86_64.

It intentionally does not install a system CUDA Toolkit or system cuDNN package. NVIDIA's native CUDA 12.9 Toolkit packages do not support Ubuntu 26.04. Instead, the lock file supplies the CUDA 12.9 and cuDNN runtime libraries required by the cu129 wheels.

Requirements:

- Ubuntu 26.04 on x86_64
- An NVIDIA GPU supported by vLLM
- NVIDIA driver 575.57.08 or newer; a current Ubuntu 26.04/WSL driver is recommended
- Internet access for system and Python dependencies

Run:

```bash
./setup.sh
```

The setup verifies the NVIDIA driver, installs uv, FFmpeg 7, and libsndfile, synchronizes the frozen dependency lock, authenticates with Hugging Face, and opens a shell with the environment activated. It also exports `VLLM_WSL2_ENABLE_PIN_MEMORY=1` for WSL2.

This context is intended for prebuilt binary execution. If a package or custom extension requires `nvcc` or a system CUDA 12.9 development toolkit, use an NVIDIA-supported Ubuntu 24.04 CUDA 12.9 environment or a suitable container instead.
