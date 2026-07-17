# vllm-cu129 on Ubuntu 26.04

This context installs CUDA Toolkit 12.9 Update 1 and runs the locked CUDA 12.9 builds of PyTorch and vLLM on Ubuntu 26.04 x86_64.

NVIDIA does not officially qualify Ubuntu 26.04 for CUDA 12.9. The setup therefore uses NVIDIA's standalone 12.9.1 runfile in silent, toolkit-only override mode. It never installs or replaces the NVIDIA driver. The download is verified against NVIDIA's published MD5 checksum before execution and removed after successful installation.

Ubuntu 26.04 defaults to GCC 15, while CUDA 12.9 supports GCC through version 14. The setup installs `gcc-14` and `g++-14` and exports them as the CUDA host compilers. System cuDNN is not installed; the frozen Python lock supplies the CUDA 12.9 and cuDNN runtime libraries required by the cu129 wheels.

Requirements:

- Ubuntu 26.04 on x86_64
- An NVIDIA GPU supported by vLLM
- NVIDIA driver 575.57.08 or newer; a current Ubuntu 26.04/WSL driver is recommended
- Enough temporary disk space for the CUDA runfile, extraction, and toolkit installation
- Internet access for system and Python dependencies

Run:

```bash
./setup.sh
```

The setup verifies the NVIDIA driver, installs GCC 14 and CUDA Toolkit 12.9 at `/usr/local/cuda-12.9`, installs uv, FFmpeg 7, and libsndfile, synchronizes the frozen dependency lock, authenticates with Hugging Face, and opens a shell with the environment activated. It exports `CUDA_HOME`, compiler variables, `PATH`, `LD_LIBRARY_PATH`, and `VLLM_WSL2_ENABLE_PIN_MEMORY=1`.

Because Ubuntu 26.04 is not an NVIDIA-qualified CUDA 12.9 platform, use Ubuntu 24.04 or a supported container if an extension remains incompatible with the newer operating-system libraries.
