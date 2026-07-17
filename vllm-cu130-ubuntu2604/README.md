# vLLM cu130 on Ubuntu 26.04

This context installs CUDA Toolkit 13.0 Update 2 and runs the locked CUDA 13.0 builds of PyTorch and vLLM on Ubuntu 26.04 x86_64.

NVIDIA does not officially qualify Ubuntu 26.04 for CUDA 13.0. The setup therefore uses NVIDIA's standalone 13.0.2 runfile in silent, toolkit-only override mode. It never installs or replaces the NVIDIA driver. The download is verified against NVIDIA's published MD5 checksum before execution and removed after successful installation.

CUDA 13.0 supports Ubuntu 26.04's GCC 15 compiler. Because Ubuntu 26.04 uses Rust coreutils by default and its `dd` behavior can break NVIDIA runfile extraction, the setup installs `gnu-coreutils` and invokes the runfile with `/usr/bin/gnudd`.

Ubuntu 26.04 also replaced the `libxml2.so.2` ABI required by CUDA 13.0's embedded installer with `libxml2.so.16`. Setup downloads checksum-pinned Ubuntu 24.04 `libxml2.so.2` and ICU 74 packages, extracts them into a private cache, and exposes them only to the CUDA installer. It does not install or symlink the older ABI system-wide. System cuDNN is not installed; the frozen Python lock supplies the runtime libraries required by the cu130 PyTorch and vLLM packages.

Requirements:

- Ubuntu 26.04 on x86_64
- An NVIDIA GPU supported by vLLM
- NVIDIA driver 580.95.05 or newer
- Enough temporary disk space for the CUDA runfile, extraction, and toolkit installation
- Internet access for system and Python dependencies

Run:

```bash
./setup.sh
```

The setup verifies the NVIDIA driver, installs build prerequisites and CUDA Toolkit 13.0 at `/usr/local/cuda-13.0`, installs uv, FFmpeg 7, and libsndfile, synchronizes the frozen dependency lock, authenticates with Hugging Face, and opens a shell with the environment activated. It exports `CUDA_HOME`, compiler variables, `PATH`, `LD_LIBRARY_PATH`, and `VLLM_WSL2_ENABLE_PIN_MEMORY=1`.

Because Ubuntu 26.04 is not an NVIDIA-qualified CUDA 13.0 platform, use Ubuntu 24.04 or a supported container if an extension remains incompatible with the newer operating-system libraries.
