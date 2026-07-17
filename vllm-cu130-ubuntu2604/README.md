# vLLM cu130 on Ubuntu 26.04

Run `./setup.sh` on Ubuntu 26.04 x86_64. The script uses an existing CUDA
Toolkit release from 13.3 onward within the 13.x series, or installs CUDA
Toolkit 13.3 Update 1 by default. It accepts an installed cuDNN release from
9.24 onward within the 9.x series, or installs cuDNN 9.24 by default.
Use NVIDIA Linux driver 610.43.02 or newer for the fully supported CUDA 13.3
configuration.

The Python environment continues to use the CUDA 13.0 builds of PyTorch and
vLLM because those are the published binary variants. NVIDIA's CUDA 13.x minor
version compatibility permits those binaries to run with the CUDA 13.3 system
toolkit and driver. The script also installs FFmpeg, then creates and opens the
`vllm-cu130-ubuntu2604` virtual environment.
