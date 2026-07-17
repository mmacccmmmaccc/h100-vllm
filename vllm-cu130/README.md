# vLLM with CUDA Toolkit 13.x

Run `./setup.sh` on Ubuntu 22.04 or 24.04. The script uses an existing CUDA
Toolkit 13.x installation when one is detected, or installs CUDA Toolkit 13.0
by default. It also installs cuDNN 9.19, FFmpeg, and the CUDA 13.0 build of
vLLM, then creates and opens the `vllm-cu130` virtual environment.
