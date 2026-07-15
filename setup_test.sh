#!/usr/bin/env bash

set -Eeuo pipefail

CUDA_VERSION="12.9"
CUDA_RELEASE="12.9.1"
CUDA_LOCAL_REPO_VERSION="12.9.1-575.57.08-1"
CUDNN_VERSION="9.17.1"
FFMPEG_VERSION="7.1.5"
MODEL="prithivMLmods/gemma-4-E4B-it-FP8"
VLLM_PORT="8080"
APP_PORT="7000"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$SCRIPT_DIR/vllm_engine"
VLLM_PID=""
APP_PID=""
INSTALL_TEMP_DIR=""
CLEANED_UP=0

log() {
    printf '[%s] [setup] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
    printf '[setup] ERROR: %s\n' "$*" >&2
    exit 1
}

as_root() {
    if (( EUID == 0 )); then
        "$@"
    else
        command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages."
        sudo "$@"
    fi
}

apt_install() {
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}

stop_process_group() {
    local pid=$1
    [[ -n "$pid" ]] || return 0

    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
}

process_group_is_alive() {
    local pid=$1
    [[ -n "$pid" ]] || return 1
    kill -0 -- "-$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null
}

cleanup() {
    local status=$?
    (( CLEANED_UP == 0 )) || return "$status"
    CLEANED_UP=1

    trap - EXIT INT TERM HUP
    if [[ -n "$INSTALL_TEMP_DIR" ]]; then
        rm -rf "$INSTALL_TEMP_DIR"
        INSTALL_TEMP_DIR=""
    fi

    if [[ -n "$APP_PID" || -n "$VLLM_PID" ]]; then
        log "Stopping app and vLLM processes..."
    fi

    stop_process_group "$APP_PID"
    stop_process_group "$VLLM_PID"

    local deadline=$((SECONDS + 15))
    while (( SECONDS < deadline )); do
        local app_alive=0
        local vllm_alive=0
        process_group_is_alive "$APP_PID" && app_alive=1
        process_group_is_alive "$VLLM_PID" && vllm_alive=1
        (( app_alive == 0 && vllm_alive == 0 )) && break
        sleep 1
    done

    for pid in "$APP_PID" "$VLLM_PID"; do
        [[ -n "$pid" ]] || continue
        if process_group_is_alive "$pid"; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done

    exit "$status"
}

handle_signal() {
    local signal=$1
    local status=$2

    log "Received $signal; stopping setup and child processes..."
    exit "$status"
}

trap cleanup EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP

if (( $# != 1 )) || [[ -z "$1" ]]; then
    die "Usage: bash setup.sh <hugging-face-token>"
fi

HF_TOKEN=$1
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# Ensure Hugging Face uses the installed hf-xet client for model downloads.
unset HF_HUB_DISABLE_XET
set --

[[ -r /etc/os-release ]] || die "This installer requires an Ubuntu system with apt."
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "Unsupported OS: ${PRETTY_NAME:-unknown}. Ubuntu is required."
command -v apt-get >/dev/null 2>&1 || die "apt-get was not found."
[[ -f "$ENGINE_DIR/pyproject.toml" ]] || die "Missing $ENGINE_DIR/pyproject.toml."
[[ -f "$ENGINE_DIR/uv.lock" ]] || die "Missing $ENGINE_DIR/uv.lock."

case "${VERSION_ID:-}" in
    22.04) CUDA_REPO_DISTRO="ubuntu2204" ;;
    24.04) CUDA_REPO_DISTRO="ubuntu2404" ;;
    *) die "CUDA 12.9 automated installation supports Ubuntu 22.04 or 24.04; found ${VERSION_ID:-unknown}." ;;
esac

[[ "$(uname -m)" == "x86_64" ]] || die "This CUDA installer currently supports x86_64 only."

ensure_download_tools() {
    if ! command -v curl >/dev/null 2>&1 \
        || ! command -v wget >/dev/null 2>&1 \
        || ! command -v aria2c >/dev/null 2>&1; then
        log "Installing download prerequisites..."
        as_root apt-get update
        apt_install aria2 ca-certificates curl wget
    fi
}

install_cuda_and_cudnn() {
    local nvcc_version=""
    if command -v nvcc >/dev/null 2>&1; then
        nvcc_version="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)"
    elif [[ -x "/usr/local/cuda-$CUDA_VERSION/bin/nvcc" ]]; then
        nvcc_version="$CUDA_VERSION"
    fi

    local cudnn_installed=0
    if dpkg-query -W -f='${Status}' cudnn9-cuda-12 2>/dev/null | grep -q 'ok installed'; then
        cudnn_installed=1
    fi

    if [[ "$nvcc_version" == "$CUDA_VERSION" && "$cudnn_installed" == 1 ]]; then
        log "CUDA Toolkit $CUDA_VERSION and cuDNN 9 are already installed."
        return
    fi

    ensure_download_tools
    log "Installing CUDA Toolkit $CUDA_VERSION and cuDNN $CUDNN_VERSION from NVIDIA's local DEB repositories..."

    local repo_name="cuda-repo-${CUDA_REPO_DISTRO}-12-9-local"
    local repo_deb="${repo_name}_${CUDA_LOCAL_REPO_VERSION}_amd64.deb"
    local cudnn_repo_name="cudnn-local-repo-${CUDA_REPO_DISTRO}-${CUDNN_VERSION}"
    local cudnn_repo_deb="${cudnn_repo_name}_1.0-1_amd64.deb"
    local download_dir
    local installer_cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/h100-vllm/installers"
    local keyring
    download_dir="$(mktemp -d)"
    INSTALL_TEMP_DIR="$download_dir"
    mkdir -p "$installer_cache_dir"

    wget -qO "$download_dir/cuda-${CUDA_REPO_DISTRO}.pin" \
        "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_DISTRO}/x86_64/cuda-${CUDA_REPO_DISTRO}.pin"
    as_root install -m 644 \
        "$download_dir/cuda-${CUDA_REPO_DISTRO}.pin" \
        /etc/apt/preferences.d/cuda-repository-pin-600

    if dpkg-query -W -f='${Status}' "$repo_name" 2>/dev/null | grep -q 'ok installed' \
        && [[ -d "/var/$repo_name" ]]; then
        log "Reusing the existing CUDA local repository in /var/$repo_name."
    else
        aria2c \
            -x 8 \
            -s 8 \
            -k 4M \
            -c \
            --console-log-level=warn \
            --summary-interval=1 \
            --auto-file-renaming=false \
            --allow-overwrite=true \
            --dir="$installer_cache_dir" \
            --out="$repo_deb" \
            "https://developer.download.nvidia.com/compute/cuda/${CUDA_RELEASE}/local_installers/${repo_deb}"
        as_root dpkg -i "$installer_cache_dir/$repo_deb"
        rm -f "$installer_cache_dir/$repo_deb" "$installer_cache_dir/${repo_deb}.aria2"
    fi

    keyring="$(find "/var/$repo_name" -maxdepth 1 -type f -name 'cuda-*-keyring.gpg' -print -quit)"
    [[ -n "$keyring" ]] || die "CUDA local repository keyring was not found in /var/$repo_name."
    as_root cp "$keyring" /usr/share/keyrings/

    if dpkg-query -W -f='${Status}' "$cudnn_repo_name" 2>/dev/null | grep -q 'ok installed' \
        && [[ -d "/var/$cudnn_repo_name" ]]; then
        log "Reusing the existing cuDNN local repository in /var/$cudnn_repo_name."
    else
        aria2c \
            -x 8 \
            -s 8 \
            -k 4M \
            -c \
            --console-log-level=warn \
            --summary-interval=1 \
            --auto-file-renaming=false \
            --allow-overwrite=true \
            --dir="$installer_cache_dir" \
            --out="$cudnn_repo_deb" \
            "https://developer.download.nvidia.com/compute/cudnn/${CUDNN_VERSION}/local_installers/${cudnn_repo_deb}"
        as_root dpkg -i "$installer_cache_dir/$cudnn_repo_deb"
        rm -f "$installer_cache_dir/$cudnn_repo_deb" "$installer_cache_dir/${cudnn_repo_deb}.aria2"
    fi

    keyring="$(find "/var/$cudnn_repo_name" -maxdepth 1 -type f -name 'cudnn-*-keyring.gpg' -print -quit)"
    [[ -n "$keyring" ]] || die "cuDNN local repository keyring was not found in /var/$cudnn_repo_name."
    as_root cp "$keyring" /usr/share/keyrings/

    as_root apt-get update
    apt_install "cuda-toolkit-12-9" "cudnn9-cuda-12"

    as_root env DEBIAN_FRONTEND=noninteractive apt-get remove -y --purge \
        "$repo_name" "$cudnn_repo_name"
    as_root rm -f /etc/apt/preferences.d/cuda-repository-pin-600
    rm -rf "$download_dir"
    INSTALL_TEMP_DIR=""

    [[ -x "/usr/local/cuda-$CUDA_VERSION/bin/nvcc" ]] || die "CUDA installation completed, but nvcc was not found."
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        log "uv is already installed: $(uv --version)"
        return
    fi

    ensure_download_tools
    log "Installing uv..."
    local installer
    installer="$(mktemp)"
    curl -LsSf https://astral.sh/uv/install.sh -o "$installer"
    env UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR="$HOME/.local/bin" sh "$installer"
    rm -f "$installer"
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv installation did not produce an executable on PATH."
}

install_ffmpeg() {
    local installed_version=""
    if command -v ffmpeg >/dev/null 2>&1; then
        installed_version="$(ffmpeg -version | awk 'NR == 1 { print $3 }')"
    fi

    if [[ "$installed_version" == 7.* ]]; then
        log "FFmpeg $installed_version is already installed."
        return
    fi

    ensure_download_tools
    log "Building and installing FFmpeg $FFMPEG_VERSION from the official source release..."
    as_root apt-get update
    apt_install build-essential nasm pkg-config xz-utils ca-certificates

    local build_dir
    build_dir="$(mktemp -d)"
    curl -LsSf "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
        -o "$build_dir/ffmpeg.tar.xz"
    tar -xJf "$build_dir/ffmpeg.tar.xz" -C "$build_dir"

    (
        cd "$build_dir/ffmpeg-$FFMPEG_VERSION"
        ./configure \
            --prefix=/usr/local \
            --disable-debug \
            --disable-doc \
            --disable-static \
            --enable-shared
        make -j"$(nproc)"
        as_root make install
    )
    as_root ldconfig
    rm -rf "$build_dir"

    command -v ffmpeg >/dev/null 2>&1 || die "FFmpeg installation completed, but ffmpeg was not found on PATH."
    [[ "$(ffmpeg -version | awk 'NR == 1 { print $3 }')" == 7.* ]] || die "FFmpeg 7 installation verification failed."
}

install_cuda_and_cudnn
export PATH="/usr/local/cuda-$CUDA_VERSION/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-$CUDA_VERSION/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

install_uv
install_ffmpeg

log "Synchronizing locked dependencies in $ENGINE_DIR..."
cd "$ENGINE_DIR"
uv sync --frozen

export VLLM_BASE_URL="http://127.0.0.1:$VLLM_PORT/v1"
export VLLM_MODEL="$MODEL"
export PYTHONUNBUFFERED=1

log "Starting vLLM on port $VLLM_PORT..."
setsid stdbuf -oL -eL uv run vllm serve "$MODEL" \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --max-model-len 8192 \
    --limit-mm-per-prompt.audio 1 \
    --mm-processor-kwargs.audio_kwargs.max_length 480000 \
    --gpu-memory-utilization 0.9 \
    --uvicorn-log-level trace &
VLLM_PID=$!

log "Waiting for vLLM to become healthy..."
while ! curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || {
        wait "$VLLM_PID" || true
        die "vLLM exited before becoming healthy."
    }
    sleep 5
done

log "vLLM is ready. Starting app.py on 0.0.0.0:$APP_PORT with trace logging..."
setsid stdbuf -oL -eL uv run uvicorn app:app \
    --host 0.0.0.0 \
    --port "$APP_PORT" \
    --log-level trace &
APP_PID=$!

log "API is listening on port $APP_PORT. Press Ctrl+C to stop app.py and vLLM."

set +e
wait -n "$VLLM_PID" "$APP_PID"
status=$?
set -e

if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    log "vLLM exited with status $status."
else
    log "app.py exited with status $status."
fi

exit "$status"
