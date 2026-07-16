#!/usr/bin/env bash

set -Eeuo pipefail

SETUP_START_SECONDS=$SECONDS

CUDA_VERSION="12.9"
CUDA_RELEASE="12.9.1"
CUDA_LOCAL_REPO_VERSION="12.9.1-575.57.08-1"
CUDNN_VERSION="9.17.1"
FFMPEG_VERSION="7.1.5"
HTTP_CONNECTIONS="8"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/.venv"
INSTALL_TEMP_DIR=""
CLEANED_UP=0
STEP_CURRENT=0
STEP_TOTAL=5

log() {
    printf '[%s] [setup] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    if (( hours > 0 )); then
        printf '%dh %02dm %02ds' "$hours" "$minutes" "$seconds"
    else
        printf '%dm %02ds' "$minutes" "$seconds"
    fi
}

step() {
    local width="${COLUMNS:-}"
    local line
    local label
    local color=""
    local reset=""

    (( STEP_CURRENT += 1 ))
    if ! [[ "$width" =~ ^[0-9]+$ ]] || (( width < 40 )); then
        width="$(tput cols 2>/dev/null || printf '80')"
    fi
    printf -v line '%*s' "$width" ''
    line=${line// /=}
    label="STEP $STEP_CURRENT/$STEP_TOTAL | $*"

    if [[ -t 1 && "${TERM:-dumb}" != "dumb" ]]; then
        color=$'\033[1;36m'
        reset=$'\033[0m'
    fi

    printf '\n%b%s%b\n' "$color" "$line" "$reset"
    printf '%b%-*s%b\n' "$color" "$width" "$label" "$reset"
    printf '%b%s%b\n' "$color" "$line" "$reset"
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

cleanup() {
    local status=$?
    (( CLEANED_UP == 0 )) || return "$status"
    CLEANED_UP=1

    trap - EXIT INT TERM HUP
    if [[ -n "$INSTALL_TEMP_DIR" ]]; then
        rm -rf "$INSTALL_TEMP_DIR"
        INSTALL_TEMP_DIR=""
    fi

    exit "$status"
}

handle_signal() {
    local signal=$1
    local status=$2

    log "Received $signal; stopping setup..."
    exit "$status"
}

trap cleanup EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP

(( $# == 0 )) || die "This script does not accept arguments."

# Keep future Hugging Face CLI/model usage on the regular HTTP path.
export HF_HUB_DISABLE_XET=1
unset HF_XET_NUM_CONCURRENT_RANGE_GETS
unset HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY

[[ -r /etc/os-release ]] || die "This installer requires an Ubuntu system with apt."
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "Unsupported OS: ${PRETTY_NAME:-unknown}. Ubuntu is required."
command -v apt-get >/dev/null 2>&1 || die "apt-get was not found."
[[ -f "$PROJECT_DIR/pyproject.toml" ]] || die "Missing $PROJECT_DIR/pyproject.toml."
[[ -f "$PROJECT_DIR/uv.lock" ]] || die "Missing $PROJECT_DIR/uv.lock."

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
    local installer_cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/h100-vllm/cu129/installers"
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
            -x "$HTTP_CONNECTIONS" \
            -s "$HTTP_CONNECTIONS" \
            -k 1M \
            -c \
            --file-allocation=falloc \
            --disk-cache=64M \
            --max-tries=10 \
            --retry-wait=3 \
            --connect-timeout=30 \
            --timeout=60 \
            --console-log-level=warn \
            --show-console-readout=true \
            --summary-interval=0 \
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
            -x "$HTTP_CONNECTIONS" \
            -s "$HTTP_CONNECTIONS" \
            -k 1M \
            -c \
            --file-allocation=falloc \
            --disk-cache=64M \
            --max-tries=10 \
            --retry-wait=3 \
            --connect-timeout=30 \
            --timeout=60 \
            --console-log-level=warn \
            --show-console-readout=true \
            --summary-interval=0 \
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

step "Install or verify CUDA Toolkit $CUDA_VERSION and cuDNN $CUDNN_VERSION"
install_cuda_and_cudnn
export PATH="/usr/local/cuda-$CUDA_VERSION/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-$CUDA_VERSION/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

step "Install or verify uv"
install_uv
step "Install or verify FFmpeg $FFMPEG_VERSION"
install_ffmpeg

step "Synchronize locked Python dependencies in $PROJECT_DIR"
cd "$PROJECT_DIR"
uv sync --frozen

step "Activate the virtual environment and open a CLI shell"
[[ -f "$VENV_DIR/bin/activate" ]] || die "uv sync completed, but $VENV_DIR/bin/activate was not found."
[[ -x "$VENV_DIR/bin/vllm" ]] || die "uv sync completed, but the vLLM CLI was not found in $VENV_DIR/bin."

SETUP_ELAPSED_SECONDS=$((SECONDS - SETUP_START_SECONDS))
log "Setup completed in $(format_duration "$SETUP_ELAPSED_SECONDS"). Opening an activated CLI shell."
log "Run 'vllm --help' to get started; run 'exit' to leave the environment."

trap - EXIT INT TERM HUP
exec "$BASH" --rcfile "$VENV_DIR/bin/activate" -i
