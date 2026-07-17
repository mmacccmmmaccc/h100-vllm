#!/usr/bin/env bash

set -Eeuo pipefail

SETUP_START_SECONDS=$SECONDS

CUDA_VERSION="13.0"
CUDA_RELEASE="13.0.2"
CUDA_LOCAL_REPO_VERSION="13.0.2-580.95.05-1"
ACTIVE_CUDA_VERSION=""
ACTIVE_CUDA_HOME=""
CUDNN_VERSION="9.19.0"
CUDNN_MIN_VERSION="9.19.0"
CUDNN_PACKAGE_VERSION="9.19.0.56-1"
ACTIVE_CUDNN_VERSION=""
ACTIVE_CUDNN_PACKAGE=""
FFMPEG_VERSION="7.1.5"
HTTP_CONNECTIONS="8"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/vllm-cu130"
INSTALL_TEMP_DIR=""
CLEANED_UP=0
STEP_CURRENT=0
STEP_TOTAL=6

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

apt_install() {
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
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
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
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
    *) die "CUDA 13.0 automated installation supports Ubuntu 22.04 or 24.04; found ${VERSION_ID:-unknown}." ;;
esac

[[ "$(uname -m)" == "x86_64" ]] || die "This CUDA installer currently supports x86_64 only."

ensure_download_tools() {
    if ! command -v curl >/dev/null 2>&1 \
        || ! command -v wget >/dev/null 2>&1 \
        || ! command -v aria2c >/dev/null 2>&1; then
        log "Installing download prerequisites..."
        sudo -n apt-get update
        apt_install aria2 ca-certificates curl wget
    fi
}

detect_compatible_cuda_toolkit() {
    local nvcc_path=""
    local nvcc_real_path=""
    local nvcc_version=""
    local -a nvcc_candidates=()

    if command -v nvcc >/dev/null 2>&1; then
        nvcc_candidates+=("$(command -v nvcc)")
    fi
    [[ -x /usr/local/cuda/bin/nvcc ]] && nvcc_candidates+=(/usr/local/cuda/bin/nvcc)
    [[ -x "/usr/local/cuda-$CUDA_VERSION/bin/nvcc" ]] \
        && nvcc_candidates+=("/usr/local/cuda-$CUDA_VERSION/bin/nvcc")

    shopt -s nullglob
    nvcc_candidates+=(/usr/local/cuda-13*/bin/nvcc)
    shopt -u nullglob

    for nvcc_path in "${nvcc_candidates[@]}"; do
        [[ -x "$nvcc_path" ]] || continue
        nvcc_version="$("$nvcc_path" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)"
        [[ "$nvcc_version" == 13.* ]] || continue

        nvcc_real_path="$(readlink -f -- "$nvcc_path")"
        ACTIVE_CUDA_VERSION="$nvcc_version"
        ACTIVE_CUDA_HOME="$(cd -- "$(dirname -- "$nvcc_real_path")/.." && pwd -P)"
        return 0
    done

    ACTIVE_CUDA_VERSION=""
    ACTIVE_CUDA_HOME=""
    return 1
}

detect_compatible_cudnn() {
    local package_name=""
    local package_version=""
    local upstream_version=""

    for package_name in cudnn9-cuda-13 libcudnn9-cuda-13; do
        package_version="$(dpkg-query -W -f='${Version}' "$package_name" 2>/dev/null || true)"
        [[ -n "$package_version" ]] || continue

        # Ignore an optional Debian epoch when comparing the upstream version.
        upstream_version="${package_version#*:}"
        if dpkg --compare-versions "$upstream_version" ge "$CUDNN_MIN_VERSION" \
            && dpkg --compare-versions "$upstream_version" lt 10; then
            ACTIVE_CUDNN_VERSION="$package_version"
            ACTIVE_CUDNN_PACKAGE="$package_name"
            return 0
        fi
    done

    ACTIVE_CUDNN_VERSION=""
    ACTIVE_CUDNN_PACKAGE=""
    return 1
}

install_cuda_and_cudnn() {
    local cuda_installed=0
    if detect_compatible_cuda_toolkit; then
        cuda_installed=1
    fi

    local cudnn_installed=0
    if detect_compatible_cudnn; then
        cudnn_installed=1
    fi

    if (( cuda_installed == 1 && cudnn_installed == 1 )); then
        log "Compatible CUDA Toolkit $ACTIVE_CUDA_VERSION and cuDNN $ACTIVE_CUDNN_VERSION are already installed."
        return
    fi

    ensure_download_tools
    if (( cuda_installed == 0 )); then
        log "No CUDA 13.x toolkit was detected; installing the default CUDA Toolkit $CUDA_VERSION."
    else
        log "Using detected CUDA Toolkit $ACTIVE_CUDA_VERSION at $ACTIVE_CUDA_HOME."
    fi
    if (( cudnn_installed == 0 )); then
        log "No supported cuDNN >=$CUDNN_MIN_VERSION,<10 was detected; installing cuDNN $CUDNN_VERSION."
    else
        log "Using detected cuDNN $ACTIVE_CUDNN_VERSION from $ACTIVE_CUDNN_PACKAGE."
    fi

    local repo_name="cuda-repo-${CUDA_REPO_DISTRO}-13-0-local"
    local repo_deb="${repo_name}_${CUDA_LOCAL_REPO_VERSION}_amd64.deb"
    local cudnn_repo_name="cudnn-local-repo-${CUDA_REPO_DISTRO}-${CUDNN_VERSION}"
    local cudnn_repo_deb="${cudnn_repo_name}_1.0-1_amd64.deb"
    local download_dir
    local installer_cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/h100-vllm/vllm-cuda_toolkit_130/installers"
    local keyring
    local -a packages=()
    local -a repo_packages=()
    download_dir="$(mktemp -d)"
    INSTALL_TEMP_DIR="$download_dir"
    mkdir -p "$installer_cache_dir"

    if (( cuda_installed == 0 )); then
        wget -qO "$download_dir/cuda-${CUDA_REPO_DISTRO}.pin" \
            "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_DISTRO}/x86_64/cuda-${CUDA_REPO_DISTRO}.pin"
        sudo -n install -m 644 \
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
            sudo -n dpkg -i "$installer_cache_dir/$repo_deb"
            rm -f "$installer_cache_dir/$repo_deb" "$installer_cache_dir/${repo_deb}.aria2"
        fi

        keyring="$(find "/var/$repo_name" -maxdepth 1 -type f -name 'cuda-*-keyring.gpg' -print -quit)"
        [[ -n "$keyring" ]] || die "CUDA local repository keyring was not found in /var/$repo_name."
        sudo -n cp "$keyring" /usr/share/keyrings/
        packages+=("cuda-toolkit-13-0")
        repo_packages+=("$repo_name")
    fi

    if (( cudnn_installed == 0 )); then
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
            sudo -n dpkg -i "$installer_cache_dir/$cudnn_repo_deb"
            rm -f "$installer_cache_dir/$cudnn_repo_deb" "$installer_cache_dir/${cudnn_repo_deb}.aria2"
        fi

        keyring="$(find "/var/$cudnn_repo_name" -maxdepth 1 -type f -name 'cudnn-*-keyring.gpg' -print -quit)"
        [[ -n "$keyring" ]] || die "cuDNN local repository keyring was not found in /var/$cudnn_repo_name."
        sudo -n cp "$keyring" /usr/share/keyrings/
        packages+=("cudnn9-cuda-13=$CUDNN_PACKAGE_VERSION")
        repo_packages+=("$cudnn_repo_name")
    fi

    sudo -n apt-get update
    apt_install "${packages[@]}"

    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get remove -y --purge \
        "${repo_packages[@]}"
    if (( cuda_installed == 0 )); then
        sudo -n rm -f /etc/apt/preferences.d/cuda-repository-pin-600
    fi
    rm -rf "$download_dir"
    INSTALL_TEMP_DIR=""

    detect_compatible_cuda_toolkit \
        || die "CUDA installation completed, but a compatible CUDA 13.x nvcc was not found."
    detect_compatible_cudnn \
        || die "cuDNN installation completed, but a supported cuDNN >=$CUDNN_MIN_VERSION,<10 was not found."
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

login_huggingface() {
    local hf_cli="$VENV_DIR/bin/hf"
    local hf_token=""
    local login_status=0

    [[ -x "$hf_cli" ]] || die "The Hugging Face CLI was not found at $hf_cli."
    IFS= read -r -s -p "Paste your Hugging Face access token: " hf_token
    printf '\n'
    [[ -n "$hf_token" ]] || die "A Hugging Face access token is required."

    "$hf_cli" auth login --token "$hf_token" || login_status=$?
    hf_token=""
    unset hf_token
    return "$login_status"
}

install_libsndfile() {
    if dpkg-query -W -f='${Status}' libsndfile1 2>/dev/null | grep -q 'ok installed'; then
        log "libsndfile1 is already installed."
        return
    fi

    ensure_download_tools
    log "Installing libsndfile1..."
    sudo -n apt-get update
    apt_install libsndfile1
    dpkg-query -W -f='${Status}' libsndfile1 2>/dev/null \
        | grep -q 'ok installed' \
        || die "libsndfile1 installation verification failed."
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
    sudo -n apt-get update
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
        sudo -n make install
    )
    sudo -n ldconfig
    rm -rf "$build_dir"

    command -v ffmpeg >/dev/null 2>&1 || die "FFmpeg installation completed, but ffmpeg was not found on PATH."
    [[ "$(ffmpeg -version | awk 'NR == 1 { print $3 }')" == 7.* ]] || die "FFmpeg 7 installation verification failed."
}

step "Install or verify CUDA Toolkit 13.x (default $CUDA_VERSION) and cuDNN >=$CUDNN_MIN_VERSION,<10"
install_cuda_and_cudnn
export CUDA_HOME="$ACTIVE_CUDA_HOME"
export PATH="$ACTIVE_CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$ACTIVE_CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

step "Install or verify uv"
install_uv
step "Install or verify FFmpeg $FFMPEG_VERSION and libsndfile1"
install_libsndfile
install_ffmpeg

step "Synchronize locked Python dependencies in $PROJECT_DIR"
cd "$PROJECT_DIR"
uv sync --frozen

step "Authenticate with the Hugging Face CLI"
login_huggingface

step "Activate the virtual environment and open a CLI shell"
[[ -f "$VENV_DIR/bin/activate" ]] || die "uv sync completed, but $VENV_DIR/bin/activate was not found."
[[ -x "$VENV_DIR/bin/vllm" ]] || die "uv sync completed, but the vLLM CLI was not found in $VENV_DIR/bin."

SETUP_ELAPSED_SECONDS=$((SECONDS - SETUP_START_SECONDS))
log "Setup completed in $(format_duration "$SETUP_ELAPSED_SECONDS"). Opening an activated CLI shell."
log "Run 'vllm --help' to get started; run 'exit' to leave the environment."

trap - EXIT INT TERM HUP
exec "$BASH" --rcfile "$VENV_DIR/bin/activate" -i
