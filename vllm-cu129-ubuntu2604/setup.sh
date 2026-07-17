#!/usr/bin/env bash

set -Eeuo pipefail

SETUP_START_SECONDS=$SECONDS

CUDA_VERSION="12.9"
CUDA_RELEASE="12.9.1"
CUDA_RUNFILE="cuda_12.9.1_575.57.08_linux.run"
CUDA_RUNFILE_MD5="a52d6c204bd4268627dfdab8bfeeb0d1"
CUDA_INSTALL_DIR="/usr/local/cuda-12.9"
CUDA_HOST_GCC_MAJOR="14"
MIN_NVIDIA_DRIVER_VERSION="575.57.08"
LIBXML2_COMPAT_DEB="libxml2_2.9.14+dfsg-1.3ubuntu3.8_amd64.deb"
LIBXML2_COMPAT_SHA256="bfd07c01d6e5ab3e327f3ca5819409b1914bbfb3f1a016d53e4dabd5f96143bb"
LIBXML2_COMPAT_URL="https://security.ubuntu.com/ubuntu/pool/main/libx/libxml2/$LIBXML2_COMPAT_DEB"
LIBICU_COMPAT_DEB="libicu74_74.2-1ubuntu3_amd64.deb"
LIBICU_COMPAT_SHA256="d29c97a21a3e3254731cfac186e4d4e611e5e67d2c9a0430f6acfbd9acaefa2e"
LIBICU_COMPAT_URL="https://archive.ubuntu.com/ubuntu/pool/main/i/icu/$LIBICU_COMPAT_DEB"
FFMPEG_VERSION="7.1.5"
HTTP_CONNECTIONS="8"
SUDO_AUTH_DURATION_SECONDS=3600
SUDO_REFRESH_INTERVAL_SECONDS=50

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/vllm-cu129-ubuntu2604"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
SUDO_KEEPALIVE_PID=""
INSTALL_TEMP_DIR=""
CUDA_INSTALLER_COMPAT_LIB_DIR=""
CLEANED_UP=0
STEP_CURRENT=0
STEP_TOTAL=7

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

authorize_sudo_for_60_minutes() {
    if (( EUID == 0 )); then
        log "Running as root; sudo authorization is not required."
        return
    fi

    command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages."
    log "Enter your sudo password once; authorization will remain active for up to 60 minutes."
    sudo -v

    (
        deadline=$(( $(date +%s) + SUDO_AUTH_DURATION_SECONDS ))
        while :; do
            now=$(date +%s)
            wait_seconds=$((deadline - now))
            if (( wait_seconds <= 0 )); then
                sudo -k
                exit 0
            fi
            if (( wait_seconds > SUDO_REFRESH_INTERVAL_SECONDS )); then
                wait_seconds=$SUDO_REFRESH_INTERVAL_SECONDS
            fi

            sleep "$wait_seconds"
            now=$(date +%s)
            if (( now >= deadline )); then
                sudo -k
                exit 0
            fi
            sudo -n -v >/dev/null 2>&1 || exit 0
        done
    ) &
    SUDO_KEEPALIVE_PID=$!
}

stop_sudo_authorization() {
    if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
        if kill -0 "$SUDO_KEEPALIVE_PID" 2>/dev/null; then
            kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
        fi
        wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
        SUDO_KEEPALIVE_PID=""
    fi

    if (( EUID != 0 )); then
        sudo -k >/dev/null 2>&1 || true
    fi
}

cleanup() {
    local status=$?
    (( CLEANED_UP == 0 )) || return "$status"
    CLEANED_UP=1

    trap - EXIT INT TERM HUP
    stop_sudo_authorization
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
unset HF_XET_NUM_CONCURRENT_RANGE_GETS
unset HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY

[[ -r /etc/os-release ]] || die "This installer requires an Ubuntu system with apt."
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "Unsupported OS: ${PRETTY_NAME:-unknown}. Ubuntu is required."
command -v apt-get >/dev/null 2>&1 || die "apt-get was not found."
[[ -f "$PROJECT_DIR/pyproject.toml" ]] || die "Missing $PROJECT_DIR/pyproject.toml."
[[ -f "$PROJECT_DIR/uv.lock" ]] || die "Missing $PROJECT_DIR/uv.lock."

[[ "${VERSION_ID:-}" == "26.04" ]] \
    || die "This cu129 binary-runtime context requires Ubuntu 26.04; found ${VERSION_ID:-unknown}."

[[ "$(uname -m)" == "x86_64" ]] || die "This setup currently supports x86_64 only."

ensure_download_tools() {
    if ! command -v curl >/dev/null 2>&1 \
        || ! command -v wget >/dev/null 2>&1 \
        || ! command -v aria2c >/dev/null 2>&1; then
        log "Installing download prerequisites..."
        as_root apt-get update
        apt_install aria2 ca-certificates curl wget
    fi
}

file_matches_sha256() {
    local file_path=$1
    local expected_sha256=$2
    local checksum_line=""

    [[ -f "$file_path" ]] || return 1
    checksum_line="$(sha256sum "$file_path")"
    [[ "${checksum_line%% *}" == "$expected_sha256" ]]
}

verify_nvidia_driver() {
    local driver_version=""

    command -v nvidia-smi >/dev/null 2>&1 \
        || die "nvidia-smi was not found. Install an NVIDIA driver that supports CUDA 12.9, then retry."

    driver_version="$(
        nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
            | head -n1 \
            | tr -d '[:space:]'
    )"
    [[ -n "$driver_version" ]] || die "nvidia-smi could not report an NVIDIA driver version."

    dpkg --compare-versions "$driver_version" ge "$MIN_NVIDIA_DRIVER_VERSION" \
        || die "NVIDIA driver $driver_version is too old; CUDA 12.9 requires driver $MIN_NVIDIA_DRIVER_VERSION or newer."

    log "NVIDIA driver $driver_version supports CUDA Toolkit $CUDA_VERSION and the locked cu129 wheels."
}

install_cuda_host_compiler() {
    local gcc_path="/usr/bin/gcc-$CUDA_HOST_GCC_MAJOR"
    local gxx_path="/usr/bin/g++-$CUDA_HOST_GCC_MAJOR"

    if [[ ! -x "$gcc_path" || ! -x "$gxx_path" || ! -x /usr/bin/gnudd ]]; then
        log "Installing GCC $CUDA_HOST_GCC_MAJOR and GNU coreutils for CUDA $CUDA_VERSION..."
        as_root apt-get update
        apt_install "gcc-$CUDA_HOST_GCC_MAJOR" "g++-$CUDA_HOST_GCC_MAJOR" gnu-coreutils
    fi

    [[ -x "$gcc_path" && -x "$gxx_path" ]] \
        || die "GCC $CUDA_HOST_GCC_MAJOR installation failed; CUDA $CUDA_VERSION does not support Ubuntu 26.04's default GCC 15."
    [[ -x /usr/bin/gnudd ]] \
        || die "GNU dd was not found at /usr/bin/gnudd; it is required for reliable CUDA runfile extraction on Ubuntu 26.04."

    log "Using $($gcc_path -dumpfullversion -dumpversion) as the CUDA host compiler."
    log "Using /usr/bin/gnudd for NVIDIA runfile extraction on Ubuntu 26.04."
}

prepare_cuda_installer_compat_libraries() {
    local compat_cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/h100-vllm/cu129/installers/ubuntu2404-libxml2-compat"
    local package_cache_dir="$compat_cache_dir/packages"
    local extract_dir="$compat_cache_dir/root"
    local lib_dir="$extract_dir/usr/lib/x86_64-linux-gnu"
    local package_path=""

    if [[ -e "$lib_dir/libxml2.so.2" \
        && -e "$lib_dir/libicuuc.so.74" \
        && -e "$lib_dir/libicudata.so.74" ]]; then
        CUDA_INSTALLER_COMPAT_LIB_DIR="$lib_dir"
        log "Reusing the private libxml2.so.2 compatibility bundle at $lib_dir."
        return
    fi

    ensure_download_tools
    mkdir -p "$package_cache_dir"

    package_path="$package_cache_dir/$LIBXML2_COMPAT_DEB"
    if ! file_matches_sha256 "$package_path" "$LIBXML2_COMPAT_SHA256"; then
        rm -f "$package_path"
        log "Downloading Ubuntu 24.04 libxml2.so.2 compatibility package..."
        curl --fail --location --retry 5 --output "$package_path" "$LIBXML2_COMPAT_URL"
    fi
    file_matches_sha256 "$package_path" "$LIBXML2_COMPAT_SHA256" \
        || die "Checksum verification failed for $LIBXML2_COMPAT_DEB."

    package_path="$package_cache_dir/$LIBICU_COMPAT_DEB"
    if ! file_matches_sha256 "$package_path" "$LIBICU_COMPAT_SHA256"; then
        rm -f "$package_path"
        log "Downloading Ubuntu 24.04 ICU 74 compatibility package..."
        curl --fail --location --retry 5 --output "$package_path" "$LIBICU_COMPAT_URL"
    fi
    file_matches_sha256 "$package_path" "$LIBICU_COMPAT_SHA256" \
        || die "Checksum verification failed for $LIBICU_COMPAT_DEB."

    rm -rf "$extract_dir"
    mkdir -p "$extract_dir"
    dpkg-deb --extract "$package_cache_dir/$LIBXML2_COMPAT_DEB" "$extract_dir"
    dpkg-deb --extract "$package_cache_dir/$LIBICU_COMPAT_DEB" "$extract_dir"

    [[ -e "$lib_dir/libxml2.so.2" \
        && -e "$lib_dir/libicuuc.so.74" \
        && -e "$lib_dir/libicudata.so.74" ]] \
        || die "The private CUDA installer compatibility bundle is incomplete."

    if LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ldd "$lib_dir/libxml2.so.2" | grep -q 'not found'; then
        die "The private libxml2.so.2 compatibility bundle has unresolved dependencies."
    fi

    CUDA_INSTALLER_COMPAT_LIB_DIR="$lib_dir"
    log "Prepared private libxml2.so.2 compatibility libraries for the CUDA installer."
}

install_cuda_toolkit() {
    local installed_version=""
    local installer_cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/h100-vllm/cu129/installers"
    local runfile_path="$installer_cache_dir/$CUDA_RUNFILE"
    local runfile_url="https://developer.download.nvidia.com/compute/cuda/$CUDA_RELEASE/local_installers/$CUDA_RUNFILE"
    local actual_md5=""

    if [[ -x "$CUDA_INSTALL_DIR/bin/nvcc" ]]; then
        installed_version="$(
            "$CUDA_INSTALL_DIR/bin/nvcc" --version \
                | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' \
                | head -n1
        )"
    fi

    if [[ "$installed_version" == "$CUDA_VERSION" ]]; then
        log "CUDA Toolkit $CUDA_VERSION is already installed at $CUDA_INSTALL_DIR."
        return
    fi

    ensure_download_tools
    mkdir -p "$installer_cache_dir"

    log "CUDA $CUDA_RELEASE does not officially qualify Ubuntu 26.04; using NVIDIA's standalone toolkit runfile with override mode."
    log "Downloading $CUDA_RUNFILE (the download can resume if interrupted)..."
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
        --out="$CUDA_RUNFILE" \
        "$runfile_url"

    actual_md5="$(md5sum "$runfile_path" | awk '{print $1}')"
    if [[ "$actual_md5" != "$CUDA_RUNFILE_MD5" ]]; then
        rm -f "$runfile_path" "${runfile_path}.aria2"
        die "CUDA runfile checksum mismatch: expected $CUDA_RUNFILE_MD5, got $actual_md5."
    fi

    prepare_cuda_installer_compat_libraries

    log "Installing CUDA Toolkit $CUDA_VERSION at $CUDA_INSTALL_DIR without installing an NVIDIA driver..."
    INSTALL_TEMP_DIR="$(mktemp -d)"
    ln -s /usr/bin/gnudd "$INSTALL_TEMP_DIR/dd"
    if ! as_root env \
        PATH="$INSTALL_TEMP_DIR:$PATH" \
        LD_LIBRARY_PATH="$CUDA_INSTALLER_COMPAT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        sh "$runfile_path" \
        --silent \
        --toolkit \
        --toolkitpath="$CUDA_INSTALL_DIR" \
        --override; then
        rm -rf "$INSTALL_TEMP_DIR"
        INSTALL_TEMP_DIR=""
        die "CUDA runfile installation failed. Check /var/log/cuda-installer.log for details."
    fi
    rm -rf "$INSTALL_TEMP_DIR"
    INSTALL_TEMP_DIR=""

    [[ -x "$CUDA_INSTALL_DIR/bin/nvcc" ]] \
        || die "CUDA installation completed, but $CUDA_INSTALL_DIR/bin/nvcc was not found."

    installed_version="$(
        "$CUDA_INSTALL_DIR/bin/nvcc" --version \
            | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' \
            | head -n1
    )"
    [[ "$installed_version" == "$CUDA_VERSION" ]] \
        || die "Expected CUDA $CUDA_VERSION after installation, but nvcc reported ${installed_version:-unknown}."

    rm -f "$runfile_path" "${runfile_path}.aria2"
    log "CUDA Toolkit $CUDA_VERSION installed successfully; the downloaded runfile was removed."
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
    as_root apt-get update
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

authorize_sudo_for_60_minutes

step "Verify NVIDIA driver compatibility with CUDA $CUDA_VERSION"
verify_nvidia_driver

step "Install or verify GCC $CUDA_HOST_GCC_MAJOR and CUDA Toolkit $CUDA_VERSION from the NVIDIA runfile"
install_cuda_host_compiler
install_cuda_toolkit
export CUDA_HOME="$CUDA_INSTALL_DIR"
export CC="/usr/bin/gcc-$CUDA_HOST_GCC_MAJOR"
export CXX="/usr/bin/g++-$CUDA_HOST_GCC_MAJOR"
export CUDAHOSTCXX="$CXX"
export NVCC_CCBIN="$CXX"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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
