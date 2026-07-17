#!/usr/bin/env bash

set -Eeuo pipefail

SETUP_START_SECONDS=$SECONDS

FFMPEG_VERSION="7.1.5"
SUDO_AUTH_DURATION_SECONDS=3600
SUDO_REFRESH_INTERVAL_SECONDS=50

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/vllm-cu130"
INSTALL_TEMP_DIR=""
SUDO_KEEPALIVE_PID=""
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

[[ "$(uname -m)" == "x86_64" ]] || die "This vLLM installer currently supports x86_64 only."

ensure_download_tools() {
    if ! command -v curl >/dev/null 2>&1; then
        log "Installing download prerequisites..."
        as_root apt-get update
        apt_install ca-certificates curl
    fi
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

    [[ -x "$hf_cli" ]] || die "The Hugging Face CLI was not found at $hf_cli."
    log "Starting Hugging Face's interactive authentication flow."
    "$hf_cli" auth login
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

log "Skipping system CUDA Toolkit and cuDNN installation; the locked Python environment supplies its cu130 runtime libraries."

step "Install or verify uv"
install_uv
step "Install or verify FFmpeg $FFMPEG_VERSION"
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
