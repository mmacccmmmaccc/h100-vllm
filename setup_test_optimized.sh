#!/usr/bin/env bash

set -Eeuo pipefail
SETUP_START_SECONDS=$SECONDS

# Fixed component versions and runtime settings.
CUDA_VERSION="12.9"
CUDA_RELEASE="12.9.1"
CUDA_LOCAL_REPO_VERSION="12.9.1-575.57.08-1"
CUDNN_VERSION="9.17.1"
FFMPEG_VERSION="7.1.5"
MODEL="prithivMLmods/gemma-4-E4B-it-FP8"
MODEL_REVISION="main"
MODEL_WEIGHTS="model.safetensors"
MODEL_WEIGHTS_SIZE="13309692724"
VLLM_PORT="8080"
APP_PORT="7000"
HTTP_CONNECTIONS="8"
BACKGROUND_PROGRESS_INTERVAL="1"

# Derived paths and mutable process state.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$SCRIPT_DIR/vllm_engine"
LOCAL_MODEL_DIR="$ENGINE_DIR/models/${MODEL//\//--}"
MODEL_DOWNLOAD_LOG="$ENGINE_DIR/.model-download.log"
MODEL_METADATA_LOG="$ENGINE_DIR/.model-metadata-download.log"
UV_SYNC_LOG="$ENGINE_DIR/.uv-sync.log"
CUDA_DOWNLOAD_LOG="$ENGINE_DIR/.cuda-download.log"
CUDNN_DOWNLOAD_LOG="$ENGINE_DIR/.cudnn-download.log"
VLLM_PID=""
APP_PID=""
MODEL_DOWNLOAD_PID=""
UV_SYNC_PID=""
CUDA_DOWNLOAD_PID=""
CUDNN_DOWNLOAD_PID=""
BACKGROUND_PROGRESS_PID=""
PROGRESS_STATE_DIR=""
MODEL_PROGRESS_STATE_FILE=""
MODEL_METADATA_PROGRESS_STATE_FILE=""
UV_PROGRESS_STATE_FILE=""
CUDA_PROGRESS_STATE_FILE=""
CUDNN_PROGRESS_STATE_FILE=""
BACKGROUND_PROGRESS_STOP_FILE=""
INSTALL_TEMP_DIR=""
CLEANED_UP=0
APT_UPDATED=0
CUDA_DOWNLOADS_STARTED=0
SUDO_KEEPALIVE_PID=""
STEP_CURRENT=0
STEP_TOTAL=5

# Output helpers.
log() {
    printf '[%s] [setup] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

print_step_banner() {
    local label=$1
    local width="${COLUMNS:-}"
    local line
    local color=""
    local reset=""

    if ! [[ "$width" =~ ^[0-9]+$ ]] || (( width < 40 )); then
        width="$(tput cols 2>/dev/null || printf '80')"
    fi
    printf -v line '%*s' "$width" ''
    line=${line// /=}

    if [[ -t 1 && "${TERM:-dumb}" != "dumb" ]]; then
        color=$'\033[1;36m'
        reset=$'\033[0m'
    fi

    printf '\n%b%s%b\n' "$color" "$line" "$reset"
    printf '%b%-*s%b\n' "$color" "$width" "$label" "$reset"
    printf '%b%s%b\n' "$color" "$line" "$reset"
}

step() {
    (( STEP_CURRENT += 1 ))
    print_step_banner "STEP $STEP_CURRENT/$STEP_TOTAL | $*"
}

die() {
    printf '[setup] ERROR: %s\n' "$*" >&2
    exit 1
}

# Privilege, subprocess, and cleanup helpers.
as_root() {
    if (( EUID == 0 )); then
        "$@"
    else
        command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages."
        sudo -n -v >/dev/null 2>&1 || die \
            "Cached sudo authorization expired. Rerun the setup so authorization can be requested at the beginning."
        sudo -n "$@"
    fi
}

initialize_sudo_session() {
    (( EUID != 0 )) || return 0
    command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages."

    log "Requesting sudo authorization once before downloads begin..."
    sudo -v || die "Unable to obtain sudo authorization."
    sudo -n -v >/dev/null 2>&1 || die \
        "This sudo policy does not allow credentials to remain cached for password-free installation."

    # Refresh the timestamp without prompting while downloads and builds run.
    (
        sleep_pid=""
        trap '
            [[ -z "$sleep_pid" ]] || kill "$sleep_pid" 2>/dev/null || true
            exit 0
        ' TERM INT HUP
        while :; do
            sleep 45 &
            sleep_pid=$!
            wait "$sleep_pid" || exit 0
            sleep_pid=""
            sudo -n -v >/dev/null 2>&1 || exit 1
        done
    ) >/dev/null 2>&1 &
    SUDO_KEEPALIVE_PID=$!
    log "Sudo authorization cached; later privileged commands will not prompt."
}

stop_sudo_keepalive() {
    [[ -n "$SUDO_KEEPALIVE_PID" ]] || return 0
    kill -TERM "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    SUDO_KEEPALIVE_PID=""
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

    # Restore the cursor and scroll region before emitting shutdown messages or
    # waiting on the larger download process groups.
    if [[ -n "$BACKGROUND_PROGRESS_PID" ]]; then
        if declare -F finish_background_download_progress >/dev/null 2>&1; then
            finish_background_download_progress
        else
            stop_process_group "$BACKGROUND_PROGRESS_PID"
            wait "$BACKGROUND_PROGRESS_PID" 2>/dev/null || true
            BACKGROUND_PROGRESS_PID=""
        fi
    fi
    stop_sudo_keepalive

    if [[ -n "$APP_PID" || -n "$VLLM_PID" || -n "$MODEL_DOWNLOAD_PID" \
        || -n "$UV_SYNC_PID" || -n "$CUDA_DOWNLOAD_PID" \
        || -n "$CUDNN_DOWNLOAD_PID" ]]; then
        log "Stopping setup child processes..."
    fi

    stop_process_group "$APP_PID"
    stop_process_group "$VLLM_PID"
    stop_process_group "$MODEL_DOWNLOAD_PID"
    stop_process_group "$UV_SYNC_PID"
    stop_process_group "$CUDA_DOWNLOAD_PID"
    stop_process_group "$CUDNN_DOWNLOAD_PID"

    local deadline=$((SECONDS + 15))
    while (( SECONDS < deadline )); do
        local app_alive=0
        local vllm_alive=0
        local model_download_alive=0
        local uv_sync_alive=0
        local cuda_download_alive=0
        local cudnn_download_alive=0
        process_group_is_alive "$APP_PID" && app_alive=1
        process_group_is_alive "$VLLM_PID" && vllm_alive=1
        process_group_is_alive "$MODEL_DOWNLOAD_PID" && model_download_alive=1
        process_group_is_alive "$UV_SYNC_PID" && uv_sync_alive=1
        process_group_is_alive "$CUDA_DOWNLOAD_PID" && cuda_download_alive=1
        process_group_is_alive "$CUDNN_DOWNLOAD_PID" && cudnn_download_alive=1
        (( app_alive == 0 && vllm_alive == 0 && model_download_alive == 0 \
            && uv_sync_alive == 0 && cuda_download_alive == 0 \
            && cudnn_download_alive == 0 )) && break
        sleep 1
    done

    for pid in "$APP_PID" "$VLLM_PID" "$MODEL_DOWNLOAD_PID" "$UV_SYNC_PID" \
        "$CUDA_DOWNLOAD_PID" "$CUDNN_DOWNLOAD_PID"; do
        [[ -n "$pid" ]] || continue
        if process_group_is_alive "$pid"; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done

    if declare -F remove_download_progress_state >/dev/null 2>&1; then
        remove_download_progress_state
    fi

    exit "$status"
}

handle_signal() {
    local signal=$1
    local status=$2

    if [[ -n "${BACKGROUND_PROGRESS_PID:-}" ]] \
        && declare -F finish_background_download_progress_for_signal >/dev/null 2>&1; then
        finish_background_download_progress_for_signal
    fi
    log "Received $signal; stopping setup and child processes..."
    exit "$status"
}

trap cleanup EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP

# CLI parsing and host validation.
usage() {
    printf 'Usage: bash %s --hf-token TOKEN [--ngrok-token TOKEN]\n' "${0##*/}"
}

HF_TOKEN=""
NGROK_AUTHTOKEN=""

while (( $# > 0 )); do
    case "$1" in
        --hf-token)
            (( $# >= 2 )) || die "--hf-token requires a value."
            [[ -z "$HF_TOKEN" ]] || die "--hf-token was specified more than once."
            [[ -n "$2" ]] || die "--hf-token requires a non-empty value."
            HF_TOKEN=$2
            shift 2
            ;;
        --ngrok-token)
            (( $# >= 2 )) || die "--ngrok-token requires a value."
            [[ -z "$NGROK_AUTHTOKEN" ]] || die "--ngrok-token was specified more than once."
            [[ -n "$2" ]] || die "--ngrok-token requires a non-empty value."
            NGROK_AUTHTOKEN=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "Unknown argument: $1"
            ;;
    esac
done

[[ -n "$HF_TOKEN" ]] || die "Missing required --hf-token TOKEN."

export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
if [[ -n "$NGROK_AUTHTOKEN" ]]; then
    export NGROK_AUTHTOKEN
else
    unset NGROK_AUTHTOKEN
fi
# Force the stable regular-HTTP path; hf-xet 1.5.2rc0 repeatedly stalled while
# decoding response bodies for this model, even with reduced concurrency.
export HF_HUB_DISABLE_XET=1
unset HF_XET_NUM_CONCURRENT_RANGE_GETS
unset HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY

[[ -r /etc/os-release ]] || die "This installer requires an Ubuntu system with apt."
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "Unsupported OS: ${PRETTY_NAME:-unknown}. Ubuntu is required."
command -v apt-get >/dev/null 2>&1 || die "apt-get was not found."
command -v script >/dev/null 2>&1 || die "The util-linux script command is required for live vLLM download progress."
[[ -f "$ENGINE_DIR/pyproject.toml" ]] || die "Missing $ENGINE_DIR/pyproject.toml."
[[ -f "$ENGINE_DIR/uv.lock" ]] || die "Missing $ENGINE_DIR/uv.lock."

case "${VERSION_ID:-}" in
    22.04) CUDA_REPO_DISTRO="ubuntu2204" ;;
    24.04) CUDA_REPO_DISTRO="ubuntu2404" ;;
    *) die "CUDA 12.9 automated installation supports Ubuntu 22.04 or 24.04; found ${VERSION_ID:-unknown}." ;;
esac

[[ "$(uname -m)" == "x86_64" ]] || die "This CUDA installer currently supports x86_64 only."

CUDA_REPO_NAME="cuda-repo-${CUDA_REPO_DISTRO}-12-9-local"
CUDA_REPO_DEB="${CUDA_REPO_NAME}_${CUDA_LOCAL_REPO_VERSION}_amd64.deb"
CUDNN_REPO_NAME="cudnn-local-repo-${CUDA_REPO_DISTRO}-${CUDNN_VERSION}"
CUDNN_REPO_DEB="${CUDNN_REPO_NAME}_1.0-1_amd64.deb"
INSTALLER_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/h100-vllm/installers"

# Download prerequisites and shared progress state.
apt_update() {
    if (( APT_UPDATED == 0 )); then
        as_root apt-get update
        APT_UPDATED=1
    fi
}

ensure_download_tools() {
    if ! command -v curl >/dev/null 2>&1 \
        || ! command -v wget >/dev/null 2>&1 \
        || ! command -v aria2c >/dev/null 2>&1; then
        log "Installing download prerequisites..."
        apt_update
        apt_install aria2 ca-certificates curl wget
    fi
}

initialize_download_progress_state() {
    PROGRESS_STATE_DIR="$(mktemp -d "$ENGINE_DIR/.download-progress.XXXXXX")"
    MODEL_PROGRESS_STATE_FILE="$PROGRESS_STATE_DIR/model"
    MODEL_METADATA_PROGRESS_STATE_FILE="$PROGRESS_STATE_DIR/model-metadata"
    UV_PROGRESS_STATE_FILE="$PROGRESS_STATE_DIR/vllm"
    CUDA_PROGRESS_STATE_FILE="$PROGRESS_STATE_DIR/cuda"
    CUDNN_PROGRESS_STATE_FILE="$PROGRESS_STATE_DIR/cudnn"
    BACKGROUND_PROGRESS_STOP_FILE="$PROGRESS_STATE_DIR/stop"
}

write_progress_state() {
    local state_file=$1
    local state=$2
    local temporary_file

    [[ -n "$state_file" && -d "$PROGRESS_STATE_DIR" ]] || return 0
    temporary_file="${state_file}.tmp.$$"
    printf '%s\n' "$state" > "$temporary_file"
    mv -f -- "$temporary_file" "$state_file"
}

read_progress_state() {
    local state_file=$1
    local state="pending"

    if [[ -f "$state_file" ]]; then
        IFS= read -r state < "$state_file" || true
    fi
    printf '%s' "${state:-pending}"
}

remove_download_progress_state() {
    [[ -n "$PROGRESS_STATE_DIR" ]] || return 0

    rm -f -- \
        "$MODEL_PROGRESS_STATE_FILE" \
        "$MODEL_METADATA_PROGRESS_STATE_FILE" \
        "$UV_PROGRESS_STATE_FILE" \
        "${UV_PROGRESS_STATE_FILE}.aggregate" \
        "$CUDA_PROGRESS_STATE_FILE" \
        "$CUDNN_PROGRESS_STATE_FILE" \
        "$BACKGROUND_PROGRESS_STOP_FILE" \
        "$PROGRESS_STATE_DIR"/*.tmp.* 2>/dev/null || true
    rmdir -- "$PROGRESS_STATE_DIR" 2>/dev/null || true
    PROGRESS_STATE_DIR=""
}

cuda_toolkit_is_installed() {
    local nvcc_version=""
    if command -v nvcc >/dev/null 2>&1; then
        nvcc_version="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)"
    elif [[ -x "/usr/local/cuda-$CUDA_VERSION/bin/nvcc" ]]; then
        nvcc_version="$CUDA_VERSION"
    fi

    [[ "$nvcc_version" == "$CUDA_VERSION" ]]
}

cudnn_is_installed() {
    dpkg-query -W -f='${Status}' cudnn9-cuda-12 2>/dev/null | grep -q 'ok installed'
}

nvidia_repo_is_ready() {
    local repo_name=$1

    dpkg-query -W -f='${Status}' "$repo_name" 2>/dev/null | grep -q 'ok installed' \
        && [[ -d "/var/$repo_name" ]]
}

# Background download process management.
start_state_tracked_background_process() {
    local state_file=$1
    local log_file=$2
    local pid_variable=$3
    shift 3

    rm -f "$log_file"
    write_progress_state "$state_file" running
    setsid bash -c '
        state_file=$1
        shift
        if "$@"; then
            status=0
            state=complete
        else
            status=$?
            state=failed
        fi
        temporary_file="${state_file}.tmp.$$"
        printf "%s\n" "$state" > "$temporary_file"
        mv -f -- "$temporary_file" "$state_file"
        exit "$status"
    ' setup-download "$state_file" "$@" >"$log_file" 2>&1 &
    printf -v "$pid_variable" '%s' "$!"
}

start_aria2_background_download() {
    local output_file=$1
    local url=$2
    local download_dir=$3
    local log_file=$4
    local state_file=$5
    local pid_variable=$6

    start_state_tracked_background_process \
        "$state_file" \
        "$log_file" \
        "$pid_variable" \
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
        --human-readable=false \
        --summary-interval=1 \
        --auto-file-renaming=false \
        --allow-overwrite=true \
        --dir="$download_dir" \
        --out="$output_file" \
        "$url"
}

start_cuda_downloads() {
    local cuda_installed=0
    local cudnn_installed=0

    (( CUDA_DOWNLOADS_STARTED == 0 )) || return 0
    CUDA_DOWNLOADS_STARTED=1

    if cuda_toolkit_is_installed; then
        cuda_installed=1
        write_progress_state "$CUDA_PROGRESS_STATE_FILE" installed
    fi
    if cudnn_is_installed; then
        cudnn_installed=1
        write_progress_state "$CUDNN_PROGRESS_STATE_FILE" installed
    fi
    if (( cuda_installed == 1 && cudnn_installed == 1 )); then
        return
    fi

    log "Starting CUDA Toolkit $CUDA_VERSION and cuDNN $CUDNN_VERSION download..."
    ensure_download_tools
    mkdir -p "$INSTALLER_CACHE_DIR"

    if (( cuda_installed == 0 )); then
        if ! nvidia_repo_is_ready "$CUDA_REPO_NAME"; then
            start_aria2_background_download \
                "$CUDA_REPO_DEB" \
                "https://developer.download.nvidia.com/compute/cuda/${CUDA_RELEASE}/local_installers/${CUDA_REPO_DEB}" \
                "$INSTALLER_CACHE_DIR" \
                "$CUDA_DOWNLOAD_LOG" \
                "$CUDA_PROGRESS_STATE_FILE" \
                CUDA_DOWNLOAD_PID
        else
            write_progress_state "$CUDA_PROGRESS_STATE_FILE" cached
        fi
    fi

    if (( cudnn_installed == 0 )); then
        if ! nvidia_repo_is_ready "$CUDNN_REPO_NAME"; then
            start_aria2_background_download \
                "$CUDNN_REPO_DEB" \
                "https://developer.download.nvidia.com/compute/cudnn/${CUDNN_VERSION}/local_installers/${CUDNN_REPO_DEB}" \
                "$INSTALLER_CACHE_DIR" \
                "$CUDNN_DOWNLOAD_LOG" \
                "$CUDNN_PROGRESS_STATE_FILE" \
                CUDNN_DOWNLOAD_PID
        else
            write_progress_state "$CUDNN_PROGRESS_STATE_FILE" cached
        fi
    fi
}

finish_aria2_background_download() {
    local pid_variable=$1
    local label=$2
    local log_file=$3
    local state_file=$4
    local pid=${!pid_variable}

    [[ -n "$pid" ]] || return 0

    if wait "$pid"; then
        printf -v "$pid_variable" '%s' ""
        write_progress_state "$state_file" complete
        rm -f "$log_file"
    else
        local status=$?
        printf -v "$pid_variable" '%s' ""
        write_progress_state "$state_file" failed
        tail -n 80 "$log_file" >&2 || true
        die "$label download failed with status $status; full output is in $log_file."
    fi
}

finish_cuda_downloads() {
    local cuda_state
    local cudnn_state

    while [[ -n "$CUDA_DOWNLOAD_PID" || -n "$CUDNN_DOWNLOAD_PID" ]]; do
        cuda_state="$(read_progress_state "$CUDA_PROGRESS_STATE_FILE")"
        cudnn_state="$(read_progress_state "$CUDNN_PROGRESS_STATE_FILE")"

        if [[ -n "$CUDA_DOWNLOAD_PID" \
            && ( "$cuda_state" == complete || "$cuda_state" == failed ) ]]; then
            finish_aria2_background_download \
                CUDA_DOWNLOAD_PID "CUDA Toolkit $CUDA_VERSION" "$CUDA_DOWNLOAD_LOG" "$CUDA_PROGRESS_STATE_FILE"
            continue
        fi
        if [[ -n "$CUDNN_DOWNLOAD_PID" \
            && ( "$cudnn_state" == complete || "$cudnn_state" == failed ) ]]; then
            finish_aria2_background_download \
                CUDNN_DOWNLOAD_PID "cuDNN $CUDNN_VERSION" "$CUDNN_DOWNLOAD_LOG" "$CUDNN_PROGRESS_STATE_FILE"
            continue
        fi
        if [[ -n "$CUDA_DOWNLOAD_PID" ]] && ! kill -0 "$CUDA_DOWNLOAD_PID" 2>/dev/null; then
            finish_aria2_background_download \
                CUDA_DOWNLOAD_PID "CUDA Toolkit $CUDA_VERSION" "$CUDA_DOWNLOAD_LOG" "$CUDA_PROGRESS_STATE_FILE"
            continue
        fi
        if [[ -n "$CUDNN_DOWNLOAD_PID" ]] && ! kill -0 "$CUDNN_DOWNLOAD_PID" 2>/dev/null; then
            finish_aria2_background_download \
                CUDNN_DOWNLOAD_PID "cuDNN $CUDNN_VERSION" "$CUDNN_DOWNLOAD_LOG" "$CUDNN_PROGRESS_STATE_FILE"
            continue
        fi
        sleep 0.2
    done
}

install_cuda_and_cudnn() {
    local cuda_needed=1
    local cudnn_needed=1
    local download_dir
    local keyring
    local -a packages=()
    local -a repo_packages=()

    if cuda_toolkit_is_installed; then
        cuda_needed=0
    fi
    if cudnn_is_installed; then
        cudnn_needed=0
    fi
    if (( cuda_needed == 0 && cudnn_needed == 0 )); then
        finish_cuda_downloads
        write_progress_state "$CUDA_PROGRESS_STATE_FILE" installed
        write_progress_state "$CUDNN_PROGRESS_STATE_FILE" installed
        log "CUDA Toolkit $CUDA_VERSION and cuDNN 9 are already installed."
        return
    fi

    (( CUDA_DOWNLOADS_STARTED == 1 )) || start_cuda_downloads
    log "Installing CUDA Toolkit $CUDA_VERSION and cuDNN $CUDNN_VERSION from NVIDIA's local DEB repositories..."

    download_dir="$(mktemp -d)"
    INSTALL_TEMP_DIR="$download_dir"

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
            --show-console-readout=false \
            --summary-interval=0 \
            --auto-file-renaming=false \
            --allow-overwrite=true \
            --dir="$installer_cache_dir" \
            --out="$repo_deb" \
            "https://developer.download.nvidia.com/compute/cuda/${CUDA_RELEASE}/local_installers/${repo_deb}"
        as_root dpkg -i "$installer_cache_dir/$repo_deb"
        rm -f "$installer_cache_dir/$repo_deb" "$installer_cache_dir/${repo_deb}.aria2"
    fi

    finish_cuda_downloads

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
            --show-console-readout=false \
            --summary-interval=0 \
            --auto-file-renaming=false \
            --allow-overwrite=true \
            --dir="$installer_cache_dir" \
            --out="$cudnn_repo_deb" \
            "https://developer.download.nvidia.com/compute/cudnn/${CUDNN_VERSION}/local_installers/${cudnn_repo_deb}"
        as_root dpkg -i "$installer_cache_dir/$cudnn_repo_deb"
        rm -f "$installer_cache_dir/$cudnn_repo_deb" "$installer_cache_dir/${cudnn_repo_deb}.aria2"
    fi

    if (( cudnn_needed == 1 )); then
        if nvidia_repo_is_ready "$CUDNN_REPO_NAME"; then
            log "Reusing the existing cuDNN local repository in /var/$CUDNN_REPO_NAME."
        else
            [[ -f "$INSTALLER_CACHE_DIR/$CUDNN_REPO_DEB" ]] || \
                die "The downloaded cuDNN repository package was not found at $INSTALLER_CACHE_DIR/$CUDNN_REPO_DEB."
            as_root dpkg -i "$INSTALLER_CACHE_DIR/$CUDNN_REPO_DEB"
            rm -f "$INSTALLER_CACHE_DIR/$CUDNN_REPO_DEB" "$INSTALLER_CACHE_DIR/${CUDNN_REPO_DEB}.aria2"
        fi

        keyring="$(find "/var/$CUDNN_REPO_NAME" -maxdepth 1 -type f -name 'cudnn-*-keyring.gpg' -print -quit)"
        [[ -n "$keyring" ]] || die "cuDNN local repository keyring was not found in /var/$CUDNN_REPO_NAME."
        as_root cp "$keyring" /usr/share/keyrings/
        packages+=("cudnn9-cuda-12")
        repo_packages+=("$CUDNN_REPO_NAME")
    fi

    APT_UPDATED=0
    apt_update
    apt_install "${packages[@]}"

    as_root env DEBIAN_FRONTEND=noninteractive apt-get remove -y --purge "${repo_packages[@]}"
    if (( cuda_needed == 1 )); then
        as_root rm -f /etc/apt/preferences.d/cuda-repository-pin-600
    fi
    rm -rf "$download_dir"
    INSTALL_TEMP_DIR=""

    cuda_toolkit_is_installed || die "CUDA installation completed, but CUDA $CUDA_VERSION was not found."
    cudnn_is_installed || die "cuDNN installation completed, but cudnn9-cuda-12 was not found."
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

# Live download dashboard.
start_uv_sync() {
    log "Starting locked Python dependency synchronization in the background..."
    rm -f "$UV_SYNC_LOG"
    setsid bash -o pipefail -c '
        stdbuf -oL -eL uv sync --frozen 2>&1 | tee "$1"
    ' bash "$UV_SYNC_LOG" &
    UV_SYNC_PID=$!
}

finish_uv_sync() {
    [[ -n "$UV_SYNC_PID" ]] || return 0

    if wait "$UV_SYNC_PID"; then
        UV_SYNC_PID=""
        write_progress_state "$UV_PROGRESS_STATE_FILE" complete
        rm -f "$UV_SYNC_LOG"
    else
        local status=$?
        UV_SYNC_PID=""
        write_progress_state "$UV_PROGRESS_STATE_FILE" failed
        tail -n 80 "$UV_SYNC_LOG" >&2 || true
        die "uv sync failed with status $status; full output is in $UV_SYNC_LOG."
    fi
}

configure_ngrok_repository() {
    if command -v ngrok >/dev/null 2>&1; then
        return
    fi

    if [[ -f /etc/apt/trusted.gpg.d/ngrok.asc && -f /etc/apt/sources.list.d/ngrok.list ]]; then
        APT_UPDATED=0
        return
    fi

    ensure_download_tools
    log "Configuring ngrok's official apt repository..."

    local install_dir
    install_dir="$(mktemp -d)"
    INSTALL_TEMP_DIR="$install_dir"

    curl -fsSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
        -o "$install_dir/ngrok.asc"
    printf '%s\n' 'deb https://ngrok-agent.s3.amazonaws.com bookworm main' \
        > "$install_dir/ngrok.list"

    as_root install -m 644 "$install_dir/ngrok.asc" \
        /etc/apt/trusted.gpg.d/ngrok.asc
    as_root install -m 644 "$install_dir/ngrok.list" \
        /etc/apt/sources.list.d/ngrok.list
    APT_UPDATED=0

    rm -rf "$install_dir"
    INSTALL_TEMP_DIR=""
}

install_ngrok() {
    if command -v ngrok >/dev/null 2>&1; then
        log "ngrok is already installed: $(ngrok version)"
        return
    fi

    configure_ngrok_repository
    log "Installing ngrok..."
    apt_update
    apt_install ngrok
    command -v ngrok >/dev/null 2>&1 || die "ngrok installation did not produce an executable on PATH."
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
    apt_update
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

start_model_download() {
    local model_url="https://huggingface.co/${MODEL}/resolve/${MODEL_REVISION}/${MODEL_WEIGHTS}?download=true"

    ensure_download_tools
    mkdir -p "$LOCAL_MODEL_DIR"

    if [[ ! -e "$LOCAL_MODEL_DIR/${MODEL_WEIGHTS}.aria2" ]] \
        && [[ -f "$LOCAL_MODEL_DIR/$MODEL_WEIGHTS" ]] \
        && [[ "$(stat -c '%s' "$LOCAL_MODEL_DIR/$MODEL_WEIGHTS")" == "$MODEL_WEIGHTS_SIZE" ]]; then
        log "$MODEL_WEIGHTS is already fully downloaded."
        write_progress_state "$MODEL_PROGRESS_STATE_FILE" cached
        return
    fi

    log "Starting $MODEL_WEIGHTS download for model $MODEL..."
    start_aria2_background_download \
        "$MODEL_WEIGHTS" \
        "$model_url" \
        "$LOCAL_MODEL_DIR" \
        "$MODEL_DOWNLOAD_LOG" \
        "$MODEL_PROGRESS_STATE_FILE" \
        MODEL_DOWNLOAD_PID
}

finish_model_weights_download() {
    [[ -n "$MODEL_DOWNLOAD_PID" ]] || return 0

    if wait "$MODEL_DOWNLOAD_PID"; then
        MODEL_DOWNLOAD_PID=""
        if [[ -e "$LOCAL_MODEL_DIR/${MODEL_WEIGHTS}.aria2" ]]; then
            write_progress_state "$MODEL_PROGRESS_STATE_FILE" failed
            die "$MODEL_WEIGHTS still has an aria2 control file and is incomplete."
        fi
        write_progress_state "$MODEL_PROGRESS_STATE_FILE" complete
        rm -f "$MODEL_DOWNLOAD_LOG"
    else
        local status=$?
        MODEL_DOWNLOAD_PID=""
        write_progress_state "$MODEL_PROGRESS_STATE_FILE" failed
        tail -n 80 "$MODEL_DOWNLOAD_LOG" >&2 || true
        die "$MODEL_WEIGHTS download failed with status $status; full output is in $MODEL_DOWNLOAD_LOG."
    fi
}

finish_model_download() {
    local downloaded_size

    if [[ "$(read_progress_state "$MODEL_PROGRESS_STATE_FILE")" == failed ]]; then
        finish_model_weights_download
    fi

    rm -f "$MODEL_METADATA_LOG"
    write_progress_state "$MODEL_METADATA_PROGRESS_STATE_FILE" running
    if uv run hf download "$MODEL" \
        --revision "$MODEL_REVISION" \
        --local-dir "$LOCAL_MODEL_DIR" \
        --exclude "$MODEL_WEIGHTS" \
        --max-workers "$HTTP_CONNECTIONS" >"$MODEL_METADATA_LOG" 2>&1; then
        write_progress_state "$MODEL_METADATA_PROGRESS_STATE_FILE" verifying
        rm -f "$MODEL_METADATA_LOG"
    else
        local status=$?
        write_progress_state "$MODEL_METADATA_PROGRESS_STATE_FILE" failed
        tail -n 80 "$MODEL_METADATA_LOG" >&2 || true
        die "Model metadata download failed with status $status; full output is in $MODEL_METADATA_LOG."
    fi

    finish_model_weights_download

    if [[ ! -f "$LOCAL_MODEL_DIR/$MODEL_WEIGHTS" ]]; then
        write_progress_state "$MODEL_PROGRESS_STATE_FILE" failed
        die "Downloaded model weights were not found at $LOCAL_MODEL_DIR/$MODEL_WEIGHTS."
    fi
    if [[ -e "$LOCAL_MODEL_DIR/${MODEL_WEIGHTS}.aria2" ]]; then
        write_progress_state "$MODEL_PROGRESS_STATE_FILE" failed
        die "Downloaded $MODEL_WEIGHTS is incomplete because its aria2 control file still exists."
    fi
    downloaded_size="$(stat -c '%s' "$LOCAL_MODEL_DIR/$MODEL_WEIGHTS")"
    if [[ "$downloaded_size" != "$MODEL_WEIGHTS_SIZE" ]]; then
        write_progress_state "$MODEL_PROGRESS_STATE_FILE" failed
        die "Downloaded $MODEL_WEIGHTS is $downloaded_size bytes; expected $MODEL_WEIGHTS_SIZE bytes."
    fi
    write_progress_state "$MODEL_PROGRESS_STATE_FILE" complete
    write_progress_state "$MODEL_METADATA_PROGRESS_STATE_FILE" complete
}


# Main setup flow.

initialize_sudo_session

# Step 1: Ensure uv is available for dependency and service commands.
step "Install or verify uv"
install_uv
export PATH="$HOME/.local/bin:$PATH"

# Step 2: Download the model, Python environment, CUDA, and cuDNN in parallel.
step "Download prerequisites"
cd "$ENGINE_DIR"
ensure_download_tools
initialize_download_progress_state
start_model_download
start_uv_sync
start_cuda_downloads
printf '\n'
start_background_download_progress
finish_uv_sync
finish_model_download
finish_cuda_downloads
finish_background_download_progress

# Step 3: Install the downloaded system prerequisites and optional tooling.
step "Install prerequisites"
log "All background downloads are complete. Starting installation with cached sudo authorization."
install_cuda_and_cudnn
export PATH="/usr/local/cuda-$CUDA_VERSION/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-$CUDA_VERSION/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -n "${NGROK_AUTHTOKEN:-}" ]]; then
    install_ngrok
else
    log "ngrok is disabled because --ngrok-token was not provided."
fi

install_ffmpeg
stop_sudo_keepalive

export VLLM_BASE_URL="http://127.0.0.1:$VLLM_PORT/v1"
export VLLM_MODEL="$MODEL"
export APP_PORT
export PYTHONUNBUFFERED=1

# Step 4: Start vLLM and wait until its health endpoint is ready.
step "Starting vLLM with $MODEL on port $VLLM_PORT"
setsid stdbuf -oL -eL uv run vllm serve "$LOCAL_MODEL_DIR" \
    --served-model-name "$MODEL" \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --max-model-len 8192 \
    --limit-mm-per-prompt.audio 1 \
    --mm-processor-kwargs.audio_kwargs.max_length 480000 \
    --gpu-memory-utilization 0.9 \
    --uvicorn-log-level trace &
VLLM_PID=$!

while ! curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || {
        wait "$VLLM_PID" || true
        die "vLLM exited before becoming healthy."
    }
    sleep 5
done

# Step 5: Start the API application and wait until its schema is available.
step "Start app.py on 0.0.0.0:$APP_PORT with trace logging"
setsid stdbuf -oL -eL uv run uvicorn app:app \
    --host 0.0.0.0 \
    --port "$APP_PORT" \
    --log-level trace &
APP_PID=$!

while ! curl -fsS "http://127.0.0.1:$APP_PORT/openapi.json" >/dev/null 2>&1; do
    kill -0 "$APP_PID" 2>/dev/null || {
        wait "$APP_PID" || true
        die "app.py exited before port $APP_PORT became ready."
    }
    sleep 1
done

setup_elapsed_seconds=$((SECONDS - SETUP_START_SECONDS))
printf -v setup_elapsed_time '%02d:%02d' \
    "$((setup_elapsed_seconds / 60))" \
    "$((setup_elapsed_seconds % 60))"
log "Time from setup start until app.py became accessible: $setup_elapsed_time."
log "API is listening on port $APP_PORT. Setup is complete. Press Ctrl+C to stop app.py and vLLM."

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
