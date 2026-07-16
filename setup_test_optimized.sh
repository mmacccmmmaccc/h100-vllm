#!/usr/bin/env bash

set -Eeuo pipefail

SETUP_START_SECONDS=$SECONDS

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
BACKGROUND_PROGRESS_DASHBOARD_ACTIVE=0
PROGRESS_STATE_DIR=""
CURRENT_STEP_STATE_FILE=""
BACKGROUND_PROGRESS_DISPLAY_STATE_FILE=""
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
INTERACTIVE_SUDO_ALLOWED=0
STEP_CURRENT=0
STEP_TOTAL=6
CURRENT_STEP_LABEL=""

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

format_progress_elapsed() {
    local state_file=$1
    local now=$2
    local started_at=""
    local finished_at=""
    local ended_at=$now

    started_at="$(read_progress_started_at "$state_file")"
    if ! [[ "$started_at" =~ ^[0-9]+$ ]] || (( started_at <= 0 )); then
        return
    fi
    finished_at="$(read_progress_finished_at "$state_file")"
    if [[ "$finished_at" =~ ^[0-9]+$ ]] && (( finished_at >= started_at )); then
        ended_at=$finished_at
    fi
    if (( ended_at < started_at )); then
        printf '0m 00s'
        return
    fi
    format_duration "$((ended_at - started_at))"
}

format_combined_progress_elapsed() {
    local now=$1
    local first_state_file=$2
    local second_state_file=$3
    local first_started=""
    local first_ended=""
    local second_started=""
    local second_ended=""
    local first_duration=0
    local second_duration=0
    local overlap_started
    local overlap_ended
    local total_duration

    first_started="$(read_progress_started_at "$first_state_file")"
    second_started="$(read_progress_started_at "$second_state_file")"
    if [[ "$first_started" =~ ^[0-9]+$ ]]; then
        first_ended="$(read_progress_finished_at "$first_state_file")"
        if ! [[ "$first_ended" =~ ^[0-9]+$ ]] || (( first_ended < first_started )); then
            first_ended=$now
        fi
        (( first_ended >= first_started )) && first_duration=$((first_ended - first_started))
    fi
    if [[ "$second_started" =~ ^[0-9]+$ ]]; then
        second_ended="$(read_progress_finished_at "$second_state_file")"
        if ! [[ "$second_ended" =~ ^[0-9]+$ ]] || (( second_ended < second_started )); then
            second_ended=$now
        fi
        (( second_ended >= second_started )) && second_duration=$((second_ended - second_started))
    fi
    if [[ ! "$first_started" =~ ^[0-9]+$ && ! "$second_started" =~ ^[0-9]+$ ]]; then
        return
    fi

    total_duration=$((first_duration + second_duration))
    if [[ "$first_started" =~ ^[0-9]+$ && "$second_started" =~ ^[0-9]+$ ]]; then
        if (( first_started > second_started )); then
            overlap_started=$first_started
        else
            overlap_started=$second_started
        fi
        if (( first_ended < second_ended )); then
            overlap_ended=$first_ended
        else
            overlap_ended=$second_ended
        fi
        if (( overlap_ended > overlap_started )); then
            total_duration=$((total_duration - overlap_ended + overlap_started))
        fi
    fi
    format_duration "$total_duration"
}

step() {
    local width="${COLUMNS:-}"
    local line
    local label
    local color=""
    local reset=""

    (( STEP_CURRENT += 1 ))
    label="STEP $STEP_CURRENT/$STEP_TOTAL | $*"
    CURRENT_STEP_LABEL=$label
    if [[ -n "${CURRENT_STEP_STATE_FILE:-}" ]]; then
        write_progress_state "$CURRENT_STEP_STATE_FILE" "$CURRENT_STEP_LABEL"
    fi
    if (( ${BACKGROUND_PROGRESS_DASHBOARD_ACTIVE:-0} == 1 )) \
        && [[ -n "${BACKGROUND_PROGRESS_DISPLAY_STATE_FILE:-}" ]] \
        && [[ "$(read_progress_state "$BACKGROUND_PROGRESS_DISPLAY_STATE_FILE")" == active ]]; then
        return 0
    fi

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

die() {
    printf '[setup] ERROR: %s\n' "$*" >&2
    exit 1
}

as_root() {
    if (( EUID == 0 )); then
        "$@"
    else
        command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages."
        if (( INTERACTIVE_SUDO_ALLOWED == 0 )); then
            sudo -n true 2>/dev/null || die \
                "A privileged download prerequisite is missing. To keep loading password-free, install aria2, ca-certificates, curl, and wget before running this script."
            sudo -n "$@"
        else
            sudo "$@"
        fi
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
    BACKGROUND_PROGRESS_DASHBOARD_ACTIVE=0

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

    log "Received $signal; stopping setup and child processes..."
    exit "$status"
}

trap cleanup EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP

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
    STEP_TOTAL=7
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
    CURRENT_STEP_STATE_FILE="$PROGRESS_STATE_DIR/current-step"
    BACKGROUND_PROGRESS_DISPLAY_STATE_FILE="$PROGRESS_STATE_DIR/display"
    MODEL_PROGRESS_STATE_FILE="$PROGRESS_STATE_DIR/model"
    MODEL_METADATA_PROGRESS_STATE_FILE="$PROGRESS_STATE_DIR/model-metadata"
    UV_PROGRESS_STATE_FILE="$PROGRESS_STATE_DIR/vllm"
    CUDA_PROGRESS_STATE_FILE="$PROGRESS_STATE_DIR/cuda"
    CUDNN_PROGRESS_STATE_FILE="$PROGRESS_STATE_DIR/cudnn"
    BACKGROUND_PROGRESS_STOP_FILE="$PROGRESS_STATE_DIR/stop"
    if [[ -n "$CURRENT_STEP_LABEL" ]]; then
        write_progress_state "$CURRENT_STEP_STATE_FILE" "$CURRENT_STEP_LABEL"
    fi
}

write_progress_state() {
    local state_file=$1
    local state=$2
    local temporary_file
    local previous_state=""
    local started_file="${state_file}.started"
    local started_temporary_file
    local finished_file="${state_file}.finished"
    local finished_temporary_file

    [[ -n "$state_file" && -d "$PROGRESS_STATE_DIR" ]] || return 0
    if [[ -f "$state_file" ]]; then
        IFS= read -r previous_state < "$state_file" || true
    fi
    temporary_file="${state_file}.tmp.$$"
    printf '%s\n' "$state" > "$temporary_file"
    case "$state" in
        running)
            rm -f -- "$finished_file"
            if [[ "$previous_state" != running || ! -f "$started_file" ]]; then
                started_temporary_file="${started_file}.tmp.$$"
                date +%s > "$started_temporary_file"
                mv -f -- "$started_temporary_file" "$started_file"
            fi
            ;;
        complete|failed|cached|installed)
            if [[ ! -f "$finished_file" ]]; then
                finished_temporary_file="${finished_file}.tmp.$$"
                date +%s > "$finished_temporary_file"
                mv -f -- "$finished_temporary_file" "$finished_file"
            fi
            ;;
        *) ;;
    esac
    # Publish the state last so readers never observe a terminal state before
    # its timing sidecars are ready.
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

read_progress_finished_at() {
    local state_file=$1
    local finished_at=""

    if [[ -f "${state_file}.finished" ]]; then
        IFS= read -r finished_at < "${state_file}.finished" || true
    fi
    [[ "$finished_at" =~ ^[0-9]+$ ]] || finished_at=""
    printf '%s' "$finished_at"
}

read_progress_started_at() {
    local state_file=$1
    local started_at=""

    if [[ -f "${state_file}.started" ]]; then
        IFS= read -r started_at < "${state_file}.started" || true
    fi
    [[ "$started_at" =~ ^[0-9]+$ ]] || started_at=""
    printf '%s' "$started_at"
}

remove_download_progress_state() {
    [[ -n "$PROGRESS_STATE_DIR" ]] || return 0

    rm -f -- \
        "$CURRENT_STEP_STATE_FILE" \
        "$BACKGROUND_PROGRESS_DISPLAY_STATE_FILE" \
        "$MODEL_PROGRESS_STATE_FILE" \
        "$MODEL_METADATA_PROGRESS_STATE_FILE" \
        "$UV_PROGRESS_STATE_FILE" \
        "$CUDA_PROGRESS_STATE_FILE" \
        "$CUDNN_PROGRESS_STATE_FILE" \
        "$BACKGROUND_PROGRESS_STOP_FILE" \
        "$PROGRESS_STATE_DIR"/*.started \
        "$PROGRESS_STATE_DIR"/*.finished \
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
        finished_file="${state_file}.finished"
        finished_temporary_file="${finished_file}.tmp.$$"
        date +%s > "$finished_temporary_file"
        mv -f -- "$finished_temporary_file" "$finished_file"
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

    ensure_download_tools
    mkdir -p "$INSTALLER_CACHE_DIR"

    if (( cuda_installed == 1 )); then
        :
    elif ! nvidia_repo_is_ready "$CUDA_REPO_NAME"; then
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

    if (( cudnn_installed == 1 )); then
        :
    elif ! nvidia_repo_is_ready "$CUDNN_REPO_NAME"; then
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

    if (( cuda_needed == 1 )); then
        wget -qO "$download_dir/cuda-${CUDA_REPO_DISTRO}.pin" \
            "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_DISTRO}/x86_64/cuda-${CUDA_REPO_DISTRO}.pin"
        as_root install -m 644 \
            "$download_dir/cuda-${CUDA_REPO_DISTRO}.pin" \
            /etc/apt/preferences.d/cuda-repository-pin-600
    fi

    finish_cuda_downloads

    if (( cuda_needed == 1 )); then
        if nvidia_repo_is_ready "$CUDA_REPO_NAME"; then
            log "Reusing the existing CUDA local repository in /var/$CUDA_REPO_NAME."
        else
            [[ -f "$INSTALLER_CACHE_DIR/$CUDA_REPO_DEB" ]] || \
                die "The downloaded CUDA repository package was not found at $INSTALLER_CACHE_DIR/$CUDA_REPO_DEB."
            as_root dpkg -i "$INSTALLER_CACHE_DIR/$CUDA_REPO_DEB"
            rm -f "$INSTALLER_CACHE_DIR/$CUDA_REPO_DEB" "$INSTALLER_CACHE_DIR/${CUDA_REPO_DEB}.aria2"
        fi

        keyring="$(find "/var/$CUDA_REPO_NAME" -maxdepth 1 -type f -name 'cuda-*-keyring.gpg' -print -quit)"
        [[ -n "$keyring" ]] || die "CUDA local repository keyring was not found in /var/$CUDA_REPO_NAME."
        as_root cp "$keyring" /usr/share/keyrings/
        packages+=("cuda-toolkit-12-9")
        repo_packages+=("$CUDA_REPO_NAME")
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

start_uv_sync() {
    log "Starting locked Python dependency synchronization in the background..."
    # uv only emits its live download bars to a terminal. Give it a private PTY
    # while keeping every raw control sequence and process message in the log.
    start_state_tracked_background_process \
        "$UV_PROGRESS_STATE_FILE" \
        "$UV_SYNC_LOG" \
        UV_SYNC_PID \
        script -q -e -f -c 'uv sync --frozen' /dev/null
}

terminal_size() {
    local rows
    local columns

    rows="$(tput lines 2>/dev/null || true)"
    columns="$(tput cols 2>/dev/null || true)"
    if ! [[ "$rows" =~ ^[0-9]+$ ]] || (( rows < 1 )); then
        rows="${LINES:-24}"
        [[ "$rows" =~ ^[0-9]+$ ]] && (( rows >= 1 )) || rows=24
    fi
    if ! [[ "$columns" =~ ^[0-9]+$ ]] || (( columns < 1 )); then
        columns="${COLUMNS:-80}"
        [[ "$columns" =~ ^[0-9]+$ ]] && (( columns >= 1 )) || columns=80
    fi

    printf '%s %s\n' "$rows" "$columns"
}

activate_background_progress_display() {
    local rows=$1
    local columns=$2

    BACKGROUND_PROGRESS_ROWS=$rows
    BACKGROUND_PROGRESS_COLUMNS=$columns
    BACKGROUND_PROGRESS_FIRST_ROW=$((rows - 3))
    BACKGROUND_PROGRESS_CONTENT_ROWS=$((rows - 4))
    BACKGROUND_PROGRESS_DISPLAY_ACTIVE=1

    # Pin the current step on row one and keep the final four rows for progress.
    # Process output scrolls only through the middle region.
    printf '\033[?25l\033[?6l\033[2;%dr\033[%d;1H' \
        "$BACKGROUND_PROGRESS_CONTENT_ROWS" "$BACKGROUND_PROGRESS_CONTENT_ROWS"
    write_progress_state "$BACKGROUND_PROGRESS_DISPLAY_STATE_FILE" active
}

start_background_progress_display() {
    local rows
    local columns

    BACKGROUND_PROGRESS_DISPLAY_ACTIVE=0
    if [[ ! -t 1 || "${TERM:-dumb}" == "dumb" ]]; then
        write_progress_state "$BACKGROUND_PROGRESS_DISPLAY_STATE_FILE" inactive
        return 1
    fi

    read -r rows columns < <(terminal_size)
    if (( rows < 8 || columns < 60 )); then
        write_progress_state "$BACKGROUND_PROGRESS_DISPLAY_STATE_FILE" inactive
        return 1
    fi
    activate_background_progress_display "$rows" "$columns"
}

refresh_background_progress_display() {
    local rows
    local columns

    [[ -t 1 && "${TERM:-dumb}" != "dumb" ]] || return 0
    read -r rows columns < <(terminal_size)

    if (( rows < 8 || columns < 60 )); then
        write_progress_state "$BACKGROUND_PROGRESS_DISPLAY_STATE_FILE" inactive
        stop_background_progress_display 0
        return
    fi
    if (( BACKGROUND_PROGRESS_DISPLAY_ACTIVE == 0 )); then
        activate_background_progress_display "$rows" "$columns"
    elif (( rows != BACKGROUND_PROGRESS_ROWS || columns != BACKGROUND_PROGRESS_COLUMNS )); then
        # Keep the shared state active across a viable geometry-only redraw so
        # step() never emits a multiline banner into the fixed-row dashboard.
        stop_background_progress_display 0
        activate_background_progress_display "$rows" "$columns"
    fi
}

compact_dashboard_progress() {
    local progress=$1
    local duration=""
    local replacement=""

    progress=${progress//" / "/"/"}
    progress=${progress//" MB"/"MB"}
    progress=${progress//" | "/" "}
    if [[ "$progress" =~ elapsed[[:space:]]([0-9]+)h[[:space:]]([0-9]{2})m[[:space:]]([0-9]{2})s ]]; then
        duration=${BASH_REMATCH[0]}
        replacement="elapsed ${BASH_REMATCH[1]}:${BASH_REMATCH[2]}:${BASH_REMATCH[3]}"
        progress=${progress/"$duration"/"$replacement"}
    elif [[ "$progress" =~ elapsed[[:space:]]([0-9]+)m[[:space:]]([0-9]{2})s ]]; then
        duration=${BASH_REMATCH[0]}
        replacement="elapsed ${BASH_REMATCH[1]}:${BASH_REMATCH[2]}"
        progress=${progress/"$duration"/"$replacement"}
    fi
    printf '%s' "$progress"
}

fit_dashboard_progress_row() {
    local full_label=$1
    local compact_label=$2
    local progress=$3
    local max_length=$4
    local compact_progress
    local percentage=""
    local text="$full_label: $progress"
    local available_label_length

    if (( ${#text} <= max_length )); then
        printf '%s' "$text"
        return
    fi

    compact_progress="$(compact_dashboard_progress "$progress")"
    text="$compact_label: $compact_progress"
    if (( ${#text} <= max_length )); then
        printf '%s' "$text"
        return
    fi
    if [[ "$compact_progress" =~ [[:space:]]\([0-9]{1,3}%\) ]]; then
        percentage=${BASH_REMATCH[0]}
        compact_progress=${compact_progress/"$percentage"/}
    fi
    text="$compact_label: $compact_progress"
    if (( ${#text} <= max_length )); then
        printf '%s' "$text"
        return
    fi
    available_label_length=$((max_length - ${#compact_progress} - 2))
    if (( available_label_length >= 2 )); then
        compact_label="${compact_label:0:available_label_length}"
        if (( ${#compact_label} == available_label_length )); then
            compact_label="${compact_label:0:available_label_length - 1}~"
        fi
        printf '%s: %s' "$compact_label" "$compact_progress"
    else
        printf '%s' "${compact_progress:0:max_length}"
    fi
}

style_progress_marker() {
    local text=$1
    local marker
    local prefix
    local suffix

    if [[ "$text" == *'[ COMPLETE ]'* ]]; then
        marker='[ COMPLETE ]'
        prefix=${text%%"$marker"*}
        suffix=${text#*"$marker"}
        text="${prefix}"$'\033[1;92m'"${marker}"$'\033[0m'"${suffix}"
    elif [[ "$text" == *'[ FAILED ]'* ]]; then
        marker='[ FAILED ]'
        prefix=${text%%"$marker"*}
        suffix=${text#*"$marker"}
        text="${prefix}"$'\033[1;91m'"${marker}"$'\033[0m'"${suffix}"
    fi
    printf '%s' "$text"
}

render_background_progress() {
    local step_text=$1
    local model_progress=$2
    local uv_progress=$3
    local cuda_progress=$4
    local cudnn_progress=$5
    local model_text
    local uv_text
    local cuda_text
    local cudnn_text
    local max_length=$((BACKGROUND_PROGRESS_COLUMNS - 1))

    (( BACKGROUND_PROGRESS_DISPLAY_ACTIVE == 1 )) || return 0
    step_text="${step_text//$'\r'/ }"
    step_text="${step_text//$'\n'/ }"
    model_progress="${model_progress//$'\r'/ }"
    model_progress="${model_progress//$'\n'/ }"
    uv_progress="${uv_progress//$'\r'/ }"
    uv_progress="${uv_progress//$'\n'/ }"
    cuda_progress="${cuda_progress//$'\r'/ }"
    cuda_progress="${cuda_progress//$'\n'/ }"
    cudnn_progress="${cudnn_progress//$'\r'/ }"
    cudnn_progress="${cudnn_progress//$'\n'/ }"
    step_text="${step_text:0:max_length}"
    model_text="$(fit_dashboard_progress_row \
        "Model ($MODEL)" "Model (${MODEL##*/})" "$model_progress" "$max_length")"
    uv_text="$(fit_dashboard_progress_row \
        'vLLM' 'vLLM' "$uv_progress" "$max_length")"
    cuda_text="$(fit_dashboard_progress_row \
        "CUDA Toolkit $CUDA_VERSION" 'CUDA Toolkit' "$cuda_progress" "$max_length")"
    cudnn_text="$(fit_dashboard_progress_row \
        "cuDNN $CUDNN_VERSION" 'cuDNN' "$cudnn_progress" "$max_length")"
    step_text=$'\033[1;36m'"${step_text}"$'\033[0m'
    model_text="$(style_progress_marker "$model_text")"
    uv_text="$(style_progress_marker "$uv_text")"
    cuda_text="$(style_progress_marker "$cuda_text")"
    cudnn_text="$(style_progress_marker "$cudnn_text")"

    printf '\0337\033[1;1H\033[2K%s\033[%d;1H\033[2K%s\033[%d;1H\033[2K%s\033[%d;1H\033[2K%s\033[%d;1H\033[2K%s\0338' \
        "$step_text" \
        "$BACKGROUND_PROGRESS_FIRST_ROW" "$model_text" \
        "$((BACKGROUND_PROGRESS_FIRST_ROW + 1))" "$uv_text" \
        "$((BACKGROUND_PROGRESS_FIRST_ROW + 2))" "$cuda_text" \
        "$BACKGROUND_PROGRESS_ROWS" "$cudnn_text"
}

stop_background_progress_display() {
    local publish_state=${1:-1}

    (( ${BACKGROUND_PROGRESS_DISPLAY_ACTIVE:-0} == 1 )) || return 0

    printf '\033[1;1H\033[2K\033[%d;1H\033[2K\033[%d;1H\033[2K\033[%d;1H\033[2K\033[%d;1H\033[2K\033[r\033[?6l\033[%d;1H\033[?25h' \
        "$BACKGROUND_PROGRESS_FIRST_ROW" \
        "$((BACKGROUND_PROGRESS_FIRST_ROW + 1))" \
        "$((BACKGROUND_PROGRESS_FIRST_ROW + 2))" \
        "$BACKGROUND_PROGRESS_ROWS" \
        "$BACKGROUND_PROGRESS_ROWS"
    BACKGROUND_PROGRESS_DISPLAY_ACTIVE=0
    if (( publish_state == 1 )); then
        write_progress_state "$BACKGROUND_PROGRESS_DISPLAY_STATE_FILE" inactive
    fi
}

format_transfer_metrics() {
    local current_value=$1
    local current_unit=$2
    local total_value=$3
    local total_unit=$4
    local percent=$5
    local rate_value=$6
    local rate_unit=$7

    LC_ALL=C awk \
        -v current_value="$current_value" \
        -v current_unit="$current_unit" \
        -v total_value="$total_value" \
        -v total_unit="$total_unit" \
        -v percent="$percent" \
        -v rate_value="$rate_value" \
        -v rate_unit="$rate_unit" '
        function unit_bytes(unit) {
            if (unit == "B")   return 1
            if (unit == "KB")  return 1000
            if (unit == "MB")  return 1000000
            if (unit == "GB")  return 1000000000
            if (unit == "TB")  return 1000000000000
            if (unit == "KiB") return 1024
            if (unit == "MiB") return 1048576
            if (unit == "GiB") return 1073741824
            if (unit == "TiB") return 1099511627776
            return 0
        }
        function display_mb(value) {
            if (value < 10) return sprintf("%.2f", value)
            return sprintf("%.1f", value)
        }
        BEGIN {
            current_factor = unit_bytes(current_unit)
            total_factor = unit_bytes(total_unit)
            rate_factor = (rate_value == "" ? 1 : unit_bytes(rate_unit))
            if (current_factor == 0 || total_factor == 0 || rate_factor == 0) exit 2
            current_mb = current_value * current_factor / 1000000
            total_mb = total_value * total_factor / 1000000
            if (percent == "" && total_mb > 0) {
                percent = int(current_mb * 100 / total_mb)
                if (current_mb < total_mb && percent >= 100) percent = 99
            }
            if (percent + 0 > 100) percent = 100
            printf "%s / %s MB", display_mb(current_mb), display_mb(total_mb)
            if (percent != "") printf " (%d%%)", percent
            if (rate_value != "") {
                rate_mb = rate_value * rate_factor / 1000000
                printf " | %s MB/s", display_mb(rate_mb)
            }
        }
    '
}

normalize_download_progress() {
    local progress_line=$1
    local current_value=""
    local current_unit=""
    local total_value=""
    local total_unit=""
    local percent=""
    local rate_value=""
    local rate_unit=""

    if [[ "$progress_line" =~ ([0-9]+([.][0-9]+)?)[[:space:]]*([KMGT]?i?B)[[:space:]]*/[[:space:]]*([0-9]+([.][0-9]+)?)[[:space:]]*([KMGT]?i?B) ]]; then
        current_value=${BASH_REMATCH[1]}
        current_unit=${BASH_REMATCH[3]}
        total_value=${BASH_REMATCH[4]}
        total_unit=${BASH_REMATCH[6]}
        if [[ "$progress_line" =~ \([[:space:]]*([0-9]{1,3})%[[:space:]]*\) ]]; then
            percent=${BASH_REMATCH[1]}
        fi
        if [[ "$progress_line" =~ DL:[[:space:]]*([0-9]+([.][0-9]+)?)[[:space:]]*([KMGT]?i?B) ]]; then
            rate_value=${BASH_REMATCH[1]}
            rate_unit=${BASH_REMATCH[3]}
        elif [[ "$progress_line" =~ ([0-9]+([.][0-9]+)?)[[:space:]]*([KMGT]?i?B)[[:space:]]*/s ]]; then
            rate_value=${BASH_REMATCH[1]}
            rate_unit=${BASH_REMATCH[3]}
        fi
        format_transfer_metrics \
            "$current_value" "$current_unit" "$total_value" "$total_unit" \
            "$percent" "$rate_value" "$rate_unit" || true
        return
    fi

    # uv announces large artifacts before its first byte-level progress update.
    if [[ "$progress_line" =~ \(([0-9]+([.][0-9]+)?)[[:space:]]*([KMGT]?i?B)\) ]]; then
        format_transfer_metrics 0 B "${BASH_REMATCH[1]}" "${BASH_REMATCH[3]}" "" "" "" || true
    fi
}

format_progress_display() {
    local progress=$1
    local elapsed=$2
    local transfer
    local rate

    case "$progress" in
        complete) progress='[ COMPLETE ]' ;;
        'ready (cached)') progress='[ COMPLETE ] cached' ;;
        'ready (installed)') progress='[ COMPLETE ] installed' ;;
        failed) progress='[ FAILED ]' ;;
        'metadata failed') progress='[ FAILED ] metadata' ;;
    esac
    if [[ -n "$elapsed" ]]; then
        if [[ "$progress" == *' | '*MB/s ]]; then
            transfer=${progress%%' | '*}
            rate=${progress#*' | '}
            progress="$transfer | elapsed $elapsed | $rate"
        else
            progress+=" | elapsed $elapsed"
        fi
    fi
    printf '%s' "$progress"
}

aria2_download_progress() {
    local log_file=$1
    local process_alive=$2
    local state=$3
    local progress_line=""
    local progress=""

    case "$state" in
        cached) printf 'ready (cached)'; return ;;
        installed) printf 'ready (installed)'; return ;;
        complete) printf 'complete'; return ;;
        failed) printf 'failed'; return ;;
        pending) printf 'queued'; return ;;
    esac
    if (( process_alive == 0 )); then
        printf 'verifying'
        return
    fi

    progress_line="$(
        tail -c 65536 -- "$log_file" 2>/dev/null \
            | tr '\r' '\n' \
            | sed -E $'s/\033\\[[0-9;?]*[ -/]*[@-~]//g' \
            | grep -E '\([[:space:]]*[0-9]{1,3}%[[:space:]]*\)' \
            | tail -n 1 \
            || true
    )"
    progress="$(normalize_download_progress "$progress_line")"
    printf '%s' "${progress:-starting}"
}

uv_download_progress() {
    local process_alive=$1
    local state=$2
    local progress_line=""
    local progress=""

    case "$state" in
        complete) printf 'complete'; return ;;
        failed) printf 'failed'; return ;;
        pending) printf 'queued'; return ;;
    esac
    if (( process_alive == 0 )); then
        printf 'verifying'
        return
    fi

    progress_line="$(
        tail -c 65536 -- "$UV_SYNC_LOG" 2>/dev/null \
            | tr '\r' '\n' \
            | sed -E $'s/\033\\[[0-9;?]*[ -/]*[@-~]//g' \
            | grep -E '[0-9]+([.][0-9]+)?[[:space:]]*[KMGT]?i?B[[:space:]]*/[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*[KMGT]?i?B' \
            | tail -n 1 \
            || true
    )"
    if [[ -z "$progress_line" ]]; then
        progress_line="$(
            tail -c 65536 -- "$UV_SYNC_LOG" 2>/dev/null \
                | tr '\r' '\n' \
                | sed -E $'s/\033\\[[0-9;?]*[ -/]*[@-~]//g' \
                | grep -E 'Downloading.*\([0-9]+([.][0-9]+)?[[:space:]]*[KMGT]?i?B\)' \
                | tail -n 1 \
                || true
        )"
    fi
    progress="$(normalize_download_progress "$progress_line")"
    printf '%s' "${progress:-starting}"
}

model_metadata_download_progress() {
    local progress_line=""

    progress_line="$(
        tail -c 65536 -- "$MODEL_METADATA_LOG" 2>/dev/null \
            | tr '\r' '\n' \
            | sed -E $'s/\033\\[[0-9;?]*[ -/]*[@-~]//g' \
            | grep -E '([0-9]+%|(^|[[:space:]])(Fetching|Downloading|Downloaded)([[:space:]]|$))' \
            | tail -n 1 \
            || true
    )"
    if [[ "$progress_line" =~ ([0-9]{1,3})% ]]; then
        printf 'metadata (%s%%)' "${BASH_REMATCH[1]}"
    else
        printf 'metadata'
    fi
}

show_background_download_progress() {
    local model_pid=$1
    local uv_pid=$2
    local cuda_pid=$3
    local cudnn_pid=$4
    local model_alive
    local uv_alive
    local cuda_alive
    local cudnn_alive
    local model_state
    local model_metadata_state
    local uv_state
    local cuda_state
    local cudnn_state
    local current_step
    local now
    local model_progress
    local uv_progress
    local cuda_progress
    local cudnn_progress
    local model_elapsed
    local uv_elapsed
    local cuda_elapsed
    local cudnn_elapsed

    BACKGROUND_PROGRESS_DISPLAY_ACTIVE=0
    BACKGROUND_PROGRESS_ROWS=0
    BACKGROUND_PROGRESS_COLUMNS=0
    BACKGROUND_PROGRESS_FIRST_ROW=0
    BACKGROUND_PROGRESS_CONTENT_ROWS=0

    start_background_progress_display || true
    trap stop_background_progress_display EXIT
    trap 'exit 0' INT TERM HUP

    while [[ ! -e "$BACKGROUND_PROGRESS_STOP_FILE" ]]; do
        now="$(date +%s)"
        model_alive=0
        uv_alive=0
        cuda_alive=0
        cudnn_alive=0
        [[ -n "$model_pid" ]] && kill -0 "$model_pid" 2>/dev/null && model_alive=1
        [[ -n "$uv_pid" ]] && kill -0 "$uv_pid" 2>/dev/null && uv_alive=1
        [[ -n "$cuda_pid" ]] && kill -0 "$cuda_pid" 2>/dev/null && cuda_alive=1
        [[ -n "$cudnn_pid" ]] && kill -0 "$cudnn_pid" 2>/dev/null && cudnn_alive=1

        model_state="$(read_progress_state "$MODEL_PROGRESS_STATE_FILE")"
        model_metadata_state="$(read_progress_state "$MODEL_METADATA_PROGRESS_STATE_FILE")"
        uv_state="$(read_progress_state "$UV_PROGRESS_STATE_FILE")"
        cuda_state="$(read_progress_state "$CUDA_PROGRESS_STATE_FILE")"
        cudnn_state="$(read_progress_state "$CUDNN_PROGRESS_STATE_FILE")"
        current_step="$(read_progress_state "$CURRENT_STEP_STATE_FILE")"

        if [[ "$model_state" == failed ]]; then
            model_progress="failed"
        elif [[ "$model_metadata_state" == failed ]]; then
            model_progress="metadata failed"
        elif [[ "$model_state" == running ]]; then
            model_progress="$(aria2_download_progress "$MODEL_DOWNLOAD_LOG" "$model_alive" "$model_state")"
        elif [[ "$model_metadata_state" == running ]]; then
            model_progress="$(model_metadata_download_progress)"
        elif [[ "$model_metadata_state" == verifying ]]; then
            model_progress="verifying"
        elif [[ "$model_metadata_state" == complete \
            && ( "$model_state" == complete || "$model_state" == cached ) ]]; then
            model_progress="complete"
        elif [[ "$model_state" == complete ]]; then
            model_progress="weights downloaded"
        elif [[ "$model_state" == cached ]]; then
            model_progress="weights cached"
        else
            model_progress="$(aria2_download_progress "$MODEL_DOWNLOAD_LOG" "$model_alive" "$model_state")"
        fi
        uv_progress="$(uv_download_progress "$uv_alive" "$uv_state")"
        cuda_progress="$(aria2_download_progress "$CUDA_DOWNLOAD_LOG" "$cuda_alive" "$cuda_state")"
        cudnn_progress="$(aria2_download_progress "$CUDNN_DOWNLOAD_LOG" "$cudnn_alive" "$cudnn_state")"

        model_elapsed="$(format_combined_progress_elapsed \
            "$now" "$MODEL_PROGRESS_STATE_FILE" "$MODEL_METADATA_PROGRESS_STATE_FILE")"
        uv_elapsed="$(format_progress_elapsed "$UV_PROGRESS_STATE_FILE" "$now")"
        cuda_elapsed="$(format_progress_elapsed "$CUDA_PROGRESS_STATE_FILE" "$now")"
        cudnn_elapsed="$(format_progress_elapsed "$CUDNN_PROGRESS_STATE_FILE" "$now")"
        model_progress="$(format_progress_display "$model_progress" "$model_elapsed")"
        uv_progress="$(format_progress_display "$uv_progress" "$uv_elapsed")"
        cuda_progress="$(format_progress_display "$cuda_progress" "$cuda_elapsed")"
        cudnn_progress="$(format_progress_display "$cudnn_progress" "$cudnn_elapsed")"

        refresh_background_progress_display
        render_background_progress \
            "$current_step" \
            "$model_progress" \
            "$uv_progress" \
            "$cuda_progress" \
            "$cudnn_progress"
        sleep "$BACKGROUND_PROGRESS_INTERVAL"
    done
}

start_background_download_progress() {
    local attempt
    local display_state

    if [[ -z "$MODEL_DOWNLOAD_PID" && -z "$UV_SYNC_PID" \
        && -z "$CUDA_DOWNLOAD_PID" && -z "$CUDNN_DOWNLOAD_PID" ]]; then
        return
    fi

    write_progress_state "$BACKGROUND_PROGRESS_DISPLAY_STATE_FILE" starting
    show_background_download_progress \
        "$MODEL_DOWNLOAD_PID" \
        "$UV_SYNC_PID" \
        "$CUDA_DOWNLOAD_PID" \
        "$CUDNN_DOWNLOAD_PID" &
    BACKGROUND_PROGRESS_PID=$!
    BACKGROUND_PROGRESS_DASHBOARD_ACTIVE=1

    # Avoid racing the next step banner against initial terminal activation.
    for attempt in {1..100}; do
        display_state="$(read_progress_state "$BACKGROUND_PROGRESS_DISPLAY_STATE_FILE")"
        [[ "$display_state" == active || "$display_state" == inactive ]] && break
        kill -0 "$BACKGROUND_PROGRESS_PID" 2>/dev/null || break
        sleep 0.02
    done
}

finish_background_download_progress() {
    if [[ -n "$BACKGROUND_PROGRESS_PID" ]]; then
        : > "$BACKGROUND_PROGRESS_STOP_FILE"
        wait "$BACKGROUND_PROGRESS_PID" 2>/dev/null || true
        BACKGROUND_PROGRESS_PID=""
    fi
    BACKGROUND_PROGRESS_DASHBOARD_ACTIVE=0
}

finish_uv_sync() {
    [[ -n "$UV_SYNC_PID" ]] || return 0

    log "Waiting for the background Python dependency synchronization..."
    if wait "$UV_SYNC_PID"; then
        UV_SYNC_PID=""
        write_progress_state "$UV_PROGRESS_STATE_FILE" complete
        rm -f "$UV_SYNC_LOG"
        log "Python dependencies are synchronized."
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

    if [[ -f "$LOCAL_MODEL_DIR/$MODEL_WEIGHTS" ]] \
        && [[ "$(stat -c '%s' "$LOCAL_MODEL_DIR/$MODEL_WEIGHTS")" == "$MODEL_WEIGHTS_SIZE" ]]; then
        log "$MODEL_WEIGHTS is already fully downloaded."
        write_progress_state "$MODEL_PROGRESS_STATE_FILE" cached
        return
    fi

    log "Starting $MODEL_WEIGHTS download in the background with $HTTP_CONNECTIONS resumable HTTP ranges..."
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

    log "Waiting for the background $MODEL_WEIGHTS download..."
    if wait "$MODEL_DOWNLOAD_PID"; then
        MODEL_DOWNLOAD_PID=""
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

    log "Downloading model metadata and tokenizer files with Hugging Face HTTP..."
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
    downloaded_size="$(stat -c '%s' "$LOCAL_MODEL_DIR/$MODEL_WEIGHTS")"
    if [[ "$downloaded_size" != "$MODEL_WEIGHTS_SIZE" ]]; then
        write_progress_state "$MODEL_PROGRESS_STATE_FILE" failed
        die "Downloaded $MODEL_WEIGHTS is $downloaded_size bytes; expected $MODEL_WEIGHTS_SIZE bytes."
    fi
    write_progress_state "$MODEL_PROGRESS_STATE_FILE" complete
    write_progress_state "$MODEL_METADATA_PROGRESS_STATE_FILE" complete
}


## Running Steps ##

step "Install or verify uv"
install_uv
export PATH="$HOME/.local/bin:$PATH"

cd "$ENGINE_DIR"
ensure_download_tools
initialize_download_progress_state
start_model_download
start_uv_sync
start_cuda_downloads
start_background_download_progress


step "Finish all background downloads before privileged installation"
finish_uv_sync
finish_model_download
finish_cuda_downloads
INTERACTIVE_SUDO_ALLOWED=1
log "All background downloads are complete. Privileged installation may now request sudo access."


step "Install or verify CUDA Toolkit $CUDA_VERSION and cuDNN $CUDNN_VERSION"
install_cuda_and_cudnn
export PATH="/usr/local/cuda-$CUDA_VERSION/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-$CUDA_VERSION/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -n "${NGROK_AUTHTOKEN:-}" ]]; then
    step "Install or verify ngrok"
    install_ngrok
else
    log "ngrok is disabled because --ngrok-token was not provided."
fi

step "Install or verify FFmpeg $FFMPEG_VERSION"
install_ffmpeg

export VLLM_BASE_URL="http://127.0.0.1:$VLLM_PORT/v1"
export VLLM_MODEL="$MODEL"
export APP_PORT
export PYTHONUNBUFFERED=1


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

SETUP_ELAPSED_SECONDS=$((SECONDS - SETUP_START_SECONDS))
log "API is listening on port $APP_PORT. Setup completed in $(format_duration "$SETUP_ELAPSED_SECONDS"). Press Ctrl+C to stop app.py and vLLM."

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
