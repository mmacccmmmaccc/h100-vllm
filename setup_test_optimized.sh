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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$SCRIPT_DIR/vllm_engine"
LOCAL_MODEL_DIR="$ENGINE_DIR/models/${MODEL//\//--}"
MODEL_DOWNLOAD_LOG="$ENGINE_DIR/.model-download.log"
UV_SYNC_LOG="$ENGINE_DIR/.uv-sync.log"
CUDA_DOWNLOAD_LOG="$ENGINE_DIR/.cuda-download.log"
VLLM_PID=""
APP_PID=""
MODEL_DOWNLOAD_PID=""
UV_SYNC_PID=""
CUDA_DOWNLOAD_PID=""
CUDA_DOWNLOADS_READY=0
UV_SYNC_START_SECONDS=0
INSTALL_TEMP_DIR=""
CLEANED_UP=0
APT_UPDATED=0
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

start_logged_process() {
    local log_file=$1
    shift

    : >"$log_file"
    setsid "$@" >"$log_file" 2>&1 &
    LOGGED_PROCESS_PID=$!
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

    if [[ -n "$APP_PID" || -n "$VLLM_PID" || -n "$MODEL_DOWNLOAD_PID" || -n "$UV_SYNC_PID" || -n "$CUDA_DOWNLOAD_PID" ]]; then
        log "Stopping setup child processes..."
    fi

    stop_process_group "$APP_PID"
    stop_process_group "$VLLM_PID"
    stop_process_group "$MODEL_DOWNLOAD_PID"
    stop_process_group "$UV_SYNC_PID"
    stop_process_group "$CUDA_DOWNLOAD_PID"

    local deadline=$((SECONDS + 15))
    while (( SECONDS < deadline )); do
        local app_alive=0
        local vllm_alive=0
        local model_download_alive=0
        local uv_sync_alive=0
        local cuda_download_alive=0
        process_group_is_alive "$APP_PID" && app_alive=1
        process_group_is_alive "$VLLM_PID" && vllm_alive=1
        process_group_is_alive "$MODEL_DOWNLOAD_PID" && model_download_alive=1
        process_group_is_alive "$UV_SYNC_PID" && uv_sync_alive=1
        process_group_is_alive "$CUDA_DOWNLOAD_PID" && cuda_download_alive=1
        (( app_alive == 0 && vllm_alive == 0 && model_download_alive == 0 && uv_sync_alive == 0 && cuda_download_alive == 0 )) && break
        sleep 1
    done

    for pid in "$APP_PID" "$VLLM_PID" "$MODEL_DOWNLOAD_PID" "$UV_SYNC_PID" "$CUDA_DOWNLOAD_PID"; do
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
    STEP_TOTAL=6
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
[[ -f "$ENGINE_DIR/pyproject.toml" ]] || die "Missing $ENGINE_DIR/pyproject.toml."
[[ -f "$ENGINE_DIR/uv.lock" ]] || die "Missing $ENGINE_DIR/uv.lock."

case "${VERSION_ID:-}" in
    22.04) CUDA_REPO_DISTRO="ubuntu2204" ;;
    24.04) CUDA_REPO_DISTRO="ubuntu2404" ;;
    *) die "CUDA 12.9 automated installation supports Ubuntu 22.04 or 24.04; found ${VERSION_ID:-unknown}." ;;
esac

[[ "$(uname -m)" == "x86_64" ]] || die "This CUDA installer currently supports x86_64 only."

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

start_cuda_downloads() {
    local nvcc_version=""
    local cudnn_installed=0
    local repo_name="cuda-repo-${CUDA_REPO_DISTRO}-12-9-local"
    local repo_deb="${repo_name}_${CUDA_LOCAL_REPO_VERSION}_amd64.deb"
    local cudnn_repo_name="cudnn-local-repo-${CUDA_REPO_DISTRO}-${CUDNN_VERSION}"
    local cudnn_repo_deb="${cudnn_repo_name}_1.0-1_amd64.deb"
    local installer_cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/h100-vllm/installers"
    local cuda_url=""
    local cudnn_url=""

    if command -v nvcc >/dev/null 2>&1; then
        nvcc_version="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)"
    elif [[ -x "/usr/local/cuda-$CUDA_VERSION/bin/nvcc" ]]; then
        nvcc_version="$CUDA_VERSION"
    fi
    if dpkg-query -W -f='${Status}' cudnn9-cuda-12 2>/dev/null | grep -q 'ok installed'; then
        cudnn_installed=1
    fi
    if [[ "$nvcc_version" == "$CUDA_VERSION" && "$cudnn_installed" == 1 ]]; then
        : >"$CUDA_DOWNLOAD_LOG"
        CUDA_DOWNLOADS_READY=1
        return
    fi

    mkdir -p "$installer_cache_dir"
    if ! dpkg-query -W -f='${Status}' "$repo_name" 2>/dev/null | grep -q 'ok installed' \
        || [[ ! -d "/var/$repo_name" ]]; then
        cuda_url="https://developer.download.nvidia.com/compute/cuda/${CUDA_RELEASE}/local_installers/${repo_deb}"
    fi
    if ! dpkg-query -W -f='${Status}' "$cudnn_repo_name" 2>/dev/null | grep -q 'ok installed' \
        || [[ ! -d "/var/$cudnn_repo_name" ]]; then
        cudnn_url="https://developer.download.nvidia.com/compute/cudnn/${CUDNN_VERSION}/local_installers/${cudnn_repo_deb}"
    fi
    if [[ -z "$cuda_url" && -z "$cudnn_url" ]]; then
        : >"$CUDA_DOWNLOAD_LOG"
        CUDA_DOWNLOADS_READY=1
        return
    fi

    : >"$CUDA_DOWNLOAD_LOG"
    setsid bash -Eeuo pipefail -c '
        log_file=$1
        connections=$2
        cache_dir=$3
        cuda_file=$4
        cuda_url=$5
        cudnn_file=$6
        cudnn_url=$7
        exec >"$log_file" 2>&1

        download() {
            local output_file=$1
            local url=$2
            aria2c -x "$connections" -s "$connections" -k 1M -c \
                --file-allocation=falloc --disk-cache=64M --max-tries=10 \
                --retry-wait=3 --connect-timeout=30 --timeout=60 \
                --console-log-level=warn --show-console-readout=true \
                --summary-interval=1 --auto-file-renaming=false \
                --allow-overwrite=true --dir="$cache_dir" \
                --out="$output_file" "$url"
        }

        phase_total=0
        [[ -n "$cuda_url" ]] && ((phase_total += 1))
        [[ -n "$cudnn_url" ]] && ((phase_total += 1))
        printf "TOTAL_PHASES:%d\n" "$phase_total"
        if [[ -n "$cuda_url" ]]; then
            printf "PHASE:1:CUDA Toolkit\n"
            download "$cuda_file" "$cuda_url"
        fi
        if [[ -n "$cudnn_url" ]]; then
            printf "PHASE:%d:cuDNN\n" "$phase_total"
            download "$cudnn_file" "$cudnn_url"
        fi
    ' bash "$CUDA_DOWNLOAD_LOG" "$HTTP_CONNECTIONS" "$installer_cache_dir" \
        "$repo_deb" "$cuda_url" "$cudnn_repo_deb" "$cudnn_url" &
    CUDA_DOWNLOAD_PID=$!
}

finish_cuda_downloads() {
    [[ -n "$CUDA_DOWNLOAD_PID" ]] || return

    if wait "$CUDA_DOWNLOAD_PID"; then
        CUDA_DOWNLOAD_PID=""
        CUDA_DOWNLOADS_READY=1
        rm -f "$CUDA_DOWNLOAD_LOG"
    else
        local status=$?
        CUDA_DOWNLOAD_PID=""
        tail -n 80 "$CUDA_DOWNLOAD_LOG" >&2 || true
        die "CUDA/cuDNN download failed with status $status; full output is in $CUDA_DOWNLOAD_LOG."
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
        if (( CUDA_DOWNLOADS_READY == 0 )) || [[ ! -f "$installer_cache_dir/$repo_deb" ]]; then
            aria2c \
                -x "$HTTP_CONNECTIONS" -s "$HTTP_CONNECTIONS" -k 1M -c \
                --file-allocation=falloc --disk-cache=64M --max-tries=10 \
                --retry-wait=3 --connect-timeout=30 --timeout=60 \
                --console-log-level=warn --show-console-readout=false \
                --auto-file-renaming=false --allow-overwrite=true \
                --dir="$installer_cache_dir" --out="$repo_deb" \
                "https://developer.download.nvidia.com/compute/cuda/${CUDA_RELEASE}/local_installers/${repo_deb}" \
                >>"$CUDA_DOWNLOAD_LOG" 2>&1
        fi
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
        if (( CUDA_DOWNLOADS_READY == 0 )) || [[ ! -f "$installer_cache_dir/$cudnn_repo_deb" ]]; then
            aria2c \
                -x "$HTTP_CONNECTIONS" -s "$HTTP_CONNECTIONS" -k 1M -c \
                --file-allocation=falloc --disk-cache=64M --max-tries=10 \
                --retry-wait=3 --connect-timeout=30 --timeout=60 \
                --console-log-level=warn --show-console-readout=false \
                --auto-file-renaming=false --allow-overwrite=true \
                --dir="$installer_cache_dir" --out="$cudnn_repo_deb" \
                "https://developer.download.nvidia.com/compute/cudnn/${CUDNN_VERSION}/local_installers/${cudnn_repo_deb}" \
                >>"$CUDA_DOWNLOAD_LOG" 2>&1
        fi
        as_root dpkg -i "$installer_cache_dir/$cudnn_repo_deb"
        rm -f "$installer_cache_dir/$cudnn_repo_deb" "$installer_cache_dir/${cudnn_repo_deb}.aria2"
    fi

    keyring="$(find "/var/$cudnn_repo_name" -maxdepth 1 -type f -name 'cudnn-*-keyring.gpg' -print -quit)"
    [[ -n "$keyring" ]] || die "cuDNN local repository keyring was not found in /var/$cudnn_repo_name."
    as_root cp "$keyring" /usr/share/keyrings/

    APT_UPDATED=0
    apt_update
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

start_uv_sync() {
    : >"$UV_SYNC_LOG"
    UV_SYNC_START_SECONDS=$SECONDS
    setsid bash -c '
        if command -v script >/dev/null 2>&1; then
            exec script -qefc "uv sync --frozen" /dev/null
        fi
        exec uv --verbose sync --frozen
    ' >"$UV_SYNC_LOG" 2>&1 &
    UV_SYNC_PID=$!
}

finish_uv_sync() {
    [[ -n "$UV_SYNC_PID" ]] || return

    if wait "$UV_SYNC_PID"; then
        UV_SYNC_PID=""
        rm -f "$UV_SYNC_LOG"
    else
        local status=$?
        UV_SYNC_PID=""
        tail -n 80 "$UV_SYNC_LOG" >&2 || true
        die "uv sync failed with status $status; full output is in $UV_SYNC_LOG."
    fi
}

latest_progress_text() {
    local log_file=$1
    [[ -s "$log_file" ]] || return 0
    tail -c 131072 "$log_file" 2>/dev/null \
        | tr '\r' '\n' \
        | sed $'s/\033\[[0-9;?]*[ -\/]*[@-~]//g'
}

aria_progress() {
    local pid=$1
    local log_file=$2
    local text=""
    local percent="0%"
    local eta="calculating"

    if [[ -z "$pid" ]]; then
        printf '100%%|done'
        return
    fi

    text="$(latest_progress_text "$log_file" | grep -E '\([0-9]+%\)' | tail -n 1 || true)"
    if [[ "$text" =~ \(([0-9]+%)\) ]]; then
        percent=${BASH_REMATCH[1]}
    fi
    if [[ "$text" =~ ETA:([^][[:space:]]+) ]]; then
        eta=${BASH_REMATCH[1]}
    fi
    if ! process_group_is_alive "$pid"; then
        percent="100%"
        eta="done"
    fi
    printf '%s|%s' "$percent" "$eta"
}

uv_progress() {
    local text=""
    local completed=0
    local total=0
    local percent=0
    local eta="calculating"
    local elapsed
    local remaining

    if [[ -z "$UV_SYNC_PID" ]]; then
        printf '100%%|done'
        return
    fi
    if ! process_group_is_alive "$UV_SYNC_PID"; then
        printf '100%%|done'
        return
    fi

    text="$(latest_progress_text "$UV_SYNC_LOG" \
        | grep -Eo 'Preparing packages[^()]*\([0-9]+/[0-9]+\)' \
        | tail -n 1 || true)"
    if [[ "$text" =~ \(([0-9]+)/([0-9]+)\) ]]; then
        completed=${BASH_REMATCH[1]}
        total=${BASH_REMATCH[2]}
    fi
    if (( total > 0 )); then
        percent=$((completed * 100 / total))
    fi
    if (( completed > 0 && total > completed )); then
        elapsed=$((SECONDS - UV_SYNC_START_SECONDS))
        remaining=$((elapsed * (total - completed) / completed))
        eta="$(format_duration "$remaining")"
    fi
    printf '%d%%|%s' "$percent" "$eta"
}

cuda_progress() {
    local value percent eta
    local percent_number=0
    local phase_number=1
    local phase_total=1
    local text=""

    value="$(aria_progress "$CUDA_DOWNLOAD_PID" "$CUDA_DOWNLOAD_LOG")"
    IFS='|' read -r percent eta <<<"$value"
    percent_number=${percent%%%}
    [[ "$percent_number" =~ ^[0-9]+$ ]] || percent_number=0

    text="$(latest_progress_text "$CUDA_DOWNLOAD_LOG")"
    phase_total="$(sed -n 's/^TOTAL_PHASES:\([0-9][0-9]*\)$/\1/p' <<<"$text" | tail -n 1)"
    phase_number="$(sed -n 's/^PHASE:\([0-9][0-9]*\):.*$/\1/p' <<<"$text" | tail -n 1)"
    [[ "$phase_total" =~ ^[1-9][0-9]*$ ]] || phase_total=1
    [[ "$phase_number" =~ ^[1-9][0-9]*$ ]] || phase_number=1
    percent_number=$((((phase_number - 1) * 100 + percent_number) / phase_total))
    (( percent_number > 100 )) && percent_number=100
    printf '%d%%|%s' "$percent_number" "$eta"
}

eta_seconds() {
    local value=${1// /}
    local hours=0 minutes=0 seconds=0
    [[ "$value" != "done" && "$value" != "calculating" ]] || return 1
    if [[ "$value" =~ ^(([0-9]+)h)?(([0-9]+)m)?(([0-9]+)s)?$ ]]; then
        hours=${BASH_REMATCH[2]:-0}
        minutes=${BASH_REMATCH[4]:-0}
        seconds=${BASH_REMATCH[6]:-0}
        printf '%d' "$((hours * 3600 + minutes * 60 + seconds))"
        return
    fi
    return 1
}

overall_eta() {
    local eta
    local seconds
    local maximum=0
    local found=0

    for eta in "$@"; do
        if seconds="$(eta_seconds "$eta")"; then
            (( seconds > maximum )) && maximum=$seconds
            found=1
        fi
    done
    if (( found == 1 )); then
        format_duration "$maximum"
    else
        printf 'calculating'
    fi
}

progress_bar() {
    local percent=$1
    local width=$2
    local filled=$((percent * width / 100))
    local empty=$((width - filled))
    local completed=""
    local remaining=""

    printf -v completed '%*s' "$filled" ''
    printf -v remaining '%*s' "$empty" ''
    completed=${completed// /#}
    remaining=${remaining// /-}
    printf '[%s%s]' "$completed" "$remaining"
}

progress_row() {
    local label=$1
    local percent=$2
    local eta=$3
    local label_width=$4
    local bar_width=$5
    local bar

    bar="$(progress_bar "$percent" "$bar_width")"
    printf '%-*.*s %s %3d%%  ETA %s' \
        "$label_width" "$label_width" "$label" "$bar" "$percent" "$eta"
}

show_download_dashboard() {
    local uv_value model_value cuda_value
    local uv_percent uv_eta model_percent model_eta cuda_percent cuda_eta
    local uv_number model_number cuda_number overall_number
    local overall_eta_value="calculating"
    local interactive=0
    local last_plain_update=0
    local dashboard_width=100
    local label_width=38
    local bar_width=28

    [[ -t 1 && "${TERM:-dumb}" != "dumb" ]] && interactive=1
    if (( interactive == 1 )); then
        dashboard_width="$(tput cols 2>/dev/null || printf '100')"
        (( dashboard_width < 72 )) && dashboard_width=72
        # Some remote terminals report the host PTY width instead of the
        # visible client width. Stay narrow enough to avoid hidden wrapping.
        (( dashboard_width > 78 )) && dashboard_width=78
        label_width=$((dashboard_width / 3))
        # Leave enough room for brackets, percentage, the longest ETA text,
        # and a safety margin so a wrapped row cannot break cursor redraws.
        bar_width=$((dashboard_width - label_width - 32))
        (( bar_width < 12 )) && bar_width=12
        # Reserve four physical rows, return to the first one, and remember
        # that exact position. Every refresh restores this cursor position.
        printf '\n\n\n\033[3A\033[s'
    fi
    while process_group_is_alive "$UV_SYNC_PID" \
        || process_group_is_alive "$MODEL_DOWNLOAD_PID" \
        || process_group_is_alive "$CUDA_DOWNLOAD_PID"; do
        uv_value="$(uv_progress)"
        model_value="$(aria_progress "$MODEL_DOWNLOAD_PID" "$MODEL_DOWNLOAD_LOG")"
        cuda_value="$(cuda_progress)"
        IFS='|' read -r uv_percent uv_eta <<<"$uv_value"
        IFS='|' read -r model_percent model_eta <<<"$model_value"
        IFS='|' read -r cuda_percent cuda_eta <<<"$cuda_value"
        uv_number=${uv_percent%%%}
        model_number=${model_percent%%%}
        cuda_number=${cuda_percent%%%}
        overall_number=$(((uv_number + model_number + cuda_number) / 3))
        overall_eta_value="$(overall_eta "$uv_eta" "$model_eta" "$cuda_eta")"

        if (( interactive == 1 )); then
            printf '\033[u'
            printf '\r\033[2K\033[1;96m%s\033[0m\n' \
                "$(progress_row 'OVERALL' "$overall_number" "$overall_eta_value" "$label_width" "$bar_width")"
            printf '\r\033[2K%s\n' \
                "$(progress_row 'vLLM' "$uv_number" "$uv_eta" "$label_width" "$bar_width")"
            printf '\r\033[2K%s\n' \
                "$(progress_row "$MODEL" "$model_number" "$model_eta" "$label_width" "$bar_width")"
            printf '\r\033[2K%s' \
                "$(progress_row 'CUDA Toolkit' "$cuda_number" "$cuda_eta" "$label_width" "$bar_width")"
        elif (( SECONDS - last_plain_update >= 10 || last_plain_update == 0 )); then
            printf '%s\n%s\n%s\n%s\n\n' \
                "$(progress_row 'OVERALL' "$overall_number" "$overall_eta_value" "$label_width" "$bar_width")" \
                "$(progress_row 'vLLM' "$uv_number" "$uv_eta" "$label_width" "$bar_width")" \
                "$(progress_row "$MODEL" "$model_number" "$model_eta" "$label_width" "$bar_width")" \
                "$(progress_row 'CUDA Toolkit' "$cuda_number" "$cuda_eta" "$label_width" "$bar_width")"
            last_plain_update=$SECONDS
        fi
        sleep 1
    done

    if (( interactive == 1 )); then
        printf '\033[u'
        printf '\r\033[2K\033[1;92m%s\033[0m\n' \
            "$(progress_row 'OVERALL' 100 done "$label_width" "$bar_width")"
        printf '\r\033[2K%s\n' "$(progress_row 'vLLM' 100 done "$label_width" "$bar_width")"
        printf '\r\033[2K%s\n' "$(progress_row "$MODEL" 100 done "$label_width" "$bar_width")"
        printf '\r\033[2K%s\n' "$(progress_row 'CUDA Toolkit' 100 done "$label_width" "$bar_width")"
    else
        printf '%s\n%s\n%s\n%s\n' \
            "$(progress_row 'OVERALL' 100 done "$label_width" "$bar_width")" \
            "$(progress_row 'vLLM' 100 done "$label_width" "$bar_width")" \
            "$(progress_row "$MODEL" 100 done "$label_width" "$bar_width")" \
            "$(progress_row 'CUDA Toolkit' 100 done "$label_width" "$bar_width")"
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
    : >"$MODEL_DOWNLOAD_LOG"

    if [[ -f "$LOCAL_MODEL_DIR/$MODEL_WEIGHTS" ]] \
        && [[ "$(stat -c '%s' "$LOCAL_MODEL_DIR/$MODEL_WEIGHTS")" == "$MODEL_WEIGHTS_SIZE" ]]; then
        return
    fi

    start_logged_process "$MODEL_DOWNLOAD_LOG" \
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
        --dir="$LOCAL_MODEL_DIR" \
        --out="$MODEL_WEIGHTS" \
        "$model_url"
    MODEL_DOWNLOAD_PID=$LOGGED_PROCESS_PID
}

finish_model_download() {
    local downloaded_size

    if [[ -n "$MODEL_DOWNLOAD_PID" ]]; then
        if wait "$MODEL_DOWNLOAD_PID"; then
            MODEL_DOWNLOAD_PID=""
        else
            local status=$?
            MODEL_DOWNLOAD_PID=""
            tail -n 80 "$MODEL_DOWNLOAD_LOG" >&2 || true
            die "$MODEL_WEIGHTS download failed with status $status; full output is in $MODEL_DOWNLOAD_LOG."
        fi
    fi

    downloaded_size="$(stat -c '%s' "$LOCAL_MODEL_DIR/$MODEL_WEIGHTS")"
    [[ "$downloaded_size" == "$MODEL_WEIGHTS_SIZE" ]] || \
        die "Downloaded $MODEL_WEIGHTS is $downloaded_size bytes; expected $MODEL_WEIGHTS_SIZE bytes."

    if uv run hf download "$MODEL" \
        --revision "$MODEL_REVISION" \
        --local-dir "$LOCAL_MODEL_DIR" \
        --exclude "$MODEL_WEIGHTS" \
        --max-workers "$HTTP_CONNECTIONS" \
        >>"$MODEL_DOWNLOAD_LOG" 2>&1; then
        rm -f "$MODEL_DOWNLOAD_LOG"
    else
        local status=$?
        tail -n 80 "$MODEL_DOWNLOAD_LOG" >&2 || true
        die "Model metadata download failed with status $status; full output is in $MODEL_DOWNLOAD_LOG."
    fi
}

step "Install or verify uv"
install_uv
export PATH="$HOME/.local/bin:$PATH"

cd "$ENGINE_DIR"
ensure_download_tools
start_model_download
start_uv_sync
start_cuda_downloads
show_download_dashboard
finish_uv_sync
finish_model_download
finish_cuda_downloads
if [[ -n "${NGROK_AUTHTOKEN:-}" ]]; then
    configure_ngrok_repository
fi

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
