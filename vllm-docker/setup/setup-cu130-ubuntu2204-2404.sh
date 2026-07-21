#!/usr/bin/env bash

set -Eeuo pipefail

readonly CUDA_VERSION="13.0"
readonly CUDA_RELEASE="13.0.2"
readonly CUDA_LOCAL_REPO_VERSION="13.0.2-580.95.05-1"
readonly CUDNN_VERSION="9.19.0"
readonly CUDNN_MIN_VERSION="9.19.0"
readonly CUDNN_PACKAGE_VERSION="9.19.0.56-1"
readonly HTTP_CONNECTIONS="8"
readonly DOCKER_KEYRING="/etc/apt/keyrings/docker.asc"
readonly DOCKER_SOURCE="/etc/apt/sources.list.d/docker.sources"
readonly DOCKER_REPO="https://download.docker.com/linux/ubuntu"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"
DOCKER_BUILD_CONTEXT="$SCRIPT_DIR"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-vllm-audio:latest}"
ACTIVE_CUDA_VERSION=""
ACTIVE_CUDA_HOME=""
ACTIVE_CUDNN_VERSION=""
ACTIVE_CUDNN_PACKAGE=""
INSTALL_TEMP_DIR=""
CLEANED_UP=0

log() {
    printf '\n\033[1;32m[Setup]\033[0m %s\n' "$1"
}

error() {
    printf '\n\033[1;31m[Error]\033[0m %s\n' "$1" >&2
    exit 1
}

apt_install() {
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
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

(( $# == 0 )) || error "This script does not accept arguments."

[[ -r /etc/os-release ]] || error "Cannot read /etc/os-release."
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || error "This installer is intended for Ubuntu. Detected: ${PRETTY_NAME:-unknown}"
command -v apt-get >/dev/null 2>&1 || error "apt-get was not found."
[[ "$(uname -m)" == "x86_64" ]] || error "This CUDA installer currently supports x86_64 only."
[[ -f "$DOCKERFILE" ]] || error "Dockerfile not found: $DOCKERFILE"

case "${VERSION_ID:-}" in
    22.04) CUDA_REPO_DISTRO="ubuntu2204" ;;
    24.04) CUDA_REPO_DISTRO="ubuntu2404" ;;
    *) error "CUDA 13.0 automated installation supports Ubuntu 22.04 or 24.04; found ${VERSION_ID:-unknown}." ;;
esac

UBUNTU_RELEASE="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
[[ -n "$UBUNTU_RELEASE" ]] || error "Unable to determine the Ubuntu release codename."

ARCHITECTURE="$(dpkg --print-architecture)"
TARGET_USER="${SUDO_USER:-$USER}"
[[ "$TARGET_USER" != "root" ]] \
    || error "Run this script as your normal user, not with sudo. The script will request sudo when needed."

ensure_download_tools() {
    if ! command -v curl >/dev/null 2>&1 \
        || ! command -v wget >/dev/null 2>&1 \
        || ! command -v aria2c >/dev/null 2>&1; then
        log "Installing download prerequisites"
        sudo apt-get update
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
    local cudnn_installed=0

    if detect_compatible_cuda_toolkit; then
        cuda_installed=1
    fi
    if detect_compatible_cudnn; then
        cudnn_installed=1
    fi

    if (( cuda_installed == 1 && cudnn_installed == 1 )); then
        log "Compatible CUDA Toolkit $ACTIVE_CUDA_VERSION and cuDNN $ACTIVE_CUDNN_VERSION are already installed"
        return
    fi

    ensure_download_tools
    if (( cuda_installed == 0 )); then
        log "No CUDA 13.x toolkit detected; installing CUDA Toolkit $CUDA_VERSION"
    else
        log "Using CUDA Toolkit $ACTIVE_CUDA_VERSION at $ACTIVE_CUDA_HOME"
    fi
    if (( cudnn_installed == 0 )); then
        log "No supported cuDNN >=$CUDNN_MIN_VERSION,<10 detected; installing cuDNN $CUDNN_VERSION"
    else
        log "Using cuDNN $ACTIVE_CUDNN_VERSION from $ACTIVE_CUDNN_PACKAGE"
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
        sudo install -m 644 \
            "$download_dir/cuda-${CUDA_REPO_DISTRO}.pin" \
            /etc/apt/preferences.d/cuda-repository-pin-600

        if dpkg-query -W -f='${Status}' "$repo_name" 2>/dev/null | grep -q 'ok installed' \
            && [[ -d "/var/$repo_name" ]]; then
            log "Reusing the existing CUDA local repository in /var/$repo_name"
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
            sudo dpkg -i "$installer_cache_dir/$repo_deb"
            rm -f "$installer_cache_dir/$repo_deb" "$installer_cache_dir/${repo_deb}.aria2"
        fi

        keyring="$(find "/var/$repo_name" -maxdepth 1 -type f -name 'cuda-*-keyring.gpg' -print -quit)"
        [[ -n "$keyring" ]] || error "CUDA local repository keyring was not found in /var/$repo_name."
        sudo cp "$keyring" /usr/share/keyrings/
        packages+=("cuda-toolkit-13-0")
        repo_packages+=("$repo_name")
    fi

    if (( cudnn_installed == 0 )); then
        if dpkg-query -W -f='${Status}' "$cudnn_repo_name" 2>/dev/null | grep -q 'ok installed' \
            && [[ -d "/var/$cudnn_repo_name" ]]; then
            log "Reusing the existing cuDNN local repository in /var/$cudnn_repo_name"
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
            sudo dpkg -i "$installer_cache_dir/$cudnn_repo_deb"
            rm -f "$installer_cache_dir/$cudnn_repo_deb" "$installer_cache_dir/${cudnn_repo_deb}.aria2"
        fi

        keyring="$(find "/var/$cudnn_repo_name" -maxdepth 1 -type f -name 'cudnn-*-keyring.gpg' -print -quit)"
        [[ -n "$keyring" ]] || error "cuDNN local repository keyring was not found in /var/$cudnn_repo_name."
        sudo cp "$keyring" /usr/share/keyrings/
        packages+=("cudnn9-cuda-13=$CUDNN_PACKAGE_VERSION")
        repo_packages+=("$cudnn_repo_name")
    fi

    sudo apt-get update
    apt_install "${packages[@]}"

    sudo env DEBIAN_FRONTEND=noninteractive apt-get remove -y --purge "${repo_packages[@]}"
    if (( cuda_installed == 0 )); then
        sudo rm -f /etc/apt/preferences.d/cuda-repository-pin-600
    fi
    rm -rf "$download_dir"
    INSTALL_TEMP_DIR=""

    detect_compatible_cuda_toolkit \
        || error "CUDA installation completed, but a compatible CUDA 13.x nvcc was not found."
    detect_compatible_cudnn \
        || error "cuDNN installation completed, but a supported cuDNN >=$CUDNN_MIN_VERSION,<10 was not found."
}

install_docker() {
    log "Updating the APT package index"
    sudo apt-get update

    log "Installing Docker repository prerequisites"
    apt_install ca-certificates curl

    log "Configuring the Docker APT repository"
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl \
        --fail \
        --silent \
        --show-error \
        --location \
        "$DOCKER_REPO/gpg" \
        --output "$DOCKER_KEYRING"
    sudo chmod a+r "$DOCKER_KEYRING"
    sudo tee "$DOCKER_SOURCE" >/dev/null <<EOF
Types: deb
URIs: ${DOCKER_REPO}
Suites: ${UBUNTU_RELEASE}
Components: stable
Architectures: ${ARCHITECTURE}
Signed-By: ${DOCKER_KEYRING}
EOF

    sudo apt-get update

    log "Installing Docker Engine and Docker Compose"
    apt_install \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin

    log "Enabling Docker and adding $TARGET_USER to the docker group"
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$TARGET_USER"
    sudo systemctl is-active --quiet docker || error "Docker did not start successfully."
}

build_docker_image_as_target_user() {
    log "Starting a fresh login as $TARGET_USER so Docker group membership is active"
    sudo -iu "$TARGET_USER" docker info >/dev/null \
        || error "Docker is not usable by $TARGET_USER after refreshing the login session."

    log "Building $DOCKER_IMAGE_TAG from $DOCKERFILE"
    sudo -iu "$TARGET_USER" env DOCKER_BUILDKIT=1 \
        docker build \
            --file "$DOCKERFILE" \
            --tag "$DOCKER_IMAGE_TAG" \
            "$DOCKER_BUILD_CONTEXT"
}

login_huggingface() {
    local hf_token=""
    local login_status=0
    local target_home=""
    local target_uid=""
    local target_gid=""
    local hf_home=""

    target_home="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
    [[ -n "$target_home" ]] || error "Unable to determine the home directory for $TARGET_USER."
    target_uid="$(id -u "$TARGET_USER")"
    target_gid="$(id -g "$TARGET_USER")"
    hf_home="$target_home/.cache/huggingface"

    IFS= read -r -s -p "Paste your Hugging Face access token: " hf_token
    printf '\n'
    [[ -n "$hf_token" ]] || error "A Hugging Face access token is required."

    sudo -u "$TARGET_USER" mkdir -p "$hf_home"
    sudo -iu "$TARGET_USER" docker run --rm \
        --user "$target_uid:$target_gid" \
        --env HF_HOME=/hf-home \
        --volume "$hf_home:/hf-home" \
        --entrypoint hf \
        "$DOCKER_IMAGE_TAG" \
        auth login --token "$hf_token" || login_status=$?

    hf_token=""
    unset hf_token
    (( login_status == 0 )) \
        || error "Hugging Face authentication failed."

    log "Hugging Face authentication saved in $hf_home"
}

log "Installing or verifying CUDA Toolkit 13.x and cuDNN"
install_cuda_and_cudnn

log "Installing Docker"
install_docker

build_docker_image_as_target_user

log "Authenticating with Hugging Face"
login_huggingface

printf '\n\033[1;32mSetup completed successfully.\033[0m\n'
printf 'CUDA Toolkit: %s (%s)\n' "$ACTIVE_CUDA_VERSION" "$ACTIVE_CUDA_HOME"
printf 'cuDNN package: %s (%s)\n' "$ACTIVE_CUDNN_PACKAGE" "$ACTIVE_CUDNN_VERSION"
printf 'Docker image: %s\n\n' "$DOCKER_IMAGE_TAG"

if [[ -t 0 && -t 1 ]]; then
    log "Opening a refreshed shell with Docker group access"
    printf 'Docker commands can now be run without sudo. Run "exit" to return to the original shell.\n\n'
    trap - EXIT INT TERM HUP
    exec newgrp docker
fi

log "Non-interactive setup complete; start a new login session before running Docker without sudo"
