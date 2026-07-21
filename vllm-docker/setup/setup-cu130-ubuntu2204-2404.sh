#!/usr/bin/env bash

set -Eeuo pipefail

readonly DOCKER_KEYRING="/etc/apt/keyrings/docker.asc"
readonly DOCKER_SOURCE="/etc/apt/sources.list.d/docker.sources"
readonly DOCKER_REPO="https://download.docker.com/linux/ubuntu"

log() {
    printf '\n\033[1;32m[Docker Setup]\033[0m %s\n' "$1"
}

error() {
    printf '\n\033[1;31m[Error]\033[0m %s\n' "$1" >&2
    exit 1
}

# Verify that this is an Ubuntu system.
if [[ ! -r /etc/os-release ]]; then
    error "Cannot read /etc/os-release."
fi

# shellcheck disable=SC1091
source /etc/os-release

if [[ "${ID:-}" != "ubuntu" ]]; then
    error "This installer is intended for Ubuntu. Detected: ${PRETTY_NAME:-unknown}"
fi

UBUNTU_RELEASE="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"

if [[ -z "${UBUNTU_RELEASE}" ]]; then
    error "Unable to determine the Ubuntu release codename."
fi

ARCHITECTURE="$(dpkg --print-architecture)"
TARGET_USER="${SUDO_USER:-$USER}"

if [[ "${TARGET_USER}" == "root" ]]; then
    error "Run this script as your normal user, not with 'sudo ./install-docker.sh'. The script will request sudo when needed."
fi

log "Updating the APT package index"
sudo apt-get update

log "Installing required packages"
sudo apt-get install -y ca-certificates curl

log "Creating the APT keyring directory"
sudo install -m 0755 -d /etc/apt/keyrings

log "Downloading Docker's repository signing key"
sudo curl \
    --fail \
    --silent \
    --show-error \
    --location \
    "${DOCKER_REPO}/gpg" \
    --output "${DOCKER_KEYRING}"

sudo chmod a+r "${DOCKER_KEYRING}"

log "Configuring the Docker APT repository"
sudo tee "${DOCKER_SOURCE}" >/dev/null <<EOF
Types: deb
URIs: ${DOCKER_REPO}
Suites: ${UBUNTU_RELEASE}
Components: stable
Architectures: ${ARCHITECTURE}
Signed-By: ${DOCKER_KEYRING}
EOF

log "Refreshing the APT package index"
sudo apt-get update

log "Installing Docker Engine and Docker Compose"
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

log "Enabling and starting Docker"
sudo systemctl enable --now docker

log "Adding ${TARGET_USER} to the docker group"
sudo usermod -aG docker "${TARGET_USER}"

log "Checking the Docker service"
sudo systemctl is-active --quiet docker || error "Docker did not start successfully."

printf '\n\033[1;32mDocker installation completed successfully.\033[0m\n'
printf '\nUser "%s" was added to the docker group.\n' "${TARGET_USER}"
printf 'You must log out completely and log back in before running Docker without sudo.\n'
printf '\nAfter logging back in, verify the installation with:\n'
printf '  docker run --rm hello-world\n'
printf '  docker compose version\n\n'
