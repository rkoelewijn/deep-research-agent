#!/bin/bash -e

# Ensure we run from repository root
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "Error: .env file not found. Run 'cp .env.example .env' and edit it first."
    exit 1
fi

source .env

if [ -z "$NFS_USER_DIR" ]; then
    echo "Error: NFS_USER_DIR is not set in .env"
    exit 1
fi

PROJECT_NFS_DIR="${NFS_USER_DIR}/deep-research-agent"
echo "Preparing cluster storage at: ${PROJECT_NFS_DIR}"

mkdir -p "${PROJECT_NFS_DIR}/data"
mkdir -p "${PROJECT_NFS_DIR}/data/hf_cache"
mkdir -p "${PROJECT_NFS_DIR}/data/torch_cache"
mkdir -p "${PROJECT_NFS_DIR}/data/pyserini_cache"
mkdir -p "${PROJECT_NFS_DIR}/logs"
mkdir -p "${PROJECT_NFS_DIR}/logs/cache"

# Create symlinks for data and logs
rm -rf data logs
ln -s "${PROJECT_NFS_DIR}/data" data
ln -s "${PROJECT_NFS_DIR}/logs" logs

echo "Created symlinks:"
ls -la data logs

# Download portable OpenJDK 21 into data/jdk-21 if not already installed (required by Pyserini)
JDK_DIR="data/jdk-21"
if [ ! -d "${JDK_DIR}/bin" ]; then
    echo "Installing portable OpenJDK 21 into ${JDK_DIR} for Pyserini..."
    mkdir -p "${JDK_DIR}"
    JDK_URL="https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.5%2B11/OpenJDK21U-jdk_x64_linux_hotspot_21.0.5_11.tar.gz"
    curl -L "${JDK_URL}" | tar -xz -C "${JDK_DIR}" --strip-components=1
    echo "OpenJDK 21 installed at ${JDK_DIR}"
else
    echo "OpenJDK 21 already present at ${JDK_DIR}"
fi

