#!/bin/bash -e

cd "$(dirname "$0")/.."

if [ -f .env ]; then
    source .env
fi

REMOTE_USER="${CSEDU_USER:-$USER}"
REMOTE_HOST="${CSEDU_HOST:-cn84.science.ru.nl}"
REMOTE_DIR="${CSEDU_REPO_DIR:-/home/${REMOTE_USER}/deep-research-agent}/logs/"

mkdir -p logs/cluster
echo "Downloading logs from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} to logs/cluster/..."
rsync -avz --exclude="cache/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" logs/cluster/
echo "Logs synced to logs/cluster/"

