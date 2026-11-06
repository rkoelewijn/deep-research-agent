#!/bin/bash -e

# Syncs the node-local /scratch virtual environment across all csedu compute nodes.
# Run this script after ./scripts/setup_virtual_environment.sh or any pip install -r requirements.txt update.

VENV_DIR="/scratch/${USER}/virtual_environments/deep-research-agent"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Error: ${VENV_DIR} does not exist. Run ./scripts/setup_virtual_environment.sh first."
    exit 1
fi

# Discover csedu nodes via sinfo (fallback to cn84 cn85 if sinfo is unavailable)
if command -v sinfo >/dev/null 2>&1; then
    NODES=$(sinfo -p csedu -N -h -o "%N" | sort -u)
else
    NODES="cn84 cn85"
fi

CURRENT_HOST=$(hostname -s)
echo "Syncing ${VENV_DIR} from ${CURRENT_HOST} to csedu nodes: ${NODES}"

for NODE in ${NODES}; do
    if [ "${NODE}" != "${CURRENT_HOST}" ]; then
        echo " -> Syncing to ${NODE}..."
        ssh "${NODE}" "mkdir -p /scratch/${USER}/virtual_environments"
        rsync -a --delete "${VENV_DIR}/" "${NODE}:${VENV_DIR}/"
    fi
done

echo "Virtual environment sync complete across all csedu nodes."

