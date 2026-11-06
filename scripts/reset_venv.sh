#!/bin/bash -e

cd "$(dirname "$0")/.."

if [ -d "/scratch/${USER}/virtual_environments/deep-research-agent" ]; then
    echo "Removing /scratch/${USER}/virtual_environments/deep-research-agent..."
    rm -rf "/scratch/${USER}/virtual_environments/deep-research-agent"
fi

rm -rf venv
./scripts/setup_virtual_environment.sh

