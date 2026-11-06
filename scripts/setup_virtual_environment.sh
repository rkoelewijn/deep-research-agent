#!/bin/bash -e

cd "$(dirname "$0")/.."

export PIP_NO_CACHE_DIR=off

# If running on the cluster (/scratch exists), place venv on local /scratch disk
if [ -d "/scratch" ]; then
    VENV_TARGET="/scratch/${USER}/virtual_environments/deep-research-agent"
    mkdir -p "$(dirname "${VENV_TARGET}")"
    if [ ! -d "${VENV_TARGET}" ]; then
        echo "Creating virtual environment on /scratch at ${VENV_TARGET}..."
        python3 -m venv "${VENV_TARGET}"
    fi
    rm -rf venv
    ln -s "${VENV_TARGET}" venv
else
    echo "/scratch not found; creating local ./venv directory..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
fi

# Export JAVA_HOME if portable JDK 21 exists so pyjnius/pyserini builds cleanly
if [ -d "$PWD/data/jdk-21/bin" ]; then
    export JAVA_HOME="$PWD/data/jdk-21"
    export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
    export PATH="$JAVA_HOME/bin:$PATH"
fi

source venv/bin/activate
pip install --no-cache-dir --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -e .

echo "Virtual environment ready. On csedu cluster, remember to run: ./scripts/sync_csedu.sh"

