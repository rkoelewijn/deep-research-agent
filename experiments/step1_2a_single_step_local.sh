#!/bin/bash -e

cd "$(dirname "$0")/.."
mkdir -p logs logs/cache

if [ -d "$PWD/data/jdk-21/bin" ]; then
    export JAVA_HOME="$PWD/data/jdk-21"
    export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
    export PATH="$JAVA_HOME/bin:$PATH"
fi

export HF_HOME="$PWD/data/hf_cache"
export TORCH_HOME="$PWD/data/torch_cache"
export PYSERINI_CACHE="$PWD/data/pyserini_cache"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 run_step1_2a_single_step.py \
    --config configs/wiki2_bm25.yaml \
    --splits validation test \
    --resume "$@"

