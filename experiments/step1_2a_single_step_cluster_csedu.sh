#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --job-name=step1_2a_rag
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/step1_2a_single_step_%j.out
#SBATCH --error=logs/step1_2a_single_step_%j.err

mkdir -p logs logs/cache

if [ -d "$PWD/data/jdk-21/bin" ]; then
    export JAVA_HOME="$PWD/data/jdk-21"
    export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
    export PATH="$JAVA_HOME/bin:$PATH"
fi

export HF_HOME="$PWD/data/hf_cache"
export TORCH_HOME="$PWD/data/torch_cache"
export PYSERINI_CACHE="$PWD/data/pyserini_cache"

source venv/bin/activate

python3 run_step1_2a_single_step.py \
    --config configs/wiki2_bm25.yaml \
    --splits validation test \
    --resume

