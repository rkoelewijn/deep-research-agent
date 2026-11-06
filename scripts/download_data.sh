#!/bin/bash -e

cd "$(dirname "$0")/.."

if [ ! -d "data" ]; then
    mkdir -p data
fi

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export HF_HOME="$PWD/data/hf_cache"
export TORCH_HOME="$PWD/data/torch_cache"
export PYSERINI_CACHE="$PWD/data/pyserini_cache"

echo "Downloading TRQA-Wiki2 queries, qrels, and Partial Wikipedia BM25 index into data/..."

python3 - <<'EOF'
import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download

data_dir = Path("data")
trqa_dir = data_dir / "trqa"
exp_dir = data_dir / "trqa_experiments"
trqa_dir.mkdir(parents=True, exist_ok=True)
exp_dir.mkdir(parents=True, exist_ok=True)

print("1. Downloading mahtaa/trqa queries and qrels for wiki2_validation and wiki2_test...")
for split in ["wiki2_validation", "wiki2_test"]:
    queries = load_dataset("mahtaa/trqa", "queries", split=split)
    queries.to_json(str(trqa_dir / f"queries_{split}.jsonl"), orient="records", lines=True)
    print(f"   Saved {len(queries)} queries to {trqa_dir / f'queries_{split}.jsonl'}")

    qrels = load_dataset("mahtaa/trqa", "qrels", split=split)
    qrels.to_json(str(trqa_dir / f"qrels_{split}.jsonl"), orient="records", lines=True)
    print(f"   Saved {len(qrels)} qrel rows to {trqa_dir / f'qrels_{split}.jsonl'}")

print("2. Downloading Partial Wikipedia BM25 Lucene index from mahtaa/trqa_experiments...")
snapshot_download(
    repo_id="mahtaa/trqa_experiments",
    repo_type="dataset",
    local_dir=str(exp_dir),
    allow_patterns=["indices/wiki_partial/bm25_partial/*"],
)
print("Done! Data and BM25 partial index are ready in ./data/")
EOF

