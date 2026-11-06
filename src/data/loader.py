from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set


@dataclass
class TRQAQuery:
    """Single TRQA query item with its gold answer and relevant document IDs."""

    query_id: str
    question: str
    answer: Any
    relevant_doc_ids: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TRQADatasetSplit:
    """Container for a loaded TRQA dataset split (e.g., wiki2_validation or wiki2_test)."""

    subset: str
    split_name: str
    hf_split_name: str
    queries: List[TRQAQuery]
    qrels: Dict[str, Set[str]]


def _extract_field(record: Dict[str, Any], candidates: List[str], default: Any = None) -> Any:
    for key in candidates:
        if key in record and record[key] is not None:
            return record[key]
    return default


def load_trqa_split(
    subset: str = "wiki2",
    split: str = "test",
    local_data_dir: str | Path = "data/trqa",
    hf_dataset_name: str = "mahtaa/trqa",
) -> TRQADatasetSplit:
    """
    Loads queries and qrels for a TRQA subset/split.
    First checks local JSONL files downloaded by `scripts/download_data.sh`,
    and falls back to Hugging Face `datasets.load_dataset` if not present locally.
    """
    hf_split = f"{subset}_{split}" if not split.startswith(subset) else split
    data_dir = Path(local_data_dir)
    queries_file = data_dir / f"queries_{hf_split}.jsonl"
    qrels_file = data_dir / f"qrels_{hf_split}.jsonl"

    if queries_file.exists() and qrels_file.exists():
        with queries_file.open("r", encoding="utf-8") as f:
            raw_queries = [json.loads(line) for line in f if line.strip()]
        with qrels_file.open("r", encoding="utf-8") as f:
            raw_qrels = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset

        raw_queries = list(load_dataset(hf_dataset_name, "queries", split=hf_split))
        raw_qrels = list(load_dataset(hf_dataset_name, "qrels", split=hf_split))

    qrels_map: Dict[str, Set[str]] = {}
    for row in raw_qrels:
        qid = str(_extract_field(row, ["query_id", "qid", "id"]))
        doc_id = str(_extract_field(row, ["corpus_id", "doc_id", "docid", "passage_id"]))
        rel_score = _extract_field(row, ["relevance", "score", "label"], default=1)
        try:
            is_rel = float(rel_score) > 0
        except (TypeError, ValueError):
            is_rel = True
        if is_rel and qid and doc_id:
            qrels_map.setdefault(qid, set()).add(doc_id)

    queries: List[TRQAQuery] = []
    for row in raw_queries:
        qid = str(_extract_field(row, ["query_id", "qid", "id"]))
        question = str(_extract_field(row, ["text", "query", "question"], default=""))
        answer = _extract_field(row, ["answer", "gold_answer", "answers", "target"], default="")
        rel_docs = qrels_map.get(qid, set())
        queries.append(
            TRQAQuery(
                query_id=qid,
                question=question,
                answer=answer,
                relevant_doc_ids=rel_docs,
                metadata=row,
            )
        )

    return TRQADatasetSplit(
        subset=subset,
        split_name=split,
        hf_split_name=hf_split,
        queries=queries,
        qrels=qrels_map,
    )

