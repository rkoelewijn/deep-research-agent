from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class JSONLCheckpoint:
    """
    Line-by-line JSONL checkpoint manager supporting fault-tolerant `--resume`
    for long-running SLURM jobs on the csedu cluster.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def load_completed(self) -> Dict[str, Dict[str, Any]]:
        """Returns a mapping of `query_id -> record` for all already-completed queries."""
        completed: Dict[str, Dict[str, Any]] = {}
        if not self.file_path.exists():
            return completed

        with self.file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    qid = str(rec.get("query_id", ""))
                    if qid:
                        completed[qid] = rec
                except json.JSONDecodeError:
                    continue
        return completed

    def append(self, record: Dict[str, Any]) -> None:
        """Appends a single completed query record and flushes immediately to disk."""
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def format_table4_markdown(split_metrics: Dict[str, Dict[str, float]]) -> str:
    """Formats Step 1.1 BM25 Recall@K results alongside Table 4 paper reference."""
    lines = [
        "# Step 1.1: Standalone BM25 Retrieval Results (Table 4 Reproduction)",
        "",
        "| Dataset / Split | Corpus | Queries | R@3 (%) | R@10 (%) | R@100 (%) | R@1000 (%) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
        "| **TRQA Paper Table 4 (Reference)** | Full Wiki (57.7M) | 1258 | 5.54 | 11.43 | 24.90 | 43.82 |",
    ]
    for split_name, m in split_metrics.items():
        lines.append(
            f"| **TRQA-Wiki2 ({split_name})** | Partial Wiki (5.9M) | {int(m.get('num_queries', 0))} "
            f"| {m.get('R@3', 0.0):.2f} | {m.get('R@10', 0.0):.2f} "
            f"| {m.get('R@100', 0.0):.2f} | {m.get('R@1000', 0.0):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def format_table8_markdown(
    method_name: str,
    split_summaries: Dict[str, Dict[str, Any]],
) -> str:
    """Formats Step 1.2 end-to-end QA & retrieval metrics alongside Table 8 paper references."""
    lines = [
        f"# Step 1.2: {method_name} Results (Table 8 Reproduction)",
        "",
        "| Method / Split | Queries | EM (%) | SM_10% | SM_20% | SM_50% | SM_90% | Union Recall (%) | Union Precision (%) | Union F1 (%) | Avg Turns |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| **Paper Ref: E5 & Qwen2.5-7B (K=50, test)** | 1258 | 3.42 | 7.00 | 8.82 | 14.47 | 24.32 | - | - | - | 1.00 |",
        "| **Paper Ref: Search-R1 (Qwen2.5-7B + E5, test)** | 1258 | 4.76 | 15.18 | 17.17 | 26.23 | 53.49 | - | - | - | - |",
    ]
    for split_name, s in split_summaries.items():
        lines.append(
            f"| **{method_name} ({split_name})** | {int(s.get('count', 0))} "
            f"| {s.get('EM', 0.0):.2f} | {s.get('SM_10%', 0.0):.2f} "
            f"| {s.get('SM_20%', 0.0):.2f} | {s.get('SM_50%', 0.0):.2f} "
            f"| {s.get('SM_90%', 0.0):.2f} | {s.get('union_recall', 0.0):.2f} "
            f"| {s.get('union_precision', 0.0):.2f} | {s.get('union_f1', 0.0):.2f} "
            f"| {s.get('avg_turns', 1.0):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)

