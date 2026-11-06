from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml

from src.data.loader import load_trqa_split
from src.evaluation.checkpoint import format_table4_markdown
from src.evaluation.ir_metrics import evaluate_batch_retrieval
from src.retrieval.bm25_session import BM25SessionRetriever, RetrievedDocument


def save_trec_run(
    run_path: Path,
    results: Dict[str, List[RetrievedDocument]],
    run_name: str = "BM25_partial_wiki",
) -> None:
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("w", encoding="utf-8") as f:
        for qid, docs in results.items():
            for doc in docs:
                f.write(f"{qid} Q0 {doc.doc_id} {doc.rank} {doc.score:.6f} {run_name}\n")


def save_top_m_cache(
    cache_path: Path,
    results: Dict[str, List[RetrievedDocument]],
    top_m: int = 100,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for qid, docs in results.items():
            serialized = [
                {
                    "doc_id": d.doc_id,
                    "score": d.score,
                    "title": d.title,
                    "text": d.text,
                    "rank": d.rank,
                }
                for d in docs[:top_m]
            ]
            f.write(json.dumps({"query_id": qid, "docs": serialized}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1.1: Reproduce BM25 Retrieval (Table 4)")
    parser.add_argument("--config", type=str, default="configs/wiki2_bm25.yaml")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to evaluate (validation test)")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap for smoke testing")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    bm25_cfg = cfg["bm25"]
    logs_dir = Path(cfg["evaluation"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    splits = args.splits or ds_cfg.get("splits", ["validation", "test"])
    cutoffs = bm25_cfg.get("recall_cutoffs", [3, 10, 100, 1000])
    max_k = int(bm25_cfg.get("max_eval_k", 1000))
    cache_top_n = int(bm25_cfg.get("session_cache_top_n", 100))

    print(f"Initializing BM25SessionRetriever from index: {ds_cfg['bm25_index_dir']}")
    retriever = BM25SessionRetriever(
        index_dir=ds_cfg["bm25_index_dir"],
        k1=float(bm25_cfg.get("k1", 0.9)),
        b=float(bm25_cfg.get("b", 0.4)),
        session_cache_top_n=cache_top_n,
    )

    split_metrics: Dict[str, Dict[str, float]] = {}

    for split_name in splits:
        print(f"\n=== Running Step 1.1 BM25 Retrieval on {ds_cfg['subset']} ({split_name}) ===")
        split_data = load_trqa_split(
            subset=ds_cfg["subset"],
            split=split_name,
            local_data_dir=ds_cfg["local_data_dir"],
            hf_dataset_name=ds_cfg["hf_dataset_name"],
        )
        queries = split_data.queries
        if args.max_queries:
            queries = queries[: args.max_queries]

        q_texts = [q.question for q in queries]
        q_ids = [q.query_id for q in queries]
        qrels_subset = {q.query_id: q.relevant_doc_ids for q in queries}

        batch_docs = retriever.batch_retrieve(
            queries=q_texts,
            qids=q_ids,
            k=max_k,
            threads=int(bm25_cfg.get("threads", 16)),
            parse_text_top_m=cache_top_n,
        )

        # Save TREC run file and top-100 parsed text cache (reused in Step 1.2a)
        trec_path = logs_dir / f"step1_1_bm25_{split_name}.trec"
        save_trec_run(trec_path, batch_docs)
        cache_path = logs_dir / "cache" / f"bm25_top{cache_top_n}_{split_name}.jsonl"
        save_top_m_cache(cache_path, batch_docs, top_m=cache_top_n)

        doc_id_lists = {qid: [d.doc_id for d in docs] for qid, docs in batch_docs.items()}
        metrics = evaluate_batch_retrieval(doc_id_lists, qrels_subset, cutoffs=cutoffs)
        split_metrics[split_name] = metrics

        print(f"Results for {split_name} ({metrics['num_queries']} queries):")
        for k in cutoffs:
            print(f"  R@{k}: {metrics[f'R@{k}']:.2f}%")

    md_report = format_table4_markdown(split_metrics)
    print("\n" + md_report)

    (logs_dir / "step1_1_table4_results.md").write_text(md_report, encoding="utf-8")
    (logs_dir / "step1_1_table4_results.json").write_text(
        json.dumps(split_metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Saved Table 4 report to {logs_dir / 'step1_1_table4_results.md'}")


if __name__ == "__main__":
    main()

