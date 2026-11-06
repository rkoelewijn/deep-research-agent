from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from src.data.loader import load_trqa_split
from src.evaluation.checkpoint import JSONLCheckpoint, format_table8_markdown
from src.evaluation.ir_metrics import compute_prf1
from src.evaluation.qa_metrics import evaluate_qa_prediction, evaluate_qa_split
from src.retrieval.bm25_session import BM25SessionRetriever, RetrievedDocument
from src.single_step.rag_runner import SingleStepRAGRunner


def load_cached_top_docs(cache_path: Path) -> Dict[str, List[RetrievedDocument]]:
    cached: Dict[str, List[RetrievedDocument]] = {}
    if not cache_path.exists():
        return cached
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["query_id"])
            cached[qid] = [
                RetrievedDocument(
                    doc_id=str(d["doc_id"]),
                    score=float(d["score"]),
                    title=str(d.get("title", "")),
                    text=str(d.get("text", "")),
                    rank=int(d.get("rank", i + 1)),
                )
                for i, d in enumerate(row.get("docs", []))
            ]
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1.2a: Reproduce Single-Step BM25 (K=50) + Qwen2.5-7B-Instruct (Table 8 Baseline)"
    )
    parser.add_argument("--config", type=str, default="configs/wiki2_bm25.yaml")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to evaluate (validation test)")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap for smoke testing")
    parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL checkpoint")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    bm25_cfg = cfg["bm25"]
    rag_cfg = cfg["single_step_rag"]
    eval_cfg = cfg["evaluation"]
    logs_dir = Path(eval_cfg["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    splits = args.splits or ds_cfg.get("splits", ["validation", "test"])
    top_k = int(rag_cfg.get("top_k_docs", 50))
    cache_top_n = int(bm25_cfg.get("session_cache_top_n", 100))
    sm_thresholds = eval_cfg.get("sm_thresholds", [10, 20, 50, 90])

    retriever: Optional[BM25SessionRetriever] = None
    print(f"Loading Single-Step RAG LLM: {rag_cfg['model_name_or_path']} ({rag_cfg.get('dtype', 'float16')})")
    runner = SingleStepRAGRunner(
        model_name_or_path=rag_cfg["model_name_or_path"],
        dtype=rag_cfg.get("dtype", "float16"),
        max_new_tokens=int(rag_cfg.get("max_new_tokens", 512)),
        temperature=float(rag_cfg.get("temperature", 0.0)),
        do_sample=bool(rag_cfg.get("do_sample", False)),
        max_doc_chars=int(rag_cfg.get("max_doc_chars", 1200)),
    )

    split_summaries: Dict[str, Dict[str, Any]] = {}

    for split_name in splits:
        print(f"\n=== Running Step 1.2a Single-Step RAG (K={top_k}) on {ds_cfg['subset']} ({split_name}) ===")
        split_data = load_trqa_split(
            subset=ds_cfg["subset"],
            split=split_name,
            local_data_dir=ds_cfg["local_data_dir"],
            hf_dataset_name=ds_cfg["hf_dataset_name"],
        )
        queries = split_data.queries
        if args.max_queries:
            queries = queries[: args.max_queries]

        cache_path = logs_dir / "cache" / f"bm25_top{cache_top_n}_{split_name}.jsonl"
        cached_docs = load_cached_top_docs(cache_path)

        ckpt_path = logs_dir / f"step1_2a_single_step_{split_name}.jsonl"
        ckpt = JSONLCheckpoint(ckpt_path)
        completed = ckpt.load_completed() if args.resume else {}
        if not args.resume and ckpt_path.exists():
            ckpt_path.unlink()

        all_records: List[Dict[str, Any]] = []

        for q in tqdm(queries, desc=f"Step 1.2a ({split_name})"):
            if q.query_id in completed:
                all_records.append(completed[q.query_id])
                continue

            if q.query_id in cached_docs and len(cached_docs[q.query_id]) >= top_k:
                docs = cached_docs[q.query_id][:top_k]
            else:
                if retriever is None:
                    retriever = BM25SessionRetriever(
                        index_dir=ds_cfg["bm25_index_dir"],
                        k1=float(bm25_cfg.get("k1", 0.9)),
                        b=float(bm25_cfg.get("b", 0.4)),
                        session_cache_top_n=max(cache_top_n, top_k),
                    )
                docs = retriever.retrieve_top_n(q.question, top_n=top_k)

            rag_out = runner.answer_question(question=q.question, docs=docs)
            qa_metrics = evaluate_qa_prediction(
                prediction=rag_out["predicted_answer"],
                gold_answer=q.answer,
                sm_thresholds=sm_thresholds,
            )
            ir_metrics = compute_prf1(
                retrieved_doc_ids=[d.doc_id for d in docs],
                relevant_doc_ids=q.relevant_doc_ids,
            )

            record = {
                "query_id": q.query_id,
                "question": q.question,
                "gold_answer": q.answer,
                "predicted_answer": rag_out["predicted_answer"],
                "response_text": rag_out["response_text"],
                "qa_metrics": qa_metrics,
                "ir_metrics": ir_metrics,
                "retrieved_doc_ids": [d.doc_id for d in docs],
            }
            ckpt.append(record)
            all_records.append(record)

        summary = evaluate_qa_split(all_records, sm_thresholds=sm_thresholds)
        n = len(all_records) or 1
        summary["union_recall"] = round(
            sum(r["ir_metrics"]["recall"] for r in all_records) / n, 2
        )
        summary["union_precision"] = round(
            sum(r["ir_metrics"]["precision"] for r in all_records) / n, 2
        )
        summary["union_f1"] = round(
            sum(r["ir_metrics"]["f1"] for r in all_records) / n, 2
        )
        summary["avg_turns"] = 1.0
        split_summaries[split_name] = summary

    md_report = format_table8_markdown(f"BM25 (K={top_k}) & Qwen2.5-7B", split_summaries)
    print("\n" + md_report)

    (logs_dir / "step1_2a_table8_results.md").write_text(md_report, encoding="utf-8")
    (logs_dir / "step1_2a_table8_results.json").write_text(
        json.dumps(split_summaries, indent=2),
        encoding="utf-8",
    )
    print(f"Saved Step 1.2a Table 8 report to {logs_dir / 'step1_2a_table8_results.md'}")


if __name__ == "__main__":
    main()

