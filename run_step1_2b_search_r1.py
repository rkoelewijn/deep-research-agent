from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm import tqdm

from src.data.loader import load_trqa_split
from src.evaluation.checkpoint import JSONLCheckpoint, format_table8_markdown
from src.evaluation.ir_metrics import evaluate_trajectory_retrieval
from src.evaluation.qa_metrics import evaluate_qa_prediction, evaluate_qa_split
from src.retrieval.bm25_session import BM25SessionRetriever
from src.search_r1.agent_loop import SearchR1AgentRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1.2b: Reproduce Search-R1 + BM25 on TRQA-Wiki2 (Table 8 DRA)"
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
    sr1_cfg = cfg["search_r1"]
    eval_cfg = cfg["evaluation"]
    logs_dir = Path(eval_cfg["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    splits = args.splits or ds_cfg.get("splits", ["validation", "test"])
    sm_thresholds = eval_cfg.get("sm_thresholds", [10, 20, 50, 90])

    print(f"Initializing BM25SessionRetriever from: {ds_cfg['bm25_index_dir']}")
    retriever = BM25SessionRetriever(
        index_dir=ds_cfg["bm25_index_dir"],
        k1=float(bm25_cfg.get("k1", 0.9)),
        b=float(bm25_cfg.get("b", 0.4)),
        session_cache_top_n=int(bm25_cfg.get("session_cache_top_n", 100)),
    )

    print(f"Loading Search-R1 model: {sr1_cfg['model_name_or_path']} ({sr1_cfg.get('dtype', 'float16')})")
    agent = SearchR1AgentRunner(
        retriever=retriever,
        model_name_or_path=sr1_cfg["model_name_or_path"],
        dtype=sr1_cfg.get("dtype", "float16"),
        top_k_per_search=int(sr1_cfg.get("top_k_per_search", 3)),
        max_turns=int(sr1_cfg.get("max_turns", 10)),
        max_new_tokens_per_turn=int(sr1_cfg.get("max_new_tokens_per_turn", 512)),
        max_doc_chars=int(sr1_cfg.get("max_doc_chars", 1200)),
        temperature=float(sr1_cfg.get("temperature", 0.0)),
        do_sample=bool(sr1_cfg.get("do_sample", False)),
    )

    split_summaries: Dict[str, Dict[str, Any]] = {}

    for split_name in splits:
        print(f"\n=== Running Step 1.2b Search-R1 + BM25 on {ds_cfg['subset']} ({split_name}) ===")
        split_data = load_trqa_split(
            subset=ds_cfg["subset"],
            split=split_name,
            local_data_dir=ds_cfg["local_data_dir"],
            hf_dataset_name=ds_cfg["hf_dataset_name"],
        )
        queries = split_data.queries
        if args.max_queries:
            queries = queries[: args.max_queries]

        ckpt_path = logs_dir / f"step1_2b_search_r1_{split_name}.jsonl"
        ckpt = JSONLCheckpoint(ckpt_path)
        completed = ckpt.load_completed() if args.resume else {}
        if not args.resume and ckpt_path.exists():
            ckpt_path.unlink()

        all_records: List[Dict[str, Any]] = []

        for q in tqdm(queries, desc=f"Step 1.2b Search-R1 ({split_name})"):
            if q.query_id in completed:
                all_records.append(completed[q.query_id])
                continue

            traj_out = agent.run_trajectory(query_id=q.query_id, question=q.question)
            qa_metrics = evaluate_qa_prediction(
                prediction=traj_out["predicted_answer"],
                gold_answer=q.answer,
                sm_thresholds=sm_thresholds,
            )
            traj_ir = evaluate_trajectory_retrieval(
                turn_logs=traj_out["turn_logs"],
                relevant_doc_ids=q.relevant_doc_ids,
            )

            record = {
                "query_id": q.query_id,
                "question": q.question,
                "gold_answer": q.answer,
                "predicted_answer": traj_out["predicted_answer"],
                "finish_reason": traj_out["finish_reason"],
                "qa_metrics": qa_metrics,
                "trajectory_ir": traj_ir,
                "generated_trace": traj_out["generated_trace"],
            }
            ckpt.append(record)
            all_records.append(record)

        summary = evaluate_qa_split(all_records, sm_thresholds=sm_thresholds)
        n = len(all_records) or 1
        summary["union_recall"] = round(
            sum(r["trajectory_ir"]["union_recall"] for r in all_records) / n, 2
        )
        summary["union_precision"] = round(
            sum(r["trajectory_ir"]["union_precision"] for r in all_records) / n, 2
        )
        summary["union_f1"] = round(
            sum(r["trajectory_ir"]["union_f1"] for r in all_records) / n, 2
        )
        summary["avg_turns"] = round(
            sum(r["trajectory_ir"]["num_search_turns"] for r in all_records) / n, 2
        )
        summary["avg_unique_queries"] = round(
            sum(r["trajectory_ir"]["num_unique_queries"] for r in all_records) / n, 2
        )
        summary["duplicate_query_rate"] = round(
            sum(r["trajectory_ir"]["duplicate_query_rate"] for r in all_records) / n, 2
        )
        summary["zero_new_doc_rate"] = round(
            sum(r["trajectory_ir"]["zero_new_doc_rate"] for r in all_records) / n, 2
        )
        split_summaries[split_name] = summary

    md_report = format_table8_markdown("Search-R1 (Qwen2.5-7B + BM25)", split_summaries)
    print("\n" + md_report)

    (logs_dir / "step1_2b_table8_results.md").write_text(md_report, encoding="utf-8")
    (logs_dir / "step1_2b_table8_results.json").write_text(
        json.dumps(split_summaries, indent=2),
        encoding="utf-8",
    )
    print(f"Saved Step 1.2b Table 8 report to {logs_dir / 'step1_2b_table8_results.md'}")


if __name__ == "__main__":
    main()

