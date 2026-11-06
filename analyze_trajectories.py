from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def analyze_split(jsonl_path: Path, max_turns: int = 10) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n = len(records)
    if n == 0:
        return {}

    total_turns = 0
    total_dup_turns = 0
    total_zero_new_turns = 0

    # Recall curve by turn (carry-forward final recall if trajectory finished earlier)
    recall_by_turn = [0.0 for _ in range(max_turns)]
    precision_by_turn = [0.0 for _ in range(max_turns)]

    # Failure taxonomy counters
    zero_recall_failures = 0
    partial_recall_failures = 0
    high_recall_reasoning_failures = 0
    exact_matches = 0

    for rec in records:
        qa = rec.get("qa_metrics", {})
        ir = rec.get("trajectory_ir", {})
        em = float(qa.get("EM", 0.0)) > 0
        union_rec = float(ir.get("union_recall", 0.0))

        if em:
            exact_matches += 1
        else:
            if union_rec == 0.0:
                zero_recall_failures += 1
            elif union_rec < 50.0:
                partial_recall_failures += 1
            else:
                high_recall_reasoning_failures += 1

        total_turns += int(ir.get("num_search_turns", 0))
        total_dup_turns += int(ir.get("duplicate_query_turns", 0))
        total_zero_new_turns += int(ir.get("zero_new_doc_turns", 0))

        per_turn = ir.get("per_turn", [])
        last_rec = 0.0
        last_prec = 0.0
        for t in range(max_turns):
            if t < len(per_turn):
                last_rec = float(per_turn[t].get("cumulative_recall", last_rec))
                last_prec = float(per_turn[t].get("cumulative_precision", last_prec))
            recall_by_turn[t] += last_rec
            precision_by_turn[t] += last_prec

    return {
        "num_queries": n,
        "EM": round(sum(float(r["qa_metrics"]["EM"]) for r in records) / n, 2),
        "SM_10%": round(sum(float(r["qa_metrics"]["SM_10%"]) for r in records) / n, 2),
        "SM_20%": round(sum(float(r["qa_metrics"]["SM_20%"]) for r in records) / n, 2),
        "SM_50%": round(sum(float(r["qa_metrics"]["SM_50%"]) for r in records) / n, 2),
        "SM_90%": round(sum(float(r["qa_metrics"]["SM_90%"]) for r in records) / n, 2),
        "union_recall": round(sum(float(r["trajectory_ir"]["union_recall"]) for r in records) / n, 2),
        "union_precision": round(
            sum(float(r["trajectory_ir"]["union_precision"]) for r in records) / n, 2
        ),
        "union_f1": round(sum(float(r["trajectory_ir"]["union_f1"]) for r in records) / n, 2),
        "avg_search_turns": round(total_turns / n, 2),
        "corpus_duplicate_turn_pct": round(
            (total_dup_turns / total_turns * 100.0) if total_turns else 0.0, 2
        ),
        "corpus_zero_new_doc_turn_pct": round(
            (total_zero_new_turns / total_turns * 100.0) if total_turns else 0.0, 2
        ),
        "recall_curve_by_turn": [round(val / n, 2) for val in recall_by_turn],
        "precision_curve_by_turn": [round(val / n, 2) for val in precision_by_turn],
        "failure_taxonomy": {
            "exact_match_pct": round(exact_matches / n * 100.0, 2),
            "zero_recall_failure_pct": round(zero_recall_failures / n * 100.0, 2),
            "partial_recall_failure_pct": round(partial_recall_failures / n * 100.0, 2),
            "high_recall_aggregation_failure_pct": round(
                high_recall_reasoning_failures / n * 100.0, 2
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Search-R1 + BM25 trajectories for failure modes and stagnation diagnostics."
    )
    parser.add_argument("--logs-dir", type=str, default="logs")
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    candidates = sorted(logs_dir.glob("**/step1_2b_search_r1_*.jsonl"))
    if not candidates:
        print(f"No trajectory files matching 'step1_2b_search_r1_*.jsonl' found in {logs_dir}")
        return

    report_lines = ["# Search-R1 + BM25 Trajectory & Stagnation Analysis", ""]
    all_stats: Dict[str, Any] = {}

    for path in candidates:
        stats = analyze_split(path, max_turns=args.max_turns)
        if not stats:
            continue
        label = path.stem.replace("step1_2b_search_r1_", "")
        all_stats[label] = stats

        report_lines.extend(
            [
                f"## Split: `{label}` ({stats['num_queries']} queries)",
                f"- **QA Accuracy:** EM = `{stats['EM']}%`, SM_10% = `{stats['SM_10%']}%`, SM_20% = `{stats['SM_20%']}%`, SM_50% = `{stats['SM_50%']}%`, SM_90% = `{stats['SM_90%']}%`",
                f"- **Trajectory Retrieval:** Union Recall = `{stats['union_recall']}%`, Union Precision = `{stats['union_precision']}%`, Union F1 = `{stats['union_f1']}%`",
                f"- **Search & Stagnation:** Avg Search Turns = `{stats['avg_search_turns']}`, Duplicate Query Turns = `{stats['corpus_duplicate_turn_pct']}%`, Zero-New-Doc Turns = `{stats['corpus_zero_new_doc_turn_pct']}%`",
                f"- **Cumulative Recall by Turn (1..{args.max_turns}):** `{stats['recall_curve_by_turn']}`",
                f"- **Error Breakdown:**",
                f"  - Correct (EM): `{stats['failure_taxonomy']['exact_match_pct']}%`",
                f"  - Zero-Recall Failure (0% gold docs found): `{stats['failure_taxonomy']['zero_recall_failure_pct']}%`",
                f"  - Partial-Recall Failure (<50% gold docs found): `{stats['failure_taxonomy']['partial_recall_failure_pct']}%`",
                f"  - High-Recall Aggregation Failure (>=50% gold docs found, wrong answer): `{stats['failure_taxonomy']['high_recall_aggregation_failure_pct']}%`",
                "",
            ]
        )

    report_md = "\n".join(report_lines)
    print(report_md)
    out_md = logs_dir / "trajectory_stagnation_analysis.md"
    out_json = logs_dir / "trajectory_stagnation_analysis.json"
    out_md.write_text(report_md, encoding="utf-8")
    out_json.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"Saved analysis report to {out_md}")


if __name__ == "__main__":
    main()

