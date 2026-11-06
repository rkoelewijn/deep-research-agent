from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set


def compute_recall_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Set[str],
    k: int,
) -> float:
    """Computes Recall@k (in percentage 0..100) for a single query."""
    if not relevant_doc_ids:
        return 0.0
    top_k_set = set(retrieved_doc_ids[:k])
    hits = len(top_k_set & relevant_doc_ids)
    return (hits / len(relevant_doc_ids)) * 100.0


def compute_prf1(
    retrieved_doc_ids: Iterable[str],
    relevant_doc_ids: Set[str],
) -> Dict[str, float]:
    """
    Computes set-based Precision, Recall, and F1-score (in percentage 0..100)
    over the unique set of retrieved documents vs. ground-truth qrels.
    """
    ret_set = set(retrieved_doc_ids)
    if not relevant_doc_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_retrieved_unique": len(ret_set),
            "num_relevant_hits": 0,
            "num_gold_relevant": 0,
        }

    hits = len(ret_set & relevant_doc_ids)
    precision = (hits / len(ret_set) * 100.0) if ret_set else 0.0
    recall = (hits / len(relevant_doc_ids)) * 100.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "num_retrieved_unique": len(ret_set),
        "num_relevant_hits": hits,
        "num_gold_relevant": len(relevant_doc_ids),
    }


def evaluate_batch_retrieval(
    run_results: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
    cutoffs: Sequence[int] = (3, 10, 100, 1000),
) -> Dict[str, float]:
    """
    Computes macro-averaged Recall@K across all queries in a split (Step 1.1 Table 4).
    """
    query_ids = [qid for qid, rel in qrels.items() if rel]
    if not query_ids:
        return {f"R@{k}": 0.0 for k in cutoffs}

    sums = {k: 0.0 for k in cutoffs}
    for qid in query_ids:
        rel_docs = qrels[qid]
        ret_docs = run_results.get(qid, [])
        for k in cutoffs:
            sums[k] += compute_recall_at_k(ret_docs, rel_docs, k=k)

    n = len(query_ids)
    metrics = {f"R@{k}": round(sums[k] / n, 2) for k in cutoffs}
    metrics["num_queries"] = n
    return metrics


def evaluate_trajectory_retrieval(
    turn_logs: List[Dict[str, Any]],
    relevant_doc_ids: Set[str],
) -> Dict[str, Any]:
    """
    Computes trajectory-level retrieval metrics across all turns of a Search-R1 rollout:
    - Union Precision, Union Recall, Union F1 across all unique retrieved documents
    - Per-turn cumulative Precision/Recall/F1 progression
    - Total search turns, unique queries, duplicate query rate, and zero-new-doc turn rate
    """
    cumulative_docs: Set[str] = set()
    per_turn_metrics: List[Dict[str, Any]] = []
    duplicate_turns = 0
    zero_new_doc_turns = 0

    for idx, turn in enumerate(turn_logs):
        turn_docs = [str(d) for d in turn.get("retrieved_doc_ids", [])]
        before_count = len(cumulative_docs)
        cumulative_docs.update(turn_docs)
        new_docs_added = len(cumulative_docs) - before_count

        if turn.get("is_duplicate_query", False):
            duplicate_turns += 1
        if new_docs_added == 0:
            zero_new_doc_turns += 1

        turn_hits = len(set(turn_docs) & relevant_doc_ids)
        cum_prf1 = compute_prf1(cumulative_docs, relevant_doc_ids)
        per_turn_metrics.append(
            {
                "turn_idx": idx + 1,
                "search_query": turn.get("search_query", ""),
                "is_duplicate_query": bool(turn.get("is_duplicate_query", False)),
                "retrieved_doc_ids": turn_docs,
                "relevant_hits_in_turn": turn_hits,
                "new_unique_docs_added": new_docs_added,
                "cumulative_unique_docs": cum_prf1["num_retrieved_unique"],
                "cumulative_relevant_hits": cum_prf1["num_relevant_hits"],
                "cumulative_precision": cum_prf1["precision"],
                "cumulative_recall": cum_prf1["recall"],
                "cumulative_f1": cum_prf1["f1"],
            }
        )

    union_prf1 = compute_prf1(cumulative_docs, relevant_doc_ids)
    num_turns = len(turn_logs)
    unique_queries = len({str(t.get("normalized_query", t.get("search_query", ""))).strip() for t in turn_logs})

    return {
        "union_precision": union_prf1["precision"],
        "union_recall": union_prf1["recall"],
        "union_f1": union_prf1["f1"],
        "num_retrieved_unique": union_prf1["num_retrieved_unique"],
        "num_relevant_hits": union_prf1["num_relevant_hits"],
        "num_gold_relevant": union_prf1["num_gold_relevant"],
        "num_search_turns": num_turns,
        "num_unique_queries": unique_queries,
        "duplicate_query_turns": duplicate_turns,
        "zero_new_doc_turns": zero_new_doc_turns,
        "duplicate_query_rate": round((duplicate_turns / num_turns * 100.0) if num_turns else 0.0, 2),
        "zero_new_doc_rate": round((zero_new_doc_turns / num_turns * 100.0) if num_turns else 0.0, 2),
        "per_turn": per_turn_metrics,
    }

