from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class RetrievedDocument:
    """Single retrieved document from the Pyserini BM25 index."""

    doc_id: str
    score: float
    title: str
    text: str
    rank: int

    def format_for_prompt(self, display_idx: int, max_chars: int = 1200) -> str:
        """Formats document in Search-R1 / TRQA passage style."""
        snippet = self.text.strip()
        if max_chars and len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + " ..."
        if self.title:
            return f'Doc {display_idx}(Title: "{self.title}") {snippet}'
        return f"Doc {display_idx}: {snippet}"


@dataclass
class QuestionSessionState:
    """Tracks multi-turn retrieval state for a single question rollout."""

    query_id: str
    query_history: List[str] = field(default_factory=list)
    query_offsets: Dict[str, int] = field(default_factory=dict)
    unique_retrieved_doc_ids: Set[str] = field(default_factory=set)
    turn_logs: List[Dict[str, Any]] = field(default_factory=list)


class BM25SessionRetriever:
    """
    Stateful Python wrapper around Pyserini's LuceneSearcher (BM25).
    - Supports multi-threaded batch retrieval for Step 1.1 (Table 4) and Step 1.2a (K=50).
    - Caches top-N (default 100) BM25 hits per unique search string in memory.
    - Tracks per-question session state (query history, rank offsets, unique doc IDs)
      for Step 1.2b Search-R1 and future Phase 2 extensions.
    """

    def __init__(
        self,
        index_dir: str | Path,
        k1: float = 0.9,
        b: float = 0.4,
        session_cache_top_n: int = 100,
    ) -> None:
        from pyserini.search.lucene import LuceneSearcher

        self.index_dir = str(index_dir)
        if not Path(self.index_dir).exists():
            raise FileNotFoundError(
                f"BM25 Lucene index not found at '{self.index_dir}'. "
                "Run './scripts/download_data.sh' first."
            )
        self.searcher = LuceneSearcher(self.index_dir)
        self.searcher.set_bm25(k1=k1, b=b)
        self.k1 = k1
        self.b = b
        self.session_cache_top_n = session_cache_top_n

        # In-memory cache: normalized_query_str -> List[RetrievedDocument]
        self._query_cache: Dict[str, List[RetrievedDocument]] = {}
        # Per-question session state: query_id -> QuestionSessionState
        self._sessions: Dict[str, QuestionSessionState] = {}

    @staticmethod
    def _normalize_query(query_text: str) -> str:
        return " ".join(query_text.strip().split())

    def _parse_hit(self, hit: Any, rank: int) -> RetrievedDocument:
        doc_id = str(hit.docid)
        score = float(hit.score)
        raw_str = getattr(hit, "raw", None)
        if not raw_str:
            doc = self.searcher.doc(doc_id)
            raw_str = doc.raw() if doc is not None else ""

        title = ""
        text = raw_str or ""
        if raw_str:
            try:
                parsed = json.loads(raw_str)
                if isinstance(parsed, dict):
                    title = str(parsed.get("title", ""))
                    text = str(
                        parsed.get("contents")
                        or parsed.get("text")
                        or parsed.get("body")
                        or raw_str
                    )
                    # If contents starts with '"Title"\nText' or 'Title\nText', separate cleanly
                    if not title and "\n" in text:
                        first_line, rest = text.split("\n", 1)
                        title = first_line.strip(' "')
                        text = rest.strip()
            except json.JSONDecodeError:
                pass
        return RetrievedDocument(
            doc_id=doc_id,
            score=score,
            title=title,
            text=text,
            rank=rank,
        )

    def retrieve_top_n(self, query_text: str, top_n: Optional[int] = None) -> List[RetrievedDocument]:
        """Retrieves and caches top-N documents for a single query string."""
        limit = top_n or self.session_cache_top_n
        norm_q = self._normalize_query(query_text)
        if not norm_q:
            return []

        cached = self._query_cache.get(norm_q)
        if cached is not None and len(cached) >= limit:
            return cached[:limit]

        hits = self.searcher.search(norm_q, k=limit)
        docs = [self._parse_hit(hit, rank=i + 1) for i, hit in enumerate(hits)]
        self._query_cache[norm_q] = docs
        return docs

    def batch_retrieve(
        self,
        queries: List[str],
        qids: List[str],
        k: int = 1000,
        threads: int = 16,
        parse_text_top_m: int = 50,
    ) -> Dict[str, List[RetrievedDocument]]:
        """
        Multi-threaded batch search across a list of queries (used in Step 1.1).
        To keep memory low when k=1000, full document text is parsed for the top
        `parse_text_top_m` hits (e.g., 50 for Step 1.2a), while ranks > parse_text_top_m
        store doc_id and score for Recall@K evaluation.
        """
        raw_results = self.searcher.batch_search(
            queries=queries,
            qids=qids,
            k=k,
            threads=threads,
        )
        results: Dict[str, List[RetrievedDocument]] = {}
        for qid, hits in raw_results.items():
            doc_list: List[RetrievedDocument] = []
            for idx, hit in enumerate(hits):
                rank = idx + 1
                if rank <= parse_text_top_m:
                    doc_list.append(self._parse_hit(hit, rank=rank))
                else:
                    doc_list.append(
                        RetrievedDocument(
                            doc_id=str(hit.docid),
                            score=float(hit.score),
                            title="",
                            text="",
                            rank=rank,
                        )
                    )
            results[qid] = doc_list
        return results

    def get_or_create_session(self, query_id: str) -> QuestionSessionState:
        if query_id not in self._sessions:
            self._sessions[query_id] = QuestionSessionState(query_id=query_id)
        return self._sessions[query_id]

    def reset_session(self, query_id: str) -> None:
        self._sessions.pop(query_id, None)

    def session_search(
        self,
        query_id: str,
        search_query: str,
        k: int = 3,
        offset: Optional[int] = None,
        advance_offset: bool = False,
    ) -> Tuple[List[RetrievedDocument], Dict[str, Any]]:
        """
        Executes a stateful search within a question's multi-turn trajectory.
        - In Phase 1 standard Search-R1 (`advance_offset=False`), returns `results[0:k]`.
        - Supports pagination/offset slicing (`offset` or `advance_offset=True`) for Phase 2 hooks.
        Returns `(selected_docs, turn_diagnostics)`.
        """
        session = self.get_or_create_session(query_id)
        norm_q = self._normalize_query(search_query)
        is_duplicate_query = norm_q in [self._normalize_query(q) for q in session.query_history]

        start_idx = offset if offset is not None else (
            session.query_offsets.get(norm_q, 0) if advance_offset else 0
        )
        needed_top_n = max(self.session_cache_top_n, start_idx + k)
        all_docs = self.retrieve_top_n(norm_q, top_n=needed_top_n)
        selected_docs = all_docs[start_idx : start_idx + k]

        # Update session state
        session.query_history.append(search_query)
        session.query_offsets[norm_q] = start_idx + len(selected_docs)

        new_doc_ids = [
            d.doc_id for d in selected_docs if d.doc_id not in session.unique_retrieved_doc_ids
        ]
        for d in selected_docs:
            session.unique_retrieved_doc_ids.add(d.doc_id)

        diagnostics = {
            "turn_idx": len(session.turn_logs) + 1,
            "search_query": search_query,
            "normalized_query": norm_q,
            "is_duplicate_query": is_duplicate_query,
            "rank_start": start_idx + 1 if selected_docs else 0,
            "rank_end": start_idx + len(selected_docs),
            "retrieved_doc_ids": [d.doc_id for d in selected_docs],
            "new_unique_doc_ids": new_doc_ids,
            "num_new_unique_docs": len(new_doc_ids),
            "cumulative_unique_docs_count": len(session.unique_retrieved_doc_ids),
        }
        session.turn_logs.append(diagnostics)
        return selected_docs, diagnostics

