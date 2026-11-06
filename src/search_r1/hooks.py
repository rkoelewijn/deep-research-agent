from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.retrieval.bm25_session import QuestionSessionState, RetrievedDocument


class BaseAgentHooks:
    """
    Modular extension hooks for Search-R1 trajectories.
    - In Phase 1 (Reproducibility), all hooks act as transparent pass-throughs.
    - In Phase 2 (Build Your Own Method), subclass `BaseAgentHooks` to intervene on:
      1. `pre_search_query_hook`: Rewrite, expand, or diversify `<search>` queries before BM25.
      2. `post_retrieval_hook`: Filter, deduplicate, rerank, or paginate retrieved BM25 passages.
      3. `on_turn_end_controller_hook`: Detect stagnation (e.g. 0 new docs added) and inject
         controller hints or reasoning prompts before the next turn.
    """

    def pre_search_query_hook(
        self,
        question: str,
        raw_search_query: str,
        session: QuestionSessionState,
        turn_idx: int,
    ) -> str:
        """Called right after `<search>...</search>` is parsed and before querying BM25."""
        return raw_search_query

    def post_retrieval_hook(
        self,
        question: str,
        search_query: str,
        retrieved_docs: List[RetrievedDocument],
        turn_diagnostics: Dict[str, Any],
        session: QuestionSessionState,
        turn_idx: int,
    ) -> List[RetrievedDocument]:
        """Called after BM25 returns top-k documents and before formatting `<information>`."""
        return retrieved_docs

    def on_turn_end_controller_hook(
        self,
        question: str,
        information_block: str,
        turn_diagnostics: Dict[str, Any],
        session: QuestionSessionState,
        turn_idx: int,
    ) -> Optional[str]:
        """
        Optional controller hook to append a guidance message or modify `<information>...</information>`
        when stagnation is detected (e.g., `turn_diagnostics['num_new_unique_docs'] == 0`).
        Returns None in Phase 1 (no modification).
        """
        return None

