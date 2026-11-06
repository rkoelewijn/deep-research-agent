from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from src.evaluation.qa_metrics import extract_answer_tag
from src.retrieval.bm25_session import BM25SessionRetriever, RetrievedDocument
from src.search_r1.hooks import BaseAgentHooks
from src.search_r1.stopping import StopOnSubstringsCriteria

SEARCH_TAG_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def build_search_r1_initial_prompt(question: str) -> str:
    """
    Standard Search-R1 prompt template from Jin et al. (2025) & TRQA evaluation:
    Instructs the model to reason inside <think>...</think>, issue search queries
    inside <search>...</search>, read <information>...</information>, and output
    the final answer inside <answer>...</answer>.
    """
    q_text = question.strip()
    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between "
        "<information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer "
        "inside <answer> and </answer>, without detailed illustrations. "
        f"For example, <answer> Beijing </answer>. Question: {q_text}\n"
    )


def format_information_block(
    docs: List[RetrievedDocument],
    max_doc_chars: int = 1200,
) -> str:
    """Formats retrieved BM25 documents inside `<information>...</information>` tags."""
    if not docs:
        return "\n\n<information>No relevant documents found.</information>\n\n"
    formatted_passages = "\n".join(
        doc.format_for_prompt(display_idx=i + 1, max_chars=max_doc_chars)
        for i, doc in enumerate(docs)
    )
    return f"\n\n<information>{formatted_passages}</information>\n\n"


class SearchR1AgentRunner:
    """
    Executes the multi-turn Search-R1 agent loop paired with `BM25SessionRetriever`.
    Supports custom Phase 2 hooks via `BaseAgentHooks`.
    """

    def __init__(
        self,
        retriever: BM25SessionRetriever,
        model_name_or_path: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        dtype: str = "float16",
        top_k_per_search: int = 3,
        max_turns: int = 10,
        max_new_tokens_per_turn: int = 512,
        max_doc_chars: int = 1200,
        temperature: float = 0.0,
        do_sample: bool = False,
        hooks: Optional[BaseAgentHooks] = None,
        stop_strings: Optional[List[str]] = None,
    ) -> None:
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.retriever = retriever
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.top_k_per_search = top_k_per_search
        self.max_turns = max_turns
        self.max_new_tokens_per_turn = max_new_tokens_per_turn
        self.max_doc_chars = max_doc_chars
        self.temperature = temperature
        self.do_sample = do_sample
        self.hooks = hooks or BaseAgentHooks()
        self.stop_strings = stop_strings or ["</search>", "</answer>"]

    @torch.inference_mode()
    def _generate_step(self, trajectory_text: str) -> str:
        inputs = self.tokenizer(trajectory_text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        stopping_criteria = StoppingCriteriaList(
            [
                StopOnSubstringsCriteria(
                    tokenizer=self.tokenizer,
                    prompt_length=prompt_len,
                    stop_strings=self.stop_strings,
                )
            ]
        )

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens_per_turn,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
        }
        if self.do_sample and self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0, prompt_len:]
        step_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # Trim cleanly right after the first matched stop string
        earliest_end: Optional[int] = None
        for stop_s in self.stop_strings:
            pos = step_text.find(stop_s)
            if pos != -1:
                end_pos = pos + len(stop_s)
                if earliest_end is None or end_pos < earliest_end:
                    earliest_end = end_pos
        if earliest_end is not None:
            step_text = step_text[:earliest_end]

        return step_text

    def run_trajectory(self, query_id: str, question: str) -> Dict[str, Any]:
        """Runs a full multi-turn Search-R1 rollout for a single TRQA question."""
        self.retriever.reset_session(query_id)
        session = self.retriever.get_or_create_session(query_id)

        initial_prompt = build_search_r1_initial_prompt(question)
        if self.tokenizer.chat_template:
            trajectory = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": initial_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            trajectory = initial_prompt

        generated_trace = ""
        finish_reason = "max_turns_reached"

        for turn_idx in range(1, self.max_turns + 1):
            step_output = self._generate_step(trajectory + generated_trace)
            generated_trace += step_output

            # 1. Check if model produced a final <answer>...</answer>
            if ANSWER_TAG_RE.search(step_output):
                finish_reason = "answered"
                break

            # 2. Check if model issued a <search>...</search> action
            search_match = SEARCH_TAG_RE.search(step_output)
            if search_match:
                raw_query = search_match.group(1).strip()
                effective_query = self.hooks.pre_search_query_hook(
                    question=question,
                    raw_search_query=raw_query,
                    session=session,
                    turn_idx=turn_idx,
                )
                docs, turn_diag = self.retriever.session_search(
                    query_id=query_id,
                    search_query=effective_query,
                    k=self.top_k_per_search,
                )
                docs = self.hooks.post_retrieval_hook(
                    question=question,
                    search_query=effective_query,
                    retrieved_docs=docs,
                    turn_diagnostics=turn_diag,
                    session=session,
                    turn_idx=turn_idx,
                )
                info_block = format_information_block(docs, max_doc_chars=self.max_doc_chars)
                override_block = self.hooks.on_turn_end_controller_hook(
                    question=question,
                    information_block=info_block,
                    turn_diagnostics=turn_diag,
                    session=session,
                    turn_idx=turn_idx,
                )
                generated_trace += override_block if override_block is not None else info_block
            else:
                # Model stopped without emitting <search> or <answer>; prompt it to conclude
                if turn_idx == self.max_turns:
                    break
                generated_trace += "\nPlease provide your final answer inside <answer> and </answer>.\n"

        # If max turns reached without an <answer> tag, force a final answer generation
        if not ANSWER_TAG_RE.search(generated_trace):
            force_prompt = trajectory + generated_trace + "\n<answer>"
            final_step = self._generate_step(force_prompt)
            if "</answer>" not in final_step:
                final_step = final_step.strip() + "</answer>"
            generated_trace += "\n<answer>" + final_step

        predicted_answer = extract_answer_tag(generated_trace)
        return {
            "query_id": query_id,
            "predicted_answer": predicted_answer,
            "finish_reason": finish_reason,
            "generated_trace": generated_trace,
            "turn_logs": list(session.turn_logs),
            "unique_retrieved_doc_ids": sorted(session.unique_retrieved_doc_ids),
        }

