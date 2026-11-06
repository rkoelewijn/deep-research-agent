from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.qa_metrics import extract_answer_tag
from src.retrieval.bm25_session import RetrievedDocument

SINGLE_STEP_SYSTEM_PROMPT = (
    "You are a helpful and precise research assistant. "
    "You are given a question and a set of retrieved Wikipedia passages. "
    "Carefully examine all relevant entities and facts across the provided passages, "
    "perform the required reasoning or numerical aggregation step-by-step inside <think>...</think> tags, "
    "and provide ONLY the final concise numerical or entity answer inside <answer>...</answer> tags."
)


def build_single_step_prompt(
    question: str,
    docs: List[RetrievedDocument],
    max_doc_chars: int = 1200,
) -> List[Dict[str, str]]:
    """Constructs the chat messages for Single-Step RAG (K=50) evaluation."""
    formatted_docs = "\n\n".join(
        doc.format_for_prompt(display_idx=i + 1, max_chars=max_doc_chars)
        for i, doc in enumerate(docs)
    )
    user_content = (
        f"Retrieved Passages:\n{formatted_docs}\n\n"
        f"Question: {question}\n\n"
        "Reason step-by-step over the passages inside <think>...</think> and output your final "
        "answer inside <answer>...</answer>."
    )
    return [
        {"role": "system", "content": SINGLE_STEP_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class SingleStepRAGRunner:
    """
    Runs Single-Step Retriever + Qwen2.5-7B-Instruct (Table 8 Baseline, K=50).
    Uses `torch.float16` and `device_map="auto"` across available GPUs (e.g. 2x RTX 2080 Ti).
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        dtype: str = "float16",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        max_doc_chars: int = 1200,
    ) -> None:
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_doc_chars = max_doc_chars

    @torch.inference_mode()
    def answer_question(
        self,
        question: str,
        docs: List[RetrievedDocument],
    ) -> Dict[str, Any]:
        messages = build_single_step_prompt(
            question=question,
            docs=docs,
            max_doc_chars=self.max_doc_chars,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample and self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = output_ids[0][input_len:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_answer = extract_answer_tag(response_text)

        return {
            "response_text": response_text,
            "predicted_answer": predicted_answer,
            "prompt_tokens": int(input_len),
            "generated_tokens": int(generated_ids.shape[0]),
        }

