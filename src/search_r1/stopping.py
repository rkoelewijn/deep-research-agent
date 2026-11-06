from __future__ import annotations

from typing import List

import torch
from transformers import PreTrainedTokenizerBase, StoppingCriteria


class StopOnSubstringsCriteria(StoppingCriteria):
    """
    Stops token generation immediately when the newly generated text suffix
    contains any of the target stop strings (e.g., `</search>` or `</answer>`).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_length: int,
        stop_strings: List[str],
        check_window_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_strings = stop_strings
        self.check_window_tokens = check_window_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_len = input_ids.shape[1] - self.prompt_length
        if generated_len <= 0:
            return False
        window_start = max(self.prompt_length, input_ids.shape[1] - self.check_window_tokens)
        tail_ids = input_ids[0, window_start:]
        tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=False)
        return any(stop_str in tail_text for stop_str in self.stop_strings)

