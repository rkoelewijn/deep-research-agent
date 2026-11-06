from __future__ import annotations

import math
import re
import string
from typing import Any, Dict, Iterable, List, Optional

WORD_TO_NUM = {
    "zero": 0.0,
    "none": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
}

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_answer_tag(text: str) -> str:
    """
    Extracts the final answer from `<answer>...</answer>`, `\\boxed{...}`,
    or falls back to the last non-empty line of the generated text.
    """
    if not text:
        return ""
    matches = ANSWER_TAG_RE.findall(text)
    if matches:
        return matches[-1].strip()
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize_text(s: Any) -> str:
    """Standard QA string normalization (lowercase, strip punctuation/articles/whitespace)."""
    text = str(s).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return " ".join(text.split())


def parse_numeric_value(val: Any) -> Optional[float]:
    """
    Parses a numeric value from a gold or predicted answer string.
    Handles integers, floats, comma-separated numbers (e.g., '1,250.5'),
    and common English number words ('zero'..'twenty').
    """
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return None if math.isnan(float(val)) else float(val)

    s = str(val).strip().lower()
    if not s:
        return None

    # Check direct word match
    clean_word = re.sub(r"[^a-z]", "", s)
    if clean_word in WORD_TO_NUM:
        return WORD_TO_NUM[clean_word]

    # Extract last numeric token in the extracted answer string
    matches = NUMBER_RE.findall(s)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except ValueError:
            pass

    # Fallback: check if any number word appears in the string
    for word, num in WORD_TO_NUM.items():
        if re.search(rf"\b{word}\b", s):
            return num

    return None


def evaluate_qa_prediction(
    prediction: Any,
    gold_answer: Any,
    sm_thresholds: Iterable[int] = (10, 20, 50, 90),
) -> Dict[str, float]:
    """
    Computes TRQA End-to-End QA metrics for a single query:
    - EM (Exact Match, in percentage 0.0 or 100.0)
    - SM_{X}% (Soft Match at tolerance X%, in percentage 0.0 or 100.0):
      |(A_pred - A_gold) / A_gold| * 100 <= X%
    """
    gold_candidates = gold_answer if isinstance(gold_answer, list) else [gold_answer]
    pred_str = str(prediction).strip()
    pred_norm = normalize_text(pred_str)
    pred_num = parse_numeric_value(pred_str)

    best_em = 0.0
    best_sm: Dict[int, float] = {t: 0.0 for t in sm_thresholds}
    min_rel_error: Optional[float] = None

    for g in gold_candidates:
        g_norm = normalize_text(g)
        g_num = parse_numeric_value(g)

        # 1. Exact Match (either normalized string match or exact numerical equality)
        is_em = False
        if pred_norm and g_norm and pred_norm == g_norm:
            is_em = True
        elif pred_num is not None and g_num is not None and math.isclose(pred_num, g_num, rel_tol=1e-6, abs_tol=1e-6):
            is_em = True

        if is_em:
            best_em = 100.0
            for t in sm_thresholds:
                best_sm[t] = 100.0
            min_rel_error = 0.0
            break

        # 2. Soft Match (SM_X%) for numerical targets
        if pred_num is not None and g_num is not None:
            if abs(g_num) < 1e-12:
                rel_error_pct = 0.0 if abs(pred_num) < 1e-12 else float("inf")
            else:
                rel_error_pct = abs(pred_num - g_num) / abs(g_num) * 100.0

            if min_rel_error is None or rel_error_pct < min_rel_error:
                min_rel_error = rel_error_pct

            for t in sm_thresholds:
                if rel_error_pct <= float(t) + 1e-9:
                    best_sm[t] = 100.0

    metrics: Dict[str, float] = {"EM": best_em}
    for t in sm_thresholds:
        metrics[f"SM_{t}%"] = best_sm[t]
    if min_rel_error is not None and not math.isinf(min_rel_error):
        metrics["rel_error_pct"] = min_rel_error
    return metrics


def evaluate_qa_split(
    records: List[Dict[str, Any]],
    sm_thresholds: Iterable[int] = (10, 20, 50, 90),
) -> Dict[str, float]:
    """Macro-averages EM and SM_{X}% across all evaluated query records in a split."""
    if not records:
        return {"EM": 0.0, **{f"SM_{t}%": 0.0 for t in sm_thresholds}, "count": 0}

    totals: Dict[str, float] = {"EM": 0.0, **{f"SM_{t}%": 0.0 for t in sm_thresholds}}
    for rec in records:
        m = rec.get("qa_metrics") or evaluate_qa_prediction(
            rec.get("predicted_answer", ""),
            rec.get("gold_answer", ""),
            sm_thresholds=sm_thresholds,
        )
        totals["EM"] += float(m.get("EM", 0.0))
        for t in sm_thresholds:
            key = f"SM_{t}%"
            totals[key] += float(m.get(key, 0.0))

    n = len(records)
    summary = {k: round(v / n, 2) for k, v in totals.items()}
    summary["count"] = n
    return summary

