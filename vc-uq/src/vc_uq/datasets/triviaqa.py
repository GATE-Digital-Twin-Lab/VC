"""TriviaQA loader.

Short free-form answers with a single reference string, which is what both
correctness criteria need: ``e_cos`` against ``a_star`` and bidirectional
entailment against ``a_star``.

``datasets`` is imported lazily, and a deterministic synthetic stand-in is
provided so the pipeline is runnable end-to-end without a download. The
stand-in is labelled ``triviaqa_synthetic`` in the dataset column so it can
never be mistaken for real data in a results table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build(hf_name: str, hf_config: str, hf_split: str, n_questions: int,
          max_question_chars: int = 400, seed: int = 0) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        return build_synthetic(n_questions, seed=seed)

    ds = load_dataset(hf_name, hf_config, split=hf_split)
    rows = []
    for rec in ds:
        q = (rec.get("question") or "").strip()
        ans = rec.get("answer") or {}
        a_star = (ans.get("value") or ans.get("normalized_value") or "").strip()
        if not q or not a_star or len(q) > max_question_chars:
            continue
        rows.append({
            "q_id": f"tqa_{rec.get('question_id', len(rows))}",
            "dataset": "triviaqa",
            "question": q,
            "a_star": a_star,
            "p_q_true": None,       # unknown for real data; that is the whole problem
        })
        if len(rows) >= n_questions:
            break
    if not rows:
        raise ValueError(f"no usable rows from {hf_name}/{hf_config}:{hf_split}")
    return pd.DataFrame(rows)


_SUBJECTS = ["the longest river in", "the highest peak in", "the currency of",
             "the national bird of", "the largest lake in", "the oldest university in"]
_REGIONS = ["Peru", "Norway", "Kenya", "Vietnam", "Portugal", "Chile", "Latvia",
            "Morocco", "Nepal", "Uruguay", "Iceland", "Ghana"]


def build_synthetic(n_questions: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic placeholder with TriviaQA's *shape*, not its content."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_questions):
        subj = _SUBJECTS[int(rng.integers(len(_SUBJECTS)))]
        reg = _REGIONS[int(rng.integers(len(_REGIONS)))]
        rows.append({
            "q_id": f"tqs_{i:05d}",
            "dataset": "triviaqa_synthetic",
            "question": f"What is {subj} {reg}?",
            "a_star": f"reference answer {i:05d}",
            "p_q_true": None,
        })
    return pd.DataFrame(rows)
