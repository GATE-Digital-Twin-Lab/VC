"""Table schemas (protocol section 2).

Two tables carry the study: one row per sampled answer, one row per question.
Declaring them explicitly means a phase that forgets to populate a column fails
loudly at write time instead of producing a silently-empty analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SPLITS = ("tau_select", "classify", "calib", "eval")

# column -> (pandas dtype, nullable, note)
ANSWERS_SCHEMA: dict[str, tuple[str, bool]] = {
    "q_id": ("string", False),
    "dataset": ("string", False),
    "split": ("string", False),
    "draw_idx": ("int32", False),
    "answer": ("string", False),
    "vc_post": ("Float64", True),          # nullable: parse failure is a finding
    "vc_post_raw": ("string", True),       # kept for the parse audit
    "h_tok_mean": ("Float64", True),       # full-vocab entropy, VC-augmented prompt
    "h_tok_max": ("Float64", True),
    "h_tok_mean_clean": ("Float64", True), # teacher-forced under a clean prompt
    "logp_mean": ("Float64", True),
    "min_token_p": ("Float64", True),
    "n_tokens": ("Int32", True),
    "temperature": ("Float64", False),
    "prompt_variant": ("string", False),
    "seed": ("Int64", False),
    "model": ("string", False),
    "cluster_id": ("Int32", True),         # filled by cluster.py
    "e_cos": ("Float64", True),            # 1 - cos(emb(a_i), emb(a_star)) -- CRITERION
    "s_anchor": ("Float64", True),         # 1 - cos(emb(anchor(q)), emb(a_i)) -- SCORE
    "s_anchor_rank": ("Float64", True),    # within-split rank transform of s_anchor
    "correct_cos": ("boolean", True),
    "correct_nli": ("boolean", True),
    "correct_human": ("boolean", True),    # Phase 0 subset only
}

QUESTIONS_SCHEMA: dict[str, tuple[str, bool]] = {
    "q_id": ("string", False),
    "dataset": ("string", False),
    "split": ("string", False),
    "question": ("string", False),
    "a_star": ("string", False),
    "p_q_true": ("Float64", True),         # known only for constructed items (fabricated: 0)
    "vc_pre": ("Float64", True),           # mean over R_pre repeats
    "vc_pre_sd": ("Float64", True),        # spread across repeats -- 5.6(c)
    "vc_pre_raw": ("string", True),
    "vc_1": ("Float64", True),
    "vc_bar": ("Float64", True),
    "vc_sd": ("Float64", True),            # ~0 means VC is a property of the prompt
    "p_hat": ("Float64", True),
    "n_draws": ("Int32", True),
    "K_q": ("Int32", True),                # -1 when censored
    "censored": ("boolean", True),
    "in_U": ("boolean", True),
    "n_clusters": ("Int32", True),
    "H_sem": ("Float64", True),
    "largest_cluster_share": ("Float64", True),
    "answer_len_mean": ("Float64", True),  # confounder -- must be controlled
    "anchor": ("string", True),
}

VC_PRE_REPEATS_SCHEMA: dict[str, tuple[str, bool]] = {
    "q_id": ("string", False),
    "dataset": ("string", False),
    "repeat_idx": ("int32", False),
    "vc_pre": ("Float64", True),
    "vc_pre_raw": ("string", True),
    "prompt_variant": ("string", False),
    "seed": ("Int64", False),
    "model": ("string", False),
}

# Generation cache key (protocol section 1). Uniqueness is enforced on write.
ANSWER_KEY = ("model", "dataset", "q_id", "draw_idx", "temperature", "prompt_variant", "seed")
VC_PRE_KEY = ("model", "dataset", "q_id", "repeat_idx", "prompt_variant", "seed")


def empty_frame(schema: dict[str, tuple[str, bool]]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=d) for c, (d, _) in schema.items()})


def conform(df: pd.DataFrame, schema: dict[str, tuple[str, bool]],
            *, name: str = "frame") -> pd.DataFrame:
    """Add declared-but-missing columns as nulls, cast dtypes, order columns.

    Raises if a non-nullable column is absent or contains nulls -- a phase that
    silently drops ``split`` or ``q_id`` would corrupt every guarantee
    downstream, so this is deliberately strict.
    """
    out = df.copy()
    for col, (dtype, nullable) in schema.items():
        if col not in out.columns:
            if not nullable:
                raise ValueError(f"{name}: required column {col!r} is missing")
            out[col] = pd.Series([pd.NA] * len(out), dtype=dtype)
        try:
            out[col] = out[col].astype(dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: column {col!r} will not cast to {dtype}: {exc}") from exc
        if not nullable and len(out) and out[col].isna().any():
            raise ValueError(f"{name}: non-nullable column {col!r} contains nulls")
    extra = [c for c in out.columns if c not in schema]
    return out[list(schema.keys()) + extra]


def check_unique_key(df: pd.DataFrame, key: tuple[str, ...], *, name: str) -> None:
    dup = df.duplicated(subset=list(key), keep=False)
    if dup.any():
        example = df.loc[dup, list(key)].head(3).to_dict("records")
        raise ValueError(
            f"{name}: {int(dup.sum())} rows share a cache key {key}. "
            f"Examples: {example}"
        )


def assert_split_by_question(answers: pd.DataFrame) -> None:
    """Splits are by question, never by draw (protocol section 4)."""
    per_q = answers.groupby("q_id")["split"].nunique()
    bad = per_q[per_q > 1]
    if len(bad):
        raise ValueError(
            f"{len(bad)} q_ids appear in more than one split; splits must be by question. "
            f"Examples: {list(bad.index[:5])}"
        )


def summarise(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "questions": int(df["q_id"].nunique()) if "q_id" in df else 0,
        "by_split": df["split"].value_counts().to_dict() if "split" in df else {},
        "by_dataset": df["dataset"].value_counts().to_dict() if "dataset" in df else {},
    }
