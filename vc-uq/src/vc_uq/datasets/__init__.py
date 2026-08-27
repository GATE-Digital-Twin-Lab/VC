"""Dataset assembly and the four-way split."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..config import Config
from ..schemas import SPLITS

_SEM_SUFFIX = " [[sem:{q_id}|gold]]"


def build_questions(cfg: Config) -> pd.DataFrame:
    frames = []
    for name in cfg.get("dataset.datasets"):
        if name == "fabricated":
            from . import fabricated
            frames.append(fabricated.build(
                n_questions=cfg.get("dataset.fabricated.n_questions"),
                seed=cfg.get("dataset.fabricated.seed"),
            ))
        elif name == "triviaqa":
            from . import triviaqa
            frames.append(triviaqa.build(
                hf_name=cfg.get("dataset.triviaqa.hf_name"),
                hf_config=cfg.get("dataset.triviaqa.hf_config"),
                hf_split=cfg.get("dataset.triviaqa.hf_split"),
                n_questions=cfg.get("dataset.triviaqa.n_questions"),
                max_question_chars=cfg.get("dataset.triviaqa.max_question_chars"),
                seed=cfg.get("run.seed"),
            ))
        else:
            raise ValueError(f"unknown dataset {name!r}")
    df = pd.concat(frames, ignore_index=True)

    if cfg.get("model.backend") == "mock":
        # The simulated world identifies answers by a semantic tag; the reference
        # answer needs the same tag so the mock embedder can score against it.
        # Real datasets are never touched by this.
        df["a_star"] = df.apply(
            lambda r: r["a_star"] + _SEM_SUFFIX.format(q_id=r["q_id"]), axis=1)

    df = assign_splits(df, cfg)
    return df


def _stable_u01(q_id: str, salt: str) -> float:
    h = hashlib.blake2b(f"{salt}::{q_id}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big") / 2**64


def assign_splits(questions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Four-way split BY QUESTION (protocol section 4), never by draw.

    Assignment is a hash of ``q_id``, so it is stable across runs and across
    machines: adding questions later cannot reshuffle the ones already
    generated, which would silently invalidate a calibration set.
    """
    fracs = {s: float(cfg.get(f"dataset.splits.{s}")) for s in SPLITS}
    total = sum(fracs.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"dataset.splits must sum to 1, got {total}")
    salt = str(cfg.get("run.seed"))
    stratify = cfg.get("dataset.splits.stratify_by", None)

    edges = np.cumsum([fracs[s] for s in SPLITS])
    out = questions.copy()

    def _assign(q_id: str, group: str) -> str:
        u = _stable_u01(q_id, f"{salt}|{group}")
        return SPLITS[int(np.searchsorted(edges, u, side="right").clip(0, len(SPLITS) - 1))]

    groups = out[stratify] if stratify and stratify in out.columns else pd.Series(
        [""] * len(out), index=out.index)
    out["split"] = [_assign(q, g) for q, g in zip(out["q_id"], groups)]
    return out


def split_summary(questions: pd.DataFrame) -> pd.DataFrame:
    return (questions.groupby(["dataset", "split"]).size()
            .rename("n_questions").reset_index()
            .pivot(index="dataset", columns="split", values="n_questions")
            .fillna(0).astype(int).reset_index())
