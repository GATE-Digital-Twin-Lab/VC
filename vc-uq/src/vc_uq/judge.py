"""Correctness criterion (``e``) and label-free score (``s``).

These are different functions and must never be merged (protocol section 0):

  ``e(a_i) = 1 - cos(emb(a_i), emb(a_star))``
      ``a_star`` is the REFERENCE and ``a_i`` the argument. On a new question
      there is nothing to sweep, so ``e`` cannot be evaluated at test time at
      all. It is a correctness criterion, confined to evaluation.

  ``s(q, a) = 1 - cos(emb(anchor(q)), emb(a))``
      ``anchor(q)`` is the reference and ``a`` is swept. At calibration you
      evaluate at ``a = a_star``; at test you sweep candidates. Same operation,
      no label. A valid nonconformity score.

Using ``e`` as a score would be undefined at deployment; thresholding the same
function for both set membership and correctness would make every retained
answer correct by construction and drive the loss identically to zero. The
circularity guard below refuses that configuration outright.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backends import build_embedder, build_nli
from .backends.base import cosine_distance, pairwise_cosine_distance
from .config import Config


class CircularityError(ValueError):
    """Set membership and correctness were defined by the same function."""


def guard_against_circularity(membership_score: str, correctness_score: str) -> None:
    if membership_score == correctness_score:
        raise CircularityError(
            f"membership and correctness both thresholded on {membership_score!r}. "
            "Every retained answer would be correct by construction and the risk "
            "would be identically zero. Membership must use s_anchor (label-free); "
            "correctness must use e_cos or NLI (a_star-anchored)."
        )


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

@dataclass
class EmbeddingSpace:
    """Embeddings for one split, with anisotropy correction applied jointly.

    Correction is fitted on the split as a whole rather than per question:
    unrelated English sentences sit at cosine 0.3-0.6 and a fixed domain can
    compress everything into 0.75-0.95, so the common component has to be
    removed from the same population that both ``e`` and ``s`` are computed in.
    """

    vectors: dict[str, np.ndarray]
    dim: int
    mean_centered: bool
    whitened: bool

    def get(self, texts) -> np.ndarray:
        return np.vstack([self.vectors[t] for t in texts])


def build_embedding_space(cfg: Config, texts, embedder=None) -> EmbeddingSpace:
    embedder = embedder if embedder is not None else build_embedder(cfg)
    uniq = list(dict.fromkeys(str(t) for t in texts))
    mat = np.asarray(embedder.embed(uniq), dtype=np.float64)
    mean_center = bool(cfg.get("embedding.postprocess.mean_center"))
    whiten = bool(cfg.get("embedding.postprocess.whiten"))

    if mean_center:
        mat = mat - mat.mean(axis=0, keepdims=True)
    if whiten:
        cov = np.cov(mat, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-8, None)
        mat = mat @ vecs @ np.diag(vals ** -0.5) @ vecs.T
    return EmbeddingSpace(vectors=dict(zip(uniq, mat)), dim=mat.shape[1],
                          mean_centered=mean_center, whitened=whiten)


# --------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------

def medoid_anchor(answers: list[str], space: EmbeddingSpace) -> str:
    """Medoid of the N draws -- deterministic given the draws.

    Preferred over a greedy decode because the radius around the medoid *is*
    semantic dispersion, so ``s`` doubles as the diversity measure used in 6.6
    instead of being computed separately.
    """
    if not answers:
        raise ValueError("cannot build an anchor from zero draws")
    if len(answers) == 1:
        return answers[0]
    mat = space.get(answers)
    d = pairwise_cosine_distance(mat)
    return answers[int(np.argmin(d.sum(axis=1)))]


def greedy_anchor(lm, cfg: Config, question_row: pd.Series) -> str:
    """T=0 decode. Also deterministic, and independent of the draw set."""
    from . import prompts
    from .generate import _mock_meta, derive_seed
    spec = prompts.get(cfg.get("generation.prompt_variant_clean"))
    msgs = spec.build(question=question_row["question"])
    if cfg.get("model.backend") == "mock":
        msgs = _mock_meta(msgs, {"qid": question_row["q_id"],
                                 "ds": question_row["dataset"],
                                 "variant": spec.variant, "scale": spec.scale,
                                 "kind": "post"})
    from .backends.base import SamplingParams
    params = SamplingParams.from_config(
        cfg, temperature=float(cfg.get("anchor.greedy_temperature")),
        max_tokens=int(cfg.get("anchor.greedy_max_tokens")))
    gen = lm.generate(msgs, params=params,
                      seed=derive_seed(int(cfg.get("run.seed")), "anchor",
                                       question_row["q_id"]))
    from .parsing import parse_answer_and_vc
    return parse_answer_and_vc(gen.text, spec.scale).answer


def build_anchors(cfg: Config, answers: pd.DataFrame, questions: pd.DataFrame,
                  space: EmbeddingSpace, lm=None) -> pd.Series:
    """One anchor per question, built IDENTICALLY at calibration and test."""
    method = cfg.get("anchor.method")
    if method == "medoid":
        return (answers.sort_values("draw_idx").groupby("q_id")["answer"]
                .apply(lambda s: medoid_anchor(list(s), space)))
    if method == "greedy":
        if lm is None:
            from .backends import build_lm
            lm = build_lm(cfg)
        idx = questions.set_index("q_id")
        return pd.Series({q: greedy_anchor(lm, cfg, idx.loc[q]) for q in idx.index})
    raise ValueError(f"unknown anchor.method {method!r}; must be medoid or greedy")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_answers(cfg: Config, answers: pd.DataFrame, questions: pd.DataFrame,
                  *, embedder=None, lm=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Populate ``e_cos``, ``s_anchor``, ``s_anchor_rank`` and the anchor column."""
    q_idx = questions.set_index("q_id")
    texts = list(answers["answer"].astype(str)) + list(q_idx["a_star"].astype(str))
    space = build_embedding_space(cfg, texts, embedder=embedder)

    anchors = build_anchors(cfg, answers, questions, space, lm=lm)
    missing = [t for t in anchors.values if t not in space.vectors]
    if missing:
        space = build_embedding_space(cfg, texts + missing, embedder=embedder)

    out = answers.copy()
    a_star_texts = out["q_id"].map(q_idx["a_star"].astype(str))
    anchor_texts = out["q_id"].map(anchors)

    emb_a = space.get(out["answer"].astype(str))
    emb_star = space.get(a_star_texts)
    emb_anchor = space.get(anchor_texts)

    out["e_cos"] = cosine_distance(emb_a, emb_star)
    out["s_anchor"] = cosine_distance(emb_anchor, emb_a)

    if bool(cfg.get("embedding.postprocess.rank_transform_s")):
        # Only the ordering of s matters downstream, and ranking within a split
        # restores dynamic range that anisotropy compresses away.
        out["s_anchor_rank"] = out.groupby("split")["s_anchor"].rank(pct=True)
    else:
        out["s_anchor_rank"] = out["s_anchor"]

    questions_out = questions.copy()
    questions_out["anchor"] = questions_out["q_id"].map(anchors)
    return out, questions_out


def apply_tau(answers: pd.DataFrame, tau: float,
              column: str = "e_cos") -> pd.DataFrame:
    out = answers.copy()
    out["correct_cos"] = out[column] <= float(tau)
    return out


def judge_nli(cfg: Config, answers: pd.DataFrame, questions: pd.DataFrame,
              nli=None) -> pd.DataFrame:
    """Bidirectional entailment against ``a_star``.

    Bidirectional, not one-way: "Paris" entails "a city in France" but the two
    are not the same answer, and a one-way test would accept the weaker string.
    """
    nli = nli if nli is not None else build_nli(cfg)
    thr = float(cfg.get("nli.entail_threshold"))
    q_idx = questions.set_index("q_id")["a_star"].astype(str)
    out = answers.copy()
    verdicts = []
    for q_id, ans in zip(out["q_id"], out["answer"].astype(str)):
        ref = q_idx.get(q_id, "")
        fwd = nli.entailment_prob(ans, ref)
        bwd = nli.entailment_prob(ref, ans)
        verdicts.append(bool(min(fwd, bwd) >= thr))
    out["correct_nli"] = verdicts
    return out


def correctness_column(cfg: Config) -> str:
    primary = cfg.get("judge.primary")
    if primary not in ("cos", "nli"):
        raise ValueError(f"judge.primary must be cos or nli, got {primary!r}")
    return "correct_cos" if primary == "cos" else "correct_nli"


def attach_correct(cfg: Config, answers: pd.DataFrame) -> pd.DataFrame:
    """Materialise the single ``correct`` column every phase reads."""
    col = correctness_column(cfg)
    if col not in answers.columns or answers[col].isna().all():
        raise ValueError(
            f"{col} is empty. Phase 0 must select tau_star (or flip judge.primary "
            "to nli) before any downstream phase can define correctness."
        )
    out = answers.copy()
    out["correct"] = out[col].astype("boolean").fillna(False).astype(bool)
    return out
