"""Semantic clustering by bidirectional entailment.

Cosine is not usable here. It is blind to negation -- "Paris" and "not Paris"
score around 0.9 -- so a cosine-clustered study would merge a right and a wrong
answer into one cluster and silently destroy the diversity 2x2 that the whole of
6.6 rests on. Entailment must hold in BOTH directions for two answers to be
called the same.

From the clusters come the sample-diversity baseline family: answer frequency
``f``, semantic entropy ``H_sem``, and the largest-cluster share used by the
self-consistency stopping rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backends import build_nli
from .config import Config


def cluster_answers(answers: list[str], nli, threshold: float = 0.5) -> list[int]:
    """Greedy bidirectional-entailment clustering, in draw order.

    Order-dependent by construction, so draws are clustered in a fixed order
    (``draw_idx``) to keep the assignment reproducible. Each answer is compared
    against one representative per cluster, which keeps this O(n * n_clusters)
    rather than O(n^2) -- the difference matters at 40 draws x thousands of
    questions with an encoder in the loop.
    """
    reps: list[str] = []
    assignment: list[int] = []
    for ans in answers:
        placed = False
        for cid, rep in enumerate(reps):
            if ans == rep:
                assignment.append(cid)
                placed = True
                break
            fwd = nli.entailment_prob(rep, ans)
            bwd = nli.entailment_prob(ans, rep)
            if min(fwd, bwd) >= threshold:
                assignment.append(cid)
                placed = True
                break
        if not placed:
            reps.append(ans)
            assignment.append(len(reps) - 1)
    return assignment


def semantic_entropy(cluster_ids) -> float:
    """Entropy over cluster frequencies (nats).

    The discrete form: clusters are the semantic units, so the entropy is over
    their empirical frequencies rather than over token sequences.
    """
    ids = np.asarray(list(cluster_ids))
    if ids.size == 0:
        return float("nan")
    _, counts = np.unique(ids, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def cluster_frame(cfg: Config, answers: pd.DataFrame, nli=None) -> pd.DataFrame:
    """Assign ``cluster_id`` per question and attach the frequency signal ``f``."""
    nli = nli if nli is not None else build_nli(cfg)
    thr = float(cfg.get("nli.entail_threshold"))
    out = answers.sort_values(["q_id", "draw_idx"]).copy()
    ids: list[int] = []
    for _, g in out.groupby("q_id", sort=False):
        ids.extend(cluster_answers(list(g["answer"].astype(str)), nli, thr))
    out["cluster_id"] = ids
    counts = out.groupby(["q_id", "cluster_id"])["draw_idx"].transform("size")
    sizes = out.groupby("q_id")["draw_idx"].transform("size")
    out["f"] = counts / sizes          # self-consistency of this answer's cluster
    return out


def question_diversity(answers: pd.DataFrame) -> pd.DataFrame:
    """Per-question diversity summary: n_clusters, H_sem, largest cluster share."""
    rows = []
    for q_id, g in answers.groupby("q_id"):
        ids = g["cluster_id"].dropna().astype(int)
        if ids.empty:
            rows.append({"q_id": q_id, "n_clusters": 0, "H_sem": np.nan,
                         "largest_cluster_share": np.nan})
            continue
        _, counts = np.unique(ids, return_counts=True)
        rows.append({
            "q_id": q_id,
            "n_clusters": int(len(counts)),
            "H_sem": semantic_entropy(ids),
            "largest_cluster_share": float(counts.max() / counts.sum()),
        })
    return pd.DataFrame(rows)


def prefix_semantic_entropy(cluster_ids: list[int]) -> list[float]:
    """H_sem over the first k draws, for k = 1..n.

    The semantic-entropy stopping rule has to decide after each draw using only
    the draws it has seen, so it needs the running value, not the final one.
    """
    out = []
    for k in range(1, len(cluster_ids) + 1):
        out.append(semantic_entropy(cluster_ids[:k]))
    return out


def prefix_largest_share(cluster_ids: list[int]) -> list[float]:
    out = []
    for k in range(1, len(cluster_ids) + 1):
        _, counts = np.unique(np.asarray(cluster_ids[:k]), return_counts=True)
        out.append(float(counts.max() / counts.sum()))
    return out
