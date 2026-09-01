"""Semantic clustering by bidirectional entailment.

Cosine is not usable here. It is blind to negation -- "Paris" and "not Paris"
score around 0.9 -- so a cosine-clustered study would merge a right and a wrong
answer into one cluster and silently destroy the diversity 2x2 that the whole of
6.6 rests on. Entailment must hold in BOTH directions for two answers to be
called the same.

From the clusters comes the sample-diversity baseline family: answer frequency
``f`` per answer, and ``H_sem`` plus the largest-cluster share per question.

These are DESCRIPTIVE comparators, not stopping rules. A diversity statistic
computed after one draw is not a low value, it is not a value -- one draw is one
cluster, so H_sem reads 0 and largest-share reads 1, the exact values a "stop
when converged" rule treats as convergence. As online rules they would fire on
the first draw at every threshold; as descriptions of a completed set of N draws
they are exactly what 6.6 needs. So ``f`` enters the Phase 2 AUROC comparison
and the per-question summaries feed the diversity 2x2, and neither appears in
``phase4.stop_rules``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backends import build_nli
from .config import Config


def _match_pairwise(ans: str, reps: list[str], nli, threshold: float) -> int | None:
    """One judge call per direction per representative, short-circuiting."""
    for cid, rep in enumerate(reps):
        if ans == rep:
            return cid
        fwd = nli.entailment_prob(rep, ans)
        bwd = nli.entailment_prob(ans, rep)
        if min(fwd, bwd) >= threshold:
            return cid
    return None


def _match_batched(ans: str, reps: list[str], batch, threshold: float) -> int | None:
    """Both directions against every representative in a single judge call.

    Returns exactly what :func:`_match_pairwise` returns -- the FIRST
    representative that either matches exactly or clears the threshold in both
    directions -- so the greedy assignment is unchanged. What changes is the
    call pattern: an encoder head run at batch size 1 wastes almost all of a
    GPU, and clustering makes on the order of half a million comparisons across
    the corpus.

    The trade is deliberate. This scores every representative instead of
    stopping at the first hit, so it runs MORE forward passes in far fewer
    calls. On an accelerator that is a large net win, because a batch of 2k
    costs about what a batch of 1 costs.
    """
    exact = {i for i, rep in enumerate(reps) if ans == rep}
    todo = [i for i in range(len(reps)) if i not in exact]
    scores: dict[int, float] = {}
    if todo:
        premises = [reps[i] for i in todo] + [ans] * len(todo)
        hypotheses = [ans] * len(todo) + [reps[i] for i in todo]
        probs = list(batch(premises, hypotheses))
        n = len(todo)
        for k, i in enumerate(todo):
            scores[i] = min(float(probs[k]), float(probs[n + k]))
    for cid in range(len(reps)):
        if cid in exact or scores.get(cid, -1.0) >= threshold:
            return cid
    return None


def cluster_answers(answers: list[str], nli, threshold: float = 0.5) -> list[int]:
    """Greedy bidirectional-entailment clustering, in draw order.

    Order-dependent by construction, so draws are clustered in a fixed order
    (``draw_idx``) to keep the assignment reproducible. Each answer is compared
    against one representative per cluster, which keeps this O(n * n_clusters)
    rather than O(n^2) -- the difference matters at 40 draws x thousands of
    questions with an encoder in the loop.

    The outer loop cannot be batched: each assignment depends on the clusters
    the previous answers created. The INNER loop can be, and a judge exposing
    ``entailment_probs`` takes that path. Both paths produce the same
    assignment; a test pins that.
    """
    batch = getattr(nli, "entailment_probs", None)
    reps: list[str] = []
    assignment: list[int] = []
    for ans in answers:
        cid = (_match_batched(ans, reps, batch, threshold) if batch is not None
               else _match_pairwise(ans, reps, nli, threshold))
        if cid is None:
            reps.append(ans)
            cid = len(reps) - 1
        assignment.append(cid)
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
