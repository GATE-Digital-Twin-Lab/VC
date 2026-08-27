"""Shared estimators.

The recurring hazard in this study is treating the ~40 draws of one question as
40 independent observations. They are not: within-question correctness
correlation is strongly positive (that is claim 1), so answer-level binomial
intervals are far too narrow and the effective sample size is closer to the
question count. Every interval here therefore resamples QUESTIONS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

NAN = float("nan")


@dataclass(frozen=True)
class Estimate:
    value: float
    lo: float
    hi: float
    n: int
    n_clusters: int = 0

    def as_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {f"{prefix}value": self.value, f"{prefix}lo": self.lo,
                f"{prefix}hi": self.hi, f"{prefix}n": self.n,
                f"{prefix}n_clusters": self.n_clusters}


# --------------------------------------------------------------------------
# AUROC
# --------------------------------------------------------------------------

def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mann-Whitney U with correct tie handling.

    Ties matter here more than usual: VC is quantised onto a handful of values,
    so a naive implementation that breaks ties by ordering would inflate the
    number by a lot.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    keep = ~np.isnan(s)
    s, y = s[keep], y[keep]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return NAN
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0     # average rank across the tie block
        i = j + 1
    r = np.empty(len(s), dtype=np.float64)
    r[order] = ranks
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_frame(df: pd.DataFrame, score: str, label: str) -> float:
    sub = df[[score, label]].dropna()
    if sub.empty:
        return NAN
    return auroc(sub[score].to_numpy(dtype=float), sub[label].to_numpy(dtype=bool))


# --------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------

def cluster_bootstrap(df: pd.DataFrame, statistic: Callable[[pd.DataFrame], float],
                      *, group: str = "q_id", n_resamples: int = 2000,
                      ci: float = 0.95, seed: int = 0) -> Estimate:
    """Resample whole ``group``s with replacement.

    Answers within a question move together, so the question is the unit that
    may be resampled. Resampling answers would understate the interval by
    roughly sqrt(draws per question).
    """
    point = statistic(df)
    groups = df[group].to_numpy()
    uniq = pd.unique(groups)
    if len(uniq) < 2:
        return Estimate(point, NAN, NAN, len(df), len(uniq))
    index_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        try:
            boot[b] = statistic(df.iloc[idx])
        except (ValueError, ZeroDivisionError):
            boot[b] = NAN
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return Estimate(point, NAN, NAN, len(df), len(uniq))
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return Estimate(point, float(np.quantile(boot, lo_q)), float(np.quantile(boot, hi_q)),
                    len(df), len(uniq))


def auroc_cluster_ci(df: pd.DataFrame, score: str, label: str, *, group: str = "q_id",
                     n_resamples: int = 2000, ci: float = 0.95,
                     seed: int = 0) -> Estimate:
    return cluster_bootstrap(df, lambda d: auroc_frame(d, score, label),
                             group=group, n_resamples=n_resamples, ci=ci, seed=seed)


def paired_delta_cluster_ci(df: pd.DataFrame, stat_a: Callable[[pd.DataFrame], float],
                            stat_b: Callable[[pd.DataFrame], float], *,
                            group: str = "q_id", n_resamples: int = 2000,
                            ci: float = 0.95, seed: int = 0) -> Estimate:
    """CI on ``stat_a - stat_b`` with both computed on the same resample.

    Pairing is what makes the pre-hoc vs post-hoc gap (protocol 5.4) testable:
    the two AUROCs share questions, so their difference has far less variance
    than the two intervals viewed separately would suggest.
    """
    return cluster_bootstrap(df, lambda d: stat_a(d) - stat_b(d),
                             group=group, n_resamples=n_resamples, ci=ci, seed=seed)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def brier_decomposition(probs: Sequence[float], labels: Sequence[bool],
                        *, bins: Sequence[float] | None = None) -> dict[str, float]:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty.

    Resolution is the term that matters: it is the variance of group outcome
    rates about the base rate, i.e. how much the forecast actually discriminates.
    A forecast can have excellent reliability and zero resolution by predicting
    the base rate every time.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(float)
    keep = ~np.isnan(p)
    p, y = p[keep], y[keep]
    if p.size == 0:
        return {"brier": NAN, "reliability": NAN, "resolution": NAN, "uncertainty": NAN}
    base = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    keys = p if bins is None else np.asarray(bins, dtype=np.float64)[keep]
    rel = res = 0.0
    for v in np.unique(keys):
        m = keys == v
        n_k = int(m.sum())
        y_k = float(y[m].mean())
        p_k = float(p[m].mean())
        rel += n_k * (p_k - y_k) ** 2
        res += n_k * (y_k - base) ** 2
    n = float(p.size)
    return {"brier": brier, "reliability": rel / n, "resolution": res / n,
            "uncertainty": base * (1 - base), "base_rate": base, "n": int(n)}


def adaptive_bins(values: Sequence[float], n_bins: int) -> np.ndarray:
    """Equal-count bins. Used only when VC is not discrete enough to bin itself."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.array([])
    qs = np.linspace(0, 1, n_bins + 1)
    return np.unique(np.quantile(v, qs))


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    x = np.asarray(a).astype(int)
    y = np.asarray(b).astype(int)
    if x.size == 0:
        return NAN
    po = float((x == y).mean())
    px, py = x.mean(), y.mean()
    pe = float(px * py + (1 - px) * (1 - py))
    if np.isclose(pe, 1.0):
        return NAN
    return float((po - pe) / (1 - pe))


# --------------------------------------------------------------------------
# Variance decomposition and within-question dependence
# --------------------------------------------------------------------------

def variance_decomposition(df: pd.DataFrame, value: str,
                           group: str = "q_id") -> dict[str, float]:
    """Split Var(value) into between-group and within-group parts.

    Within-question variance of ``vc_post`` near zero means VC attaches to the
    prompt regardless of what string follows it (protocol 5.3).
    """
    sub = df[[group, value]].dropna()
    if sub.empty:
        return {"total": NAN, "between": NAN, "within": NAN, "icc": NAN}
    grand = float(sub[value].mean())
    means = sub.groupby(group)[value].mean()
    sizes = sub.groupby(group)[value].size()
    between = float((sizes * (means - grand) ** 2).sum() / len(sub))
    within = float(sub.groupby(group)[value].transform("mean").rsub(sub[value]).pow(2).sum()
                   / len(sub))
    total = between + within
    return {"total": total, "between": between, "within": within,
            "within_share": (within / total) if total > 0 else NAN,
            "icc": (between / total) if total > 0 else NAN,
            "n": int(len(sub)), "n_groups": int(sub[group].nunique())}


def within_question_correlation(df: pd.DataFrame, value: str = "correct",
                                group: str = "q_id") -> dict[str, float]:
    """Corr(c_i, c_j), i != j, pooled over questions.

    This is the independence assumption the product rule needs, as one number.
    Computed from the intraclass form so it uses all pairs without materialising
    them: for binary outcomes it equals the average pairwise correlation.
    """
    sub = df[[group, value]].dropna()
    sub = sub[sub.groupby(group)[value].transform("size") >= 2]
    if sub.empty:
        return {"rho": NAN, "n_questions": 0}
    y = sub[value].astype(float)
    grand = float(y.mean())
    var = float(y.var(ddof=0))
    if var <= 0:
        return {"rho": NAN, "n_questions": int(sub[group].nunique())}
    num = den = 0.0
    for _, g in sub.groupby(group):
        v = g[value].astype(float).to_numpy() - grand
        n = len(v)
        num += (v.sum() ** 2 - np.sum(v ** 2))    # sum over ordered pairs i != j
        den += n * (n - 1)
    if den == 0:
        return {"rho": NAN, "n_questions": int(sub[group].nunique())}
    return {"rho": float((num / den) / var), "n_questions": int(sub[group].nunique()),
            "base_rate": grand}


def within_question_auroc(df: pd.DataFrame, score: str, label: str,
                          group: str = "q_id",
                          min_per_class: int = 1) -> dict[str, float]:
    """For fixed q, does ``score`` rank correct draws above incorrect ones?

    Near 0.5 means the signal is a property of the prompt, not of the answer it
    ostensibly scores (protocol 5.5). Averaged over questions that contain both
    classes -- pooling across questions would conflate this with the
    between-question question.
    """
    vals = []
    for _, g in df.groupby(group):
        sub = g[[score, label]].dropna()
        y = sub[label].astype(bool)
        if int(y.sum()) < min_per_class or int((~y).sum()) < min_per_class:
            continue
        a = auroc(sub[score].to_numpy(dtype=float), y.to_numpy())
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return {"mean_auroc": NAN, "n_questions": 0, "sd": NAN}
    arr = np.asarray(vals)
    return {"mean_auroc": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1
            else NAN, "n_questions": int(len(arr)),
            "se": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else NAN}


# --------------------------------------------------------------------------
# The product rule, accumulated in log space
# --------------------------------------------------------------------------

#: Default clamp on (1 - vc) before taking its log. See ``log_one_minus_vc``.
ONE_MINUS_VC_FLOOR = 1e-6


def log_one_minus_vc(vc, floor: float = ONE_MINUS_VC_FLOOR) -> np.ndarray:
    """``log(1 - vc)``, with ``1 - vc`` clamped below at ``floor``.

    The clamp is a modelling decision, not numerical hygiene, and that is why it
    is a declared parameter rather than a hidden epsilon.

    A stated confidence of exactly 1.0 asserts a failure probability of zero. In
    the linear form that sets the running product to exactly 0, so a single
    overconfident answer satisfies every threshold at once and the stopping rule
    collapses to "stop as soon as any answer says 1.0". Under the clamp that
    answer instead contributes a large but finite ``log(floor)``, so it can be
    outweighed by subsequent evidence. ``floor = 1e-6`` caps any one answer's
    claim at a one-in-a-million failure rate.
    """
    x = 1.0 - np.asarray(vc, dtype=np.float64)
    return np.log(np.clip(x, floor, 1.0))


def log_product_claim(vc, floor: float = ONE_MINUS_VC_FLOOR) -> float:
    """``sum_i log(1 - vc_i)`` -- the log of the claimed "all k wrong" probability.

    Accumulating in log space rather than multiplying keeps the statistic
    representable at any k. The linear product decays geometrically: at
    ``vc = 0.99`` it underflows float64 by roughly k = 160, and every question
    past that point becomes an indistinguishable 0.0, which silently merges the
    grid quantiles that Phase 4 searches over.
    """
    lv = log_one_minus_vc(vc, floor)
    return float(np.sum(lv)) if lv.size else 0.0


def log_to_prob(log_p: float) -> float:
    """Exponentiate back to a probability, saturating at 0 instead of raising."""
    if not np.isfinite(log_p):
        return 0.0 if log_p < 0 else float("inf")
    if log_p <= -745.0:
        return 0.0
    with np.errstate(over="ignore"):
        # A discount of 1e400 is a legitimate reading here, not an error; the
        # log10 form is what should be quoted when this saturates.
        return float(np.exp(log_p))


def wilson_interval(k: int, n: int, ci: float = 0.95) -> tuple[float, float]:
    """Binomial interval -- valid ONLY for question-level counts."""
    if n == 0:
        return NAN, NAN
    from scipy.stats import norm
    z = float(norm.ppf(1 - (1 - ci) / 2))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)
