"""Phase 2 -- descriptive calibration. Run on ``eval``.

Answer-level grouping is PRIMARY. Post-hoc VC attaches to an answer, so the
claim it makes is: of all answers assigned 0.8, 80% are correct. Pooling all
draws by ``vc_post`` value tests exactly that claim, and VC's discreteness
supplies canonical bins for free -- the model's own partition rather than a
binning heuristic -- which sidesteps the arbitrary-bin objection to ECE.

Fixed-width ECE is refused outright: on a support of {0.7, 0.8, 0.9, 0.95} it is
meaningless.

Intervals come from a cluster bootstrap over QUESTIONS. Binomial intervals over
40 x 1000 answers would be far too narrow, because draws within a question are
not independent -- which 5.7 measures directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .stats import (Estimate, NAN, auroc_frame, brier_decomposition,
                    cluster_bootstrap, paired_delta_cluster_ci,
                    variance_decomposition, within_question_auroc,
                    within_question_correlation)


def vc_histogram(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Support and group sizes. n_g is always reported: high-VC groups will
    dominate and low-VC groups may be too small to say anything about."""
    sub = df[[col]].dropna()
    out = (sub[col].round(6).value_counts().rename_axis("value")
           .rename("n").reset_index().sort_values("value"))
    out["share"] = out["n"] / out["n"].sum()
    out["column"] = col
    return out.reset_index(drop=True)


def discreteness_summary(df: pd.DataFrame, col: str,
                         top_k: int = 5) -> dict:
    hist = vc_histogram(df, col)
    if hist.empty:
        return {"column": col, "n_distinct": 0}
    top = hist.nlargest(top_k, "n")
    return {
        "column": col,
        "n_distinct": int(len(hist)),
        "share_in_top_k": float(top["share"].sum()),
        "top_values": top["value"].tolist(),
        "min": float(hist["value"].min()), "max": float(hist["value"].max()),
        "share_in_0.7_1.0": float(hist.loc[hist["value"] >= 0.7, "share"].sum()),
    }


def reliability(df: pd.DataFrame, cfg: Config, *, vc_col: str = "vc_post",
                label_col: str = "correct") -> pd.DataFrame:
    """Reliability table with cluster-bootstrap intervals on each group's rate."""
    binning = cfg.get("phase2.binning")
    n_res = int(cfg.get("phase2.bootstrap.n_resamples"))
    ci = float(cfg.get("phase2.bootstrap.ci"))
    seed = int(cfg.get("run.seed"))
    sub = df.dropna(subset=[vc_col, label_col]).copy()
    if sub.empty:
        return pd.DataFrame()

    if binning == "model_defined":
        sub["group"] = sub[vc_col].round(6)
    elif binning == "adaptive":
        n_bins = int(cfg.get("phase2.adaptive_n_bins"))
        sub["group"] = pd.qcut(sub[vc_col].rank(method="first"),
                               min(n_bins, max(1, sub[vc_col].nunique())),
                               labels=False, duplicates="drop")
    else:
        raise ValueError(
            f"phase2.binning = {binning!r}. Fixed-width binning is not offered: "
            "on discrete VC values it is meaningless.")

    rows = []
    for g, grp in sub.groupby("group"):
        est = cluster_bootstrap(grp, lambda d: float(d[label_col].astype(bool).mean()),
                                group="q_id", n_resamples=n_res, ci=ci, seed=seed)
        v_g = float(grp[vc_col].mean())
        rows.append({
            "v_g": v_g, "group": g, "n_g": int(len(grp)),
            "n_questions": int(grp["q_id"].nunique()),
            "p_hat_g": est.value, "lo": est.lo, "hi": est.hi,
            "calibration_gap": est.value - v_g,
            "abs_gap": abs(est.value - v_g),
        })
    out = pd.DataFrame(rows).sort_values("v_g").reset_index(drop=True)
    min_n = int(cfg.get("phase2.min_group_n"))
    out["reportable"] = out["n_g"] >= min_n
    return out


def monotonicity(rel: pd.DataFrame) -> dict:
    """Is p_hat_g increasing in v_g? A calibration curve that is not monotone
    cannot be repaired by any monotone recalibration either."""
    sub = rel[rel["reportable"]] if "reportable" in rel.columns else rel
    if len(sub) < 2:
        return {"monotone": None, "n_groups": int(len(sub))}
    diffs = np.diff(sub["p_hat_g"].to_numpy())
    from scipy.stats import spearmanr
    rho, p = spearmanr(sub["v_g"], sub["p_hat_g"])
    return {"monotone": bool(np.all(diffs >= -1e-12)),
            "n_inversions": int(np.sum(diffs < -1e-12)),
            "spearman_rho": float(rho), "spearman_p": float(p),
            "n_groups": int(len(sub))}


def model_defined_ece(rel: pd.DataFrame) -> float:
    """ECE over the model's OWN bins. Not fixed-width, by construction."""
    if rel.empty:
        return NAN
    w = rel["n_g"] / rel["n_g"].sum()
    return float((w * rel["abs_gap"]).sum())


def information_gain(questions: pd.DataFrame, cfg: Config) -> dict:
    """AUROC(vc_pre) vs AUROC(vc_1) on the same questions, paired.

    Pre-hoc VC cannot see the answer, so any gap is the information post-hoc VC
    extracts from reading its own output. A gap near zero means post-hoc VC is
    not reading the answer either -- it is a question-difficulty estimate
    wearing an answer-confidence label.
    """
    sub = questions.dropna(subset=["vc_pre", "vc_1", "p_hat"]).copy()
    if sub.empty:
        return {"n_questions": 0}
    sub["_label"] = sub["p_hat"] > 0     # question has at least one correct draw
    if sub["_label"].nunique() < 2:
        return {"n_questions": int(len(sub)), "note": "single class; AUROC undefined"}

    a_pre = auroc_frame(sub, "vc_pre", "_label")
    a_1 = auroc_frame(sub, "vc_1", "_label")
    delta = paired_delta_cluster_ci(
        sub, lambda d: auroc_frame(d, "vc_1", "_label"),
        lambda d: auroc_frame(d, "vc_pre", "_label"),
        group="q_id", n_resamples=int(cfg.get("phase2.bootstrap.n_resamples")),
        ci=float(cfg.get("phase2.bootstrap.ci")), seed=int(cfg.get("run.seed")))
    return {
        "auroc_vc_pre": a_pre, "auroc_vc_1": a_1,
        "delta_post_minus_pre": delta.value, "delta_lo": delta.lo, "delta_hi": delta.hi,
        "n_questions": int(len(sub)),
        "reading_own_answer_adds_nothing": bool(delta.lo <= 0 <= delta.hi),
    }


def two_aurocs(answers: pd.DataFrame, questions: pd.DataFrame,
               cfg: Config) -> pd.DataFrame:
    """Between-question and within-question AUROC, reported SEPARATELY.

    Pooling them conflates two distinct claims: whether VC identifies hard
    questions, and whether it ranks correct draws above incorrect ones for a
    fixed question. Within-question near 0.5 means VC is a property of the
    prompt rather than of the answer it ostensibly scores -- and it predicts
    that VC-based stopping will tie with fixed_k in Phase 4.
    """
    n_res = int(cfg.get("phase2.bootstrap.n_resamples"))
    ci = float(cfg.get("phase2.bootstrap.ci"))
    seed = int(cfg.get("run.seed"))
    min_pc = int(cfg.get("phase2.within_question_auroc_min_per_class"))
    rows = []

    # The between-question half needs a question table; callers that only want
    # the within-question half (which is computed from answers alone) may pass
    # an empty frame.
    q = (questions.dropna(subset=["p_hat"]).copy()
         if len(questions) and "p_hat" in questions.columns else pd.DataFrame())
    if len(q):
        q["_label"] = q["p_hat"] > 0
    for col in ("vc_pre", "vc_1", "vc_bar"):
        if not len(q) or col not in q.columns or q[col].isna().all() \
                or q["_label"].nunique() < 2:
            continue
        sub = q.dropna(subset=[col])
        est = cluster_bootstrap(sub, lambda d, c=col: auroc_frame(d, c, "_label"),
                                group="q_id", n_resamples=n_res, ci=ci, seed=seed)
        rows.append({"level": "between_question", "signal": col, "auroc": est.value,
                     "lo": est.lo, "hi": est.hi, "n": int(len(sub))})

    for col in ("vc_post", "h_tok_mean", "min_token_p", "f", "s_anchor"):
        if col not in answers.columns or answers[col].isna().all():
            continue
        # h_tok and s_anchor point the other way: more entropy / more distance
        # from the anchor should mean LESS likely correct.
        direction = -1.0 if col in ("h_tok_mean", "s_anchor") else 1.0
        tmp = answers.dropna(subset=[col, "correct"]).copy()
        tmp["_score"] = direction * tmp[col].astype(float)
        res = within_question_auroc(tmp, "_score", "correct", min_per_class=min_pc)
        rows.append({"level": "within_question", "signal": col,
                     "auroc": res["mean_auroc"],
                     "lo": res["mean_auroc"] - 1.96 * res["se"]
                     if not np.isnan(res.get("se", NAN)) else NAN,
                     "hi": res["mean_auroc"] + 1.96 * res["se"]
                     if not np.isnan(res.get("se", NAN)) else NAN,
                     "n": res["n_questions"]})
    return pd.DataFrame(rows)


def vc_variance_decomposition(answers: pd.DataFrame,
                              cfg: Config | None = None) -> dict:
    """Between- vs within-question variance of vc_post.

    Within-question variance near zero means answer-level grouping degenerates
    to question-level: VC attaches to the prompt regardless of what string
    follows it.
    """
    out = variance_decomposition(answers, "vc_post", group="q_id")
    threshold = (float(cfg.get("phase2.vc_within_share_prompt_threshold"))
                 if cfg is not None else 0.10)
    out["within_share_threshold"] = threshold
    out["interpretation"] = (
        "VC attaches to the prompt, not the answer"
        if out.get("within_share", 1) < threshold
        else "VC varies across answers to the same question")
    return out


def brier(answers: pd.DataFrame, vc_col: str = "vc_post",
          label_col: str = "correct") -> dict:
    sub = answers.dropna(subset=[vc_col, label_col])
    return brier_decomposition(sub[vc_col].to_numpy(dtype=float),
                               sub[label_col].astype(bool).to_numpy(),
                               bins=sub[vc_col].round(6).to_numpy())


def error_dependence(answers: pd.DataFrame) -> dict:
    """Corr(c_i, c_j) -- the independence assumption the product rule needs."""
    out = within_question_correlation(answers, value="correct", group="q_id")
    out["common_mode"] = bool(out.get("rho", 0) > 0.1) if not np.isnan(
        out.get("rho", NAN)) else None
    return out
