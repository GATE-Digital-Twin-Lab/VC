"""Phase 3 -- survival, the U/A partition, diversity, and the product rule. CORE.

The organising fact is censoring. ``K_q``, the draw index of the first correct
answer, is right-censored at ``N_MAX``: on questions where nothing correct ever
appears there is no value to average. Averaging ``K_q`` over successes only
silently deletes exactly the hard questions, which is the classic bias in this
setting, so Kaplan-Meier is used throughout and ``beta = S(N_MAX)`` -- the
fraction of questions with no admissible answer at any budget -- is reported as
a first-class quantity.

``beta`` must be known BEFORE any ``alpha`` is chosen. If ``alpha < beta`` no
stopping rule of any kind can be certified and Phase 4 returns a blank table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .stats import (NAN, auroc, auroc_cluster_ci, log_product_claim,
                    wilson_interval)


# --------------------------------------------------------------------------
# Question-level aggregation
# --------------------------------------------------------------------------

def question_stats(answers: pd.DataFrame, n_max: int | None = None) -> pd.DataFrame:
    """p_hat, K_q, censoring, and the three question-level VC statistics."""
    if "correct" not in answers.columns:
        raise ValueError("answers must carry a 'correct' column; call judge.attach_correct")
    n_max = n_max or int(answers.groupby("q_id")["draw_idx"].size().max())
    rows = []
    for q_id, g in answers.sort_values("draw_idx").groupby("q_id"):
        c = g["correct"].astype(bool).to_numpy()
        first = np.flatnonzero(c)
        k_q = int(first[0]) + 1 if first.size else -1     # 1-indexed draw count
        vc = g["vc_post"].astype("Float64")
        rows.append({
            "q_id": q_id,
            "dataset": g["dataset"].iloc[0],
            "split": g["split"].iloc[0],
            "n_draws": int(len(g)),
            "p_hat": float(c.mean()),
            "K_q": k_q,
            "censored": bool(k_q < 0),
            "vc_1": float(vc.iloc[0]) if pd.notna(vc.iloc[0]) else np.nan,
            "vc_bar": float(vc.mean()) if vc.notna().any() else np.nan,
            "vc_sd": float(vc.std()) if vc.notna().sum() > 1 else np.nan,
            "answer_len_mean": float(g["answer"].astype(str).str.len().mean()),
        })
    return pd.DataFrame(rows)


def p_hat_se(n_draws: int) -> float:
    """SE of p_hat is at most 0.5/sqrt(N) -- at N=10 that is 0.16, which is
    comparable to VC's own granularity. N >= 30 is the floor for this study."""
    return 0.5 / np.sqrt(max(1, n_draws))


# --------------------------------------------------------------------------
# Kaplan-Meier
# --------------------------------------------------------------------------

def kaplan_meier(k_values, censored, n_max: int) -> pd.DataFrame:
    """S(k) = P(K_q > k) with Greenwood standard errors.

    Here every censored observation is censored at exactly ``n_max``, so the
    estimator reduces to a clean product-limit form, but it is written in the
    general way so a variable budget can be dropped in without changing callers.
    """
    k = np.asarray(list(k_values), dtype=float)
    cens = np.asarray(list(censored), dtype=bool)
    n = len(k)
    if n == 0:
        return pd.DataFrame(columns=["k", "at_risk", "events", "S", "se", "lo", "hi"])

    event_time = np.where(cens, np.inf, k)
    rows = []
    surv = 1.0
    cum_var = 0.0
    at_risk = n
    for kk in range(1, n_max + 1):
        events = int(np.sum(event_time == kk))
        if at_risk > 0:
            surv *= (1.0 - events / at_risk)
            if at_risk - events > 0 and events > 0:
                cum_var += events / (at_risk * (at_risk - events))
        se = surv * np.sqrt(cum_var) if surv > 0 else 0.0
        rows.append({"k": kk, "at_risk": at_risk, "events": events, "S": surv,
                     "se": se, "lo": max(0.0, surv - 1.96 * se),
                     "hi": min(1.0, surv + 1.96 * se)})
        at_risk -= events
    return pd.DataFrame(rows)


def beta_from_km(km: pd.DataFrame) -> float:
    """Plateau height: the fraction with no admissible answer at any budget."""
    return float(km["S"].iloc[-1]) if len(km) else NAN


def km_flatness(km: pd.DataFrame, cfg: Config) -> dict:
    """Is S(k) actually flat by N_MAX?

    If it is still declining, ``U`` is an artifact of the sampling budget and
    everything downstream describes the budget rather than the model. This
    returns the verdict rather than raising, so the number can be reported.
    """
    window = int(cfg.get("phase3.km_flatness.tail_window"))
    max_slope = float(cfg.get("phase3.km_flatness.max_tail_slope"))
    if len(km) < window + 1:
        return {"flat": False, "tail_slope": NAN, "reason": "curve shorter than window"}
    tail = km.tail(window)
    slope = float((tail["S"].iloc[0] - tail["S"].iloc[-1]) / max(1, window - 1))
    flat = slope <= max_slope
    return {"flat": bool(flat), "tail_slope": slope, "max_tail_slope": max_slope,
            "reason": "" if flat else
            "S(k) is still declining at N_MAX: U is a censoring artifact, and "
            "every downstream statement is about the budget, not the model."}


def hazard(answers: pd.DataFrame, max_k: int) -> pd.DataFrame:
    """h(k) = P(c_k = 1 | c_1..c_{k-1} = 0).

    Conditionally-i.i.d. draws with a single p_q predict a flat hazard. Pooled
    over heterogeneous p_q it must decline: a run of failures is evidence of
    being in a low-p_q question. Sharp decay to ~0 by k=3 refutes the "just
    sample more" premise outright.
    """
    wide = (answers.sort_values("draw_idx")
            .assign(correct=lambda d: d["correct"].astype(bool))
            .pivot_table(index="q_id", columns="draw_idx", values="correct",
                         aggfunc="first"))
    rows = []
    alive = np.ones(len(wide), dtype=bool)
    for k in range(min(max_k, wide.shape[1])):
        col = wide.iloc[:, k].to_numpy()
        observed = alive & ~pd.isna(col)
        n_at_risk = int(observed.sum())
        if n_at_risk == 0:
            break
        successes = int(np.sum(np.where(observed, col == True, False)))  # noqa: E712
        lo, hi = wilson_interval(successes, n_at_risk)
        rows.append({"k": k + 1, "at_risk": n_at_risk, "successes": successes,
                     "hazard": successes / n_at_risk, "lo": lo, "hi": hi})
        alive = observed & ~(np.where(observed, col == True, False))     # noqa: E712
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# U / A partition -- ORDER MATTERS
# --------------------------------------------------------------------------

def partition_U(answers: pd.DataFrame, *, classify_split: str = "classify") -> pd.DataFrame:
    """Membership in U decided from the CLASSIFY split's draws only.

    "Unanswerable" is not a label anyone has; it is defined by the outcome of
    generation. Deciding membership with the same draws that later calibrate the
    stopping rule is selection on the outcome being certified and voids the LTT
    guarantee, so classification and calibration are given disjoint draws.
    """
    sub = answers[answers["split"] == classify_split]
    if sub.empty:
        raise ValueError(
            f"the {classify_split!r} split has no draws. U must be decided on draws "
            "that are never reused for calibration or evaluation.")
    grp = sub.groupby("q_id")["correct"]
    out = grp.max().rename("any_correct").reset_index()
    out["in_U"] = ~out["any_correct"].astype(bool)
    out["n_classify_draws"] = grp.size().to_numpy()
    return out[["q_id", "in_U", "n_classify_draws"]]


def assert_disjoint_draws(classify: pd.DataFrame, calib: pd.DataFrame) -> None:
    key = ["q_id", "draw_idx", "seed"]
    overlap = classify.merge(calib, on=key, how="inner")
    if len(overlap):
        raise ValueError(
            f"{len(overlap)} draws appear in both the classify and calibration sets. "
            "Reusing them selects on the outcome being certified.")


def beta_vs_tau(answers: pd.DataFrame, e_col: str, taus) -> pd.DataFrame:
    """beta as a function of tau -- a strict tau inflates it mechanically."""
    rows = []
    for tau in taus:
        corr = answers[e_col] <= float(tau)
        any_correct = corr.groupby(answers["q_id"]).max()
        rows.append({"tau": float(tau), "beta": float((~any_correct.astype(bool)).mean()),
                     "n_questions": int(len(any_correct))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Predicted vs observed budget -- the dynamic-range collapse
# --------------------------------------------------------------------------

def predicted_budget(vc: float, alpha: float) -> float:
    """N_hat = ceil(log alpha / log(1 - VC)), the budget VC's claim implies."""
    v = float(vc)
    if not np.isfinite(v) or v <= 0:
        return np.inf
    if v >= 1.0:
        return 1.0
    return float(np.ceil(np.log(alpha) / np.log(1.0 - v)))


def budget_table(questions: pd.DataFrame, cfg: Config,
                 vc_cols=("vc_pre", "vc_1", "vc_bar")) -> pd.DataFrame:
    """Compare the budget VC implies against the budget actually needed.

    The observed VC range of roughly [0.7, 1.0] maps to N_hat in {1, 2}: VC's
    entire confidence vocabulary spans about one sample of operational
    difference, while true difficulty spans one sample to never. Reported in
    units of compute, which is where it bites.
    """
    alpha = float(cfg.get("phase3.budget_alpha"))
    rows = []
    for col in vc_cols:
        if col not in questions.columns:
            continue
        sub = questions.dropna(subset=[col])
        n_hat = sub[col].map(lambda v: predicted_budget(v, alpha))
        observed = sub["K_q"].where(sub["K_q"] > 0, np.inf)
        finite = np.isfinite(n_hat) & np.isfinite(observed)
        rows.append({
            "vc_statistic": col,
            "alpha": alpha,
            "vc_min": float(sub[col].min()), "vc_max": float(sub[col].max()),
            "n_hat_min": float(np.nanmin(n_hat)), "n_hat_max": float(np.nanmax(n_hat)),
            "n_hat_distinct_values": int(pd.Series(n_hat).nunique()),
            "observed_K_median": float(np.median(observed[np.isfinite(observed)]))
            if np.isfinite(observed).any() else NAN,
            "observed_K_p95": float(np.quantile(observed[np.isfinite(observed)], 0.95))
            if np.isfinite(observed).any() else NAN,
            "frac_never_correct": float(np.mean(~np.isfinite(observed))),
            "mean_underprediction": float(np.mean(observed[finite] - n_hat[finite]))
            if finite.any() else NAN,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The headline comparison (6.5)
# --------------------------------------------------------------------------

def u_detection(questions: pd.DataFrame, cfg: Config,
                signals=("vc_pre", "vc_1", "vc_bar", "H_sem", "h_tok_mean",
                         "largest_cluster_share")) -> pd.DataFrame:
    """AUROC of each signal for predicting in_U, plus the legible overlap.

    Stated in the falsifiable direction: clean separation is a POSITIVE result
    for the signal and is reported as such. ``vc_pre`` is the operationally
    right signal, since deciding to abstain before spending compute is exactly
    the pre-hoc question.
    """
    n_res = int(cfg.get("phase2.bootstrap.n_resamples"))
    ci = float(cfg.get("phase2.bootstrap.ci"))
    rows = []
    for sig in signals:
        if sig not in questions.columns:
            continue
        sub = questions.dropna(subset=[sig, "in_U"])
        if sub.empty or sub["in_U"].nunique() < 2:
            continue
        # Higher confidence should mean LESS likely to be in U, so the signal is
        # negated: AUROC > 0.5 then means the signal detects the floor.
        direction = -1.0 if sig in ("vc_pre", "vc_1", "vc_bar", "largest_cluster_share") else 1.0
        tmp = sub.assign(_score=direction * sub[sig].astype(float),
                         _label=sub["in_U"].astype(bool))
        est = auroc_cluster_ci(tmp, "_score", "_label", group="q_id",
                               n_resamples=n_res, ci=ci,
                               seed=int(cfg.get("run.seed")))
        in_u = sub[sub["in_U"].astype(bool)]
        # Below 0.5 the signal is ANTI-predictive: it is systematically more
        # confident on the questions where nothing ever comes out right. For
        # the diversity signals that is the low-diversity-U cell showing up as
        # a number, and it is a stronger finding than mere blindness.
        verdict = ("anti-predictive: most confident exactly where nothing is correct"
                   if est.hi < 0.5 else
                   "blind: interval covers chance" if est.lo <= 0.5 <= est.hi else
                   "detects U")
        rows.append({
            "signal": sig, "direction": "lower is worse" if direction < 0 else "higher is worse",
            "auroc_in_U": est.value, "lo": est.lo, "hi": est.hi, "verdict": verdict,
            "n_questions": int(len(sub)), "n_in_U": int(len(in_u)),
            "overlap_high_conf_in_U": float((in_u[sig] >= 0.8).mean())
            if direction < 0 and len(in_u) else NAN,
        })
    return pd.DataFrame(rows)


def overlap_statement(questions: pd.DataFrame, signal: str = "vc_pre",
                      threshold: float = 0.8) -> str:
    sub = questions.dropna(subset=[signal, "in_U"])
    in_u = sub[sub["in_U"].astype(bool)]
    if in_u.empty:
        return f"U is empty at this tau; {signal} overlap is undefined."
    share = float((in_u[signal] >= threshold).mean())
    return (f"{share:.0%} of never-correct questions received {signal} >= {threshold} "
            f"(n = {len(in_u)} of {len(sub)}).")


# --------------------------------------------------------------------------
# Diversity 2x2 (6.6)
# --------------------------------------------------------------------------

def diversity_2x2(questions: pd.DataFrame, *, diversity_col: str = "H_sem",
                  quantile: float = 0.5) -> pd.DataFrame:
    """The 2x2 of answerability against diversity.

    The low-diversity-U cell is the dangerous one: self-consistency, semantic
    entropy and answer frequency are ALL diversity measures, so none of them can
    distinguish confidently-right from confidently-wrong. If that cell is a
    meaningful fraction of U, the limitation belongs to the field's dominant
    approach, not only to VC.
    """
    sub = questions.dropna(subset=[diversity_col, "in_U"]).copy()
    if sub.empty:
        return pd.DataFrame()
    cut = float(sub[diversity_col].quantile(quantile))
    sub["diversity"] = np.where(sub[diversity_col] <= cut, "low", "high")
    sub["answerable"] = np.where(sub["in_U"].astype(bool), "U (never correct)",
                                 "A (answerable)")
    labels = {
        ("low", "A (answerable)"): "genuine knowledge",
        ("low", "U (never correct)"): "systematic misconception",
        ("high", "A (answerable)"): "recoverable",
        ("high", "U (never correct)"): "fabrication",
    }
    agg = {"n": ("q_id", "size")}
    for col in ("vc_pre", "vc_1", "vc_bar", "p_hat", "h_tok_mean"):
        if col in sub.columns:
            agg[f"mean_{col}"] = (col, "mean")
    out = sub.groupby(["diversity", "answerable"]).agg(**agg).reset_index()
    out["cell"] = [labels.get((d, a), "") for d, a in
                   zip(out["diversity"], out["answerable"])]
    out["share_of_total"] = out["n"] / out["n"].sum()
    return out


# --------------------------------------------------------------------------
# Product-rule reliability (6.7)
# --------------------------------------------------------------------------

def product_rule_curve(answers: pd.DataFrame, cfg: Config, *,
                       subset: str = "all") -> pd.DataFrame:
    """Claimed vs observed probability that the first k draws are ALL wrong.

    Honest aggregation puts the curves on the diagonal. Common-mode errors
    flatten them, and -- this is the signature -- the gap WIDENS with k. A single
    VC of 0.8 on a wrong answer is off by about 5x in failure probability; after
    five draws the claim of 0.2^5 = 3.2e-4 stands against a truth of 1, off by
    about 3000x. Sampling makes the estimate monotonically worse, and no
    monotone recalibration repairs it: an isotonic map shrinks the values but
    preserves the compounding.
    """
    ks = list(cfg.get("phase3.product_rule.k_values"))
    n_bins = int(cfg.get("phase3.product_rule.n_bins"))
    floor = float(cfg.get("phase3.product_rule.log_floor"))
    vc_floor = float(cfg.get("aggregation.one_minus_vc_floor"))
    log_floor = float(np.log(floor))
    rows = []
    for k in ks:
        recs = []
        for q_id, g in answers.sort_values("draw_idx").groupby("q_id"):
            g = g.head(k)
            if len(g) < k:
                continue
            vc = g["vc_post"].astype(float)
            if vc.isna().any():
                continue
            # Sum of logs, matching the Phase 4 stopping statistic exactly. The
            # diagnostic and the rule must accumulate the claim the same way, or
            # the curve would not describe the rule being certified.
            log_claimed = log_product_claim(vc.to_numpy(), vc_floor)
            all_wrong = bool(~g["correct"].astype(bool).any())
            recs.append({"q_id": q_id, "log_claimed": log_claimed,
                         "all_wrong": all_wrong})
        if not recs:
            continue
        df = pd.DataFrame(recs)
        # Bin on the log: the ordering is identical, but the linear product
        # underflows to a wall of exact zeros at large k, which would collapse
        # the low bins into one indistinguishable group.
        df["bin"] = pd.qcut(df["log_claimed"].rank(method="first"),
                            min(n_bins, max(1, df["log_claimed"].nunique())),
                            labels=False, duplicates="drop")
        agg = df.groupby("bin").agg(log_claimed=("log_claimed", "mean"),
                                    observed=("all_wrong", "mean"),
                                    n=("q_id", "size")).reset_index()
        agg["k"] = k
        agg["subset"] = subset
        # Displayed on log axes anyway; the floor only guards the plot.
        agg["claimed"] = np.exp(np.clip(agg["log_claimed"], log_floor, 0.0))
        agg["log_ratio"] = (np.log(agg["observed"].clip(lower=floor))
                            - agg["log_claimed"])
        agg["log10_ratio"] = agg["log_ratio"] / np.log(10.0)
        agg["ratio_observed_over_claimed"] = np.exp(np.clip(agg["log_ratio"], -700, 700))
        rows.append(agg)
    if not rows:
        return pd.DataFrame(columns=["bin", "log_claimed", "claimed", "observed",
                                     "n", "k", "subset"])
    out = pd.concat(rows, ignore_index=True)
    return out


def product_rule_divergence(curve: pd.DataFrame) -> pd.DataFrame:
    """Per-k summary of the gap, and the empirical floor where curves plateau."""
    rows = []
    for (k, subset), g in curve.groupby(["k", "subset"]):
        w = g["n"] / g["n"].sum()
        rows.append({
            "k": int(k), "subset": subset,
            "mean_log_claimed": float((g["log_claimed"] * w).sum()),
            "mean_claimed": float((g["claimed"] * w).sum()),
            "mean_observed": float((g["observed"] * w).sum()),
            # log10 is the primary reporting form: the linear ratio reaches 1e5
            # by k = 10 in the U subset and overflows entirely at larger k.
            "median_log10_ratio": float(g["log10_ratio"].median()),
            "max_log10_ratio": float(g["log10_ratio"].max()),
            "median_ratio": float(g["ratio_observed_over_claimed"].median()),
            "max_ratio": float(g["ratio_observed_over_claimed"].max()),
            "empirical_floor": float(g["observed"].min()),
        })
    return pd.DataFrame(rows).sort_values(["subset", "k"]).reset_index(drop=True)
