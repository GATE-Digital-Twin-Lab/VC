"""Phase 6 -- recalibration transfer.

The obvious rebuttal to any miscalibration finding is that miscalibration is
trivially fixable: fit an isotonic map and move on. This pre-empts it. Fit on
dataset A, apply to dataset B, and report the calibration gap before and after.
If the fitted map does not transfer, VC is not a stable instrument and there is
no single correction to apply.

Note the limit of what recalibration could ever buy, from 6.7: an isotonic map
is monotone and pointwise, so it shrinks the values but preserves the
compounding structure of the product rule. The divergence in ``k`` survives it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import model_defined_ece, reliability
from .config import Config


def fit_isotonic(vc: np.ndarray, correct: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(vc, dtype=float), np.asarray(correct, dtype=float))
    return iso


def calibration_gap(df: pd.DataFrame, cfg: Config, vc_col: str,
                    label_col: str = "correct") -> dict:
    rel = reliability(df, cfg, vc_col=vc_col, label_col=label_col)
    if rel.empty:
        return {"ece": float("nan"), "mae_gap": float("nan"), "n_groups": 0}
    metric = cfg.get("phase6.report_gap_metric")
    w = rel["n_g"] / rel["n_g"].sum()
    return {
        "ece_model_bins": model_defined_ece(rel),
        "mae_gap": float(rel["abs_gap"].mean()),
        "weighted_gap": float((w * rel["abs_gap"]).sum()),
        "metric": metric,
        "n_groups": int(len(rel)),
        "n": int(rel["n_g"].sum()),
    }


def transfer_study(cfg: Config, answers: pd.DataFrame, *,
                   vc_col: str = "vc_post") -> tuple[pd.DataFrame, dict]:
    """Fit on one dataset, apply to another, both directions."""
    fit_on = cfg.get("phase6.fit_on")
    apply_to = cfg.get("phase6.apply_to")
    sub = answers.dropna(subset=[vc_col, "correct"])
    rows = []
    substituted = None

    # A transfer study needs two populated datasets. If the configured names are
    # absent -- a stand-in loader labels its data differently, say -- fall back
    # to the two largest present and record the substitution, rather than
    # returning an empty table that looks like a null result.
    present = set(sub["dataset"].unique())
    if not {fit_on, apply_to} <= present:
        largest = (sub["dataset"].value_counts().index.tolist())
        if len(largest) >= 2:
            substituted = {"configured": [fit_on, apply_to], "used": largest[:2]}
            fit_on, apply_to = largest[0], largest[1]

    for source, target in ((fit_on, apply_to), (apply_to, fit_on)):
        src = sub[sub["dataset"] == source]
        tgt = sub[sub["dataset"] == target]
        if src.empty or tgt.empty:
            rows.append({"fit_on": source, "apply_to": target, "n_source": len(src),
                         "n_target": len(tgt), "note": "dataset absent"})
            continue

        iso = fit_isotonic(src[vc_col].to_numpy(dtype=float),
                           src["correct"].astype(float).to_numpy())

        before_in = calibration_gap(src, cfg, vc_col)
        src_cal = src.assign(**{f"{vc_col}_cal": iso.predict(
            src[vc_col].to_numpy(dtype=float))})
        after_in = calibration_gap(src_cal, cfg, f"{vc_col}_cal")

        before_out = calibration_gap(tgt, cfg, vc_col)
        tgt_cal = tgt.assign(**{f"{vc_col}_cal": iso.predict(
            tgt[vc_col].to_numpy(dtype=float))})
        after_out = calibration_gap(tgt_cal, cfg, f"{vc_col}_cal")

        rows.append({
            "fit_on": source, "apply_to": target,
            "n_source": int(len(src)), "n_target": int(len(tgt)),
            "in_sample_gap_before": before_in["weighted_gap"],
            "in_sample_gap_after": after_in["weighted_gap"],
            "transfer_gap_before": before_out["weighted_gap"],
            "transfer_gap_after": after_out["weighted_gap"],
            "in_sample_improvement": before_in["weighted_gap"] - after_in["weighted_gap"],
            "transfer_improvement": before_out["weighted_gap"] - after_out["weighted_gap"],
        })

    table = pd.DataFrame(rows)
    summary: dict = {"fit_on": fit_on, "apply_to": apply_to,
                     "datasets_present": sorted(present)}
    if substituted:
        summary["dataset_substitution"] = substituted
    if "transfer_improvement" in table.columns and table["transfer_improvement"].notna().any():
        in_s = float(table["in_sample_improvement"].mean())
        out_s = float(table["transfer_improvement"].mean())
        summary.update({
            "mean_in_sample_improvement": in_s,
            "mean_transfer_improvement": out_s,
            "transfer_retention": (out_s / in_s) if abs(in_s) > 1e-12 else float("nan"),
            "interpretation": (
                "the isotonic map does not transfer: the correction is dataset-specific, "
                "so VC is not a stable instrument that one recalibration repairs"
                if out_s < 0.5 * in_s else
                "recalibration transfers; miscalibration is at least partly a fixed offset"),
        })
    return table, summary


def isotonic_preserves_compounding(cfg: Config, answers: pd.DataFrame,
                                   vc_col: str = "vc_post") -> pd.DataFrame:
    """Product-rule divergence before and after recalibration.

    The point of running it after isotonic regression is that the gap still
    widens in k. A monotone pointwise map cannot undo a dependence structure.
    """
    from .survival import product_rule_curve, product_rule_divergence

    sub = answers.dropna(subset=[vc_col, "correct"])
    if sub.empty:
        return pd.DataFrame()
    iso = fit_isotonic(sub[vc_col].to_numpy(dtype=float),
                       sub["correct"].astype(float).to_numpy())
    raw = product_rule_divergence(product_rule_curve(sub, cfg, subset="raw"))
    cal_answers = sub.assign(vc_post=iso.predict(sub[vc_col].to_numpy(dtype=float)))
    cal = product_rule_divergence(product_rule_curve(cal_answers, cfg,
                                                     subset="isotonic"))
    return pd.concat([raw, cal], ignore_index=True)
