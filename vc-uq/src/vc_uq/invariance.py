"""Phase 5 -- invariance. This is what carries the thesis.

Phases 2-4 establish poor calibration *quality*, which isotonic regression could
in principle repair. These experiments test properties any estimator of ``p_q``
must satisfy regardless of quality:

  8.1 ``p_q`` is definitionally a function of the decoder, so it must move with
      temperature. VC has no argument for T and cannot. Cheapest experiment,
      strongest payoff -- run it first.
  8.2 Test-retest reliability across semantically equivalent elicitations.

Both are MATCHED comparisons -- the sweep holds the prompt fixed and moves only
``T``; the paraphrase set holds the question fixed and moves only wording. The
probes that were once 8.3-8.6 (scale reframing, sycophancy, forced decode,
pre-hoc metacognition) are out of scope: each varied more than one thing at a
time and would have needed its own control arm to be interpretable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .generate import Generator
from .stats import NAN


# --------------------------------------------------------------------------
# 8.1 Temperature sweep -- run this first
# --------------------------------------------------------------------------

def temperature_sweep(cfg: Config, gen: Generator, questions: pd.DataFrame,
                      judge_fn) -> pd.DataFrame:
    """p_hat(T) against VC(T), prompt held fixed.

    ``judge_fn(answers_df) -> answers_df`` supplies the correctness column;
    passed in rather than imported so the sweep can run before Phase 0 has
    selected tau, using whatever criterion is available.
    """
    temps = list(cfg.get("phase5.temperature_sweep.temperatures"))
    n_q = int(cfg.get("phase5.temperature_sweep.n_questions"))
    n_draws = int(cfg.get("phase5.temperature_sweep.n_draws"))
    sub = questions.head(n_q)

    frames = []
    for T in temps:
        df = gen.draw_answers(sub, n_max=n_draws, temperature=float(T),
                              store_per_position=False)
        df = judge_fn(df)
        frames.append(df.assign(temperature=float(T)))
    allf = pd.concat(frames, ignore_index=True)

    per_q = (allf.groupby(["q_id", "temperature"])
             .agg(p_hat=("correct", "mean"), vc_bar=("vc_post", "mean"))
             .reset_index())
    return per_q


def temperature_summary(per_q: pd.DataFrame) -> pd.DataFrame:
    """Slopes of p_hat and VC in T, plus the ratio between them."""
    rows = []
    for col in ("p_hat", "vc_bar"):
        agg = per_q.groupby("temperature")[col].agg(["mean", "std", "count"])
        slope = _ols_slope(agg.index.to_numpy(dtype=float), agg["mean"].to_numpy())
        rows.append({"quantity": col, "slope_per_unit_T": slope,
                     "range": float(agg["mean"].max() - agg["mean"].min()),
                     "min": float(agg["mean"].min()), "max": float(agg["mean"].max())})
    out = pd.DataFrame(rows)
    p_range = float(out.loc[out["quantity"] == "p_hat", "range"].iloc[0])
    v_range = float(out.loc[out["quantity"] == "vc_bar", "range"].iloc[0])
    out["interpretation"] = [
        "", ("VC is not a function of the decoder: p_hat moves "
             f"{p_range / v_range:.1f}x more than VC across the same sweep")
        if v_range > 1e-9 else "VC is constant in T while p_hat is not"]
    return out


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return NAN
    x = x - x.mean()
    denom = float((x ** 2).sum())
    return float((x * (y - y.mean())).sum() / denom) if denom > 0 else NAN


# --------------------------------------------------------------------------
# 8.2 Elicitation paraphrase
# --------------------------------------------------------------------------

def paraphrase_reliability(cfg: Config, gen: Generator,
                           questions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-question variance of VC across equivalent phrasings.

    Test-retest reliability is a basic psychometric requirement: an instrument
    that returns a different number when the same question is asked in different
    words is not measuring a stable quantity.
    """
    variants = list(cfg.get("phase5.paraphrase.variants"))
    frames = []
    for v in variants:
        df = gen.draw_answers(questions, n_max=1, prompt_variant=v,
                              store_per_position=False)
        frames.append(df[["q_id", "vc_post"]].assign(prompt_variant=v))
    allf = pd.concat(frames, ignore_index=True)

    per_q = (allf.groupby("q_id")["vc_post"]
             .agg(mean="mean", sd="std", vmin="min", vmax="max", n="count")
             .reset_index())
    total_var = float(allf["vc_post"].var())
    within_var = float((per_q["sd"] ** 2).mean())
    summary = {
        "n_variants": len(variants),
        "mean_within_question_sd": float(per_q["sd"].mean()),
        "mean_range": float((per_q["vmax"] - per_q["vmin"]).mean()),
        "total_variance": total_var,
        "variance_from_prompt_wording": within_var,
        # Reliability in the psychometric sense: the share of variance that is
        # signal about the question rather than about the wording.
        "reliability_icc": float(1 - within_var / total_var) if total_var > 0 else NAN,
    }
    return per_q, summary
