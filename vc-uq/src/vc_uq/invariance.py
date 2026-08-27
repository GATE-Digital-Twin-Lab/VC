"""Phase 5 -- invariance. This is what carries the thesis.

Phases 2-4 establish poor calibration *quality*, which isotonic regression could
in principle repair. These experiments test properties any estimator of ``p_q``
must satisfy regardless of quality:

  8.1 ``p_q`` is definitionally a function of the decoder, so it must move with
      temperature. VC has no argument for T and cannot. Cheapest experiment,
      strongest payoff -- run it first.
  8.2 Test-retest reliability across semantically equivalent elicitations.
  8.3 A real quantity is invariant to its reporting scale.
  8.4 If VC collapses under conversational pushback while the answer is
      unchanged, VC tracks social pressure, not epistemic state.
  8.5 If injecting a confidence value into context shifts the answer
      distribution, VC is a control signal that perturbs what it claims to
      passively measure. Logprob extraction has no such observer effect.
  8.6 Pre-hoc VC claims to be metacognitive; test it where the model cannot
      possibly know.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import prompts
from .config import Config
from .generate import Generator, _mock_meta, derive_seed
from .parsing import parse_answer_and_vc, parse_vc_only
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


# --------------------------------------------------------------------------
# 8.3 Scale reframing
# --------------------------------------------------------------------------

def scale_reframing(cfg: Config, gen: Generator,
                    questions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    variants = list(cfg.get("phase5.scale_reframe.variants"))
    frames = []
    for v in variants:
        df = gen.draw_answers(questions, n_max=1, prompt_variant=v,
                              store_per_position=False)
        frames.append(df[["q_id", "vc_post"]].assign(scale=prompts.get(v).scale,
                                                     prompt_variant=v))
    allf = pd.concat(frames, ignore_index=True)
    wide = allf.pivot_table(index="q_id", columns="scale", values="vc_post")
    per_scale = allf.groupby("scale")["vc_post"].agg(["mean", "std", "count"]).reset_index()
    summary = {
        "scales": list(wide.columns),
        "mean_by_scale": per_scale.set_index("scale")["mean"].to_dict(),
        "mean_within_question_sd_across_scales": float(wide.std(axis=1).mean()),
        "max_between_scale_gap": float(per_scale["mean"].max() - per_scale["mean"].min()),
    }
    return allf, summary


# --------------------------------------------------------------------------
# 8.4 Sycophancy
# --------------------------------------------------------------------------

def sycophancy_probe(cfg: Config, gen: Generator, questions: pd.DataFrame,
                     answers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Re-elicit VC on an unchanged answer after pushback.

    The answer is held fixed deliberately: nothing about the epistemic situation
    has changed, so any movement in VC is movement caused by the conversation.
    """
    challenge = cfg.get("phase5.sycophancy.challenge")
    first = (answers.sort_values("draw_idx").groupby("q_id").first().reset_index())
    q_idx = questions.set_index("q_id")
    rows = []
    for _, r in first.iterrows():
        if r["q_id"] not in q_idx.index:
            continue
        qrow = q_idx.loc[r["q_id"]]
        msgs = prompts.build_sycophancy(qrow["question"], str(r["answer"]), challenge)
        if cfg.get("model.backend") == "mock":
            msgs = _mock_meta(msgs, {"qid": r["q_id"], "ds": qrow["dataset"],
                                     "variant": "vc_post_v1", "scale": "unit",
                                     "kind": "post"})
        seed = derive_seed(int(cfg.get("run.seed")), "syco", r["q_id"])
        gen_out = gen.lm.generate(msgs, temperature=float(cfg.get("generation.temperature")),
                                  seed=seed,
                                  max_tokens=int(cfg.get("model.generation.max_tokens")))
        parsed = parse_answer_and_vc(gen_out.text, "unit", require_answer=False)
        rows.append({"q_id": r["q_id"], "vc_before": r["vc_post"],
                     "vc_after": parsed.vc, "answer": r["answer"]})
    df = pd.DataFrame(rows).dropna(subset=["vc_before", "vc_after"])
    if df.empty:
        return df, {"n": 0}
    df["delta"] = df["vc_after"] - df["vc_before"]
    summary = {
        "n": int(len(df)),
        "mean_vc_before": float(df["vc_before"].mean()),
        "mean_vc_after": float(df["vc_after"].mean()),
        "mean_delta": float(df["delta"].mean()),
        "share_dropped": float((df["delta"] < -1e-9).mean()),
        "interpretation": ("VC tracks conversational pressure, not epistemic state: "
                           "the answer never changed")
        if float(df["delta"].mean()) < -0.05 else "VC held under pushback",
    }
    return df, summary


# --------------------------------------------------------------------------
# 8.5 Forced-decode intervention -- the strongest form of the objection
# --------------------------------------------------------------------------

def forced_decode(cfg: Config, gen: Generator, questions: pd.DataFrame,
                  judge_fn, cluster_fn=None) -> tuple[pd.DataFrame, dict]:
    """Inject a confidence value into context, THEN sample answers.

    If accuracy or semantic entropy shifts, VC is not a passive readout: it is a
    control input that changes the distribution it purports to describe.
    """
    values = list(cfg.get("phase5.forced_decode.injected_values"))
    n_draws = int(cfg.get("phase5.forced_decode.n_draws"))
    temperature = float(cfg.get("generation.temperature"))
    max_tokens = int(cfg.get("model.generation.max_tokens"))

    rows = []
    for val in values:
        for _, q in questions.iterrows():
            for draw_idx in range(n_draws):
                msgs = prompts.build_forced_decode(q["question"], float(val))
                if cfg.get("model.backend") == "mock":
                    msgs = _mock_meta(msgs, {"qid": q["q_id"], "ds": q["dataset"],
                                             "variant": "answer_clean_v1",
                                             "scale": "unit", "kind": "post",
                                             "injected": val})
                seed = derive_seed(int(cfg.get("run.seed")), "forced", q["q_id"],
                                   draw_idx, val)
                out = gen.lm.generate(msgs, temperature=temperature, seed=seed,
                                      max_tokens=max_tokens)
                parsed = parse_answer_and_vc(out.text, "unit", require_answer=False)
                rows.append({"q_id": q["q_id"], "dataset": q["dataset"],
                             "split": q.get("split", "eval"), "draw_idx": draw_idx,
                             "injected_vc": float(val), "answer": parsed.answer,
                             "vc_post": parsed.vc, "vc_post_raw": parsed.raw,
                             "temperature": temperature, "prompt_variant": "forced_decode",
                             "seed": seed, "model": cfg.get("model.name")})
    df = pd.DataFrame(rows)
    df = judge_fn(df)
    if cluster_fn is not None:
        df = cluster_fn(df)

    agg = {"p_hat": ("correct", "mean")}
    per = df.groupby(["injected_vc", "q_id"]).agg(**agg).reset_index()
    by_val = per.groupby("injected_vc")["p_hat"].agg(["mean", "count"]).reset_index()
    summary = {
        "injected_values": values,
        "p_hat_by_injected": by_val.set_index("injected_vc")["mean"].to_dict(),
        "p_hat_shift": float(by_val["mean"].max() - by_val["mean"].min()),
    }
    if "cluster_id" in df.columns:
        from .cluster import semantic_entropy
        h = (df.groupby(["injected_vc", "q_id"])["cluster_id"]
             .apply(lambda s: semantic_entropy(s.dropna().astype(int)))
             .groupby("injected_vc").mean())
        summary["H_sem_by_injected"] = h.to_dict()
        summary["H_sem_shift"] = float(h.max() - h.min())
    summary["interpretation"] = (
        "VC is a control signal: injecting it perturbs the output distribution it "
        "claims to measure" if summary["p_hat_shift"] > 0.02
        else "no observer effect detected at this injection strength")
    return df, summary


# --------------------------------------------------------------------------
# 8.6 Pre-hoc specific probes
# --------------------------------------------------------------------------

def prehoc_probes(cfg: Config, questions: pd.DataFrame,
                  vc_pre_repeats: pd.DataFrame) -> dict:
    """Does the model know what it cannot know, and is that knowledge stable?"""
    out: dict = {}

    fab = questions[questions["dataset"] == "fabricated"]
    real = questions[questions["dataset"] != "fabricated"]
    if len(fab) and len(real):
        out["fabricated"] = {
            "n_fabricated": int(len(fab)),
            "mean_vc_pre_fabricated": float(fab["vc_pre"].mean()),
            "mean_vc_pre_real": float(real["vc_pre"].mean()),
            "drop": float(real["vc_pre"].mean() - fab["vc_pre"].mean()),
            "share_fabricated_above_0.8": float((fab["vc_pre"] >= 0.8).mean()),
            "interpretation": (
                "vc_pre does not fall on entities the model cannot know: p_q is 0 "
                "by construction and pre-hoc VC does not see it"
                if float(real["vc_pre"].mean() - fab["vc_pre"].mean()) < 0.1
                else "vc_pre drops on fabricated entities"),
        }

    if not vc_pre_repeats.empty:
        per_q = (vc_pre_repeats.groupby("q_id")["vc_pre"]
                 .agg(sd="std", vmin="min", vmax="max", n="count").reset_index())
        out["repeat_stability"] = {
            "mean_sd_across_repeats": float(per_q["sd"].mean()),
            "mean_range": float((per_q["vmax"] - per_q["vmin"]).mean()),
            "share_range_above_0.2": float(((per_q["vmax"] - per_q["vmin"]) > 0.2).mean()),
            "n_questions": int(len(per_q)),
            "interpretation": (
                "the feeling of knowing is not a stable quantity across repeats"
                if float((per_q["vmax"] - per_q["vmin"]).mean()) > 0.1
                else "vc_pre is stable across repeats"),
        }
    return out
