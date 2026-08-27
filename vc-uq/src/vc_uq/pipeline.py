"""Phase orchestration, in the protocol's build order (section 10).

  1. Phase 5.1 temperature sweep      -- does the headline effect exist at all?
  2. Phase 0 instrument validation    -- GATE: kappa >= 0.7, AUROC >= 0.85
  3. Phase 1 full generation + cache  -- GATE: KM flat by N_MAX
  4. Phase 3 survival, beta, U/A, 2x2 -- beta known BEFORE any alpha is chosen
  5. Phase 2 descriptive
  6. Phase 4 CLM + LTT                -- GATE: Lambda_hat non-empty, non-vacuous
  7. Phases 5.2-5.6, Phase 6

The order is not cosmetic. The temperature sweep is first because it is the
cheapest signal and the strongest claim. Phase 0 is second because nothing
downstream is interpretable through a blunt instrument. ``beta`` precedes any
choice of ``alpha`` because choosing alpha below beta produces a blank table for
arithmetic reasons that have nothing to do with VC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import calibration, cluster, invariance, judge, phase0, plots, survival, transfer
from .backends import build_embedder, build_lm, build_nli
from .clm import build_traces, run_phase4, stratified_discount
from .config import Config
from .generate import Generator, run_phase1
from .pitfalls import run_checks
from .schemas import QUESTIONS_SCHEMA, assert_split_by_question
from .store import Store


@dataclass
class PipelineState:
    cfg: Config
    store: Store
    questions: pd.DataFrame | None = None
    answers: pd.DataFrame | None = None
    gate: phase0.GateResult | None = None
    beta: float | None = None
    km: pd.DataFrame | None = None
    results: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _oracle(answers: pd.DataFrame) -> pd.DataFrame:
    """Known-truth column, available only under the simulated backend."""
    from .backends.mock import SEM_TAG
    tags = answers["answer"].astype(str).map(
        lambda t: (SEM_TAG.search(t).group(1) if SEM_TAG.search(t) else ""))
    out = answers.copy()
    out["correct_oracle"] = tags.str.endswith("|gold")
    return out


# --------------------------------------------------------------------------
# Step 1 -- temperature sweep
# --------------------------------------------------------------------------

def step1_temperature(state: PipelineState) -> PipelineState:
    cfg, store = state.cfg, state.store
    gen = Generator(cfg, store)
    embedder = build_embedder(cfg)
    questions = state.questions

    # tau is fixed ONCE for the whole sweep. Re-deriving it at each temperature
    # would let the correctness threshold move with T and confound the very
    # effect being measured -- p_hat would shift partly because the ruler did.
    tau_holder: dict[str, float] = {}

    def judge_fn(df: pd.DataFrame) -> pd.DataFrame:
        scored, _ = judge.score_answers(cfg, df, questions, embedder=embedder)
        if "tau" not in tau_holder:
            tau = cfg.get("judge.tau_star", None)
            if tau is None:
                tau = float(scored["e_cos"].median())
                state.notes.append(
                    f"temperature sweep used a provisional tau = {tau:.3f} (median "
                    "e_cos at the first temperature, held fixed across the sweep); "
                    "Phase 0 must select tau_star before any headline number is quoted")
            tau_holder["tau"] = float(tau)
        scored = judge.apply_tau(scored, tau_holder["tau"])
        scored["correct"] = scored["correct_cos"].astype(bool)
        return scored

    per_q = invariance.temperature_sweep(cfg, gen, questions, judge_fn)
    summary = invariance.temperature_summary(per_q)
    store.write_table("phase5_1_temperature_per_question", per_q)
    store.write_table("phase5_1_temperature_summary", summary)
    plots.temperature_sweep(per_q, store.figure_path("temperature_sweep"), cfg)
    state.results["step1_temperature"] = summary.to_dict("records")
    return state


# --------------------------------------------------------------------------
# Step 2 -- Phase 0 gate
# --------------------------------------------------------------------------

def step2_gate(state: PipelineState, *, simulate_labels: bool = False) -> PipelineState:
    cfg, store = state.cfg, state.store
    answers, questions = state.answers, state.questions
    embedder = build_embedder(cfg)

    scored, questions = judge.score_answers(cfg, answers, questions, embedder=embedder)
    scored = judge.judge_nli(cfg, scored, questions)
    state.questions = questions

    sheet = phase0.build_label_sheet(cfg, scored, questions)
    if simulate_labels:
        labelled = phase0.simulate_human_labels(sheet, _oracle(scored))
    else:
        path = store.table_path("phase0_label_sheet")
        store.write_table("phase0_label_sheet", sheet)
        raise SystemExit(
            f"Phase 0 needs hand labels. Wrote {len(sheet)} stratified pairs to {path}.\n"
            "Fill the correct_human column (TRUE/FALSE) and re-run with "
            "--labels <path>. Do not skip this: nothing downstream is interpretable "
            "through an unvalidated criterion.")

    labelled = labelled.merge(
        scored[["q_id", "draw_idx", "correct_nli"]], on=["q_id", "draw_idx"], how="left")
    gate = phase0.run_gate(cfg, store, labelled, scored, questions, embedder=embedder)
    state.gate = gate

    # tau_star and the primary criterion are OUTPUTS of the gate, so they are
    # written back into the live config here. Nothing downstream may define
    # correctness before this point.
    state.cfg = cfg.with_overrides([f"judge.tau_star={gate.tau_star}",
                                    f"judge.primary={gate.primary}"])
    scored = judge.apply_tau(scored, gate.tau_star)
    state.answers = scored
    store.write_processed("answers", scored)

    state.results["step2_gate"] = gate.as_dict()
    if not gate.passed:
        state.notes.append(
            "Phase 0 gate FAILED: " + "; ".join(gate.reasons) +
            " -- downstream numbers describe the instrument, not the model.")
    return state


# --------------------------------------------------------------------------
# Step 3 -- Phase 1 generation
# --------------------------------------------------------------------------

def step3_generate(state: PipelineState) -> PipelineState:
    cfg, store = state.cfg, state.store
    info = run_phase1(cfg, store, state.questions)
    state.answers = store.read_answers()
    state.questions = store.read_questions()
    assert_split_by_question(state.answers)
    state.results["step3_generation"] = info
    return state


# --------------------------------------------------------------------------
# Step 4 -- Phase 3 survival, U/A, diversity, product rule
# --------------------------------------------------------------------------

def step4_survival(state: PipelineState) -> PipelineState:
    cfg, store = state.cfg, state.store
    answers = judge.attach_correct(cfg, state.answers)
    nli = build_nli(cfg)
    answers = cluster.cluster_frame(cfg, answers, nli=nli)
    state.answers = answers

    qs = survival.question_stats(answers)
    div = cluster.question_diversity(answers)
    # The question schema declares these columns as nullable placeholders; drop
    # them before merging so pandas does not produce _x/_y pairs.
    recomputed = [c for c in list(qs.columns) + list(div.columns) if c != "q_id"]
    questions = (state.questions
                 .drop(columns=[c for c in recomputed if c in state.questions.columns],
                       errors="ignore")
                 .merge(qs, on="q_id", how="left")
                 .merge(div, on="q_id", how="left"))

    n_max = int(cfg.get("generation.n_max"))
    km = survival.kaplan_meier(questions["K_q"], questions["censored"], n_max)
    beta = survival.beta_from_km(km)
    flat = survival.km_flatness(km, cfg)

    part = survival.partition_U(answers)
    questions = questions.drop(columns=["in_U"], errors="ignore").merge(
        part[["q_id", "in_U"]], on="q_id", how="left")
    questions["in_U"] = questions["in_U"].astype("boolean").fillna(
        questions["censored"].astype("boolean"))

    # h_tok is a per-answer quantity; the U-detection comparison needs it at
    # question level, using the same summary the stopping rule would see.
    htok = answers.groupby("q_id")["h_tok_mean"].mean().rename("h_tok_mean").reset_index()
    questions = questions.drop(columns=["h_tok_mean"], errors="ignore").merge(
        htok, on="q_id", how="left")

    hz = survival.hazard(answers, int(cfg.get("phase3.hazard_max_k")))
    budget = survival.budget_table(questions, cfg)
    u_det = survival.u_detection(questions, cfg)
    twobytwo = survival.diversity_2x2(questions)

    eval_answers = answers[answers["split"] == "eval"]
    u_ids = set(questions.loc[questions["in_U"].astype("boolean").fillna(False), "q_id"])
    curves = [survival.product_rule_curve(eval_answers, cfg, subset="all")]
    for name, sub in (("U", eval_answers[eval_answers["q_id"].isin(u_ids)]),
                      ("A", eval_answers[~eval_answers["q_id"].isin(u_ids)])):
        if len(sub):
            curves.append(survival.product_rule_curve(sub, cfg, subset=name))
    curve = pd.concat([c for c in curves if len(c)], ignore_index=True)
    diverg = survival.product_rule_divergence(curve) if len(curve) else pd.DataFrame()

    if cfg.get("judge.primary") == "cos" and state.gate is not None:
        taus = phase0.tau_sensitivity_values(cfg, state.gate.tau_star)
        store.write_table("phase3_beta_vs_tau",
                          survival.beta_vs_tau(answers, "e_cos", taus))

    store.write_table("phase3_km", km)
    store.write_table("phase3_hazard", hz)
    store.write_table("phase3_budget", budget)
    store.write_table("phase3_u_detection", u_det)
    store.write_table("phase3_diversity_2x2", twobytwo)
    store.write_table("phase3_product_rule_curve", curve)
    store.write_table("phase3_product_rule_divergence", diverg)
    store.write_json("phase3_summary", {
        "beta": beta, "km_flatness": flat,
        "n_in_U": int(questions["in_U"].astype("boolean").fillna(False).sum()),
        "n_questions": int(len(questions)),
        "overlap": survival.overlap_statement(questions, "vc_pre", 0.8),
    })
    plots.km_curve(km, store.figure_path("km"), cfg, beta=beta)
    if len(hz):
        plots.hazard_curve(hz, store.figure_path("hazard"), cfg)
    if len(curve):
        plots.product_rule(curve[curve["subset"] == "all"],
                           store.figure_path("product_rule"), cfg)
    plots.budget_collapse(questions, store.figure_path("budget_collapse"), cfg)

    state.questions = questions
    state.km = km
    state.beta = beta
    state.results["step4_survival"] = {
        "beta": beta, "km_flat": flat, "u_detection": u_det.to_dict("records"),
        "diversity_2x2": twobytwo.to_dict("records"),
        "product_rule_divergence": diverg.to_dict("records"),
        "budget": budget.to_dict("records"),
    }
    if not flat["flat"]:
        state.notes.append(flat["reason"])
    store.write_questions(questions)
    return state


# --------------------------------------------------------------------------
# Step 5 -- Phase 2 descriptive
# --------------------------------------------------------------------------

def step5_descriptive(state: PipelineState) -> PipelineState:
    cfg, store = state.cfg, state.store
    answers = state.answers[state.answers["split"] == "eval"]
    questions = state.questions[state.questions["split"] == "eval"]

    hist_post = calibration.vc_histogram(answers, "vc_post")
    hist_pre = calibration.vc_histogram(questions, "vc_pre")
    rel = calibration.reliability(answers, cfg)
    mono = calibration.monotonicity(rel)
    var = calibration.vc_variance_decomposition(answers, cfg)
    info = calibration.information_gain(questions, cfg)
    aurocs = calibration.two_aurocs(answers, questions, cfg)
    br = calibration.brier(answers)
    rho = calibration.error_dependence(answers)

    store.write_table("phase2_vc_hist_post", hist_post)
    store.write_table("phase2_vc_hist_pre", hist_pre)
    store.write_table("phase2_reliability", rel)
    store.write_table("phase2_aurocs", aurocs)
    store.write_json("phase2_summary", {
        "discreteness_post": calibration.discreteness_summary(answers, "vc_post"),
        "discreteness_pre": calibration.discreteness_summary(questions, "vc_pre"),
        "monotonicity": mono, "variance_decomposition": var,
        "information_gain": info, "brier": br, "error_dependence": rho,
        "ece_model_bins": calibration.model_defined_ece(rel),
    })
    if len(rel):
        plots.reliability_diagram(rel, store.figure_path("reliability"), cfg)
    if len(hist_post):
        plots.vc_histogram(hist_post, store.figure_path("vc_hist_post"), cfg, "vc_post")

    state.results["step5_descriptive"] = {
        "reliability": rel.to_dict("records"), "monotonicity": mono,
        "variance_decomposition": var, "information_gain": info,
        "aurocs": aurocs.to_dict("records"), "brier": br, "error_dependence": rho,
    }
    return state


# --------------------------------------------------------------------------
# Step 6 -- Phase 4 CLM + LTT
# --------------------------------------------------------------------------

def step6_clm(state: PipelineState) -> PipelineState:
    cfg, store = state.cfg, state.store
    answers, questions = state.answers, state.questions
    n_max = int(cfg.get("generation.n_max"))
    embedder = build_embedder(cfg)

    if bool(cfg.get("phase4.restrict_to_A")):
        keep = set(questions.loc[~questions["in_U"].astype("boolean").fillna(False),
                                 "q_id"])
        state.notes.append(
            "Phase 4 primary run is restricted to A, so beta_eff = 0 and small alpha is "
            "feasible. This conditions on solvability, which is information "
            "unavailable at deployment.")
        beta_eff = 0.0
    else:
        keep = set(questions["q_id"])
        beta_eff = float(state.beta or 0.0)

    def traces_for(split: str):
        sub = answers[(answers["split"] == split) & (answers["q_id"].isin(keep))]
        qs = questions[questions["q_id"].isin(set(sub["q_id"]))]
        return build_traces(cfg, sub, qs, embedder=embedder) if len(sub) else []

    calib_traces = traces_for("calib")
    eval_traces = traces_for("eval")
    if not calib_traces or not eval_traces:
        state.notes.append("Phase 4 skipped: calib or eval split is empty after "
                           "restriction to A.")
        return state

    all_rows, per_alpha = [], {}
    for alpha in cfg.get("phase4.alphas"):
        res = run_phase4(cfg, calib_traces, eval_traces, beta=beta_eff,
                         alpha=float(alpha), n_max=n_max)
        per_alpha[str(alpha)] = {"feasibility": res.feasibility,
                                 "vacuity": res.vacuity, "notes": res.notes}
        if len(res.headline):
            all_rows.append(res.headline.assign(alpha=float(alpha)))
        state.notes.extend(res.notes)

        if len(res.headline):
            certified = res.headline[res.headline["certified"]]
            vc_row = certified[certified["rule"] == "vc_product"]
            if len(vc_row):
                r = vc_row.iloc[0]
                lam = (float(r["lambda_qual_hat"]), float(r["lambda_div_hat"]),
                       float(r["lambda_stop_hat"]))
                for by in ("dataset", "p_hat"):
                    strat = stratified_discount(cfg, eval_traces, "vc_product",
                                                float(alpha), lam, by=by)
                    if len(strat):
                        store.write_table(
                            f"phase4_discount_stability__alpha{alpha}__{by}", strat)

    headline = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    store.write_table("phase4_headline", headline)
    store.write_json("phase4_summary", {"beta_used": beta_eff,
                                        "restricted_to_A": bool(cfg.get("phase4.restrict_to_A")),
                                        "per_alpha": per_alpha})
    state.results["step6_clm"] = {"headline": headline.to_dict("records"),
                                  "per_alpha": per_alpha}
    return state


# --------------------------------------------------------------------------
# Step 7 -- remaining invariance probes and transfer
# --------------------------------------------------------------------------

def step7_invariance_transfer(state: PipelineState) -> PipelineState:
    cfg, store = state.cfg, state.store
    gen = Generator(cfg, store)
    embedder = build_embedder(cfg)
    questions = state.questions
    eval_q = questions[questions["split"] == "eval"]
    probe_q = eval_q.head(int(cfg.get("phase5.temperature_sweep.n_questions")))

    def judge_fn(df: pd.DataFrame) -> pd.DataFrame:
        scored, _ = judge.score_answers(cfg, df, questions, embedder=embedder)
        tau = cfg.get("judge.tau_star", None)
        tau = float(tau) if tau is not None else float(
            state.gate.tau_star if state.gate else scored["e_cos"].median())
        scored = judge.apply_tau(scored, tau)
        scored["correct"] = scored["correct_cos"].astype(bool)
        return scored

    out: dict = {}

    per_q, para = invariance.paraphrase_reliability(cfg, gen, probe_q)
    store.write_table("phase5_2_paraphrase_per_question", per_q)
    out["paraphrase"] = para

    scales_df, scales = invariance.scale_reframing(cfg, gen, probe_q)
    store.write_table("phase5_3_scale_reframing", scales_df)
    out["scale_reframing"] = scales

    syco_df, syco = invariance.sycophancy_probe(
        cfg, gen, probe_q, state.answers[state.answers["q_id"].isin(set(probe_q["q_id"]))])
    store.write_table("phase5_4_sycophancy", syco_df)
    out["sycophancy"] = syco

    nli = build_nli(cfg)
    forced_df, forced = invariance.forced_decode(
        cfg, gen, probe_q, judge_fn,
        cluster_fn=lambda d: cluster.cluster_frame(cfg, d, nli=nli))
    store.write_table("phase5_5_forced_decode", forced_df)
    out["forced_decode"] = forced

    out["prehoc"] = invariance.prehoc_probes(cfg, questions,
                                             store.read_vc_pre_repeats())

    eval_answers = state.answers[state.answers["split"] == "eval"]
    table, tsum = transfer.transfer_study(cfg, eval_answers)
    store.write_table("phase6_transfer", table)
    compounding = transfer.isotonic_preserves_compounding(cfg, eval_answers)
    store.write_table("phase6_isotonic_compounding", compounding)
    out["transfer"] = tsum

    store.write_json("phase5_6_summary", out)
    state.results["step7"] = out
    return state


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------

def run_all(cfg: Config, *, simulate_labels: bool = False,
            skip: tuple[str, ...] = (), store: Store | None = None) -> PipelineState:
    from .datasets import build_questions, split_summary

    # A full run always gets its own timestamped directory; per-phase CLI
    # commands attach to an existing one instead.
    store = store if store is not None else Store(cfg)
    state = PipelineState(cfg=cfg, store=store)
    state.questions = build_questions(cfg)
    store.write_table("dataset_splits", split_summary(state.questions))

    # Generation must precede the gate (the gate labels real draws), and the
    # gate precedes the sweep so the sweep can use the validated tau_star rather
    # than a provisional one. The protocol's ordering of the sweep first is
    # about it being the cheapest STANDALONE probe -- `vc_uq step1` still runs
    # it on its own, before anything else exists.
    steps = [
        ("step3_generate", step3_generate),
        ("step2_gate", lambda s: step2_gate(s, simulate_labels=simulate_labels)),
        ("step1_temperature", step1_temperature),
        ("step4_survival", step4_survival),
        ("step5_descriptive", step5_descriptive),
        ("step6_clm", step6_clm),
        ("step7", step7_invariance_transfer),
    ]
    for name, fn in steps:
        if name in skip:
            state.notes.append(f"{name} skipped by request")
            continue
        state = fn(state)

    report = run_checks(
        cfg, answers=state.answers, questions=state.questions, km=state.km,
        beta=state.beta,
        gate=state.results.get("step2_gate"),
        reliability_table=pd.DataFrame(state.results.get("step5_descriptive", {})
                                       .get("reliability", [])),
        error_rho=state.results.get("step5_descriptive", {})
        .get("error_dependence", {}).get("rho"),
    )
    store.write_table("pitfalls", report.to_frame())
    store.write_json("pitfalls_summary", report.summary())
    state.results["pitfalls"] = report.summary()
    store.write_json("run_notes", {"notes": state.notes})
    store.finalize(status="complete", command="all", extra={
        "pitfalls": report.summary(),
        "beta": state.beta,
        "gate_passed": bool(state.gate.passed) if state.gate else None,
        "n_questions": int(len(state.questions)) if state.questions is not None else None,
        "n_answers": int(len(state.answers)) if state.answers is not None else None,
        "notes": state.notes,
    })
    return state
