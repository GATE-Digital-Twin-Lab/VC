"""Phase orchestration.

**Step names encode the protocol's section 10 numbering, NOT the order they
run in.** ``run_all`` executes them like this:

  step3_generate     Phase 1 full generation + cache  -- GATE: KM flat by N_MAX
  step2_gate         Phase 0 instrument validation    -- GATE: kappa >= 0.7, AUROC >= 0.85
  step1_temperature  Phase 5.1 temperature sweep      -- does the headline effect exist at all?
  step4_survival     Phase 3 survival, beta, U/A, 2x2 -- beta known BEFORE any alpha is chosen
  step5_descriptive  Phase 2 descriptive
  step6_clm          Phase 4 CLM + LTT                -- GATE: Lambda_hat non-empty, non-vacuous
  step7              Phases 5.2-5.6, Phase 6

The first three are swapped relative to section 10 because of two hard
dependencies: the gate labels REAL DRAWS, so generation must precede it; and
once the gate has run, the sweep can use the validated ``tau_star`` instead of a
provisional one. Section 10 puts the sweep first for a different reason -- it is
the cheapest STANDALONE probe, answering "is there an effect at all?" before
anyone commits to a full generation. That path is still available and still
first: ``vc_uq step1`` runs the sweep alone, against a provisional tau, and says
so in a note.

Nothing after that is negotiable. Phase 0 gates everything because no downstream
number is interpretable through an unvalidated instrument. ``beta`` is fixed in
step 4, before step 6 chooses any ``alpha``, because an alpha below beta produces
a blank certification table for arithmetic reasons that have nothing to do with
VC -- and a reader who saw the blank table without beta would draw exactly the
wrong conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import calibration, cluster, invariance, judge, phase0, plots, survival, transfer
from .backends import build_embedder, build_nli
from .clm import build_traces, run_phase4, stratified_discount
from .config import Config
from .generate import Generator, run_phase1
from .pitfalls import run_checks
from .schemas import assert_split_by_question
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


def _pass(answers: pd.DataFrame, name: str, split: str | None = None) -> pd.DataFrame:
    """Rows from ONE generation pass (protocol 6.4), optionally one split.

    Every phase states which pass it reads. ``classify`` decides U and the
    survival picture; ``downstream`` is what Phase 2, Phase 4 and 6.7 are
    calibrated and evaluated on. A phase that took the union would be averaging
    over 2*N_MAX draws no deployed caller ever observes together, and -- for
    anything conditioned on ``in_U`` -- would be measuring the draws that
    defined the condition.
    """
    out = answers[answers["draw_set"] == name]
    if split is not None:
        out = out[out["split"] == split]
    return out


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
    cfg = state.cfg
    # The generator needs the run-level store (it appends to the shared raw
    # cache); only the derived tables are phase-scoped.
    gen = Generator(cfg, state.store)
    store = state.store.phase("invariance")
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
    store.write_table("temperature_per_question", per_q)
    store.write_table("temperature_summary", summary)
    plots.temperature_sweep(per_q, store.figure_path("temperature_sweep"), cfg)
    state.results["step1_temperature"] = summary.to_dict("records")
    return state


# --------------------------------------------------------------------------
# Step 2 -- Phase 0 gate
# --------------------------------------------------------------------------

def step2_gate(state: PipelineState, *, simulate_labels: bool = False,
               labels_path: str | None = None) -> PipelineState:
    cfg, store = state.cfg, state.store.phase("gate")
    answers, questions = state.answers, state.questions
    embedder = build_embedder(cfg)

    scored, questions = judge.score_answers(cfg, answers, questions, embedder=embedder)
    scored = judge.judge_nli(cfg, scored, questions)
    state.questions = questions

    # The sheet is rebuilt from the CURRENT scored answers every time, and a
    # label file contributes only its correct_human column. e_cos moves whenever
    # the scored population changes, so it must never be carried in from a file.
    sheet = phase0.build_label_sheet(cfg, scored, questions)
    label_report = None
    if simulate_labels:
        labelled = phase0.simulate_human_labels(sheet, _oracle(scored))
    elif labels_path:
        try:
            labelled, label_report = phase0.load_hand_labels(cfg, sheet, labels_path)
        except phase0.LabelCoverageError as exc:
            # Persist the coverage table before failing: the point of refusing is
            # to say which strata still need labelling.
            store.write_json("label_coverage", exc.report)
            raise
        store.write_json("label_coverage", label_report)
        for w in label_report["warnings"]:
            state.notes.append(f"Phase 0 labels: {w}")
    else:
        path = store.table_path("label_sheet")
        store.write_table("label_sheet", sheet)
        raise SystemExit(
            f"Phase 0 needs hand labels. Wrote {len(sheet)} stratified pairs to {path}.\n"
            "Fill the correct_human column (TRUE/FALSE), then re-run with\n"
            f"    vc_uq gate --labels {path}\n"
            "Rows are shuffled, so a partially labelled sheet still covers the "
            "e_cos range. Only q_id, draw_set, draw_idx and correct_human are "
            "read back. "
            "Do not skip this: nothing downstream is interpretable through an "
            "unvalidated criterion.")

    # On phase0.LABEL_KEY, not (q_id, draw_idx): the latter matches two rows
    # per label since generation grew a second pass, which would duplicate every
    # labelled pair and silently double-weight it in kappa and tau selection.
    labelled = labelled.merge(
        scored[[*phase0.LABEL_KEY, "correct_nli"]], on=phase0.LABEL_KEY, how="left")
    gate = phase0.run_gate(cfg, store, labelled, scored, questions, embedder=embedder)
    state.gate = gate

    # tau_star and the primary criterion are OUTPUTS of the gate, so they are
    # written back into the live config here. Nothing downstream may define
    # correctness before this point.
    state.cfg = cfg.with_overrides([f"judge.tau_star={gate.tau_star}",
                                    f"judge.primary={gate.primary}"])
    scored = judge.apply_tau(scored, gate.tau_star)
    state.answers = scored
    state.store.write_processed("answers", scored)

    state.results["step2_gate"] = {**gate.as_dict(), "labels": label_report}
    if not gate.passed:
        state.notes.append(
            "Phase 0 gate FAILED: " + "; ".join(gate.reasons) +
            " -- downstream numbers describe the instrument, not the model.")
    return state


# --------------------------------------------------------------------------
# Step 3 -- Phase 1 generation
# --------------------------------------------------------------------------

def step3_generate(state: PipelineState,
                   splits: list[str] | None = None,
                   shard: tuple[int, int] | None = None) -> PipelineState:
    cfg, store = state.cfg, state.store
    info = run_phase1(cfg, store, state.questions, splits=splits, shard=shard)
    state.answers = store.read_answers()
    state.questions = store.read_questions()
    assert_split_by_question(state.answers)
    state.results["step3_generation"] = info
    if shard is not None:
        state.notes.append(
            f"Phase 1 ran shard {info['shard']} only. The other shard(s) must "
            "complete before any phase reads the cache -- a partial shard set is "
            "a question table with holes, not a smaller study.")
    if splits is not None:
        state.notes.append(
            f"Phase 1 generated {info['n_questions_generated']} of "
            f"{info['n_questions']} questions -- split(s) {info['splits']} only. "
            "The cache is a strict subset of the full pass, so an unrestricted "
            "run later reuses these draws; but no phase past the gate should be "
            "quoted until the rest exists.")
    for table, r in info["resume"].items():
        if r["reused_from_cache"]:
            state.notes.append(
                f"Phase 1 {table}: reused {r['reused_from_cache']} of "
                f"{r['requested']} draws from the shared cache and generated "
                f"{r['generated']}.")
    return state


# --------------------------------------------------------------------------
# Step 4 -- Phase 3 survival, U/A, diversity, product rule
# --------------------------------------------------------------------------

def step4_survival(state: PipelineState) -> PipelineState:
    cfg, store = state.cfg, state.store.phase("survival")
    answers = judge.attach_correct(cfg, state.answers)
    nli = build_nli(cfg)
    answers = cluster.cluster_frame(cfg, answers, nli=nli)
    state.answers = answers
    # A prompted judge can emit something that is not one of the three labels.
    # That is a parse failure like any other and is reported, not absorbed.
    if hasattr(nli, "audit"):
        audit = nli.audit()
        store.write_json("nli_audit", audit)
        if audit.get("unparsed_rate", 0.0) > 0.01:
            state.notes.append(
                f"NLI judge returned an unrecognised verdict on "
                f"{audit['unparsed_rate']:.1%} of comparisons; clustering treated "
                "those as 'not the same answer'.")

    # Protocol 6.4: the draws that DECIDE membership in U are never the draws
    # anything downstream is calibrated or evaluated on. Checked, not assumed --
    # the failure is silent, and it looks like a spectacular result.
    classify = _pass(answers, "classify")
    downstream = _pass(answers, "downstream")
    survival.assert_disjoint_draws(classify, downstream)
    if classify.empty:
        raise ValueError(
            "no draws in the classification pass; U cannot be decided. Re-run "
            "Phase 1 -- a cache written before generation.downstream_splits "
            "existed carries no draw_set and will not load at all.")

    qs = survival.question_stats(classify)
    div = cluster.question_diversity(classify)
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
    questions["in_U"] = questions["in_U"].astype("boolean")
    # No fallback to `censored`. The classification pass covers every question by
    # construction, so a null here means draws are missing -- and the fallback
    # that used to fill it silently defined in_U from the question's OWN
    # downstream draws, which is exactly the circularity 6.4 exists to prevent.
    unlabelled = questions.loc[questions["in_U"].isna(), "q_id"]
    if len(unlabelled):
        raise ValueError(
            f"{len(unlabelled)} questions have no classification draws, so U is "
            f"undefined for them (e.g. {list(unlabelled[:5])}). Re-run Phase 1: "
            "the classify pass must cover every question.")

    # h_tok is a per-answer quantity; the U-detection comparison needs it at
    # question level, using the same summary the stopping rule would see.
    htok = (classify.groupby("q_id")["h_tok_mean"].mean()
            .rename("h_tok_mean").reset_index())
    questions = questions.drop(columns=["h_tok_mean"], errors="ignore").merge(
        htok, on="q_id", how="left")

    hz = survival.hazard(classify, int(cfg.get("phase3.hazard_max_k")))
    budget = survival.budget_table(questions, cfg)
    u_det = survival.u_detection(questions, cfg)
    twobytwo = survival.diversity_2x2(questions)

    # 6.7 runs on the DOWNSTREAM pass while in_U came from the classification
    # pass, so "all k wrong" is a measurement on the U subset rather than its
    # definition restated. Read together with the previous block: if these two
    # ever came from the same draws, every U bin would report observed = 1.0.
    eval_answers = _pass(answers, "downstream", split="eval")
    u_ids = set(questions.loc[questions["in_U"].astype("boolean").fillna(False), "q_id"])
    if eval_answers.empty:
        state.notes.append(
            "6.7 product-rule curves skipped: the eval split has no downstream "
            "draws. Add 'eval' to generation.downstream_splits.")
        curve, diverg = pd.DataFrame(), pd.DataFrame()
    else:
        curves = [survival.product_rule_curve(eval_answers, cfg, subset="all")]
        for name, sub in (("U", eval_answers[eval_answers["q_id"].isin(u_ids)]),
                          ("A", eval_answers[~eval_answers["q_id"].isin(u_ids)])):
            if len(sub):
                curves.append(survival.product_rule_curve(sub, cfg, subset=name))
        curve = pd.concat([c for c in curves if len(c)], ignore_index=True)
        diverg = (survival.product_rule_divergence(curve) if len(curve)
                  else pd.DataFrame())

    if cfg.get("judge.primary") == "cos" and state.gate is not None:
        taus = phase0.tau_sensitivity_values(cfg, state.gate.tau_star)
        store.write_table("beta_vs_tau",
                          survival.beta_vs_tau(classify, "e_cos", taus))

    store.write_table("km", km)
    store.write_table("hazard", hz)
    store.write_table("budget", budget)
    store.write_table("u_detection", u_det)
    store.write_table("diversity_2x2", twobytwo)
    store.write_table("product_rule_curve", curve)
    store.write_table("product_rule_divergence", diverg)
    store.write_json("summary", {
        # Both describe the classification pass, and only that pass: beta is the
        # KM plateau over its K_q, n_in_U counts the questions it never got
        # right. Reported together so a reader can see they agree.
        "beta": beta, "km_flatness": flat,
        "draw_set_for_U": "classify",
        "n_classify_draws": int(len(classify)),
        "n_downstream_draws": int(len(downstream)),
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

    # Persist the enriched answers, not just the scored ones. step2 wrote this
    # table BEFORE clustering, so `cluster_id`, `f` and `correct` existed only in
    # memory -- which made every per-phase command after `survival` fail on a
    # missing column, and would have put every draw of every question into one
    # cluster had it not. Section 1 requires each phase to be re-runnable from
    # the cache; that only holds if each phase writes what the next one reads.
    state.store.write_processed("answers", answers)

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
    state.store.write_questions(questions)
    return state


# --------------------------------------------------------------------------
# Step 5 -- Phase 2 descriptive
# --------------------------------------------------------------------------

def step5_descriptive(state: PipelineState) -> PipelineState:
    cfg, store = state.cfg, state.store.phase("descriptive")
    # The downstream pass, not the classification pass: Phase 2 describes the
    # draws the rest of the study is certified on.
    answers = _pass(state.answers, "downstream", split="eval")
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

    store.write_table("vc_hist_post", hist_post)
    store.write_table("vc_hist_pre", hist_pre)
    store.write_table("reliability", rel)
    store.write_table("aurocs", aurocs)
    store.write_json("summary", {
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
    cfg, store = state.cfg, state.store.phase("clm")
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
        # Downstream draws only. `keep` is derived from in_U, which the
        # classification pass decided -- so restricting to A here conditions on
        # information from draws these traces do not contain, which is what
        # makes the LTT guarantee hold.
        sub = _pass(answers, "downstream", split=split)
        sub = sub[sub["q_id"].isin(keep)]
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
                                 "vacuity": res.vacuity, "notes": res.notes,
                                 "ltt": res.ltt_meta, "traces": res.trace_audit}
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
                            f"discount_stability__alpha{alpha}__{by}", strat)

    headline = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    store.write_table("headline", headline)

    # Robustness pass (protocol 7.1). The primary table conditions on
    # solvability, which is information unavailable at deployment. This repeats
    # the comparison on ALL questions -- U included -- at alphas above beta,
    # where certification is arithmetically possible without that conditioning.
    if bool(cfg.get("phase4.restrict_to_A")):
        beta_all = float(state.beta or 0.0)
        rob_alphas = [float(a) for a in cfg.get("phase4.robustness_alphas")
                      if float(a) > beta_all]
        skipped = [float(a) for a in cfg.get("phase4.robustness_alphas")
                   if float(a) <= beta_all]
        rob_rows, rob_meta = [], {"beta": beta_all, "alphas_run": rob_alphas,
                                  "alphas_skipped_below_beta": skipped}
        if skipped:
            state.notes.append(
                f"Phase 4 robustness: alpha in {skipped} is at or below beta = "
                f"{beta_all:.3f}, so no rule of any kind is certifiable there; "
                "skipped rather than reported as a blank table.")
        for alpha in rob_alphas:
            sub = _pass(answers, "downstream")
            sub = sub[sub["split"].isin(["calib", "eval"])]
            all_q = questions[questions["q_id"].isin(set(sub["q_id"]))]
            rob_calib = build_traces(
                cfg, _pass(answers, "downstream", split="calib"), all_q,
                embedder=embedder)
            rob_eval = build_traces(
                cfg, _pass(answers, "downstream", split="eval"), all_q,
                embedder=embedder)
            if not rob_calib or not rob_eval:
                continue
            res = run_phase4(cfg, rob_calib, rob_eval, beta=beta_all,
                             alpha=alpha, n_max=n_max)
            if len(res.headline):
                rob_rows.append(res.headline.assign(alpha=alpha))
        if rob_rows:
            store.write_table("robustness_headline",
                              pd.concat(rob_rows, ignore_index=True))
        store.write_json("robustness_summary", rob_meta)
    store.write_json("summary", {"beta_used": beta_eff,
                                        "restricted_to_A": bool(cfg.get("phase4.restrict_to_A")),
                                        "quality_score": cfg.get("phase4.quality_score"),
                                        "per_alpha": per_alpha})
    state.results["step6_clm"] = {"headline": headline.to_dict("records"),
                                  "per_alpha": per_alpha}
    return state


# --------------------------------------------------------------------------
# Step 7 -- paraphrase reliability (8.2) and recalibration transfer (Phase 6)
# --------------------------------------------------------------------------

def step7_invariance_transfer(state: PipelineState) -> PipelineState:
    cfg, store = state.cfg, state.store
    gen = Generator(cfg, store)
    questions = state.questions
    eval_q = questions[questions["split"] == "eval"]
    probe_q = eval_q.head(int(cfg.get("phase5.temperature_sweep.n_questions")))

    out: dict = {}

    per_q, para = invariance.paraphrase_reliability(cfg, gen, probe_q)
    state.store.phase("invariance").write_table("paraphrase_per_question", per_q)
    out["paraphrase"] = para

    eval_answers = _pass(state.answers, "downstream", split="eval")
    table, tsum = transfer.transfer_study(cfg, eval_answers)
    xfer = state.store.phase("transfer")
    xfer.write_table("transfer", table)
    compounding = transfer.isotonic_preserves_compounding(cfg, eval_answers)
    xfer.write_table("isotonic_compounding", compounding)
    out["transfer"] = tsum

    state.store.phase("invariance").write_json("summary", out)
    state.results["step7"] = out
    return state


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------

def run_all(cfg: Config, *, simulate_labels: bool = False,
            labels_path: str | None = None,
            skip: tuple[str, ...] = (), store: Store | None = None) -> PipelineState:
    from .datasets import build_questions, split_deviation, split_summary

    # A full run always gets its own timestamped directory; per-phase CLI
    # commands attach to an existing one instead.
    store = store if store is not None else Store(cfg)
    state = PipelineState(cfg=cfg, store=store)
    state.questions = build_questions(cfg)
    store.write_table("dataset_splits", split_summary(state.questions))
    store.write_table("dataset_split_deviation",
                      split_deviation(state.questions, cfg))

    # Execution order, not section-10 order -- see the module docstring. The
    # first three are swapped: the gate labels real draws (so generation first),
    # and the sweep wants the gate's validated tau_star (so the gate second).
    steps = [
        ("step3_generate", step3_generate),
        ("step2_gate", lambda s: step2_gate(s, simulate_labels=simulate_labels,
                                            labels_path=labels_path)),
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
