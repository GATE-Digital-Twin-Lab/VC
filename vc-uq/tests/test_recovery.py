"""Does the analysis recover a truth it was not told?

The mock backend has a latent world with hand-set parameters. These tests run
the real analysis code over draws from that world and check that each estimator
lands on the value it is supposed to estimate.

The second half is the more important half. It builds a world where VC IS a
good estimator and checks that the analysis says so. Without it, every negative
finding in this repository would be unfalsifiable: code that always concludes
"VC fails" would pass a suite that only ever tested failing worlds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vc_uq import calibration, cluster, judge, survival
from vc_uq.backends import build_embedder, build_nli
from vc_uq.backends.mock import SEM_TAG
from vc_uq.datasets import build_questions
from vc_uq.generate import Generator
from vc_uq.store import Store


def _run_world(cfg, overrides, n_max=12):
    """Generate and score one world end to end, returning (answers, questions)."""
    cfg = cfg.with_overrides(list(overrides) + [f"generation.n_max={n_max}"])
    store = Store(cfg)
    questions = build_questions(cfg)
    gen = Generator(cfg, store)
    answers = gen.draw_answers(questions, store_per_position=False)

    embedder = build_embedder(cfg)
    answers, questions = judge.score_answers(cfg, answers, questions,
                                             embedder=embedder)
    # Correctness comes from the KNOWN truth here, so these tests measure the
    # analysis rather than the embedding instrument. Phase 0 is what validates
    # the instrument; conflating the two would make a failure ambiguous.
    tags = answers["answer"].astype(str).map(
        lambda t: SEM_TAG.search(t).group(1) if SEM_TAG.search(t) else "")
    answers["correct"] = tags.str.endswith("|gold")
    answers = cluster.cluster_frame(cfg, answers, nli=build_nli(cfg))

    qs = survival.question_stats(answers)
    div = cluster.question_diversity(answers)
    drop = [c for c in list(qs.columns) + list(div.columns) if c != "q_id"]
    questions = (questions.drop(columns=[c for c in drop if c in questions.columns])
                 .merge(qs, on="q_id", how="left").merge(div, on="q_id", how="left"))
    return cfg, answers, questions


# --------------------------------------------------------------------------
# World 1: the protocol's predictions
# --------------------------------------------------------------------------

PREDICTED_WORLD = [
    "dataset.triviaqa.n_questions=50", "dataset.fabricated.n_questions=15",
    "model.mock.vc_within_sd=0.01", "model.mock.vc_reads_answer=0.0",
    "model.mock.error_correlation=0.85", "model.mock.vc_floor=0.7",
]


def test_fabricated_questions_are_never_correct(cfg):
    """p_q = 0 by construction, so U is non-empty and its membership is known."""
    _, answers, questions = _run_world(cfg, PREDICTED_WORLD)
    fab = answers[answers["dataset"] == "fabricated"]
    assert len(fab) > 0
    assert not fab["correct"].any()
    fab_q = questions[questions["dataset"] == "fabricated"]
    assert (fab_q["p_hat"] == 0).all()
    assert bool(fab_q["censored"].all())


def test_vc_attaches_to_the_prompt_when_within_variance_is_zero(cfg):
    cfg2, answers, _ = _run_world(cfg, PREDICTED_WORLD)
    d = calibration.vc_variance_decomposition(answers, cfg2)
    assert d["within_share"] < d["within_share_threshold"]
    assert "attaches to the prompt" in d["interpretation"]


def test_within_question_auroc_is_at_chance_when_vc_ignores_the_answer(cfg):
    _, answers, _ = _run_world(cfg, PREDICTED_WORLD)
    res = calibration.two_aurocs(answers, pd.DataFrame(), cfg)
    row = res[(res["level"] == "within_question") & (res["signal"] == "vc_post")]
    assert len(row) == 1
    assert row["auroc"].iloc[0] == pytest.approx(0.5, abs=0.06)


def test_error_correlation_tracks_p_q_heterogeneity(cfg):
    """What Corr(c_i, c_j) actually measures, checked in both directions.

    Draws of one question are positively correlated whenever ``p_q`` varies
    across questions -- a failure is evidence of sitting in a low-``p_q``
    question. That is precisely the dependence the product rule ignores. Under a
    homogeneous ``p_q`` the same estimator must return ~0, so the statistic is
    reading heterogeneity rather than reporting a constant.
    """
    _, hetero, _ = _run_world(cfg, PREDICTED_WORLD)
    assert calibration.error_dependence(hetero)["rho"] > 0.1
    assert calibration.error_dependence(hetero)["common_mode"] is True

    homo = PREDICTED_WORLD + ["model.mock.p_q_beta=[40, 40]",   # p_q ~ 0.5 for all
                              "dataset.fabricated.n_questions=0"]
    _, flat, _ = _run_world(cfg, homo)
    assert abs(calibration.error_dependence(flat)["rho"]) < 0.1


def test_vc_dynamic_range_collapses_to_one_or_two_samples(cfg):
    cfg2, _, questions = _run_world(cfg, PREDICTED_WORLD)
    budget = survival.budget_table(questions, cfg2)
    row = budget[budget["vc_statistic"] == "vc_1"].iloc[0]
    assert row["vc_min"] >= 0.65
    assert row["n_hat_max"] <= 3
    # ... while the truth spans one draw to never.
    assert row["frac_never_correct"] > 0.0


def test_temperature_moves_p_hat_far_more_than_it_moves_vc(cfg):
    """8.1, measured. p_q is a function of the decoder; VC has no argument for T."""
    rows = []
    for T in (0.2, 0.8, 1.4):
        _, answers, _ = _run_world(
            cfg, PREDICTED_WORLD + [f"generation.temperature={T}"], n_max=10)
        rows.append({"T": T, "p_hat": answers["correct"].mean(),
                     "vc": answers["vc_post"].mean()})
    df = pd.DataFrame(rows)
    p_range = df["p_hat"].max() - df["p_hat"].min()
    vc_range = df["vc"].max() - df["vc"].min()
    assert p_range > 0.1
    assert vc_range < 0.02
    assert p_range > 5 * vc_range


def test_low_diversity_U_cell_is_populated(cfg):
    """The dangerous cell: fluent, consistent, confident, and wrong."""
    _, answers, questions = _run_world(cfg, PREDICTED_WORLD)
    part = survival.partition_U(answers.assign(split="classify"))
    questions = questions.drop(columns=["in_U"], errors="ignore").merge(
        part[["q_id", "in_U"]], on="q_id", how="left")
    table = survival.diversity_2x2(questions)
    cell = table[table["cell"] == "systematic misconception"]
    assert len(cell) == 1
    assert cell["n"].iloc[0] > 0


# --------------------------------------------------------------------------
# World 2: VC is a good estimator. The analysis must SAY SO.
# --------------------------------------------------------------------------

GOOD_VC_WORLD = [
    "dataset.triviaqa.n_questions=60", "dataset.fabricated.n_questions=0",
    "model.mock.vc_reads_answer=1.0",      # VC reads the answer it just produced
    "model.mock.vc_within_sd=0.0",
    "model.mock.vc_informativeness=1.0",
    "model.mock.vc_floor=0.0",             # full dynamic range, not [0.7, 1]
    "model.mock.error_correlation=0.0",    # independent errors
    "model.mock.vc_answer_signal=[0.05, 0.95]",
    # A fine support: the coarse default quantises 0.05 up to 0.5, which would
    # look like miscalibration when it is only the reporting grid.
    "model.mock.vc_support=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]",
]


def test_within_question_auroc_rises_when_vc_does_read_the_answer(cfg):
    """The negative finding must be falsifiable: here it must NOT hold."""
    _, answers, _ = _run_world(cfg, GOOD_VC_WORLD)
    res = calibration.two_aurocs(answers, pd.DataFrame(), cfg)
    row = res[(res["level"] == "within_question") & (res["signal"] == "vc_post")]
    assert row["auroc"].iloc[0] > 0.9


def test_variance_decomposition_finds_within_question_signal_when_present(cfg):
    cfg2, answers, _ = _run_world(cfg, GOOD_VC_WORLD)
    d = calibration.vc_variance_decomposition(answers, cfg2)
    assert d["within_share"] > 0.2
    assert "attaches to the prompt" not in d["interpretation"]


def test_product_rule_gap_does_not_diverge_when_vc_is_honest(cfg):
    """With honest, answer-sensitive VC and independent errors, the curves stay
    much closer to the diagonal than in the common-mode world."""
    cfg_g, good, _ = _run_world(cfg, GOOD_VC_WORLD)
    cfg_b, bad, _ = _run_world(cfg, PREDICTED_WORLD)
    conf = ["phase3.product_rule.k_values=[1, 3, 5]", "phase3.product_rule.n_bins=4"]
    good_d = survival.product_rule_divergence(
        survival.product_rule_curve(good, cfg_g.with_overrides(conf)))
    bad_d = survival.product_rule_divergence(
        survival.product_rule_curve(bad, cfg_b.with_overrides(conf)))
    assert good_d["max_ratio"].max() < bad_d["max_ratio"].max()


def test_reliability_gap_is_small_when_vc_is_calibrated(cfg):
    _, answers, _ = _run_world(cfg, GOOD_VC_WORLD)
    rel = calibration.reliability(answers, cfg)
    reportable = rel[rel["reportable"]]
    assert len(reportable) >= 2
    weighted = float((reportable["n_g"] * reportable["abs_gap"]).sum()
                     / reportable["n_g"].sum())
    assert weighted < 0.15


def test_reliability_gap_is_large_in_the_overconfident_world(cfg):
    """Same estimator, opposite verdict -- the measure discriminates."""
    _, answers, _ = _run_world(cfg, PREDICTED_WORLD)
    rel = calibration.reliability(answers, cfg)
    reportable = rel[rel["reportable"]]
    weighted = float((reportable["n_g"] * reportable["abs_gap"]).sum()
                     / reportable["n_g"].sum())
    assert weighted > 0.2
    # Overconfidence specifically: stated confidence exceeds observed accuracy.
    assert (reportable["calibration_gap"] < 0).mean() > 0.5
