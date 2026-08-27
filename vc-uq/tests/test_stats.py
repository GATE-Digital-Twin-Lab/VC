"""Estimators checked against values computable by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vc_uq.stats import (auroc, brier_decomposition, cluster_bootstrap,
                         cohens_kappa, variance_decomposition,
                         within_question_auroc, within_question_correlation)


def test_auroc_perfect_and_inverted():
    assert auroc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert auroc([4, 3, 2, 1], [False, False, True, True]) == 0.0


def test_auroc_all_ties_is_exactly_half():
    """VC is quantised, so ties are the common case, not an edge case.

    A tie-breaking implementation would report 1.0 here and silently inflate
    every AUROC in the study.
    """
    assert auroc([0.8] * 6, [True, True, True, False, False, False]) == 0.5


def test_auroc_partial_ties():
    # scores 1,1,2 with labels F,T,T -> the tied pair contributes 0.5
    assert auroc([1, 1, 2], [False, True, True]) == pytest.approx(0.75)


def test_auroc_single_class_is_nan():
    assert np.isnan(auroc([1, 2, 3], [True, True, True]))


def test_brier_decomposition_identity():
    rng = np.random.default_rng(0)
    p = rng.choice([0.2, 0.5, 0.8], size=500)
    y = rng.random(500) < p
    d = brier_decomposition(p, y)
    # Murphy: brier = reliability - resolution + uncertainty
    assert d["brier"] == pytest.approx(
        d["reliability"] - d["resolution"] + d["uncertainty"], abs=1e-9)


def test_perfect_forecast_has_zero_reliability_term():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([False, False, True, True])
    d = brier_decomposition(p, y)
    assert d["reliability"] == pytest.approx(0.0)
    assert d["resolution"] == pytest.approx(d["uncertainty"])


def test_cohens_kappa_bounds():
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)


def test_within_question_correlation_positive_when_errors_are_common_mode(answers_frame):
    """q1 is always wrong and q2 always right: draws move together."""
    out = within_question_correlation(answers_frame, "correct")
    assert out["rho"] > 0.3


def test_within_question_correlation_zero_when_independent():
    rng = np.random.default_rng(1)
    rows = []
    for q in range(200):
        for i in range(10):
            rows.append({"q_id": f"q{q}", "correct": bool(rng.random() < 0.5)})
    out = within_question_correlation(pd.DataFrame(rows), "correct")
    assert abs(out["rho"]) < 0.05


def test_within_question_auroc_is_half_when_vc_is_constant(answers_frame):
    """The 5.5 prediction, as a unit test: constant VC cannot rank anything."""
    res = within_question_auroc(answers_frame, "vc_post", "correct")
    assert res["mean_auroc"] == pytest.approx(0.5)


def test_variance_decomposition_all_between_when_vc_is_per_question(answers_frame):
    d = variance_decomposition(answers_frame, "vc_post")
    assert d["within"] == pytest.approx(0.0)
    assert d["icc"] == pytest.approx(1.0)


def test_cluster_bootstrap_is_wider_than_naive_binomial():
    """The central methodological point of section 5.2, as a test.

    Resampling questions must produce a materially wider interval than
    resampling answers when draws within a question are perfectly correlated.
    """
    rows = []
    rng = np.random.default_rng(3)
    for q in range(40):
        val = bool(rng.random() < 0.5)
        for i in range(25):                     # 25 identical draws per question
            rows.append({"q_id": f"q{q}", "correct": val})
    df = pd.DataFrame(rows)

    by_question = cluster_bootstrap(df, lambda d: d["correct"].mean(),
                                    group="q_id", n_resamples=400, seed=0)
    by_answer = cluster_bootstrap(df.assign(row_id=range(len(df))),
                                  lambda d: d["correct"].mean(),
                                  group="row_id", n_resamples=400, seed=0)
    q_width = by_question.hi - by_question.lo
    a_width = by_answer.hi - by_answer.lo
    assert q_width > 3 * a_width
