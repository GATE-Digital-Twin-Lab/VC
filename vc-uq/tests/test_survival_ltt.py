"""Survival, censoring, and the LTT machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vc_uq.ltt import (feasibility, fixed_sequence_test, h1,
                       hoeffding_bentkus_p, run_ltt)
from vc_uq.survival import (beta_from_km, hazard, kaplan_meier,
                            predicted_budget, product_rule_curve,
                            question_stats)


def test_km_recovers_known_censoring_fraction():
    """Half the questions never succeed; beta must be exactly 0.5."""
    k = [1, 2, 3, 4, -1, -1, -1, -1]
    cens = [False] * 4 + [True] * 4
    km = kaplan_meier(k, cens, n_max=10)
    assert beta_from_km(km) == pytest.approx(0.5)


def test_km_is_not_biased_by_dropping_censored():
    """Averaging K over successes only is the classic bias here.

    The mean of the uncensored K is 2.5, but the true survival at N_MAX is 0.5:
    the two numbers describe different populations and the survival estimate is
    the one that keeps the hard questions.
    """
    k = [1, 2, 3, 4, -1, -1, -1, -1]
    cens = [False] * 4 + [True] * 4
    naive = np.mean([x for x, c in zip(k, cens) if not c])
    km = kaplan_meier(k, cens, n_max=10)
    assert naive == pytest.approx(2.5)
    assert beta_from_km(km) == pytest.approx(0.5)
    assert km["S"].iloc[0] == pytest.approx(0.875)   # 1 - 1/8 after the first draw


def test_hazard_flat_for_homogeneous_iid_draws():
    """A single p_q across all questions predicts a flat hazard."""
    rng = np.random.default_rng(0)
    rows = []
    for q in range(600):
        for i in range(6):
            rows.append({"q_id": f"q{q}", "draw_idx": i, "split": "eval",
                         "correct": bool(rng.random() < 0.3)})
    hz = hazard(pd.DataFrame(rows), max_k=5)
    assert hz["hazard"].std() < 0.05
    assert hz["hazard"].mean() == pytest.approx(0.3, abs=0.05)


def test_hazard_declines_under_heterogeneous_p_q():
    """Pooled over a mix of easy and impossible questions it must decline:
    a run of failures is evidence of being in a low-p_q question."""
    rng = np.random.default_rng(1)
    rows = []
    for q in range(600):
        p = 0.9 if q % 2 == 0 else 0.0
        for i in range(6):
            rows.append({"q_id": f"q{q}", "draw_idx": i, "split": "eval",
                         "correct": bool(rng.random() < p)})
    hz = hazard(pd.DataFrame(rows), max_k=5)
    assert hz["hazard"].iloc[0] > hz["hazard"].iloc[-1] + 0.3


def test_predicted_budget_matches_the_protocol_table():
    """Section 6.3's table, at alpha = 0.1."""
    expected = {0.95: 1, 0.9: 1, 0.8: 2, 0.7: 2, 0.5: 4, 0.3: 7}
    for vc, n_hat in expected.items():
        assert predicted_budget(vc, 0.1) == n_hat


def test_dynamic_range_collapse_over_the_observed_vc_support():
    """The whole observed VC range spans one sample of operational difference."""
    n_hats = {predicted_budget(v, 0.1) for v in (0.7, 0.8, 0.9, 0.95, 1.0)}
    assert n_hats <= {1.0, 2.0}


def test_h1_is_zero_at_equality_and_positive_otherwise():
    assert h1(0.3, 0.3) == pytest.approx(0.0)
    assert h1(0.1, 0.3) > 0


def test_bentkus_dominates_for_small_nonzero_risk():
    """Why plain Hoeffding must not be used at alpha = 0.05.

    At exactly R_hat = 0 the two bounds coincide up to the factor of e that
    Bentkus carries, and Hoeffding is the tighter of the two. The moment R_hat
    lifts off zero -- which is the realistic case, a handful of failures in a
    few hundred calibration questions -- Bentkus becomes much the stronger, and
    a Hoeffding-only implementation would silently fail to certify
    configurations that are in fact certifiable.
    """
    n, alpha = 200, 0.05
    for r_hat in (0.01, 0.02, 0.03):
        hoeff = float(np.exp(-n * h1(min(r_hat, alpha), alpha)))
        combined = hoeffding_bentkus_p(r_hat, n, alpha)
        assert combined < hoeff, f"Bentkus should bind at R_hat = {r_hat}"

    at_zero = hoeffding_bentkus_p(0.0, n, alpha)
    assert at_zero == pytest.approx(float(np.exp(-n * h1(0.0, alpha))))


def test_p_value_is_monotone_in_empirical_risk():
    ps = [hoeffding_bentkus_p(r, 300, 0.1) for r in (0.0, 0.05, 0.1, 0.2, 0.4)]
    assert all(a <= b for a, b in zip(ps, ps[1:]))


def test_p_value_never_rejects_when_risk_exceeds_alpha():
    assert hoeffding_bentkus_p(0.4, 500, 0.1) > 0.1


def test_fixed_sequence_halts_at_first_non_rejection():
    """Validity comes from stopping at the first failure, not from skipping it."""
    risks = pd.DataFrame({
        "lambda_qual": [0.0] * 4, "lambda_div": [0.0] * 4,
        "lambda_stop": [4.0, 3.0, 2.0, 1.0],       # most to least conservative
        "risk_hat": [0.0, 0.0, 0.5, 0.0],          # third one fails
        "n": [500] * 4,
    })
    out = fixed_sequence_test(risks, alpha=0.1, delta=0.1,
                              group_cols=("lambda_qual", "lambda_div"))
    out = out.sort_values("lambda_stop", ascending=False)
    assert list(out["rejected"]) == [True, True, False, False]


def test_feasibility_blocks_alpha_below_beta():
    f = feasibility(beta=0.3, alpha=0.1)
    assert not f["feasible"]
    assert "beta" in f["reason"]
    assert feasibility(beta=0.05, alpha=0.1)["feasible"]


def test_run_ltt_returns_empty_when_nothing_is_certifiable():
    risks = pd.DataFrame({"lambda_qual": [0.0], "lambda_div": [0.0],
                          "lambda_stop": [1.0], "risk_hat": [0.9], "n": [100]})
    res = run_ltt(risks, alpha=0.05, delta=0.1, method="fixed_sequence")
    assert res.is_empty


def test_product_rule_gap_widens_with_k_under_common_mode_errors():
    """The 6.7 signature. Questions are either always right or always wrong,
    while VC is a constant 0.8: the claim compounds, the truth does not."""
    from vc_uq.config import load_config
    cfg = load_config(overrides=["phase3.product_rule.k_values=[1, 2, 5]",
                                 "phase3.product_rule.n_bins=1"])
    rows = []
    for q in range(200):
        always_right = q % 2 == 0
        for i in range(5):
            rows.append({"q_id": f"q{q}", "draw_idx": i, "vc_post": 0.8,
                         "correct": always_right, "split": "eval"})
    curve = product_rule_curve(pd.DataFrame(rows), cfg)
    by_k = curve.set_index("k")
    # Observed failure rate is 0.5 at every k; the claim shrinks as 0.2^k.
    assert by_k.loc[1, "observed"] == pytest.approx(0.5)
    assert by_k.loc[5, "observed"] == pytest.approx(0.5)
    r1 = by_k.loc[1, "ratio_observed_over_claimed"]
    r5 = by_k.loc[5, "ratio_observed_over_claimed"]
    assert r5 > 50 * r1


def test_question_stats_marks_censoring_not_success():
    df = pd.DataFrame([
        {"q_id": "a", "dataset": "d", "split": "eval", "draw_idx": i,
         "answer": "x", "vc_post": 0.8, "correct": False} for i in range(5)
    ])
    qs = question_stats(df)
    assert qs["K_q"].iloc[0] == -1
    assert bool(qs["censored"].iloc[0])
    assert qs["p_hat"].iloc[0] == 0.0


# --------------------------------------------------------------------------
# The product rule as a sum of logs
# --------------------------------------------------------------------------

def test_log_sum_and_linear_product_agree_where_both_are_representable():
    """The change is numerical, not statistical: same statistic, same boundary."""
    from vc_uq.stats import log_product_claim
    rng = np.random.default_rng(0)
    vc = rng.uniform(0.0, 0.99, size=12)
    assert np.exp(log_product_claim(vc)) == pytest.approx(float(np.prod(1 - vc)),
                                                          rel=1e-9)


def test_stopping_decision_is_unchanged_by_the_reparameterisation():
    """`prod <= t` and `sum log <= log t` must select the same draw index."""
    from vc_uq.clm import stop_index
    from vc_uq.stats import log_one_minus_vc
    rng = np.random.default_rng(1)
    vc = rng.uniform(0.1, 0.9, size=20)
    linear = np.cumprod(1 - vc)
    logs = np.cumsum(log_one_minus_vc(vc))
    for t in (0.5, 0.1, 0.01, 1e-4, 1e-8):
        assert stop_index(linear, t, "le") == stop_index(logs, float(np.log(t)), "le")


def test_log_sum_survives_where_the_linear_product_underflows():
    """At vc = 0.99 the product flushes to exactly 0 well before k = 400.

    Every question past that point becomes an indistinguishable 0.0, which
    silently merges the grid quantiles Phase 4 searches over. The log form keeps
    ranking them.
    """
    from vc_uq.stats import log_product_claim
    vc = np.full(400, 0.99)
    assert float(np.prod(1 - vc)) == 0.0
    log_claim = log_product_claim(vc)
    assert np.isfinite(log_claim)
    assert log_claim == pytest.approx(400 * np.log(0.01))
    # Still strictly ordered against a slightly weaker claim.
    assert log_claim < log_product_claim(np.full(399, 0.99))


def test_certainty_is_clamped_rather_than_annihilating_the_claim():
    """vc = 1.0 asserts zero failure probability.

    In the linear form that single answer zeroes the product and satisfies every
    threshold at once. The clamp keeps its contribution large but finite, so
    later evidence can still move the statistic.
    """
    from vc_uq.stats import log_one_minus_vc, log_product_claim
    assert float(np.prod(1 - np.array([1.0, 0.2, 0.3]))) == 0.0

    claim = log_product_claim(np.array([1.0, 0.2, 0.3]), floor=1e-6)
    assert np.isfinite(claim)
    assert claim == pytest.approx(np.log(1e-6) + np.log(0.8) + np.log(0.7))
    # The floor sets exactly how strong one answer's claim may be.
    assert log_one_minus_vc(1.0, floor=1e-3) == pytest.approx(np.log(1e-3))


def test_discount_factor_exponentiates_the_log_threshold():
    """c_discount is defined against a PROBABILITY, so a nats threshold must be
    converted back before dividing -- otherwise the units error still yields a
    plausible-looking number."""
    from vc_uq.clm import discount_factor
    alpha, lam_prob = 0.1, 0.01
    out = discount_factor("vc_product", alpha, float(np.log(lam_prob)))
    assert out["c_discount"] == pytest.approx(alpha / lam_prob)
    assert out["log10_c_discount"] == pytest.approx(np.log10(alpha / lam_prob))
    assert out["lambda_stop_hat_as_prob"] == pytest.approx(lam_prob)


def test_discount_factor_is_one_when_vc_is_truthful():
    from vc_uq.clm import discount_factor
    out = discount_factor("vc_product", 0.1, float(np.log(0.1)))
    assert out["c_discount"] == pytest.approx(1.0)


def test_log_discount_stays_finite_where_the_linear_form_overflows():
    from vc_uq.clm import discount_factor
    out = discount_factor("vc_product", 0.1, -800.0)   # exp(800) overflows
    assert np.isinf(out["c_discount"]) or out["c_discount"] > 1e300
    assert np.isfinite(out["log10_c_discount"])
    assert out["log10_c_discount"] == pytest.approx((np.log(0.1) + 800) / np.log(10))


def test_discount_is_not_defined_for_non_product_rules():
    from vc_uq.clm import discount_factor
    assert np.isnan(discount_factor("vc_max", 0.1, 0.9)["c_discount"])
