"""Survival, censoring, and the LTT machinery."""

from __future__ import annotations

import numpy as np
import pathlib

import pandas as pd
import pytest

from vc_uq.ltt import (check_monotone_order, feasibility, fixed_sequence_test,
                       h1, hoeffding_bentkus_p, order_is_ascending, run_ltt)
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
    out = fixed_sequence_test(risks, alpha=0.1, delta=0.1, ascending=False,
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
    res = run_ltt(risks, alpha=0.05, delta=0.1, method="fixed_sequence",
                  ascending=False)
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


def test_discount_exponentiates_the_nats_threshold():
    """c_discount is defined against a PROBABILITY.

    vc_product's lambda_stop is a log-probability -- the one lambda component
    not on [0, 1], because a compounding claim is only legible in nats -- so it
    must be exponentiated before dividing. Skipping that is a units error that
    still yields a plausible-looking number.
    """
    from vc_uq.clm import RULE_UNITS, discount_factor

    assert RULE_UNITS["vc_product"] == "log-probability (nats)"
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
    """A certified threshold far below alpha is the regime this exists for."""
    from vc_uq.clm import discount_factor
    out = discount_factor("vc_product", 0.1, -800.0)   # exp(800) overflows
    assert np.isinf(out["c_discount"]) or out["c_discount"] > 1e300
    assert np.isfinite(out["log10_c_discount"])
    assert out["log10_c_discount"] == pytest.approx((np.log(0.1) + 800) / np.log(10))
    # -inf means "never stop early": the rule never fires, so it makes no claim.
    assert np.isnan(discount_factor("vc_product", 0.1, -np.inf)["c_discount"])


def test_discount_is_not_defined_for_non_product_rules():
    from vc_uq.clm import discount_factor
    assert np.isnan(discount_factor("vc_max", 0.1, 0.9)["c_discount"])


# --------------------------------------------------------------------------
# The certification order, and what delta actually buys
# --------------------------------------------------------------------------

def _monotone_risks(*, ascending_risk: bool, n_cells: int = 1, n: int = 500):
    """One family per cell, risk monotone in lambda_stop in the given direction."""
    stops = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    risk = [0.0, 0.0, 0.0, 0.02, 0.4, 0.8]
    if not ascending_risk:
        risk = risk[::-1]
    rows = []
    for c in range(n_cells):
        for s, r in zip(stops, risk):
            rows.append({"lambda_qual": float(c), "lambda_div": 0.0,
                         "lambda_stop": s, "risk_hat": r, "n": n})
    return pd.DataFrame(rows)


def test_the_order_comes_from_the_rules_direction_not_from_the_data():
    """A 'le' rule's conservative end is the LOW threshold.

    It stops when its statistic falls to lambda_stop, so a lower threshold is
    harder to reach, stops later, retains more and carries less risk. Deriving
    this from the observed risks instead would make the order data-dependent,
    which is exactly what fixed-sequence validity forbids -- so it is read off
    the rule.
    """
    from vc_uq.clm import RULE_DIRECTION

    assert order_is_ascending("le") is True
    assert order_is_ascending("ge") is False
    with pytest.raises(ValueError):
        order_is_ascending("lt")

    # Every configured rule declares a direction the order can be read from.
    for rule, direction in RULE_DIRECTION.items():
        assert isinstance(order_is_ascending(direction), bool), rule


def test_a_reversed_sequence_reports_a_certifiable_rule_as_uncertifiable():
    """The failure is silent: it halts on test one and returns nothing.

    Measured on the smoke calibration set this was the whole difference between
    vc_product certifying 27 configurations and certifying none -- the study's
    headline result arriving as a sort order.
    """
    # A "le" rule: risk rises with lambda_stop, so the sequence must ascend.
    risks = _monotone_risks(ascending_risk=True)

    right = run_ltt(risks, alpha=0.1, delta=0.1, method="fixed_sequence",
                    ascending=order_is_ascending("le"))
    wrong = run_ltt(risks, alpha=0.1, delta=0.1, method="fixed_sequence",
                    ascending=order_is_ascending("ge"))

    assert len(right.valid) == 4, "the four low-risk thresholds are certifiable"
    assert wrong.is_empty, "walking down opens on the worst point and halts"

    # And symmetrically for a "ge" rule, so neither order is simply better.
    ge = _monotone_risks(ascending_risk=False)
    assert len(run_ltt(ge, alpha=0.1, delta=0.1, method="fixed_sequence",
                       ascending=order_is_ascending("ge")).valid) == 4
    assert run_ltt(ge, alpha=0.1, delta=0.1, method="fixed_sequence",
                   ascending=order_is_ascending("le")).is_empty


def test_delta_is_split_across_the_parallel_sequences():
    """G fixed sequences at delta each control FWER at G*delta, not delta.

    Each (lambda_qual, lambda_div) cell is its own sequence and the default grid
    has 100 of them, so the promise that any selection from Lambda_hat is safe
    at 1 - delta is off by two orders of magnitude without this.
    """
    risks = _monotone_risks(ascending_risk=True, n_cells=8)
    res = run_ltt(risks, alpha=0.1, delta=0.1, method="fixed_sequence",
                  ascending=True)

    assert res.n_families == 8
    assert res.delta_family == pytest.approx(0.1 / 8)
    assert set(res.table["delta_family"].round(12)) == {round(0.1 / 8, 12)}

    # Every rejection cleared the corrected level, not the nominal one.
    assert (res.valid["p_value"] < res.delta_family).all()

    # And the correction bites: a configuration between the two levels is
    # certified at delta and refused at delta/G.
    borderline = pd.DataFrame([{"lambda_qual": float(c), "lambda_div": 0.0,
                                "lambda_stop": 1.0, "risk_hat": 0.04, "n": 100}
                               for c in range(8)])
    one = run_ltt(borderline.head(1), alpha=0.1, delta=0.1,
                  method="fixed_sequence", ascending=True)
    many = run_ltt(borderline, alpha=0.1, delta=0.1,
                   method="fixed_sequence", ascending=True)
    assert one.delta_family > many.delta_family
    p = float(one.table["p_value"].iloc[0])
    assert one.delta_family > p > many.delta_family, (
        "the fixture must sit between the two levels for this to test anything")
    assert len(one.valid) == 1 and many.is_empty


def test_a_non_monotone_family_is_reported_not_reordered():
    """Risk is exactly monotone in the sample, so a violation is a defect.

    Within a family the retention path is fixed: a stricter threshold can only
    stop later, and stopping later can only turn a loss of 1 into a 0. Sorting
    the grid to repair it would make the order data-dependent, which is the one
    thing that would actually void the guarantee.
    """
    risks = _monotone_risks(ascending_risk=True)
    assert check_monotone_order(risks, order_col="lambda_stop", ascending=True,
                                group_cols=("lambda_qual", "lambda_div")) == 0
    assert check_monotone_order(risks, order_col="lambda_stop", ascending=False,
                                group_cols=("lambda_qual", "lambda_div")) == 1

    scrambled = risks.copy()
    scrambled.loc[scrambled["lambda_stop"] == 3.0, "risk_hat"] = 0.9
    res = run_ltt(scrambled, alpha=0.1, delta=0.1, method="fixed_sequence",
                  ascending=True)
    assert res.nonmonotone_families == 1
    # Reported, not acted on: the order is unchanged, so the sequence still
    # halts where the p-values say it does.
    assert len(res.valid) == 2


def test_phase4_reads_the_order_off_each_rule(cfg):
    """The wiring, not just the helper: every rule must get its own direction."""
    from vc_uq import clm

    for rule, direction in clm.RULE_DIRECTION.items():
        assert clm.order_is_ascending(direction) == (direction == "le"), rule

    src = (pathlib.Path(clm.__file__)).read_text(encoding="utf-8")
    assert "order_is_ascending(RULE_DIRECTION[rule])" in src, \
        "run_phase4 must derive the sequence order from the rule it is testing"


def test_the_sequence_order_cannot_be_defaulted():
    """A forgotten `ascending` must be a TypeError, not a silent wrong answer.

    There is no correct default: "ge" rules walk down and "le" rules walk up, so
    whichever is chosen is wrong for half the rules -- and being wrong does not
    raise, it returns an empty Lambda_hat that reads as "not certifiable". The
    only safe design is to make the caller say.
    """
    import inspect

    from vc_uq import ltt

    for fn in (ltt.fixed_sequence_test, ltt.run_ltt):
        param = inspect.signature(fn).parameters["ascending"]
        assert param.default is inspect.Parameter.empty, (
            f"{fn.__name__} gives `ascending` a default; there is no safe one")
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


# --------------------------------------------------------------------------
# Phase 4 retention: what admits an answer, and what is done with a draw whose
# statistics are missing
# --------------------------------------------------------------------------

def _phase4_frame(vc=(0.9, 0.8, 0.7, 0.6), logp=(-0.4, -0.9, -1.2, -1.6),
                  correct=(True, False, False, False)):
    import pandas as pd

    answers = pd.DataFrame([{
        "q_id": "q0", "dataset": "d", "split": "calib", "draw_set": "downstream",
        "draw_idx": i, "answer": f"answer number {i}", "vc_post": v,
        "logp_mean": lp, "h_tok_mean": h, "min_token_p": 0.3,
        "s_anchor": 0.1 * i, "cluster_id": i, "correct": c,
    } for i, (v, lp, c, h) in enumerate(zip(vc, logp, correct, (1.8, 1.4, 0.9, 0.3)))])
    questions = pd.DataFrame([{"q_id": "q0", "a_star": "answer number 0",
                               "vc_pre": 0.8, "p_hat": 0.25, "in_U": False}])
    return answers, questions


def test_retention_admits_on_likelihood_not_on_the_signal_under_test(cfg):
    """Phase 4 tests VC as a STOPPING rule. It must not also gate the set.

    If VC decided retention, token_entropy, min_token_p and fixed_k would all be
    scored on a set VC had already filtered -- there would be no VC-free arm to
    compare against, and a difference in draws could not be attributed to the
    stopping rule. Conformal language modelling admits on the length-normalised
    log-likelihood, and so does this.
    """
    from vc_uq.clm import build_traces
    from vc_uq.backends.mock import MockEmbedder

    assert cfg.get("phase4.quality_score") == "likelihood", \
        "the shipped default must not be vc_post"

    answers, questions = _phase4_frame()
    tr = build_traces(cfg, answers, questions, embedder=MockEmbedder(dim=32, seed=1))[0]
    # p(y|x)^(1/T) = exp(mean_t log p_t): a probability, not a log-probability.
    assert list(tr.quality) == pytest.approx(list(np.exp([-0.4, -0.9, -1.2, -1.6])))
    assert np.all((tr.quality >= 0) & (tr.quality <= 1)), "quality is in [0, 1]"
    assert list(tr.logp_mean) == pytest.approx([-0.4, -0.9, -1.2, -1.6])
    assert list(tr.vc_post) == pytest.approx([0.9, 0.8, 0.7, 0.6]), \
        "vc_post is still carried -- it is what the stopping rules read"


def test_pitfall_refuses_vc_as_both_gate_and_stopping_signal(cfg):
    from vc_uq.pitfalls import run_checks

    name = "retention and stopping use different signals"
    assert next(c for c in run_checks(cfg).checks if c.name == name).passed is True

    bad = cfg.with_overrides(["phase4.quality_score=vc_post"])
    check = next(c for c in run_checks(bad).checks if c.name == name)
    assert check.passed is False and check.severity == "fatal"
    assert "logp_mean" in check.detail


def test_a_draw_with_no_parseable_confidence_is_dropped_not_imputed(cfg):
    """An unparsed VC is not a confidence of zero.

    Filling one in puts a number the model never produced into the statistic the
    study is about, and it is not neutral: 0.0 reads as "maximally unsure", which
    pushes every VC rule toward drawing more and inflates the one metric the
    headline table compares.
    """
    import numpy as np

    from vc_uq.clm import build_traces, trace_audit
    from vc_uq.backends.mock import MockEmbedder

    answers, questions = _phase4_frame()
    answers.loc[answers["draw_idx"] == 2, "vc_post"] = np.nan
    tr = build_traces(cfg, answers, questions,
                      embedder=MockEmbedder(dim=32, seed=1))[0]

    assert tr.n == 3, "the unparsed draw is gone, not filled"
    assert 0.0 not in set(tr.vc_post), "no imputed confidence anywhere"
    assert list(tr.vc_post) == pytest.approx([0.9, 0.8, 0.6])
    assert tr.n_dropped == 1

    audit = trace_audit([tr])
    assert audit["n_draws_dropped_missing_stats"] == 1
    assert audit["frac_draws_dropped"] == pytest.approx(0.25)


def test_a_question_whose_every_draw_is_unusable_is_skipped(cfg):
    import numpy as np

    from vc_uq.clm import build_traces
    from vc_uq.backends.mock import MockEmbedder

    answers, questions = _phase4_frame()
    answers["h_tok_mean"] = np.nan          # e.g. an answer that emitted no tokens
    assert build_traces(cfg, answers, questions,
                        embedder=MockEmbedder(dim=32, seed=1)) == []


def test_every_lambda_is_searched_over_the_unit_interval(cfg):
    """One search space for both retention dimensions and six of seven rules.

    Each underlying score is rescaled into [0, 1] first -- a length-normalised
    likelihood, a cosine distance halved, a geometric mean token probability, a
    fraction of the budget -- so a certified threshold reads without a units
    table. vc_product is the one exception, in nats, because a claim that
    compounds is only legible on a log axis.
    """
    import numpy as np

    from vc_uq.clm import (LOG_SPACE_RULES, RULE_DIRECTION, build_grid,
                           diversity_grid, quality_grid)

    assert quality_grid(5).tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert diversity_grid(5).tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    for rule in RULE_DIRECTION:
        grid = build_grid(cfg.with_overrides(
            ["phase4.grid.lambda_qual.num=4", "phase4.grid.lambda_div.num=4",
             "phase4.grid.lambda_stop.num=6"]), rule)
        pts = np.asarray(grid, dtype=float)
        # Retention is always [0, 1], whatever the rule.
        assert pts[:, 0].min() >= 0.0 and pts[:, 0].max() <= 1.0, f"{rule} qual"
        assert pts[:, 1].min() >= 0.0 and pts[:, 1].max() <= 1.0, f"{rule} div"
        stop = pts[:, 2]
        if rule in LOG_SPACE_RULES:
            assert stop.max() <= 0.0, f"{rule} lambda_stop is a log-probability"
            assert np.isneginf(stop.min()), "-inf anchors the conservative end"
        else:
            assert stop.min() >= 0.0 and stop.max() <= 1.0, f"{rule} leaves [0, 1]"


def test_the_scores_being_thresholded_are_in_zero_one(cfg):
    """The grid being [0, 1] only means something if the scores are too."""
    import numpy as np

    from vc_uq.clm import build_traces, running_statistic, retention_path
    from vc_uq.backends.mock import MockEmbedder

    answers, questions = _phase4_frame()
    for score in ("likelihood", "anchor_sim", "vc_post"):
        tr = build_traces(cfg.with_overrides([f"phase4.quality_score={score}"]),
                          answers, questions,
                          embedder=MockEmbedder(dim=32, seed=1))[0]
        assert np.all((tr.quality >= 0.0) & (tr.quality <= 1.0)), score
        # Cosine distance is halved, not clipped, so [0, 2] maps onto [0, 1]
        # without collapsing the anti-correlated pairs.
        assert np.all((tr.dist >= 0.0) & (tr.dist <= 1.0)), score

    tr = build_traces(cfg, answers, questions,
                      embedder=MockEmbedder(dim=32, seed=1))[0]
    ret = retention_path(tr, 0.0, 0.0)
    for rule in ("vc_max", "token_entropy"):
        stat = running_statistic(tr, rule, ret)
        assert np.all((stat >= 0.0) & (stat <= 1.0)), rule

    # token_entropy carries exp(-H), not H. Entropy is unbounded above -- this
    # fixture starts at 1.8 nats -- so leaving it in nats would put lambda_stop
    # on a scale no [0, 1] grid can reach, and the rule would never fire.
    from vc_uq.clm import RULE_DIRECTION
    assert RULE_DIRECTION["token_entropy"] == "ge",         "higher exp(-H) means more confident, so the comparison rises"
    te = running_statistic(tr, "token_entropy", ret)
    assert te[0] == pytest.approx(np.exp(-1.8))
    assert np.all(np.diff(te) >= 0), "most confident draw so far is a running MAX"
    assert te[-1] == pytest.approx(np.exp(-0.3))


def test_the_stopping_grid_follows_the_statistic_scale(cfg):
    """[0, 1] for six rules; nats for the one whose claim compounds.

    Evenly spaced in nats is orders of magnitude in the claim, which is what a
    product needs: as probabilities those thresholds bunch against 0 and stop
    being readable, while -5 and -20 nats are two legible numbers.
    """
    import numpy as np

    from vc_uq.clm import rule_stop_grid

    for rule in ("vc_max", "vc_first", "vc_prehoc", "token_entropy",
                 "min_token_p", "fixed_k"):
        assert rule_stop_grid(rule, 6).tolist() == pytest.approx(
            np.linspace(0, 1, 6).tolist()), rule

    prod = rule_stop_grid("vc_product", 6, product_floor=1e-6)
    assert np.isneginf(prod[0]), "-inf means 'never stop early' -- the anchor"
    assert prod[1] == pytest.approx(np.log(1e-6))
    assert prod[-1] == pytest.approx(0.0)
    assert np.all(np.diff(prod[1:]) > 0), "monotone, so the sequence order holds"
    # Evenly spaced in nats is geometric in the claim it represents.
    claims = np.exp(prod[1:])
    assert claims.min() == pytest.approx(1e-6) and claims.max() == pytest.approx(1.0)
    assert int(np.sum(claims < 1e-3)) >= 2

    with pytest.raises(ValueError):
        rule_stop_grid("vc_product", 6, product_floor=0.0)


def test_the_threshold_is_mapped_onto_the_statistic(cfg):
    """Never the statistic onto the threshold.

    Exponentiating Lambda_k to meet a probability threshold would underflow to a
    wall of exact zeros at large k and collapse every low threshold into one
    test -- the same failure 6.7 warns about. vc_product's threshold is in nats
    so no mapping is needed; fixed_k's [0, 1] fraction is scaled by the budget.
    """
    import numpy as np

    from vc_uq.clm import LOG_SPACE_RULES, stop_threshold

    assert "vc_product" in LOG_SPACE_RULES
    assert stop_threshold("vc_product", np.log(1e-6), 40) == pytest.approx(np.log(1e-6))
    assert stop_threshold("vc_product", -np.inf, 40) == -np.inf
    assert stop_threshold("fixed_k", 0.25, 40) == pytest.approx(10.0)
    assert stop_threshold("vc_max", 0.8, 40) == pytest.approx(0.8)

    # Monotone increasing in lambda for every rule, which is what lets the
    # certification order be read off RULE_DIRECTION in lambda space.
    from vc_uq.clm import RULE_DIRECTION, rule_stop_grid
    for rule in RULE_DIRECTION:
        grid = rule_stop_grid(rule, 8)
        vals = [stop_threshold(rule, lam, 40) for lam in grid]
        assert all(b >= a for a, b in zip(vals, vals[1:])), rule


# --------------------------------------------------------------------------
# Every stopping rule reads the retained set, not the raw draw stream
# --------------------------------------------------------------------------

def _trace(vc, h_tok=None, correct=None, n_dist=None, quality=None):
    """A bare QuestionTrace with mutually distinct answers unless told otherwise.

    ``quality`` defaults to ``vc`` for brevity, but the two are separate columns
    in the real pipeline and the gating tests below set them independently -- an
    answer can be fluent and wrong, or hesitant and right.
    """
    import numpy as np

    from vc_uq.clm import QuestionTrace

    n = len(vc)
    vc = np.asarray(vc, dtype=float)
    # Entropy tracks confidence unless overridden, so the token-level rule has
    # something to distinguish draws by.
    h = np.asarray(h_tok if h_tok is not None else (1.0 - vc), dtype=float)
    c = np.asarray(correct if correct is not None else [False] * n, dtype=bool)
    qual = np.asarray(quality if quality is not None else vc, dtype=float)
    dist = n_dist if n_dist is not None else (1.0 - np.eye(n)) * 0.5
    return QuestionTrace(
        q_id="q0", dataset="d", correct=c, quality=qual, vc_post=vc,
        logp_mean=np.log(np.clip(vc, 1e-9, 1.0)), h_tok=h,
        min_token_p=np.full(n, 0.3), cluster_id=np.arange(n),
        dist=np.asarray(dist, dtype=float), vc_pre=0.5, p_hat=0.5, in_U=False)


@pytest.mark.parametrize("rule", ["vc_product", "vc_max", "token_entropy", "fixed_k"])
def test_a_rejected_answer_cannot_end_sampling(rule):
    """Stopping on an answer you then throw away certifies a set without it.

    The gate decides what is in C(q); the rule then reads C(q). An answer the
    quality gate rejected is not in the set, so it must not move the statistic --
    and while the statistic ran over all draws instead, lambda_qual and
    lambda_div could change the set but never the draw count, which is why
    retention was never once selected.
    """
    import numpy as np

    from vc_uq.clm import running_statistic

    # Draw 1 is the confident one, and it is the one the gate rejects.
    tr = _trace(vc=[0.2, 0.99, 0.3])
    everything = running_statistic(tr, rule, [0, 1, 2])
    without_the_confident_one = running_statistic(tr, rule, [0, 2])

    assert not np.allclose(everything, without_the_confident_one), (
        f"{rule} gives the same statistic whether or not draw 1 is retained, so "
        "it is reading the draw stream rather than the set")


def test_vc_max_reads_the_best_answer_in_the_set(cfg):
    import numpy as np

    from vc_uq.clm import running_statistic, stop_index

    tr = _trace(vc=[0.2, 0.99, 0.3])
    stat = running_statistic(tr, "vc_max", [0, 2])
    assert stat.tolist() == pytest.approx([0.2, 0.2, 0.3]), \
        "the rejected 0.99 never enters the running max"
    assert np.isneginf(running_statistic(tr, "vc_max", [])[0]), \
        "an empty set cannot satisfy any threshold"

    # And the rule then fires on the set, not on the draw that was discarded.
    assert stop_index(running_statistic(tr, "vc_max", [0, 1, 2]), 0.9, "ge") == 1
    assert stop_index(running_statistic(tr, "vc_max", [0, 2]), 0.9, "ge") == 2


def test_fixed_k_counts_the_set_not_the_draws(cfg):
    """|C(q)| >= k, which is what makes it a fair null.

    Counting draws instead would let the baseline ignore the gate that every
    other rule pays for, so it would be playing a different game rather than
    marking the floor.
    """
    from vc_uq.clm import running_statistic, stop_index, stop_threshold

    tr = _trace(vc=[0.9] * 6)
    assert running_statistic(tr, "fixed_k", [0, 1, 2, 3, 4, 5]).tolist() == \
        pytest.approx([1, 2, 3, 4, 5, 6])
    # Gate out two answers and the same k takes two more draws to reach.
    assert running_statistic(tr, "fixed_k", [0, 2, 4, 5]).tolist() == \
        pytest.approx([1, 1, 2, 2, 3, 4])

    thr = stop_threshold("fixed_k", 3 / 6, n_max=6)     # |C(q)| >= 3
    assert stop_index(running_statistic(tr, "fixed_k", [0, 1, 2, 3, 4, 5]), thr, "ge") == 2
    assert stop_index(running_statistic(tr, "fixed_k", [0, 2, 4, 5]), thr, "ge") == 4


def test_gating_buys_a_cleaner_set_by_spending_draws(cfg):
    """The trade-off retention is supposed to offer, end to end.

    A stricter gate admits less, so the statistic advances more slowly and the
    rule spends more draws -- and in exchange the low-quality answers that would
    have triggered a premature stop are gone, so the risk falls. Neither half of
    that was visible while the rules read the raw draw stream.
    """
    from vc_uq.clm import risk_table

    # The first two answers are fluent, confident and WRONG, and the model gives
    # them a low likelihood; the right ones come later and score well.
    traces = [_trace(vc=[0.95, 0.95, 0.40, 0.95],
                     quality=[0.2, 0.2, 0.9, 0.9],
                     correct=[False, False, True, True]) for _ in range(20)]
    lam_stop = 0.9

    loose = risk_table(traces, "vc_max", [(0.0, 0.0, lam_stop)], n_max=4).iloc[0]
    tight = risk_table(traces, "vc_max", [(0.5, 0.0, lam_stop)], n_max=4).iloc[0]

    assert loose["mean_draws"] == 1.0 and loose["risk_hat"] == 1.0,         "ungated it stops on the first confident wrong answer, every time"
    assert tight["mean_draws"] > loose["mean_draws"], "the gate costs draws"
    assert tight["risk_hat"] < loose["risk_hat"], "and buys a set that is right"


def test_the_stopping_family_is_four_rules_that_all_read_the_set(cfg):
    """vc_first, vc_prehoc and min_token_p are deliberately gone.

    The first two fix the budget from a single number -- the first draw's VC, or
    a pre-hoc estimate -- so they cannot read C(q) at all and belong to a
    different family from everything here. min_token_p duplicated the
    token-level mechanism token_entropy already carries, and a second
    near-identical baseline costs multiplicity without adding an argument.
    """
    from vc_uq import clm

    expected = {"vc_product", "vc_max", "token_entropy", "fixed_k"}
    assert set(cfg.get("phase4.stop_rules")) == expected
    assert set(clm.RULE_DIRECTION) == expected
    assert set(clm.RULE_UNITS) == expected
    for gone in ("vc_first", "vc_prehoc", "min_token_p"):
        assert gone not in clm.RULE_DIRECTION
        with pytest.raises(ValueError, match="unknown stop rule"):
            clm.running_statistic(_trace(vc=[0.5]), gone, [0])
    assert "fixed_k" in expected, "the null baseline is mandatory"
