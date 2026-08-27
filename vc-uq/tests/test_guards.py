"""The protections in section 11 must actually fire.

A guard that never triggers is indistinguishable from no guard at all, so each
one is tested by constructing the situation it exists to catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vc_uq.config import ConfigError, load_config
from vc_uq.judge import CircularityError, guard_against_circularity
from vc_uq.parsing import (OK, NO_CONFIDENCE_FIELD, OUT_OF_RANGE,
                           parse_answer_and_vc, parse_audit, parse_vc_only)
from vc_uq.pitfalls import run_checks
from vc_uq.prompts import get, variants
from vc_uq.schemas import assert_split_by_question, conform, ANSWERS_SCHEMA
from vc_uq.survival import km_flatness, partition_U


def test_circularity_guard_rejects_shared_function():
    with pytest.raises(CircularityError):
        guard_against_circularity("s_anchor", "s_anchor")
    guard_against_circularity("s_anchor", "e_cos")     # different functions: fine


def test_config_refuses_undeclared_knobs(cfg):
    """No magic numbers in code means an undeclared key must raise, not default."""
    with pytest.raises(ConfigError):
        cfg.get("phase4.some_knob_nobody_declared")
    assert cfg.get("phase4.delta") == 0.1


def test_split_by_question_guard_catches_draw_level_split():
    df = pd.DataFrame([
        {"q_id": "a", "split": "calib"}, {"q_id": "a", "split": "eval"},
    ])
    with pytest.raises(ValueError, match="splits must be by question"):
        assert_split_by_question(df)


def test_conform_refuses_missing_required_column():
    with pytest.raises(ValueError, match="required column"):
        conform(pd.DataFrame({"q_id": ["a"]}), ANSWERS_SCHEMA, name="answers")


def test_partition_U_refuses_to_use_calibration_draws():
    """U membership must come from the classify split and nothing else."""
    df = pd.DataFrame([{"q_id": "a", "split": "calib", "correct": False,
                        "draw_idx": 0}])
    with pytest.raises(ValueError, match="classify"):
        partition_U(df)


def test_partition_U_uses_only_classify_draws():
    df = pd.DataFrame([
        {"q_id": "a", "split": "classify", "draw_idx": 0, "correct": False},
        {"q_id": "a", "split": "classify", "draw_idx": 1, "correct": False},
        # A correct answer in another split must NOT rescue the question from U:
        # membership is decided on the classify draws alone.
        {"q_id": "a", "split": "eval", "draw_idx": 0, "correct": True},
    ])
    out = partition_U(df)
    assert bool(out.loc[out["q_id"] == "a", "in_U"].iloc[0])


def test_km_flatness_flags_a_still_declining_curve(cfg):
    km = pd.DataFrame({"k": range(1, 11),
                       "S": np.linspace(1.0, 0.5, 10)})     # steadily declining
    verdict = km_flatness(km, cfg)
    assert not verdict["flat"]
    assert "censoring artifact" in verdict["reason"]

    flat_km = pd.DataFrame({"k": range(1, 11),
                            "S": [1.0, 0.8, 0.7, 0.65, 0.64] + [0.64] * 5})
    assert km_flatness(flat_km, cfg)["flat"]


def test_reliability_refuses_fixed_width_binning(cfg):
    from vc_uq.calibration import reliability
    bad = cfg.with_overrides(["phase2.binning=fixed_width"])
    df = pd.DataFrame([{"q_id": "a", "vc_post": 0.8, "correct": True}])
    with pytest.raises(ValueError, match="Fixed-width binning is not offered"):
        reliability(df, bad)


def test_pitfall_report_flags_missing_fixed_k_baseline(cfg):
    bad = cfg.with_overrides(["phase4.stop_rules=[vc_product, vc_max]"])
    report = run_checks(bad)
    failed = [c.name for c in report.checks if c.passed is False]
    assert "fixed_k null baseline is included" in failed
    assert not report.ok


def test_pitfall_report_flags_cosine_clustering(cfg):
    bad = cfg.with_overrides(["nli.backend=cosine"])
    report = run_checks(bad)
    failed = [c.name for c in report.checks if c.passed is False]
    assert "clustering uses entailment, not cosine" in failed


def test_pitfall_report_flags_stochastic_anchor(cfg):
    bad = cfg.with_overrides(["anchor.method=sample"])
    report = run_checks(bad)
    failed = [c.name for c in report.checks if c.passed is False]
    assert "anchor is deterministic" in failed


def test_pitfall_report_passes_on_the_default_config(cfg):
    """The checks must not be unconditionally pessimistic either."""
    report = run_checks(cfg)
    assert report.ok, [c.as_dict() for c in report.fatal_failures]


def test_vc_pre_prompt_contains_no_answer():
    """Contaminating vc_pre with an answer destroys the construct."""
    msgs = get("vc_pre_v1").build(question="What is the capital of Peru?")
    joined = " ".join(m["content"].lower() for m in msgs)
    assert "do not answer" in joined
    assert not any(m["role"] == "assistant" for m in msgs)


def test_clean_prompt_carries_no_confidence_instruction():
    """h_tok must be measurable under a prompt that never mentions confidence."""
    spec = get("answer_clean_v1")
    assert not spec.elicits_vc
    text = " ".join(m["content"].lower() for m in spec.build(question="q"))
    assert "confidence" not in text


def test_ten_paraphrase_variants_exist():
    post = [v for v in variants("post") if v.startswith("vc_post_v")]
    assert len(post) == 10


# -- parsing ---------------------------------------------------------------

def test_parse_failure_returns_none_not_a_default():
    """A failed parse must never become 0.5; that would fabricate data."""
    p = parse_answer_and_vc("Answer: Lima\nI would rather not say.")
    assert p.vc is None
    assert p.status == NO_CONFIDENCE_FIELD


def test_parse_scales():
    assert parse_answer_and_vc("Answer: x\nConfidence: 0.8").vc == pytest.approx(0.8)
    assert parse_answer_and_vc("Answer: x\nConfidence: 80%", "percent").vc == \
        pytest.approx(0.8)
    assert parse_answer_and_vc("Answer: x\nConfidence: 8", "outof10").vc == \
        pytest.approx(0.8)
    assert parse_answer_and_vc("Answer: x\nConfidence: highly confident",
                               "verbal").vc == pytest.approx(0.95)


def test_verbal_scale_prefers_the_longest_matching_label():
    assert parse_answer_and_vc("Answer: x\nConfidence: highly confident",
                               "verbal").vc > \
        parse_answer_and_vc("Answer: x\nConfidence: confident", "verbal").vc


def test_out_of_range_is_recorded_not_rescaled():
    """A model answering 85 on a 0-1 scale is a scale-invariance violation and
    must be visible as one, not quietly divided by 100."""
    p = parse_answer_and_vc("Answer: x\nConfidence: 85", "unit")
    assert p.vc is None
    assert p.status == OUT_OF_RANGE


def test_parse_audit_reports_the_failure_rate():
    parsed = [parse_answer_and_vc("Answer: a\nConfidence: 0.9"),
              parse_answer_and_vc("Answer: a\nno confidence here")]
    audit = parse_audit(parsed)
    assert audit["failure_rate"] == pytest.approx(0.5)
    assert audit["n"] == 2


def test_pre_hoc_parser_needs_no_answer():
    p = parse_vc_only("Confidence: 0.42")
    assert p.vc == pytest.approx(0.42)
    assert p.status == OK
