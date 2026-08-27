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


# --------------------------------------------------------------------------
# Phase 0 hand labels (the real, non-simulated path)
# --------------------------------------------------------------------------

def _sheet_and_scored(cfg):
    """A scored tau_select population and the label sheet built from it."""
    from vc_uq import judge, phase0
    from vc_uq.backends import build_embedder
    from vc_uq.datasets import build_questions
    from vc_uq.generate import Generator
    from vc_uq.store import Store

    cfg = cfg.with_overrides(["dataset.triviaqa.n_questions=40",
                              "dataset.fabricated.n_questions=12",
                              "generation.n_max=4", "phase0.n_hand_label=40",
                              "phase0.stratify_bins=3"])
    store = Store(cfg)
    questions = build_questions(cfg)
    answers = Generator(cfg, store).draw_answers(questions, store_per_position=False)
    scored, questions = judge.score_answers(cfg, answers, questions,
                                            embedder=build_embedder(cfg))
    sheet = phase0.build_label_sheet(cfg, scored, questions)
    return cfg, sheet, scored


def test_label_sheet_is_shuffled_not_stratum_ordered(cfg):
    """Unshuffled, the sheet is written in stratum order, so labelling the first
    half would cover only the low-e_cos bins of the first dataset."""
    _, sheet, _ = _sheet_and_scored(cfg)
    strata = list(sheet["stratum"])
    runs = sum(1 for a, b in zip(strata, strata[1:]) if a != b)
    # Stratum-ordered would give exactly (n_strata - 1) transitions.
    assert runs > sheet["stratum"].nunique()


def test_hand_labels_merge_on_key_and_ignore_stale_e_cos(cfg, tmp_path):
    """Only correct_human is read back.

    e_cos is not a stable property of a pair -- the embedding space is
    mean-centred over whatever was scored together -- so a stale column in the
    CSV must not reach tau selection.
    """
    from vc_uq import phase0
    cfg2, sheet, _ = _sheet_and_scored(cfg)
    filled = sheet.copy()
    filled["correct_human"] = [i % 2 == 0 for i in range(len(filled))]
    filled["e_cos"] = -999.0          # poisoned; must be ignored
    path = tmp_path / "labels.csv"
    filled.to_csv(path, index=False)

    out, report = phase0.load_hand_labels(cfg2, sheet, path)
    assert report["n_joined"] == len(sheet)
    assert report["coverage"] == pytest.approx(1.0)
    assert (out["e_cos"] != -999.0).all()
    assert out["correct_human"].dtype == bool


def test_hand_labels_accept_common_truth_spellings(cfg, tmp_path):
    from vc_uq import phase0
    cfg2, sheet, _ = _sheet_and_scored(cfg)
    tokens = ["TRUE", "false", "yes", "no", "1", "0", "Y", "n"]
    filled = sheet.copy()
    filled["correct_human"] = [tokens[i % len(tokens)] for i in range(len(filled))]
    path = tmp_path / "labels.csv"
    filled.to_csv(path, index=False)
    out, report = phase0.load_hand_labels(cfg2, sheet, path)
    assert report["n_joined"] == len(sheet)
    assert report["n_unrecognised_values"] == 0


def test_unrecognised_label_counts_as_unlabelled_not_as_false(cfg, tmp_path):
    """Guessing on a value nobody wrote would fabricate ground truth."""
    from vc_uq import phase0
    cfg2, sheet, _ = _sheet_and_scored(cfg)
    filled = sheet.copy()
    vals = [True] * len(filled)
    vals[0] = "maybe?"
    filled["correct_human"] = vals
    path = tmp_path / "labels.csv"
    filled.to_csv(path, index=False)
    out, report = phase0.load_hand_labels(cfg2, sheet, path)
    assert report["n_unrecognised_values"] == 1
    assert report["n_joined"] == len(sheet) - 1


def test_partial_labelling_is_allowed_but_reported(cfg, tmp_path):
    from vc_uq import phase0
    cfg2, sheet, _ = _sheet_and_scored(cfg)
    filled = sheet.copy()
    # Label one pair per stratum, plus a few more: covers every stratum.
    keep = filled.groupby("stratum").head(2).index
    filled["correct_human"] = pd.Series([pd.NA] * len(filled), dtype="boolean")
    filled.loc[keep, "correct_human"] = True
    path = tmp_path / "labels.csv"
    filled.to_csv(path, index=False)

    out, report = phase0.load_hand_labels(cfg2, sheet, path)
    assert 0 < report["n_joined"] < len(sheet)
    assert report["n_strata_below_min"] == 0
    assert len(out) == report["n_joined"]


def test_empty_stratum_is_refused(cfg, tmp_path):
    """Overall coverage would hide this; per-stratum coverage is the check."""
    from vc_uq import phase0
    cfg2, sheet, _ = _sheet_and_scored(cfg)
    dropped = sorted(sheet["stratum"].unique())[0]
    filled = sheet.copy()
    filled["correct_human"] = pd.Series([True] * len(filled), dtype="boolean")
    filled.loc[filled["stratum"] == dropped, "correct_human"] = pd.NA
    path = tmp_path / "labels.csv"
    filled.to_csv(path, index=False)

    with pytest.raises(phase0.LabelCoverageError, match="strata"):
        phase0.load_hand_labels(cfg2, sheet, path)


def test_rows_that_match_nothing_are_reported_not_silently_dropped(cfg, tmp_path):
    """A moved split assignment shows up here rather than as a quiet shortfall."""
    from vc_uq import phase0
    cfg2, sheet, _ = _sheet_and_scored(cfg)
    filled = sheet.copy()
    filled["correct_human"] = True
    ghost = filled.iloc[[0]].copy()
    ghost["q_id"] = "q_that_does_not_exist"
    ghost["draw_idx"] = 0
    path = tmp_path / "labels.csv"
    pd.concat([filled, ghost], ignore_index=True).to_csv(path, index=False)

    out, report = phase0.load_hand_labels(cfg2, sheet, path)
    assert report["n_unjoined"] == 1
    assert "q_that_does_not_exist#0" in report["unjoined_examples"]
    assert any("matched no pair" in w for w in report["warnings"])


def test_missing_required_column_is_rejected(cfg, tmp_path):
    from vc_uq import phase0
    cfg2, sheet, _ = _sheet_and_scored(cfg)
    path = tmp_path / "labels.csv"
    sheet.drop(columns=["draw_idx"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="draw_idx"):
        phase0.load_hand_labels(cfg2, sheet, path)


def test_blank_cells_are_unlabelled_not_unrecognised(cfg, tmp_path):
    """The string accessors propagate NA, so a naive membership test reports
    every deliberately blank row as an unrecognised value."""
    from vc_uq import phase0
    cfg2, sheet, _ = _sheet_and_scored(cfg)
    filled = sheet.copy()
    vals = ["TRUE"] * len(filled)
    blanks = filled.groupby("stratum").filter(lambda g: len(g) > 2).index[:3]
    for i in blanks:
        vals[filled.index.get_loc(i)] = pd.NA
    filled["correct_human"] = pd.Series(vals, dtype="object")
    path = tmp_path / "labels.csv"
    filled.to_csv(path, index=False)

    _, report = phase0.load_hand_labels(cfg2, sheet, path)
    assert report["n_unrecognised_values"] == 0
    assert report["n_joined"] == len(sheet) - len(blanks)
    assert not any("not recognised" in w for w in report["warnings"])


# --------------------------------------------------------------------------
# The decoder must be unrestricted, and every declared knob must be read
# --------------------------------------------------------------------------

def test_sampling_params_default_to_an_unrestricted_decoder(cfg):
    from vc_uq.backends.base import SamplingParams
    p = SamplingParams.from_config(cfg)
    assert not p.truncates_tail
    assert (p.top_p, p.top_k, p.min_p, p.repeat_penalty) == (1.0, 0, 0.0, 1.0)


def test_truncation_is_detected_and_named(cfg):
    """llama.cpp's own defaults, which would apply if we passed nothing."""
    from vc_uq.backends.base import SamplingParams
    p = SamplingParams.from_config(cfg).replace(top_k=40, top_p=0.95, min_p=0.05,
                                                repeat_penalty=1.1)
    assert p.truncates_tail
    for token in ("top_k=40", "top_p=0.95", "min_p=0.05", "repeat_penalty=1.1"):
        assert token in p.truncation_reason()


def test_pitfall_check_fires_on_a_truncated_decoder(cfg):
    bad = cfg.with_overrides(["model.generation.top_k=40",
                              "model.generation.top_p=0.95"])
    report = run_checks(bad)
    failed = [c.name for c in report.checks if c.passed is False]
    assert "decoder is unrestricted, so p_q is a function of T alone" in failed
    assert not report.ok


def test_sampling_params_reach_the_backend(cfg):
    """The bug this guards: knobs declared in config but never passed on."""
    from vc_uq.datasets import build_questions
    from vc_uq.generate import Generator
    from vc_uq.store import Store

    c = cfg.with_overrides(["dataset.triviaqa.n_questions=2",
                            "dataset.fabricated.n_questions=0",
                            "generation.n_max=1", "model.generation.top_k=7",
                            "model.generation.top_p=0.5"])
    gen = Generator(c, Store(c))
    gen.draw_answers(build_questions(c), store_per_position=False)
    seen = gen.lm.last_params
    assert seen.top_k == 7 and seen.top_p == 0.5
    assert seen.temperature == pytest.approx(c.get("generation.temperature"))


def test_temperature_is_the_only_thing_that_varies_across_a_sweep(cfg):
    """8.1 is only a measurement of T if T is the only thing that moves."""
    from vc_uq.datasets import build_questions
    from vc_uq.generate import Generator
    from vc_uq.store import Store

    c = cfg.with_overrides(["dataset.triviaqa.n_questions=2",
                            "dataset.fabricated.n_questions=0",
                            "generation.n_max=1"])
    gen = Generator(c, Store(c))
    qs = build_questions(c)
    seen = []
    for T in (0.2, 1.4):
        gen.draw_answers(qs, temperature=T, store_per_position=False)
        seen.append(gen.lm.last_params)
    a, b = seen
    assert a.temperature != b.temperature
    assert a.replace(temperature=0.0) == b.replace(temperature=0.0)


def test_dependent_draws_are_refused(cfg):
    from vc_uq.generate import Generator
    from vc_uq.store import Store
    bad = cfg.with_overrides(["generation.fresh_context_per_draw=false"])
    with pytest.raises(ValueError, match="fresh_context_per_draw"):
        Generator(bad, Store(bad))


def test_no_dead_config_keys(cfg):
    """Every declared knob must be referenced somewhere in src/.

    The config enforces one direction already -- reading an undeclared key
    raises. This is the other direction: a key that looks like a switch but is
    never read is worse than no comment, because changing it appears to do
    something.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "vc_uq"
    blob = "\n".join(p.read_text(encoding="utf-8")
                     for p in src.rglob("*.py"))

    def leaves(node, prefix=""):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict) and v:
                yield from leaves(v, path)
            else:
                yield path, k

    unread = []
    for dotted, key in leaves(cfg.data):
        # A key counts as read if its full dotted path appears, or if its bare
        # name appears (sections are often splatted into dataclasses).
        if dotted in blob or re.search(rf"\b{re.escape(key)}\b", blob):
            continue
        unread.append(dotted)
    assert not unread, f"declared but never read: {unread}"


# --------------------------------------------------------------------------
# Resumable generation
#
# Generation is the only expensive step in the study; everything downstream is
# a pure function of its cache. A run that dies at hour nine of twelve must
# restart where it stopped and -- more importantly -- must produce the table an
# uninterrupted run would have produced. These pin both halves of that.
# --------------------------------------------------------------------------

def _resume_cfg(cfg):
    return cfg.with_overrides(["dataset.triviaqa.n_questions=6",
                               "dataset.fabricated.n_questions=4",
                               "generation.n_max=3", "generation.r_pre=2"])


class _CountingLM:
    """Wraps a backend and counts generate() calls."""

    def __init__(self, inner):
        self.inner, self.n = inner, 0

    def generate(self, *a, **kw):
        self.n += 1
        return self.inner.generate(*a, **kw)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _fresh_generator(cfg, store):
    from vc_uq.generate import Generator
    gen = Generator(cfg, store)
    gen.lm = _CountingLM(gen.lm)
    return gen


def test_resume_second_pass_generates_nothing_and_returns_the_same_table(cfg):
    """The whole point: a complete cache means zero model calls, same output."""
    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    c = _resume_cfg(cfg)
    store = Store(c)
    questions = build_questions(c)

    first = _fresh_generator(c, store)
    a1 = first.draw_answers(questions, store_per_position=False, resume=True)
    store.append_answers(a1)
    assert first.lm.n == len(a1) > 0

    second = _fresh_generator(c, store)
    a2 = second.draw_answers(questions, store_per_position=False, resume=True)

    assert second.lm.n == 0, "a complete cache must not cost a single model call"
    assert second.last_resume["generated"] == 0
    assert second.last_resume["reused_from_cache"] == len(a1)

    key = ["q_id", "draw_idx"]
    left = a1.sort_values(key).reset_index(drop=True)
    right = a2.sort_values(key).reset_index(drop=True)
    for col in ("q_id", "draw_idx", "answer", "vc_post", "seed", "temperature"):
        assert list(left[col]) == list(right[col]), f"{col} changed on resume"


def test_partial_cache_generates_only_the_missing_draws(cfg):
    """A job killed midway resumes at the boundary, not from zero."""
    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    c = _resume_cfg(cfg)
    store = Store(c)
    questions = build_questions(c)

    # Simulate a crash after the first two of three draws per question.
    partial = _fresh_generator(c, store)
    store.append_answers(partial.draw_answers(questions, n_max=2,
                                              store_per_position=False))

    rest = _fresh_generator(c, store)
    full = rest.draw_answers(questions, store_per_position=False, resume=True)

    n_q = len(questions)
    assert rest.lm.n == n_q, "only the third draw of each question was missing"
    assert rest.last_resume == {"requested": 3 * n_q, "generated": n_q,
                                "reused_from_cache": 2 * n_q}
    assert len(full) == 3 * n_q
    assert full.groupby("q_id")["draw_idx"].nunique().eq(3).all()


def test_resumed_table_matches_an_uninterrupted_one(cfg):
    """Resumption must not perturb the sample.

    Seeds are hashed from (q_id, draw_idx, T, variant), so the draw a question
    receives cannot depend on how many draws preceded it in this process.
    """
    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    c = _resume_cfg(cfg)
    questions = build_questions(c)

    uninterrupted = _fresh_generator(c, Store(c, run_id="clean"))
    clean = uninterrupted.draw_answers(questions, store_per_position=False)

    store = Store(c, run_id="crashy")
    crashed = _fresh_generator(c, store)
    store.append_answers(crashed.draw_answers(questions, n_max=1,
                                              store_per_position=False))
    resumed = _fresh_generator(c, store).draw_answers(
        questions, store_per_position=False, resume=True)

    key = ["q_id", "draw_idx"]
    a = clean.sort_values(key).reset_index(drop=True)
    b = resumed.sort_values(key).reset_index(drop=True)
    for col in ("q_id", "draw_idx", "answer", "vc_post", "seed"):
        assert list(a[col]) == list(b[col]), f"{col} differs after a crash+resume"


def test_per_position_arrays_survive_a_crash_and_resume(cfg):
    """The 10% entropy subsample is generation output, so it must not be lost.

    It costs a decode to recreate and a resumed run only produces it for the
    draws it actually makes, so it lives in the append-only raw cache rather
    than a per-run results directory that a crash would leave half-written.

    Membership in the subsample is hashed per question rather than drawn from a
    shared RNG stream, so it is a property of the question and not of the batch
    it happened to be generated in. Under a stream draw, generating a subset of
    the questions -- which is exactly what a retry of the failed ones is --
    re-rolls the subsample and silently changes which diagnostics exist.
    """
    import pandas as pd

    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    base = _resume_cfg(cfg).with_overrides(
        ["generation.token_stats.store_per_position_fraction=0.5"])
    key = ["q_id", "draw_idx"]

    clean_cfg = base.with_overrides(["run.name=clean"])
    clean_store = Store(clean_cfg)
    questions = build_questions(clean_cfg)
    _fresh_generator(clean_cfg, clean_store).draw_answers(questions, resume=True)
    clean = pd.read_parquet(clean_store.raw_path("per_position"))

    crash_cfg = base.with_overrides(["run.name=crashy"])
    crash_store = Store(crash_cfg)
    first = _fresh_generator(crash_cfg, crash_store)
    crash_store.append_answers(first.draw_answers(questions, n_max=1))
    _fresh_generator(crash_cfg, crash_store).draw_answers(questions, resume=True)
    crashed = pd.read_parquet(crash_store.raw_path("per_position"))

    assert 0 < clean["q_id"].nunique() < len(questions), "fraction=0.5 must split"
    assert (sorted(map(tuple, clean[key].to_numpy())) ==
            sorted(map(tuple, crashed[key].to_numpy()))),         "a crash+resume lost or shifted the per-position subsample"

    # Re-batching must not re-roll membership either.
    subset = questions.iloc[::2].reset_index(drop=True)
    sub_cfg = base.with_overrides(["run.name=subset"])
    sub_store = Store(sub_cfg)
    _fresh_generator(sub_cfg, sub_store).draw_answers(subset, resume=True)
    sub_pp = pd.read_parquet(sub_store.raw_path("per_position"))

    expected = set(clean["q_id"]) & set(subset["q_id"])
    assert expected, "fixture must leave some subsampled question in the subset"
    assert set(sub_pp["q_id"]) == expected,         "the subsample was re-rolled when the batch changed"


def test_resume_restamps_split_from_the_current_question_table(cfg):
    """``split`` is not in the cache key, so a cached row can carry a stale one.

    Re-splitting must not invalidate expensive generation, but a resumed run
    must not mix two split assignments either -- that would silently break the
    calib/eval disjointness every downstream guarantee rests on.
    """
    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    c = _resume_cfg(cfg)
    store = Store(c)
    questions = build_questions(c)
    store.append_answers(_fresh_generator(c, store).draw_answers(
        questions, store_per_position=False, resume=True))
    assert set(questions["split"]) != {"eval"}, "fixture must exercise a change"

    moved = questions.assign(split="eval")
    out = _fresh_generator(c, store).draw_answers(moved, store_per_position=False,
                                                  resume=True)
    assert set(out["split"]) == {"eval"}, "cached rows kept a stale split"


def test_vc_pre_resumes_too(cfg):
    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    c = _resume_cfg(cfg)
    store = Store(c)
    questions = build_questions(c)

    first = _fresh_generator(c, store)
    pre1 = first.draw_vc_pre(questions, resume=True)
    store.append_vc_pre_repeats(pre1)
    assert first.lm.n == len(pre1) == 2 * len(questions)

    second = _fresh_generator(c, store)
    pre2 = second.draw_vc_pre(questions, resume=True)
    assert second.lm.n == 0
    assert (list(pre1.sort_values(["q_id", "repeat_idx"])["vc_pre"]) ==
            list(pre2.sort_values(["q_id", "repeat_idx"])["vc_pre"]))


def test_parse_audit_covers_cached_rows_not_just_new_ones(cfg):
    """Otherwise the reported failure rate depends on how much was already done."""
    import json as _json

    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    c = _resume_cfg(cfg)
    store = Store(c)
    questions = build_questions(c)
    store.append_answers(_fresh_generator(c, store).draw_answers(
        questions, store_per_position=False, resume=True))
    _fresh_generator(c, store).draw_answers(questions, store_per_position=False,
                                            resume=True)

    variant = c.get("generation.prompt_variant_post")
    T = c.get("generation.temperature")
    audit = _json.loads((store.tables_dir / f"parse_audit__{variant}__T{T}.json")
                        .read_text(encoding="utf-8"))
    assert audit["n"] == 3 * len(questions)
    assert audit["resume"]["generated"] == 0
