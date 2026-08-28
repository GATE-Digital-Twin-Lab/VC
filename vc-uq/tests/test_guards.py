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


# --------------------------------------------------------------------------
# Dataset assembly, splits, and backend compatibility
# --------------------------------------------------------------------------

def test_mock_embedder_refuses_a_real_model(cfg):
    """The mock embedder reads a tag that real model output does not carry.

    Paired with a real backend it silently returns noise, the Phase 0 gate
    fails, and the failure message blames the instrument -- which is true and
    completely misleading. Nothing crashes, so the combination is refused here.
    """
    from vc_uq.backends import build_embedder, build_nli

    for key, build in (("embedding.backend", build_embedder),
                       ("nli.backend", build_nli)):
        c = cfg.with_overrides(["model.backend=llamacpp", f"{key}=mock"])
        with pytest.raises(ValueError, match="cannot be used with model.backend"):
            build(c)

    # The valid pairing still works, and does not reach llama-cpp-python.
    c = cfg.with_overrides(["model.backend=mock", "embedding.backend=mock",
                            "nli.backend=mock"])
    assert build_embedder(c) is not None
    assert build_nli(c) is not None


def test_sem_tag_is_present_exactly_when_the_mock_embedder_can_read_it(cfg):
    """The guard above is only sound if the tag really does track the LM backend."""
    from vc_uq.datasets import build_questions

    mock_q = build_questions(cfg.with_overrides(["model.backend=mock"]))
    assert mock_q["a_star"].str.contains(r"\[\[sem:", regex=True).all()

    real_q = build_questions(cfg.with_overrides(["model.backend=llamacpp"]))
    assert not real_q["a_star"].str.contains(r"\[\[sem:", regex=True).any()


def test_splits_are_stable_when_questions_are_added(cfg):
    """The reason assignment is a hash rather than a shuffle.

    Re-splitting must never move an already-generated question between calib and
    eval; that would invalidate a calibration set built from expensive draws.
    """
    from vc_uq.datasets import assign_splits, fabricated

    c = cfg.with_overrides(["model.backend=llamacpp"])
    small = assign_splits(fabricated.build(50, 991), c)
    big = assign_splits(fabricated.build(200, 991), c)

    merged = small.merge(big, on="q_id", suffixes=("_small", "_big"))
    assert len(merged) == 50
    assert (merged["split_small"] == merged["split_big"]).all(), \
        "adding questions reshuffled the ones already assigned"


def test_split_deviation_reports_the_gap_it_cannot_remove(cfg):
    """Hash assignment is binomial, not quota-based, so sizes miss their targets.

    The imbalance is accepted (the alternative breaks add-later stability); what
    is not accepted is failing to say so. Small strata deviate most, and the
    fabricated set -- carrying the p_q = 0 population -- is the small one.
    """
    import pandas as pd

    from vc_uq.datasets import assign_splits, fabricated, split_deviation
    from vc_uq.datasets.triviaqa import build_synthetic

    c = cfg.with_overrides(["model.backend=llamacpp", "run.seed=20260826"])
    qs = pd.concat([fabricated.build(200, 991), build_synthetic(1200, 0)],
                   ignore_index=True)
    dev = split_deviation(assign_splits(qs, c), c)

    assert set(dev["stratum"]) == {"fabricated", "triviaqa_synthetic"}
    assert len(dev) == 8
    for _, row in dev.iterrows():
        assert row["actual_n"] == pytest.approx(
            row["actual_share"] * row["n_stratum"], abs=1e-6)

    small = dev[dev["stratum"] == "fabricated"]["share_deviation"].abs().max()
    large = dev[dev["stratum"] == "triviaqa_synthetic"]["share_deviation"].abs().max()
    assert small > large, "the small stratum should deviate more, not less"
    assert small > 0.04, "this seed is the one that exercises a material gap"


def test_pitfall_flags_a_lopsided_split(cfg):
    """The deviation has to reach the report, not just a CSV."""
    import pandas as pd

    from vc_uq.datasets import assign_splits, fabricated
    from vc_uq.datasets.triviaqa import build_synthetic
    from vc_uq.pitfalls import run_checks

    c = cfg.with_overrides(["model.backend=llamacpp", "run.seed=20260826"])
    qs = assign_splits(pd.concat([fabricated.build(200, 991),
                                  build_synthetic(1200, 0)], ignore_index=True), c)
    name = "split sizes are close to their targets"

    strict = run_checks(c.with_overrides(["dataset.splits.max_share_deviation=0.01"]),
                        questions=qs)
    check = next(x for x in strict.checks if x.name == name)
    assert check.passed is False and check.severity == "warn"
    assert "of a target" in check.detail

    lenient = run_checks(c.with_overrides(["dataset.splits.max_share_deviation=0.5"]),
                         questions=qs)
    assert next(x for x in lenient.checks if x.name == name).passed is True


def test_fabricated_questions_are_unique_and_known_unanswerable(cfg):
    from vc_uq.datasets import fabricated

    df = fabricated.build(200, 991)
    assert len(df) == 200
    assert df["question"].nunique() == 200
    assert (df["p_q_true"] == 0.0).all(), "p_q = 0 is the whole point of this set"
    assert (df["a_star"] == fabricated.A_STAR).all()

    with pytest.raises(ValueError, match=r"exhausted at \d+ unique questions.*reach 5000"):
        fabricated.build(5000, 991)


# --------------------------------------------------------------------------
# Prompt registry
# --------------------------------------------------------------------------

def test_verbal_confidence_is_mapped_onto_the_ladder_not_discarded():
    """No prompt asks for a word -- every elicitation asks for a number in [0, 1].

    Models answer "fairly confident" anyway. Mapping such a reply onto the ladder
    keeps it as data instead of booking it as a parse failure, which would
    understate the elicitation rate. The ladder lives with the prompts because it
    is elicitation vocabulary; parsing imports the one definition.
    """
    from vc_uq import prompts
    from vc_uq.parsing import VERBAL_SCALE, parse_vc_value

    assert VERBAL_SCALE is prompts.VERBAL_SCALE
    assert list(VERBAL_SCALE.values()) == sorted(VERBAL_SCALE.values()),         "insertion order is the ladder, so the values must ascend"

    for word, value in VERBAL_SCALE.items():
        assert parse_vc_value(word, "unit") == (value, "ok"), word
    # Longest label first, so the substring match cannot swallow the qualifier.
    assert parse_vc_value("highly confident", "unit")[0] == 0.95
    assert parse_vc_value("confident", "unit")[0] == 0.85


def test_unknown_prompt_variant_fails_before_the_model_loads(cfg):
    """A typo must not surface hours into Phase 5, after a full generation."""
    from vc_uq import prompts
    from vc_uq.generate import Generator
    from vc_uq.store import Store

    bad = cfg.with_overrides(["phase5.paraphrase.variants=[vc_post_v1,vc_post_v99]"])
    assert prompts.unknown_config_variants(bad) == {
        "phase5.paraphrase.variants": ["vc_post_v99"]}

    store = Store(bad)
    with pytest.raises(KeyError, match="unknown prompt variant"):
        Generator(bad, store)

    assert prompts.unknown_config_variants(cfg) == {}
    assert Generator(cfg, Store(cfg)) is not None


def test_every_config_named_variant_is_registered(cfg):
    """The shipped config must not name a variant that does not exist."""
    from vc_uq import prompts

    for key in prompts.VARIANT_CONFIG_KEYS:
        value = cfg.get(key, None)
        assert value is not None, f"{key} vanished from the config"
    prompts.check_config_variants(cfg)


def test_pitfall_reports_an_unregistered_variant(cfg):
    from vc_uq.pitfalls import run_checks

    name = "every prompt variant named in config is registered"
    bad = cfg.with_overrides(["generation.prompt_variant_post=nope_v1"])
    check = next(c for c in run_checks(bad).checks if c.name == name)
    assert check.passed is False and check.severity == "fatal"
    assert next(c for c in run_checks(cfg).checks if c.name == name).passed is True


def test_prehoc_prompt_carries_no_answer_and_is_marked_pre():
    """vc_pre and vc_post are different constructs; the registry encodes which."""
    from vc_uq import prompts

    spec = prompts.get("vc_pre_v1")
    assert spec.kind == "pre"
    body = " ".join(m["content"] for m in spec.build(question="Q"))
    assert "Do NOT answer" in body
    assert "Answer:" not in body, "a pre-hoc prompt must not solicit an answer"

    assert all(prompts.get(v).kind == "post"
               for v in prompts.variants("post"))
    assert prompts.variants("pre") == ["vc_pre_v1"]


def test_paraphrase_variants_differ_only_in_wording():
    """Per-question spread across them is test-retest reliability, so message
    structure has to be held constant or the spread measures something else."""
    from vc_uq import prompts

    built = [prompts.get(v).build(question="Q") for v in
             [f"vc_post_v{i}" for i in range(1, 11)]]
    assert all(len(m) == 2 for m in built)
    assert len({tuple(x["role"] for x in m) for m in built}) == 1
    assert len({m[1]["content"] for m in built}) == 1, "the user turn must be identical"
    assert len({m[0]["content"] for m in built}) == 10, "systems must all differ"
