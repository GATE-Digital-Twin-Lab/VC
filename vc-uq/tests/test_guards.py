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
from vc_uq.parsing import (OK, NO_CONFIDENCE_FIELD, OUT_OF_RANGE, UNPARSEABLE_VALUE,
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


def test_partition_U_refuses_to_run_without_classification_draws():
    """No classification pass means U is undefined, not empty."""
    df = pd.DataFrame([{"q_id": "a", "split": "calib", "draw_set": "downstream",
                        "correct": False, "draw_idx": 0}])
    with pytest.raises(ValueError, match="classify"):
        partition_U(df)


def test_partition_U_keys_on_the_pass_not_the_split():
    """The held-out thing is the PASS, not the question.

    Splits are by q_id, so filtering on split == "classify" decides U for the
    classify-split questions using every draw they have -- the same draws, under
    another name -- and leaves the calib/eval questions unlabelled, which is how
    the fallback to `censored` crept in and made 6.7's U curve tautological.
    """
    df = pd.DataFrame([
        # An eval-split question: it has both passes, and only the first counts.
        {"q_id": "a", "split": "eval", "draw_set": "classify",
         "draw_idx": 0, "correct": False},
        {"q_id": "a", "split": "eval", "draw_set": "classify",
         "draw_idx": 1, "correct": False},
        {"q_id": "a", "split": "eval", "draw_set": "downstream",
         "draw_idx": 0, "correct": True},
    ])
    out = partition_U(df)
    assert list(out["q_id"]) == ["a"], "every question gets a verdict, not just 30%"
    assert bool(out["in_U"].iloc[0]), \
        "a correct DOWNSTREAM draw must not rescue the question from U"
    assert int(out["n_classify_draws"].iloc[0]) == 2


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


@pytest.mark.parametrize("text,answer", [
    ("Answer: 1", "1"),
    ("Answer: 0", "0"),
    ("How many moons does Earth have?\nAnswer: 1", "1"),
    ("Answer: Lisbon\n0.9", "Lisbon"),
])
def test_a_number_is_only_a_confidence_when_it_follows_the_label(text, answer):
    """There used to be a fallback that read a bare number off the last line.

    With no confidence line the last line is the ANSWER, so "Answer: 1" was
    read as confidence 1.0 and marked ok -- a trivia answer became a confidence
    score. The value must follow a Confidence label or it is not a value.
    """
    p = parse_answer_and_vc(text)
    assert p.answer == answer
    assert p.vc is None
    assert p.status == NO_CONFIDENCE_FIELD


def test_parse_scales():
    assert parse_answer_and_vc("Answer: x\nConfidence: 0.8").vc == pytest.approx(0.8)
    assert parse_answer_and_vc("Answer: x\nConfidence: 1").vc == pytest.approx(1.0)
    assert parse_answer_and_vc("Answer: x\nConfidence: 0").vc == pytest.approx(0.0)
    assert parse_answer_and_vc("Answer: x\nConfidence: .5").vc == pytest.approx(0.5)


@pytest.mark.parametrize("token,status", [
    ("confident", UNPARSEABLE_VALUE),
    ("not confident", UNPARSEABLE_VALUE),
    ("not certain", UNPARSEABLE_VALUE),
    ("highly confident", UNPARSEABLE_VALUE),
    ("high", UNPARSEABLE_VALUE),
    ("90%", OUT_OF_RANGE),
    ("90", OUT_OF_RANGE),
])
def test_only_a_plain_number_in_range_is_a_confidence(token, status):
    """Word ladders are gone, and this is why.

    They were matched as substrings, so "not confident" read 0.85 and "not
    certain" read 1.00 -- an INVERTED value marked ok, in the direction that
    makes VC look worse calibrated than it is. Negation-blindness is the same
    failure that rules cosine out of clustering; it has no place in the
    measurement either. A worded reply is now a recorded failure.
    """
    p = parse_answer_and_vc(f"Answer: x\nConfidence: {token}")
    assert p.vc is None, f"{token!r} must not become a number"
    assert p.status == status
    assert p.answer == "x", "a bad confidence must not cost us the answer"


def test_out_of_range_is_recorded_not_rescaled():
    """A model answering 85 when asked for a number in [0, 1] has not followed
    the format. Dividing by 100 would launder that into a valid reading, so it
    is recorded as out of range and shows up in the parse audit instead."""
    p = parse_answer_and_vc("Answer: x\nConfidence: 85")
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


def _sheet_and_scored_with_duplicates(cfg):
    """The same fixture with per-draw rows, i.e. unique_answers off.

    The simulated world emits few distinct strings, so deduplicating collapses
    its sheet to a handful of rows -- too few for the shuffling and partial
    coverage tests below to say anything. Those tests are about sheet ORDER and
    coverage accounting, not about deduplication, so they pin the un-deduplicated
    path explicitly rather than depending on whichever default is current.
    """
    return _sheet_and_scored(cfg.with_overrides(["phase0.labels.unique_answers=false"]))


def test_label_sheet_holds_each_question_answer_pair_once(cfg):
    """N_MAX draws repeat themselves, and a repeat is not a second observation.

    Identical text scores an identical e_cos and earns an identical human
    verdict, so labelling it twice spends the budget on a row that cannot
    disagree -- and then kappa and AUROC count it as independent evidence and
    report more precision than the labels contain.
    """
    _, deduped, _ = _sheet_and_scored(cfg)
    assert not deduped.duplicated(subset=["q_id", "answer"]).any()

    _, per_draw, _ = _sheet_and_scored_with_duplicates(cfg)
    assert per_draw.duplicated(subset=["q_id", "answer"]).any(), \
        "fixture no longer produces repeats, so the dedup assertion proves nothing"

    # A question may still appear more than once -- with DIFFERENT answers.
    # Collapsing to one row per question would drop exactly the questions the
    # model is inconsistent on, which are the informative ones.
    assert deduped["q_id"].duplicated().any()


def test_label_sheet_is_shuffled_not_stratum_ordered(cfg):
    """Unshuffled, the sheet is written in stratum order, so labelling the first
    half would cover only the low-e_cos bins of the first dataset."""
    _, sheet, _ = _sheet_and_scored_with_duplicates(cfg)
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
    cfg2, sheet, _ = _sheet_and_scored_with_duplicates(cfg)
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
    cfg2, sheet, _ = _sheet_and_scored_with_duplicates(cfg)
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
    assert "q_that_does_not_exist#classify#0" in report["unjoined_examples"]
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
    audit = _json.loads((store.phase_dir("generation")
                         / f"parse_audit__{variant}__T{T}__classify.json")
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


# --------------------------------------------------------------------------
# llama.cpp backend
#
# llama-cpp-python is not installed here (and will not be on a 4 GB laptop),
# so these exercise the parts that are pure Python: the entailment verdict
# mapping and the teacher-forcing index arithmetic. Both are reached with
# duck-typed stand-ins; neither imports llama_cpp.
# --------------------------------------------------------------------------

class _ScriptedLM:
    """Returns a fixed verdict string, whatever it is asked."""

    name = "scripted"

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    def generate(self, messages, *, params, seed):
        from vc_uq.backends.base import Generation, TokenStats
        self.calls += 1
        empty = np.zeros(0)
        return Generation(text=self.verdict,
                          stats=TokenStats(empty, empty, empty))


def test_neutral_verdict_does_not_merge_two_answers():
    """A three-way judge emits a LABEL; only "entailment" is entailment.

    Returning 0.5 for neutral was worse than vague: clustering merges when
    min(fwd, bwd) >= nli.entail_threshold, and that threshold is 0.5, so every
    neutral verdict merged two distinct answers -- deflating H_sem and inflating
    largest_cluster_share, the two quantities the diversity 2x2 is built from.
    """
    from vc_uq.backends.llamacpp import LlamaCppNLI
    from vc_uq.cluster import cluster_answers

    for verdict, expected in [("entailment", 1.0), ("neutral", 0.0),
                              ("contradiction", 0.0), ("Entailment.", 1.0)]:
        judge = LlamaCppNLI(_ScriptedLM(verdict))
        assert judge.entailment_prob("a", "b") == expected, verdict

    neutral = LlamaCppNLI(_ScriptedLM("neutral"))
    ids = cluster_answers(["Lisbon", "Madrid", "Oslo"], neutral, threshold=0.5)
    assert ids == [0, 1, 2], "neutral verdicts must not collapse distinct answers"

    entailing = LlamaCppNLI(_ScriptedLM("entailment"))
    assert cluster_answers(["Lisbon", "Lisboa"], entailing, threshold=0.5) == [0, 0]


def test_unrecognised_verdict_is_counted_not_absorbed():
    """Judge parse failure is a reported rate, like VC parse failure."""
    from vc_uq.backends.llamacpp import LlamaCppNLI

    judge = LlamaCppNLI(_ScriptedLM("I am not sure about that one"))
    assert judge.entailment_prob("a", "b") == 0.0
    judge.entailment_prob("c", "d")
    assert judge.audit() == {"judge": "scripted-nli", "n_calls": 2,
                             "n_unparsed": 2, "unparsed_rate": 1.0}

    clean = LlamaCppNLI(_ScriptedLM("entailment"))
    clean.entailment_prob("a", "b")
    assert clean.audit()["unparsed_rate"] == 0.0


class _RefusesWholeBufferConversion:
    """Stands in for llama.cpp's [n_ctx, n_vocab] score buffer.

    Slicing works; converting the whole thing does not. At n_ctx=4096 and a
    128k vocabulary that buffer is 2.1 GB of float32, so a float64 conversion
    of all of it costs 4.2 GB per call -- twice per draw, tens of thousands of
    draws -- to read a few dozen rows.
    """

    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, key):
        return self._arr[key]

    def __array__(self, *a, **k):
        raise AssertionError("whole scores buffer converted; slice it first")


class _FakeLlama:
    def __init__(self, scores, prompt_len, cont_ids, added_special=False):
        self.scores = _RefusesWholeBufferConversion(scores)
        self._prompt_len = prompt_len
        self._cont_ids = cont_ids
        self.evaluated = None
        self.prompt_add_bos = None
        # Reports added_special, as a real ChatFormatterResponse does. A handler
        # that returns a bare string is refused -- see the double-BOS test.
        self.chat_handler = type("H", (), {
            "to_chat_completion_prompt": staticmethod(
                lambda m: ("PROMPT", added_special))})()

    def tokenize(self, text, add_bos=False, special=False):
        # Keyed on the TEXT, not on add_bos: add_bos is the thing under test.
        if text == b"PROMPT":
            self.prompt_add_bos = add_bos
            return list(range(self._prompt_len))
        return list(self._cont_ids)

    def reset(self):
        pass

    def eval(self, ids):
        self.evaluated = list(ids)


def test_teacher_force_reads_only_the_rows_it_needs():
    """Index arithmetic and buffer handling, without llama-cpp-python.

    Position i predicts token i+1, so the row scoring the first continuation
    token is at len(prompt_ids) - 1. Getting this off by one would silently
    attribute each token's entropy to its neighbour.
    """
    from vc_uq.backends.llamacpp import LlamaCppLM, resolve_chat_formatter

    rng = np.random.default_rng(0)
    n_ctx, vocab, prompt_len = 16, 8, 5
    cont_ids = [2, 5, 1]
    scores = rng.normal(size=(n_ctx, vocab)).astype(np.float32)

    lm = object.__new__(LlamaCppLM)          # no GGUF, no llama_cpp import
    lm._llm = _FakeLlama(scores, prompt_len, cont_ids)
    lm._format_prompt, _ = resolve_chat_formatter(lm._llm)

    stats = lm.teacher_force([{"role": "user", "content": "q"}], "irrelevant")

    assert lm._llm.evaluated == list(range(prompt_len)) + cont_ids
    assert stats.n_tokens == len(cont_ids)

    # Rows 4, 5, 6 score continuation tokens 0, 1, 2.
    for i, tok in enumerate(cont_ids):
        z = scores[prompt_len - 1 + i].astype(np.float64)
        p = np.exp(z - z.max())
        p /= p.sum()
        assert stats.entropies[i] == pytest.approx(float(-(p * np.log(p)).sum()))
        assert stats.chosen_probs[i] == pytest.approx(float(p[tok]))
        assert stats.logprobs[i] == pytest.approx(float(np.log(p[tok])))


@pytest.mark.parametrize("added_special, expect_add_bos", [(True, False),
                                                           (False, True)])
def test_teacher_force_prepends_a_bos_exactly_when_sampling_does(added_special,
                                                                 expect_add_bos):
    """The BOS must be added on the same terms as create_chat_completion.

    llama-cpp-python tokenises a chat prompt with ``add_bos=not added_special``
    (llama_chat_format.py). A template that emits its own ``<bos>`` -- gemma-3
    and Llama-3 both do -- sets the flag, so an unconditional ``add_bos=True``
    here prepends a SECOND BOS and teacher forcing scores the answer under a
    prompt one token longer than the one that produced it. llama.cpp warns on
    stderr and proceeds, so nothing fails and every h_tok_mean, logp_mean and
    min_token_p in the study is measured against the wrong context.

    That is not symmetric noise. h_tok is the token-level baseline VC is
    compared against, so corrupting it flatters VC -- which is the hypothesis
    under test.
    """
    from vc_uq.backends.llamacpp import LlamaCppLM, resolve_chat_formatter

    lm = object.__new__(LlamaCppLM)
    lm._llm = _FakeLlama(np.zeros((8, 3), dtype=np.float32), 4, [1, 2],
                         added_special=added_special)
    lm._format_prompt, _ = resolve_chat_formatter(lm._llm)
    lm.teacher_force([{"role": "user", "content": "q"}], "answer")

    assert lm._llm.prompt_add_bos is expect_add_bos


def test_teacher_force_on_an_empty_continuation_is_empty_not_zero():
    from vc_uq.backends.llamacpp import LlamaCppLM, resolve_chat_formatter

    lm = object.__new__(LlamaCppLM)
    lm._llm = _FakeLlama(np.zeros((4, 3), dtype=np.float32), 2, [])
    lm._format_prompt, _ = resolve_chat_formatter(lm._llm)
    stats = lm.teacher_force([{"role": "user", "content": "q"}], "")
    assert stats.n_tokens == 0
    assert stats.summary()["h_tok_mean"] is None


# --------------------------------------------------------------------------
# Chat-template resolution
#
# teacher_force must rebuild exactly the string create_chat_completion used.
# Guessing produces a complete run whose every token statistic describes the
# wrong context -- and since the token-entropy family is what VC is compared
# against, a degraded baseline flatters VC. So the resolver refuses to guess.
# --------------------------------------------------------------------------

class _Llm:
    """Minimal stand-in: only what resolve_chat_formatter inspects."""

    def __init__(self, *, handler=None, template=None):
        if handler is not None:
            self.chat_handler = handler
        else:
            self.chat_handler = None
        self.metadata = {"tokenizer.chat_template": template} if template else {}

    def token_bos(self):
        return 1

    def token_eos(self):
        return 2

    def detokenize(self, ids):
        return b"<bos>" if ids == [1] else b"<eos>"


def test_formatter_prefers_the_models_own_handler():
    from vc_uq.backends.llamacpp import resolve_chat_formatter

    handler = type("H", (), {
        "to_chat_completion_prompt": staticmethod(
            lambda m: ("FROM HANDLER", True))})()
    render, source = resolve_chat_formatter(
        _Llm(handler=handler, template="{{ 'FROM TEMPLATE' }}"))

    assert render([{"role": "user", "content": "q"}]) == ("FROM HANDLER", True)
    assert source == "chat_handler.to_chat_completion_prompt"


def test_formatter_refuses_a_handler_that_hides_whether_it_added_the_bos():
    """A bare string is not enough to reconstruct the conditioning.

    ``added_special`` decides whether a BOS is prepended. Defaulting it either
    way is a coin flip on whether every token statistic is aligned with the
    text that produced it, and being wrong costs nothing visible at runtime.
    """
    from vc_uq.backends.llamacpp import resolve_chat_formatter

    handler = type("H", (), {
        "to_chat_completion_prompt": staticmethod(lambda m: "BARE STRING")})()
    render, _ = resolve_chat_formatter(_Llm(handler=handler))
    with pytest.raises(RuntimeError, match="bare string"):
        render([{"role": "user", "content": "q"}])


def test_formatter_falls_back_to_the_gguf_own_template(monkeypatch):
    """Model-agnostic: the GGUF carries the template it was trained with.

    The previous implementation hardcoded format_llama3 here, which is right by
    coincidence for a Llama GGUF and wrong for every other model.
    """
    from vc_uq.backends import llamacpp

    seen = {}

    def fake_jinja(template, *, bos_token, eos_token):
        seen.update(template=template, bos=bos_token, eos=eos_token)
        return lambda msgs: (f"RENDERED[{template}]", True)

    monkeypatch.setattr(llamacpp, "_jinja_formatter", fake_jinja)
    render, source = llamacpp.resolve_chat_formatter(_Llm(template="QWEN-TEMPLATE"))

    assert render([]) == ("RENDERED[QWEN-TEMPLATE]", True)
    assert source == "gguf tokenizer.chat_template"
    assert seen == {"template": "QWEN-TEMPLATE", "bos": "<bos>", "eos": "<eos>"}


def test_formatter_refuses_to_guess(monkeypatch):
    """No handler, no template, no configured format -> stop, do not improvise."""
    from vc_uq.backends import llamacpp

    with pytest.raises(RuntimeError, match="cannot reconstruct"):
        llamacpp.resolve_chat_formatter(_Llm())

    # An explicit chat_format that matches nothing is also a refusal, not a guess.
    monkeypatch.setattr(llamacpp, "_named_formatter", lambda name: None)
    with pytest.raises(RuntimeError, match="matches no formatter"):
        llamacpp.resolve_chat_formatter(_Llm(), chat_format="not-a-real-format")


def test_configured_chat_format_is_used_when_the_gguf_has_no_template(monkeypatch):
    from vc_uq.backends import llamacpp

    monkeypatch.setattr(llamacpp, "_named_formatter",
                        lambda name: (lambda msgs: (f"NAMED[{name}]", False)))
    render, source = llamacpp.resolve_chat_formatter(_Llm(), chat_format="chatml")
    assert render([]) == ("NAMED[chatml]", False)
    assert source == "chat_format='chatml'"


# --------------------------------------------------------------------------
# Batched entailment
#
# An encoder head at batch size 1 wastes almost all of a GPU, and clustering
# makes on the order of half a million comparisons across the corpus. The
# batched path exists for that -- but it must not change a single cluster
# assignment, because H_sem and largest_cluster_share are built from them.
# --------------------------------------------------------------------------

class _PairJudge:
    """Deterministic judge exposing only the one-pair API."""

    def __init__(self, table, default=0.0):
        self.table = table
        self.default = default
        self.calls = 0

    def entailment_prob(self, premise, hypothesis):
        self.calls += 1
        return self.table.get((premise, hypothesis), self.default)


class _BatchJudge(_PairJudge):
    """The same judge, plus the batched API HFNLI exposes."""

    def __init__(self, table, default=0.0):
        super().__init__(table, default)
        self.batches = 0
        self.batch_sizes = []

    def entailment_probs(self, premises, hypotheses):
        self.batches += 1
        self.batch_sizes.append(len(premises))
        return np.array([self.table.get(pair, self.default)
                         for pair in zip(premises, hypotheses)])


def _symmetric(pairs, value=0.9):
    """Entailment in BOTH directions -- the only thing that merges clusters."""
    table = {}
    for a, b in pairs:
        table[(a, b)] = value
        table[(b, a)] = value
    return table


def test_batched_and_pairwise_clustering_agree():
    from vc_uq.cluster import cluster_answers

    answers = ["Lisbon", "Lisboa", "Madrid", "Lisbon", "Oslo", "Madrid city",
               "Lisboa", "Reykjavik"]
    table = _symmetric([("Lisbon", "Lisboa"), ("Madrid", "Madrid city")])

    pair = cluster_answers(answers, _PairJudge(table), threshold=0.5)
    batched = cluster_answers(answers, _BatchJudge(table), threshold=0.5)

    assert pair == batched == [0, 0, 1, 0, 2, 1, 0, 3]


def test_batched_path_keeps_first_qualifying_rep_when_several_qualify():
    """Entailment is not transitive, so more than one cluster can qualify.

    "the capital of Portugal" and "Lisboa" need not entail each other, yet both
    can entail "Lisbon" in both directions. The pairwise path short-circuits and
    takes the first; the batched path scores every representative before
    choosing, so it must scan in index order rather than taking the strongest
    match. Picking argmax instead would reassign answers between clusters and
    move n_clusters, H_sem and largest_cluster_share.
    """
    from vc_uq.cluster import cluster_answers

    table = _symmetric([("the capital of Portugal", "Lisbon"), ("Lisboa", "Lisbon")])
    table[("the capital of Portugal", "Lisboa")] = 0.02   # not each other
    table[("Lisboa", "the capital of Portugal")] = 0.02
    # Make the LATER representative the stronger match, so argmax would pick it.
    table[("Lisboa", "Lisbon")] = 0.99
    table[("Lisbon", "Lisboa")] = 0.99
    table[("the capital of Portugal", "Lisbon")] = 0.60
    table[("Lisbon", "the capital of Portugal")] = 0.60

    answers = ["the capital of Portugal", "Lisboa", "Lisbon"]
    pair = cluster_answers(answers, _PairJudge(table), threshold=0.5)
    batched = cluster_answers(answers, _BatchJudge(table), threshold=0.5)

    assert pair == [0, 1, 0], "the pairwise path takes the first qualifying rep"
    assert batched == pair, "the batched path must not prefer the stronger match"


def test_one_sided_entailment_does_not_merge_in_either_path():
    """Bidirectional is the whole point: "Paris" implies "a city in France",
    not the reverse, and merging on the one-way relation would collapse a
    specific answer into a vague one."""
    from vc_uq.cluster import cluster_answers

    table = {("Paris", "a city in France"): 0.99,
             ("a city in France", "Paris"): 0.05}
    answers = ["Paris", "a city in France"]

    assert cluster_answers(answers, _PairJudge(table), threshold=0.5) == [0, 1]
    assert cluster_answers(answers, _BatchJudge(table), threshold=0.5) == [0, 1]


def test_batched_path_collapses_the_call_count():
    """One judge call per answer instead of up to two per representative."""
    from vc_uq.cluster import cluster_answers

    answers = [f"answer {i}" for i in range(12)]     # all distinct, no merging
    table = {}

    pair = _PairJudge(table)
    cluster_answers(answers, pair, threshold=0.5)

    batch = _BatchJudge(table)
    cluster_answers(answers, batch, threshold=0.5)

    # 12 answers, no merges: the pairwise path makes 2 calls per existing
    # representative, the batched path one call per answer that has any.
    assert pair.calls == 2 * sum(range(12))
    assert batch.batches == 11
    assert batch.calls == 0, "the batched path must not fall back per pair"
    # Each batch carries both directions for every current representative.
    assert batch.batch_sizes == [2 * i for i in range(1, 12)]


def test_cluster_frame_uses_whichever_api_the_judge_offers(cfg):
    """cluster_frame must not care which judge it was handed."""
    import pandas as pd

    from vc_uq.cluster import cluster_frame

    rows = [{"q_id": "q0", "draw_idx": i, "answer": a}
            for i, a in enumerate(["Lisbon", "Lisboa", "Madrid"])]
    df = pd.DataFrame(rows)
    table = _symmetric([("Lisbon", "Lisboa")])

    a = cluster_frame(cfg, df, nli=_PairJudge(table))
    b = cluster_frame(cfg, df, nli=_BatchJudge(table))
    assert list(a["cluster_id"]) == list(b["cluster_id"]) == [0, 0, 1]
    assert list(a["f"]) == list(b["f"])


# --------------------------------------------------------------------------
# Elicitation robustness
#
# Two different failures are easy to confuse. The PARSER locates fields by
# name, so preamble, blank lines, markdown and trailing chatter are all
# harmless. The STOP SEQUENCE is upstream of the parser: it can halt generation
# before the confidence field is ever emitted, and no regex recovers that.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Answer: Lisbon\nConfidence: 0.9",
    "Answer: Lisbon\n\n\n\nConfidence: 0.9",
    "Sure! Happy to help.\n\nAnswer: Lisbon\n\nConfidence: 0.9",
    "Answer: Lisbon\nConfidence: 0.9\n\nLet me know if you need anything else.",
    "Confidence: 0.9\nAnswer: Lisbon",
    "**Answer:** Lisbon\n**Confidence:** 0.9",
    "**Answer:** **Lisbon**\n\n**Confidence:** 0.9",
    "- Answer: Lisbon\n- Confidence: 0.9",
    "### Answer: Lisbon\n### Confidence: 0.9",
    "> Answer: Lisbon\n> Confidence: 0.9",
    "Answer - Lisbon\nConfidence - 0.9",
    "  answer:   Lisbon  \n  CONFIDENCE:   0.9  ",
])
def test_fields_are_found_by_name_whatever_the_layout(text):
    """Layout is the model's business; the fields are located by name."""
    p = parse_answer_and_vc(text)
    assert p.status == "ok", text
    assert p.answer == "Lisbon", text
    assert p.vc == pytest.approx(0.9)


def test_markup_is_stripped_from_the_answer_text():
    """The answer feeds e_cos against a_star, so stray asterisks are noise in
    the correctness criterion, not cosmetic."""
    assert parse_answer_and_vc("**Answer:** **Lisbon**\nConfidence: 0.9").answer == "Lisbon"
    assert parse_answer_and_vc("- **Lisbon**\nConfidence: 0.9").answer == "Lisbon"
    assert parse_answer_and_vc("Answer: `Lisbon`\nConfidence: 0.9").answer == "Lisbon"


def test_truncated_generation_is_a_failure_the_parser_cannot_fix():
    """What a stop sequence removes is gone before parsing begins."""
    emitted = "Answer: Lisbon\n\nConfidence: 0.9"
    assert parse_answer_and_vc(emitted).vc == pytest.approx(0.9)

    truncated = emitted[:emitted.index("\n\n")]      # what a "\n\n" stop returns
    p = parse_answer_and_vc(truncated)
    assert p.answer == "Lisbon"
    assert p.vc is None
    assert p.status == NO_CONFIDENCE_FIELD


def test_no_stop_sequence_can_cut_the_reply_in_half(cfg):
    """The shipped config must not contain a whitespace-only stop."""
    from vc_uq.pitfalls import run_checks

    name = "no stop sequence can cut the reply in half"
    assert not [s for s in cfg.get("model.generation.stop") if s.strip() == ""]
    assert next(c for c in run_checks(cfg).checks if c.name == name).passed is True

    bad = cfg.with_overrides(['model.generation.stop=["\n\n"]'])
    check = next(c for c in run_checks(bad).checks if c.name == name)
    assert check.passed is False and check.severity == "fatal"


def test_vc_parse_failure_rate_is_fatal_above_the_limit(cfg):
    """Elicitation IS the measurement; a run that cannot read VC has no result.

    Without this the failure is silent: reliability() drops null rows, so Phase
    2 would emit empty tables rather than an error.
    """
    import pandas as pd

    from vc_uq.pitfalls import run_checks

    name = "verbalised confidence parsed off the draws"

    def frame(n_null, n_ok):
        rows = [{"q_id": f"q{i}", "vc_post": None, "parse_status": "no_confidence_field"}
                for i in range(n_null)]
        rows += [{"q_id": f"q{i}", "vc_post": 0.8, "parse_status": "ok"}
                 for i in range(n_null, n_null + n_ok)]
        return pd.DataFrame(rows)

    good = next(c for c in run_checks(cfg, answers=frame(1, 99)).checks if c.name == name)
    assert good.passed is True

    bad = next(c for c in run_checks(cfg, answers=frame(40, 60)).checks if c.name == name)
    assert bad.passed is False and bad.severity == "fatal"
    assert "40.0%" in bad.detail
    assert "no_confidence_field" in bad.detail, "say WHICH failure dominates"


# --------------------------------------------------------------------------
# judge_nli batching, and undefined vs wrong
# --------------------------------------------------------------------------

def _answers_and_questions():
    import pandas as pd

    answers = pd.DataFrame([
        {"q_id": "q0", "draw_idx": 0, "answer": "Lisbon"},
        {"q_id": "q0", "draw_idx": 1, "answer": "Lisboa"},
        {"q_id": "q0", "draw_idx": 2, "answer": "Madrid"},
        {"q_id": "q1", "draw_idx": 0, "answer": "Oslo"},
    ])
    questions = pd.DataFrame([{"q_id": "q0", "a_star": "Lisbon"},
                              {"q_id": "q1", "a_star": "Reykjavik"}])
    return answers, questions


def test_judge_nli_agrees_whether_or_not_the_judge_batches(cfg):
    """Correctness must not depend on which API the judge happens to expose."""
    from vc_uq.judge import judge_nli

    answers, questions = _answers_and_questions()
    table = _symmetric([("Lisbon", "Lisboa")])
    for a in ("Lisbon", "Lisboa", "Madrid", "Oslo", "Reykjavik"):
        table[(a, a)] = 0.99

    pair = judge_nli(cfg, answers, questions, nli=_PairJudge(table))
    batched = judge_nli(cfg, answers, questions, nli=_BatchJudge(table))

    assert list(pair["correct_nli"]) == list(batched["correct_nli"])
    assert list(pair["correct_nli"]) == [True, True, False, False]


def test_judge_nli_sends_every_pair_in_one_call(cfg):
    """2N forward passes at batch size 1 is most of the corpus wasted."""
    from vc_uq.judge import judge_nli

    answers, questions = _answers_and_questions()
    judge = _BatchJudge({})
    judge_nli(cfg, answers, questions, nli=judge)

    assert judge.batches == 1, "one call for the whole column"
    assert judge.batch_sizes == [2 * len(answers)], "both directions, all rows"
    assert judge.calls == 0, "must not fall back to the per-pair API"


@pytest.mark.parametrize("answer,a_star", [
    ("Paris", "a city in France"),        # answer entails reference, not back
    ("a city in France", "Paris"),        # reference entails answer, not back
])
def test_judge_nli_is_bidirectional(cfg, answer, a_star):
    """One-way entailment is not sameness, in EITHER direction.

    "Paris" entails "a city in France" but the two are not the same answer, so
    accepting the one-way relation would mark a vague reply correct against a
    specific reference (or the reverse). Both orientations are parametrised
    because checking only one lets an implementation that drops fwd or bwd pass.
    """
    import pandas as pd

    from vc_uq.judge import judge_nli

    answers = pd.DataFrame([{"q_id": "q0", "draw_idx": 0, "answer": answer}])
    questions = pd.DataFrame([{"q_id": "q0", "a_star": a_star}])
    one_way = {("Paris", "a city in France"): 0.99,
               ("a city in France", "Paris"): 0.02}

    for judge in (_PairJudge(one_way), _BatchJudge(one_way)):
        got = judge_nli(cfg, answers, questions, nli=judge)["correct_nli"].tolist()
        assert got == [False], f"{answer!r} vs {a_star!r} with {type(judge).__name__}"


def test_an_undefined_criterion_is_marked_not_absorbed(cfg):
    """`correct` must be a plain bool, so NA becomes False -- but visibly.

    An undefined criterion recorded as a wrong answer pushes p_hat down and
    beta up, and inflates U with an instrument artifact rather than a finding
    about the model.
    """
    import pandas as pd

    from vc_uq.judge import attach_correct

    df = pd.DataFrame({
        "q_id": ["q0", "q1", "q2"],
        "correct_cos": pd.array([True, None, False], dtype="boolean"),
    })
    out = attach_correct(cfg.with_overrides(["judge.primary=cos"]), df)

    assert list(out["correct"]) == [True, False, False]
    assert list(out["correct_undefined"]) == [False, True, False]


def test_pitfall_reports_undefined_correctness(cfg):
    import pandas as pd

    from vc_uq.pitfalls import run_checks

    name = "correctness is defined for every scored answer"
    clean = pd.DataFrame({"correct_undefined": [False, False]})
    assert next(c for c in run_checks(cfg, answers=clean).checks
                if c.name == name).passed is True

    dirty = pd.DataFrame({"correct_undefined": [False, True, True]})
    check = next(c for c in run_checks(cfg, answers=dirty).checks if c.name == name)
    assert check.passed is False and check.severity == "warn"
    assert "2 of 3" in check.detail


def test_medoid_anchor_is_the_most_central_actual_draw(cfg):
    """The anchor must be a real draw, deterministic, and label-free.

    Deterministic because CLM's guarantee is conditional on the anchor being a
    fixed function of the draws; label-free because a_star does not exist at
    test time; an actual draw rather than a centroid because a mean vector
    corresponds to no text and cannot be embedded or compared.
    """
    from vc_uq.backends.mock import MockEmbedder
    from vc_uq.judge import EmbeddingSpace, medoid_anchor

    draws = ["Lisbon [[sem:a]]", "Lisboa [[sem:a]]", "Lisbon, Portugal [[sem:a]]",
             "Porto [[sem:b]]", "Madrid [[sem:c]]"]
    emb = MockEmbedder(dim=64, noise=0.25, seed=3)
    space = EmbeddingSpace(vectors=dict(zip(draws, emb.embed(draws))), dim=64,
                           mean_centered=False, whitened=False)

    pick = medoid_anchor(draws, space)
    assert pick in draws, "the anchor is a member of the set, not a mean vector"
    assert "sem:a" in pick, "it must land in the dominant cluster, not on an outlier"
    assert medoid_anchor(draws, space) == pick, "deterministic"
    assert medoid_anchor(list(reversed(draws)), space) == pick, \
        "order of the draw list must not change which draw is most central"
    assert medoid_anchor(["only one"], space) == "only one"


# --------------------------------------------------------------------------
# Sample-diversity signals: descriptive, never a stopping rule
# --------------------------------------------------------------------------

def test_diversity_signals_are_not_stopping_rules(cfg):
    """A diversity statistic over one draw is not a value.

    One draw is one cluster, so H_sem reads 0 and largest-share reads 1 -- the
    exact values a "stop when converged" rule treats as convergence. As online
    rules they fire at k=1 for every threshold, which makes them duplicates of
    fixed_k=1 under names that imply otherwise. A minimum draw count would fix
    that by changing the efficiency each rule reports, so they are kept out of
    the stopping table entirely.
    """
    from vc_uq import clm

    for name in ("self_consistency", "semantic_entropy"):
        assert name not in cfg.get("phase4.stop_rules")
        assert name not in clm.RULE_DIRECTION
        assert name not in clm.RULE_UNITS

    # Every configured rule is implemented and declares a direction and a unit.
    for rule in cfg.get("phase4.stop_rules"):
        assert rule in clm.RULE_DIRECTION, rule
        assert rule in clm.RULE_UNITS, rule
    assert "fixed_k" in cfg.get("phase4.stop_rules"), "the null baseline is mandatory"


def test_diversity_signals_survive_as_descriptive_comparators(cfg):
    """Dropping them as rules must not drop them as baselines.

    The claim that diversity-based signals also fail on low-diversity U rests on
    these, so they have to keep reaching the 6.6 2x2 and the Phase 2 AUROC
    comparison.
    """
    import pandas as pd

    from vc_uq.cluster import cluster_frame, question_diversity, semantic_entropy

    rows = [{"q_id": "q0", "draw_idx": i, "answer": a} for i, a in
            enumerate(["Lisbon", "Lisbon", "Madrid", "Lisbon"])]
    df = cluster_frame(cfg, pd.DataFrame(rows), nli=_PairJudge({}))

    assert "cluster_id" in df.columns
    assert "f" in df.columns, "per-answer self-consistency feeds the Phase 2 AUROCs"

    div = question_diversity(df).iloc[0]
    assert div["n_clusters"] == 2
    assert div["largest_cluster_share"] == pytest.approx(0.75)
    assert div["H_sem"] == pytest.approx(semantic_entropy([0, 0, 1, 0]))
    assert div["H_sem"] > 0


def test_the_per_phase_workflow_carries_its_state_forward(cfg):
    """Each phase must write what the next one reads (protocol section 1).

    step2 wrote the answers table BEFORE clustering, so cluster_id, f and
    correct lived only in memory: every per-phase command after `survival`
    failed on a missing column, and had it not, every draw of every question
    would have landed in one cluster.
    """
    from vc_uq import pipeline
    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    c = cfg.with_overrides(["dataset.triviaqa.n_questions=24",
                            "dataset.fabricated.n_questions=8",
                            "generation.n_max=6", "phase0.n_hand_label=40",
                            "phase0.stratify_bins=3",
                            "phase0.null_band.n_mismatched_pairs=200"])
    store = Store(c)
    state = pipeline.PipelineState(cfg=c, store=store)
    state.questions = build_questions(c)
    state = pipeline.step3_generate(state)
    state = pipeline.step2_gate(state, simulate_labels=True)
    state = pipeline.step4_survival(state)

    # What a later `vc_uq clm` would actually load off disk.
    reloaded = store.read_answers()
    for col in ("cluster_id", "f", "correct"):
        assert col in reloaded.columns, f"{col} was not persisted by survival"
    assert reloaded["cluster_id"].notna().all(), "cluster_id must not be all null"
    assert reloaded.groupby("q_id")["cluster_id"].nunique().max() > 1, \
        "every draw landing in one cluster is the failure this guards against"


# --------------------------------------------------------------------------
# Two generation passes (protocol 6.4): U is decided on draws nothing is
# certified on. Without this the 6.7 U-curve is its own definition restated.
# --------------------------------------------------------------------------

def _two_pass_cfg(cfg):
    return cfg.with_overrides(["dataset.triviaqa.n_questions=24",
                               "dataset.fabricated.n_questions=8",
                               "generation.n_max=6", "phase0.n_hand_label=40",
                               "phase0.stratify_bins=3",
                               "phase0.null_band.n_mismatched_pairs=200"])


def test_the_second_pass_is_fresh_draws_not_a_relabelled_copy(cfg):
    """"Fresh draws" has to mean different tokens, not a second name.

    The prompt, the decoder and the model are identical between the passes, so
    the ONLY thing that can make them independent is the seed. If the pass name
    is not salted in, the downstream pass replays the classification pass token
    for token and the disjointness is cosmetic.
    """
    from vc_uq.generate import Generator
    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    c = _two_pass_cfg(cfg)
    store = Store(c)
    gen = Generator(c, store)
    qs = build_questions(c)

    a = gen.answer_keys(qs, n_max=3, temperature=0.8, variant="vc_post_v1",
                        draw_set="classify")
    b = gen.answer_keys(qs, n_max=3, temperature=0.8, variant="vc_post_v1",
                        draw_set="downstream")

    assert len(a) == len(b) == 3 * len(qs)
    assert set(a["seed"]).isdisjoint(set(b["seed"])), \
        "the two passes must not share a single seed"
    # And the key really does tell them apart, so the cache cannot merge them.
    merged = a.merge(b, on=["q_id", "draw_idx", "seed"], how="inner")
    assert merged.empty


def test_phase1_runs_both_passes_over_the_right_questions(cfg):
    from vc_uq.datasets import build_questions
    from vc_uq.generate import run_phase1
    from vc_uq.store import Store

    c = _two_pass_cfg(cfg)
    store = Store(c)
    qs = build_questions(c)
    info = run_phase1(c, store, qs)

    answers = store.read_answers()
    cls = answers[answers["draw_set"] == "classify"]
    dwn = answers[answers["draw_set"] == "downstream"]

    assert set(cls["q_id"]) == set(qs["q_id"]), \
        "the classification pass must cover EVERY question -- in_U has no fallback"
    assert set(dwn["q_id"]) == set(qs.loc[qs["split"].isin(["calib", "eval"]), "q_id"])
    assert info["n_downstream_draws"] > 0
    assert set(dwn["split"]) <= {"calib", "eval"}


def test_staged_generation_is_a_strict_subset_of_the_full_pass(cfg):
    """`generate --splits tau_select` must not cost the full run anything.

    The Phase 0 gate needs only tau_select, and a failed gate ends the study, so
    staging it first is the difference between spending an afternoon and
    spending a week to find out the criterion is blunt. That is only true if the
    staged draws are IDENTICAL to the ones the full pass would have made -- the
    seed is a hash of (draw_set, q_id, draw_idx, T, variant) and `split` is
    deliberately not in the cache key, so they are. Checked, because the failure
    is silent: a differing seed would just regenerate, and the run would look
    fine while having thrown the staged pass away.
    """
    from vc_uq.datasets import build_questions
    from vc_uq.generate import run_phase1
    from vc_uq.store import Store

    c = _two_pass_cfg(cfg)
    qs = build_questions(c)

    store = Store(c)
    staged = run_phase1(c, store, qs, splits=["tau_select"])
    partial = store.read_answers()
    tau_ids = set(qs.loc[qs["split"] == "tau_select", "q_id"])

    assert set(partial["q_id"]) == tau_ids
    assert staged["n_questions_generated"] == len(tau_ids)
    assert staged["n_questions"] == len(qs), \
        "the question table is written whole; a staged pass must not delete rows"
    # vc_pre is elicited only where it was asked for, and left null elsewhere
    # rather than imputed.
    written = store.read_questions()
    assert written.loc[written["split"] == "tau_select", "vc_pre"].notna().all()
    assert written.loc[written["split"] != "tau_select", "vc_pre"].isna().all()

    full = run_phase1(c, store, qs)
    assert full["resume"]["answers[classify]"]["reused_from_cache"] >= len(partial), \
        "the full pass must reuse every staged draw, not resample it"

    after = store.read_answers()
    key = ["q_id", "draw_set", "draw_idx"]
    rejoined = partial.merge(after, on=key, suffixes=("_staged", "_full"))
    assert len(rejoined) == len(partial)
    assert (rejoined["answer_staged"] == rejoined["answer_full"]).all()
    assert (rejoined["seed_staged"] == rejoined["seed_full"]).all()


def test_generation_checkpoints_so_a_crash_keeps_completed_draws(cfg):
    """A kill signal at hour three must not cost hours one and two.

    The failure this replaces was total: draws accumulated in memory and the
    parquet was written once, at the end, so anything short of a clean exit left
    data/raw/ empty. At 27B and ~2.4 s/draw that is the difference between
    losing minutes and losing an afternoon.
    """
    from vc_uq.datasets import build_questions
    from vc_uq.generate import Generator
    from vc_uq.store import Store

    c = cfg.with_overrides(["generation.checkpoint_every=5"])
    store = Store(c)
    qs = build_questions(c).head(4)

    gen = Generator(c, store)
    boom = RuntimeError("node evicted")
    n_calls = {"n": 0}
    real_call = gen._call

    def die_after_12(*a, **k):
        n_calls["n"] += 1
        if n_calls["n"] > 12:
            raise boom
        return real_call(*a, **k)

    gen._call = die_after_12
    with pytest.raises(RuntimeError, match="node evicted"):
        gen.draw_answers(qs, n_max=8, resume=True, store_per_position=False)

    survived = store.read_answers()
    assert len(survived) >= 10, \
        f"only {len(survived)} draws survived the crash; checkpointing did nothing"

    # And the resumed run regenerates only what is genuinely missing.
    gen2 = Generator(c, store)
    out = gen2.draw_answers(qs, n_max=8, resume=True, store_per_position=False)
    assert len(out) == 4 * 8
    assert gen2.last_resume["reused_from_cache"] == len(survived)
    assert gen2.last_resume["generated"] == 4 * 8 - len(survived)


def test_vc_pre_checkpoints_so_a_crash_keeps_completed_elicitations(cfg):
    """R_pre elicitations on every question is thousands of calls.

    The post-hoc path was fixed for this; the pre-hoc path kept accumulating in
    memory, so a kill signal still cost the whole vc_pre pass.
    """
    from vc_uq.datasets import build_questions
    from vc_uq.generate import Generator
    from vc_uq.store import Store

    c = cfg.with_overrides(["generation.checkpoint_every=4", "generation.r_pre=6"])
    store = Store(c)
    qs = build_questions(c).head(4)

    gen = Generator(c, store)
    n = {"n": 0}
    real_call = gen._call

    def die_after_10(*a, **k):
        n["n"] += 1
        if n["n"] > 10:
            raise RuntimeError("node evicted")
        return real_call(*a, **k)

    gen._call = die_after_10
    with pytest.raises(RuntimeError, match="node evicted"):
        gen.draw_vc_pre(qs, resume=True)

    survived = store.read_vc_pre_repeats()
    assert len(survived) >= 8, \
        f"only {len(survived)} elicitations survived; checkpointing did nothing"

    out = Generator(c, store).draw_vc_pre(qs, resume=True)
    assert len(out) == 4 * 6


def test_progress_bar_is_silent_when_stderr_is_not_a_terminal(cfg):
    """A bar sized by cache HITS would read 95% instantly and then crawl.

    It is sized by the pending set, and it stays out of redirected logs: at 84k
    draws an unthrottled bar writes 84k lines into the log file.
    """
    from vc_uq.generate import _NullBar, _progress

    # pytest captures stderr, so isatty() is False -> auto means silent.
    assert isinstance(_progress(100, "d", cfg), _NullBar)
    assert isinstance(_progress(0, "d", cfg.with_overrides(
        ["generation.progress=always"])), _NullBar), "nothing pending, no bar"
    assert isinstance(_progress(100, "d", cfg.with_overrides(
        ["generation.progress=never"])), _NullBar)

    bar = _progress(100, "d", cfg.with_overrides(["generation.progress=always"]))
    assert not isinstance(bar, _NullBar)
    assert bar.total == 100
    bar.close()


def test_shards_are_disjoint_exhaustive_and_order_independent():
    from vc_uq.generate import shard_questions

    qs = pd.DataFrame({"q_id": [f"q{i:03d}" for i in range(37)]})
    shards = [shard_questions(qs, (i, 3)) for i in range(3)]
    ids = [set(sh["q_id"]) for sh in shards]

    assert set().union(*ids) == set(qs["q_id"])
    assert sum(len(i) for i in ids) == len(qs), "shards overlap"
    # Two workers may build the frame in different orders; they must still agree
    # on who owns what, or a question is generated twice and another never.
    shuffled = qs.sample(frac=1.0, random_state=0)
    for i in range(3):
        assert set(shard_questions(shuffled, (i, 3))["q_id"]) == ids[i]

    assert list(shard_questions(qs, None)["q_id"]) == list(qs["q_id"])
    with pytest.raises(ValueError, match="shard must be"):
        shard_questions(qs, (3, 3))


def test_concurrent_shards_do_not_lose_each_others_draws(cfg):
    """Two workers append to one cache; the lock is what keeps both.

    append_raw is read-modify-write, so an unlocked second writer would read a
    snapshot from before the first one's rows landed and overwrite them. This
    drives the two shards from separate PROCESSES, because the lock is an flock
    and a single-process test would not exercise it.
    """
    import subprocess
    import sys as _sys
    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    c = cfg.with_overrides(["generation.checkpoint_every=3"])
    store = Store(c)
    qs = build_questions(c)
    n_q = len(qs)

    script = (
        "import sys;"
        "from vc_uq.config import load_config;"
        "from vc_uq.datasets import build_questions;"
        "from vc_uq.generate import run_phase1;"
        "from vc_uq.store import Store;"
        "c = load_config(overrides=["
        f"'run.data_root={c.data['run']['data_root']}',"
        f"'run.results_root={c.data['run']['results_root']}',"
        "'run.name=test','model.backend=mock','embedding.backend=mock',"
        "'nli.backend=mock','generation.n_max=4','generation.r_pre=2',"
        "'generation.checkpoint_every=3',"
        "'dataset.triviaqa.n_questions=30','dataset.fabricated.n_questions=10']);"
        "s = Store(c, run_id='shardtest');"
        "run_phase1(c, s, build_questions(c), shard=(int(sys.argv[1]), 2))"
    )
    procs = [subprocess.Popen([_sys.executable, "-c", script, str(i)])
             for i in range(2)]
    for pr in procs:
        assert pr.wait() == 0, "a shard worker failed"

    answers = store.read_answers()
    cls = answers[answers["draw_set"] == "classify"]
    assert set(cls["q_id"]) == set(qs["q_id"]), \
        "a shard's draws were overwritten by the other shard's write"
    assert len(cls) == n_q * 4


def test_staged_generation_refuses_a_split_that_holds_nothing(cfg):
    from vc_uq.datasets import build_questions
    from vc_uq.generate import run_phase1
    from vc_uq.store import Store

    c = _two_pass_cfg(cfg)
    with pytest.raises(ValueError, match="no questions in split"):
        run_phase1(c, Store(c), build_questions(c), splits=["nonesuch"])


def test_step4_reads_the_classification_pass_and_6_7_reads_the_downstream_pass(cfg):
    """Which frame reaches which analysis is the whole guarantee.

    Recorded rather than inferred from the numbers: the failure mode is that
    6.7 is handed the draws that defined its own subset, and that shows up as a
    spectacular result (observed = 1.0 in every U bin) rather than as an error.
    """
    from vc_uq import pipeline, survival
    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    c = _two_pass_cfg(cfg)
    store = Store(c)
    state = pipeline.PipelineState(cfg=c, store=store)
    state.questions = build_questions(c)
    state = pipeline.step3_generate(state)
    state = pipeline.step2_gate(state, simulate_labels=True)

    seen = {"partition": [], "product": []}
    real_partition, real_product = survival.partition_U, survival.product_rule_curve

    def spy_partition(answers, **kw):
        out = real_partition(answers, **kw)
        seen["partition"].append(answers)
        return out

    def spy_product(answers, cfg_, **kw):
        seen["product"].append(answers)
        return real_product(answers, cfg_, **kw)

    survival.partition_U, survival.product_rule_curve = spy_partition, spy_product
    try:
        state = pipeline.step4_survival(state)
    finally:
        survival.partition_U, survival.product_rule_curve = real_partition, real_product

    assert seen["product"], "6.7 never ran"
    for frame in seen["product"]:
        assert set(frame["draw_set"]) == {"downstream"}, \
            "6.7 must never see the draws that decided U"
        assert set(frame["split"]) == {"eval"}

    # And U itself was decided on the other pass, for every question.
    part = real_partition(state.answers)
    assert set(part["q_id"]) == set(state.questions["q_id"])
    assert state.questions["in_U"].notna().all(), "no question falls back to censored"


def test_u_membership_is_falsifiable_by_the_draws_that_measure_it(cfg):
    """The 6.7 U-curve must be able to come out below 1.0.

    A question can be never-correct in the classification pass and still get a
    correct answer downstream. Under the old single-pass partition that was
    impossible by construction -- "none of the first k was correct" held for
    every member of the subset at every k -- so the observed frequency read 1.0
    whatever VC had claimed, and the resulting 1e16 ratio was arithmetic, not a
    finding.
    """
    import pandas as pd

    from vc_uq.survival import partition_U, product_rule_curve

    rows = []
    for i in range(4):
        rows.append({"q_id": "q0", "dataset": "d", "split": "eval",
                     "draw_set": "classify", "draw_idx": i,
                     "correct": False, "vc_post": 0.8})
    for i in range(4):
        rows.append({"q_id": "q0", "dataset": "d", "split": "eval",
                     "draw_set": "downstream", "draw_idx": i,
                     "correct": i == 2, "vc_post": 0.8})
    df = pd.DataFrame(rows)

    assert bool(partition_U(df)["in_U"].iloc[0]), \
        "never correct in the classification pass, so it is in U"

    curve = product_rule_curve(df[df["draw_set"] == "downstream"],
                               cfg.with_overrides(["phase3.product_rule.k_values=[2,3]"]),
                               subset="U")
    obs = dict(zip(curve["k"], curve["observed"]))
    assert obs[2] == 1.0, "the first two downstream draws really were both wrong"
    assert obs[3] == 0.0, "the third was right -- measured, not assumed"


def test_the_product_rule_curve_reports_what_it_dropped(cfg):
    """A question dropped for an unparsed VC is not a random question.

    The draws that fail to state a confidence are plausibly the ones the model
    is least sure of, so the denominator has to travel with the bin frequency.
    """
    import pandas as pd

    from vc_uq.survival import product_rule_curve

    rows = []
    for q, vc in (("keep", 0.8), ("unparsed", None)):
        for i in range(3):
            rows.append({"q_id": q, "dataset": "d", "split": "eval",
                         "draw_set": "downstream", "draw_idx": i,
                         "correct": False,
                         "vc_post": vc if not (q == "unparsed" and i == 1) else None})
    rows.append({"q_id": "short", "dataset": "d", "split": "eval",
                 "draw_set": "downstream", "draw_idx": 0, "correct": False,
                 "vc_post": 0.5})
    df = pd.DataFrame(rows)

    curve = product_rule_curve(df, cfg.with_overrides(
        ["phase3.product_rule.k_values=[3]"]), subset="all")
    row = curve.iloc[0]
    assert int(row["n_questions_binned"]) == 1
    assert int(row["n_dropped_unparsed_vc"]) == 1
    assert int(row["n_dropped_short_of_k"]) == 1
    assert row["frac_dropped_unparsed_vc"] == pytest.approx(1 / 3)


def test_the_two_passes_are_clustered_and_anchored_separately(cfg):
    """f, H_sem and the medoid describe ONE set of draws.

    Pooling the passes would compute them over 2*N_MAX draws that no caller ever
    observes together, and would let the classification pass shift the diversity
    of the pass being certified.
    """
    import pandas as pd

    from vc_uq.cluster import cluster_frame
    from vc_uq.judge import anchor_group_keys

    rows = ([{"q_id": "q0", "draw_set": "classify", "draw_idx": i, "answer": a}
             for i, a in enumerate(["Lisbon", "Lisbon", "Lisbon", "Lisbon"])]
            + [{"q_id": "q0", "draw_set": "downstream", "draw_idx": i, "answer": a}
               for i, a in enumerate(["Madrid", "Porto", "Rome", "Oslo"])])
    out = cluster_frame(cfg, pd.DataFrame(rows), nli=_PairJudge({}))

    cls = out[out["draw_set"] == "classify"]
    dwn = out[out["draw_set"] == "downstream"]
    assert (cls["f"] == 1.0).all(), "the classify pass is unanimous on its own"
    assert dwn["f"].tolist() == pytest.approx([0.25] * 4), \
        "the downstream pass is four distinct answers on its own"

    assert anchor_group_keys(pd.DataFrame(rows)) == ["q_id", "draw_set"]
    assert anchor_group_keys(pd.DataFrame({"q_id": ["a"]})) == ["q_id"]


def test_hand_labels_identify_one_answer_not_two(cfg):
    """(q_id, draw_idx) stopped being a key when the second pass arrived."""
    import pandas as pd

    from vc_uq.phase0 import LABEL_KEY, simulate_human_labels

    assert LABEL_KEY == ["q_id", "draw_set", "draw_idx"]
    answers = pd.DataFrame([
        {"q_id": "q0", "draw_set": "classify", "draw_idx": 0, "correct_oracle": True},
        {"q_id": "q0", "draw_set": "downstream", "draw_idx": 0, "correct_oracle": False},
    ])
    sheet = pd.DataFrame([{"q_id": "q0", "draw_set": "downstream", "draw_idx": 0}])
    out = simulate_human_labels(sheet, answers, error_rate=0.0)
    assert list(out["correct_human"]) == [False], \
        "the label must follow the pass it was written for"


def test_pitfall_reports_a_missing_or_shared_second_pass(cfg):
    import pandas as pd

    from vc_uq.pitfalls import run_checks

    name = "U is decided on draws that are never certified on"

    def frame(sets):
        return pd.DataFrame([{"q_id": "q0", "split": "eval", "draw_set": s,
                              "draw_idx": i, "seed": seed}
                             for s, i, seed in sets])

    clean = frame([("classify", 0, 1), ("downstream", 0, 2)])
    assert next(c for c in run_checks(cfg, answers=clean).checks
                if c.name == name).passed is True

    # Same seed on both sides: the passes are the same draws under two labels.
    shared = frame([("classify", 0, 1), ("downstream", 0, 1)])
    bad = next(c for c in run_checks(cfg, answers=shared).checks if c.name == name)
    assert bad.passed is False and bad.severity == "fatal"

    # Second pass missing entirely, while eval questions exist.
    only_one = frame([("classify", 0, 1)])
    bad2 = next(c for c in run_checks(cfg, answers=only_one).checks if c.name == name)
    assert bad2.passed is False


def test_wilson_interval_always_brackets_the_estimate():
    """At p = 0 or 1 the algebra cancels to the bound and float64 overshoots.

    It reaches matplotlib as a negative error bar and aborts the figure several
    phases after the number itself was fine.
    """
    from vc_uq.stats import wilson_interval

    for n in (1, 3, 7, 20, 100):
        for k in (0, n):
            lo, hi = wilson_interval(k, n)
            p = k / n
            assert 0.0 <= lo <= p <= hi <= 1.0, (k, n, lo, hi)


def _state_ready_for_survival(cfg):
    """A pipeline state generated and gated, one step short of Phase 3."""
    from vc_uq import pipeline
    from vc_uq.datasets import build_questions
    from vc_uq.store import Store

    c = _two_pass_cfg(cfg)
    state = pipeline.PipelineState(cfg=c, store=Store(c))
    state.questions = build_questions(c)
    state = pipeline.step3_generate(state)
    return pipeline.step2_gate(state, simulate_labels=True)


def test_a_question_with_no_classification_draws_stops_the_run(cfg):
    """The fallback this replaced was the whole bug.

    `in_U` used to be filled from the question's own `censored` flag whenever
    the partition did not cover it -- which was 70% of questions, since splits
    are by q_id. That fallback reads the downstream draws to decide the
    condition those same draws are then measured under. Missing classification
    draws must stop the run, not be quietly imputed.
    """
    from vc_uq import pipeline

    state = _state_ready_for_survival(cfg)
    victim = state.answers["q_id"].iloc[0]
    state.answers = state.answers[
        ~((state.answers["q_id"] == victim)
          & (state.answers["draw_set"] == "classify"))]

    with pytest.raises(ValueError, match="classification draws"):
        pipeline.step4_survival(state)


def test_two_passes_sharing_a_draw_stop_the_run(cfg):
    """assert_disjoint_draws runs on every run, not as documentation.

    If the passes ever coincide, nothing raises on its own: the 6.7 U-curve
    simply reports an observed frequency of 1.0 in every bin, which reads as a
    spectacular result and is only the subset's definition restated.
    """
    import pandas as pd

    from vc_uq import pipeline

    state = _state_ready_for_survival(cfg)
    cls = state.answers[state.answers["draw_set"] == "classify"]
    # The same draw, relabelled as the other pass: identical q_id/draw_idx/seed.
    forged = cls.head(3).copy()
    forged["draw_set"] = "downstream"
    state.answers = pd.concat(
        [state.answers[state.answers["draw_set"] == "classify"], forged],
        ignore_index=True)

    with pytest.raises(ValueError, match="both the classification pass"):
        pipeline.step4_survival(state)


def test_censoring_and_U_are_the_same_question_answered_once(cfg):
    """`censored` and `in_U` must be decided by the SAME frame.

    Both mean "no correct answer in the classification pass", so they are the
    same column and phase3 reports beta and n_in_U as commensurable numbers. The
    old bug was that they were not: question_stats ran on every draw while the
    partition ran on the classify SPLIT, which left 70% of questions unlabelled
    and let `in_U` be filled from `censored` -- that is, from the question's own
    downstream draws, the ones 6.7 then measured it against.

    Holding this identity is what makes that fallback unreachable rather than
    merely absent: when the partition does not cover a question, neither does
    question_stats, so there is nothing to fall back to and the run stops.
    """
    import json as _json

    from vc_uq import pipeline

    state = pipeline.step4_survival(_state_ready_for_survival(cfg))
    q = state.questions

    assert q["in_U"].notna().all() and q["censored"].notna().all()
    assert list(q["in_U"].astype(bool)) == list(q["censored"].astype(bool)), \
        "in_U and censored disagree, so one of them is reading the wrong pass"

    summary = _json.loads((state.store.phase_dir("survival") / "summary.json")
                          .read_text(encoding="utf-8"))
    assert summary["draw_set_for_U"] == "classify"
    assert summary["n_in_U"] == int(q["censored"].astype(bool).sum()), \
        "beta and n_in_U are reported side by side; they must describe one thing"


# --------------------------------------------------------------------------
# Answer normalisation and equivalence overrides
# --------------------------------------------------------------------------

def test_normalisation_collapses_the_formatting_the_criterion_should_ignore():
    """The observed failure class, pinned case by case.

    Every pair here was scored WRONG by cosine at tau_star = 0.37 on a real run,
    and bidirectional NLI agreed on all of them -- so this is not something a
    change of criterion fixes.
    """
    from vc_uq.judge import normalise_answer as n

    for a, b in [("Seven", "7"), ("Four", "4"), ("Twenty", "20"),
                 ("The Wash", "Wash"), ("St. Martin's", "St Martins"),
                 ("A Fury", "a fury"), ("Music Man", "The Music Man"),
                 # num2words supplies both the cardinal and the spoken-year
                 # form; a hand-listed table had neither.
                 ("twenty-one", "21"), ("Twenty One", "21"),
                 ("nineteen sixty-nine", "1969"),
                 ("One Thousand, Nine Hundred and Sixty-Nine", "1969")]:
        assert n(a) == n(b), f"{a!r} and {b!r} are the same answer"

    # ...and it must not collapse answers that genuinely differ.
    assert n("Paris") != n("London")
    assert n("Joe Orton") != n("Orton"), \
        "token-level differences are the equivalence rule's job, not normalisation's"


def test_numerals_match_the_whole_string_only():
    """The rejected design, pinned so it cannot come back.

    Parsing arbitrary text for numbers (word2number) accepts 8 of the 4,323
    distinct answers on the real corpus and is WRONG on 5 of them. Collapsing a
    film title to a digit would merge unrelated answers and make each correct
    for the other -- a false positive in the correctness criterion itself, which
    is the one place this study cannot afford one.
    """
    from vc_uq.judge import normalise_answer as n

    for title in ["Seven Samurai", "Marine One", "Million Dollar Baby",
                  "Four Weddings and a Funeral", "One Direction"]:
        assert not n(title).isdigit(), f"{title!r} was parsed as a number"
    assert n("Seven Samurai") != n("7 Samurai")


def test_normalisation_off_is_exactly_the_old_behaviour(cfg):
    """The disabled path is identity, not a near-identity.

    A normaliser that still lowercased when switched off would silently change
    every historical run's e_cos while claiming to be inert.
    """
    from vc_uq.judge import normaliser

    off = normaliser(cfg.with_overrides(["judge.normalise.enabled=false"]))
    for t in ("Seven", "The Wash", "St. Martin's", "  spaced  ", "MiXeD"):
        assert off(t) == str(t)


def test_token_subset_is_off_by_default_and_declared_when_on(cfg):
    """It rescues surnames and accepts incomplete titles. Both, or neither."""
    from vc_uq.judge import equivalence_mask, token_subset_equivalent

    assert token_subset_equivalent("Orton", "Joe Orton")
    assert token_subset_equivalent("Jones", "James Jones")
    # The price of the rule above, pinned so it cannot be forgotten.
    assert token_subset_equivalent("Bridge", "Bridge Over Troubled Water")
    assert not token_subset_equivalent("Paris", "London")

    answers, refs = ["Orton"], ["Joe Orton"]
    assert not equivalence_mask(cfg, answers, refs).any(), \
        "token_subset must default to off: it trades false negatives for false positives"
    on = cfg.with_overrides(["judge.equivalence.token_subset=true"])
    assert equivalence_mask(on, answers, refs).all()


def test_tau_is_selected_against_the_predicate_the_phases_apply(cfg):
    """`tau_sweep` and `apply_tau` must be one rule.

    With an equivalence override enabled, a sweep on `e_cos <= tau` alone would
    choose tau_star to optimise a rule no downstream phase uses -- and nothing
    would raise, because both halves are individually sensible.
    """
    from vc_uq import phase0
    from vc_uq.judge import apply_tau

    on = cfg.with_overrides(["judge.equivalence.token_subset=true"])
    # One labelled pair the human called correct that the DISTANCE calls wrong,
    # and that only the override rescues.
    labelled = pd.DataFrame({
        "q_id": ["q0", "q1"], "draw_set": ["classify"] * 2, "draw_idx": [0, 1],
        "e_cos": [0.9, 0.01], "answer_equivalent": [True, False],
        "correct_human": [True, True],
    })
    sweep = phase0.tau_sweep(labelled, on, "e_cos")
    at_low_tau = sweep[np.isclose(sweep["tau"], 0.10)].iloc[0]
    assert at_low_tau["accuracy"] == 1.0, \
        "the sweep ignored the override, so tau_star optimises the wrong rule"

    scored = apply_tau(labelled, 0.10)
    assert list(scored["correct_cos"].astype(bool)) == [True, True], \
        "apply_tau and tau_sweep disagree about what correct means"


def test_equivalence_column_is_written_even_when_the_rule_is_off(cfg):
    """Present and all-False, so the parquet records which criterion ran."""
    from vc_uq.judge import equivalence_mask

    mask = equivalence_mask(cfg, ["Orton", "Paris"], ["Joe Orton", "Paris"])
    assert mask.dtype == bool and len(mask) == 2
    assert not mask.any()


def test_normalisation_makes_spelling_variants_score_identically(cfg):
    """End to end through the embedding space, not just the string helper.

    The space is keyed on normalised text and looked up through the same
    function, so a caller passing raw answers cannot desynchronise the two.
    """
    from vc_uq.judge import build_embedding_space

    space = build_embedding_space(cfg, ["Seven", "7", "Paris"])
    assert np.allclose(space.get(["Seven"]), space.get(["7"])), \
        "'Seven' and '7' must land on one vector once normalisation is on"
    assert not np.allclose(space.get(["Seven"]), space.get(["Paris"]))


def test_accent_folding_unifies_spellings_without_erasing_scripts():
    """Diacritics are a spelling difference, not an answer difference.

    "Le Carre" written with the acute scored e_cos = 1.058 against the gold
    "John Le Carre" -- further apart than two unrelated strings -- and the
    question was counted as never-correct on that basis.
    """
    from vc_uq.judge import normalise_answer as n

    for a, b in [("La Bohème", "La Boheme"),
                 ("Ilie Năstase", "Ilie Nastase"),
                 ("Nadia Comăneci", "Nadia Comaneci"),
                 ("Le Carré", "Le Carre"),
                 # NFKD leaves these alone; the explicit ligature map catches them.
                 ("Le Cœur", "Le Coeur"), ("Straße", "Strasse")]:
        assert n(a) == n(b), f"{a!r} and {b!r} differ only in spelling"

    # Folding must not be "delete the non-ASCII range". Two unrelated answers
    # that both collapsed to "" would be identical, which is a false positive in
    # the correctness criterion -- the one error this study cannot absorb.
    assert n("Пари́ж") == "париж"
    assert n("東京") == "東京"
    assert n("Москва") != n("東京")
    assert n("Paris") != n("London")


def test_accent_folding_can_be_switched_off(cfg):
    from vc_uq.judge import normaliser

    off = normaliser(cfg.with_overrides(["judge.normalise.accents=false"]))
    assert off("La Bohème") != off("La Boheme")
    on = normaliser(cfg)
    assert on("La Bohème") == on("La Boheme")
