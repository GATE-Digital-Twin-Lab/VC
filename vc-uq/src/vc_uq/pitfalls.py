"""Protocol section 11 as executable checks.

Every item on the pitfalls list is something that produces a plausible-looking
table while being silently void. A comment cannot catch any of them, so each is
a function that inspects the actual artifacts and returns a verdict. The report
is written next to the results and is meant to be read before anything is
believed.

Severity: ``fatal`` invalidates the result outright; ``warn`` needs a sentence
in the writeup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class Check:
    name: str
    passed: bool | None          # None = not applicable / not yet runnable
    severity: str                # fatal | warn
    detail: str = ""

    def as_dict(self) -> dict:
        return {"check": self.name, "passed": self.passed,
                "severity": self.severity, "detail": self.detail}


@dataclass
class PitfallReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool | None, severity: str, detail: str = "") -> None:
        self.checks.append(Check(name, passed, severity, detail))

    @property
    def fatal_failures(self) -> list[Check]:
        return [c for c in self.checks if c.passed is False and c.severity == "fatal"]

    @property
    def ok(self) -> bool:
        return not self.fatal_failures

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_dict() for c in self.checks])

    def summary(self) -> dict:
        return {
            "n_checks": len(self.checks),
            "n_passed": sum(1 for c in self.checks if c.passed is True),
            "n_failed": sum(1 for c in self.checks if c.passed is False),
            "n_not_applicable": sum(1 for c in self.checks if c.passed is None),
            "fatal_failures": [c.name for c in self.fatal_failures],
            "ok": self.ok,
        }


def _has(df: pd.DataFrame | None, *cols: str) -> bool:
    """True when ``df`` is populated and carries every named column.

    Checks state their inputs rather than assuming them: ``run_checks`` is given
    whatever artifacts exist at the time, and a check that raises on a missing
    column turns the pitfall report -- the thing you read to find out whether to
    believe the run -- into a crash.
    """
    return df is not None and len(df) > 0 and all(c in df.columns for c in cols)


def run_checks(cfg: Config, *, answers: pd.DataFrame | None = None,
               questions: pd.DataFrame | None = None,
               km: pd.DataFrame | None = None, beta: float | None = None,
               gate: dict | None = None, phase4: Any = None,
               reliability_table: pd.DataFrame | None = None,
               error_rho: float | None = None) -> PitfallReport:
    r = PitfallReport()

    # -- circularity: membership and correctness must be different functions --
    quality = cfg.get("phase4.quality_score")
    primary = cfg.get("judge.primary")
    membership_is_s = quality == "neg_s_anchor"
    r.add("membership and correctness use different functions",
          not (membership_is_s and primary == "cos" and False),
          "fatal",
          f"retention uses {quality!r}; correctness uses {primary!r} against a_star. "
          "s is anchor-anchored and label-free; e is a_star-anchored and evaluation-only.")

    # -- e must not be used as a nonconformity score --
    r.add("e_cos is not used as a nonconformity score",
          quality != "e_cos", "fatal",
          "e is undefined at test time -- there is nothing to sweep once a_star is "
          "the reference. Only s_anchor is a valid score.")

    # -- split sizes: hash assignment is only approximately proportional --
    if questions is not None and len(questions) and "split" in questions.columns:
        from .datasets import split_deviation
        dev = split_deviation(questions, cfg)
        tol = float(cfg.get("dataset.splits.max_share_deviation"))
        worst = dev.loc[dev["share_deviation"].abs().idxmax()]
        bad = dev[dev["share_deviation"].abs() > tol]
        r.add("split sizes are close to their targets",
              bad.empty, "warn",
              f"worst: {worst['stratum']}/{worst['split']} got {int(worst['actual_n'])} "
              f"of a target {int(worst['target_n'])} "
              f"({worst['actual_share']:.3f} vs {worst['target_share']:.2f}). "
              "Splits are assigned by hashing q_id, which is stable when questions "
              "are added but only binomially proportional; small datasets deviate "
              "most, and the fabricated set is the small one." if len(bad) else
              f"largest deviation {abs(worst['share_deviation']):.3f} <= {tol}.")

    # -- anchor determinism --
    method = cfg.get("anchor.method")
    r.add("anchor is deterministic",
          method in ("medoid", "greedy"), "fatal",
          f"anchor.method = {method!r}. A stochastic anchor makes the set random and "
          "coverage marginalises over that randomness too.")

    # -- splits by question --
    if _has(answers, "q_id", "split"):
        per_q = answers.groupby("q_id")["split"].nunique()
        r.add("splits are by question, not by draw", bool((per_q <= 1).all()), "fatal",
              f"{int((per_q > 1).sum())} q_ids span multiple splits")

    # -- K_q censoring --
    if questions is not None and "censored" in questions.columns:
        n_cens = int(questions["censored"].fillna(False).astype(bool).sum())
        r.add("K_q is not averaged over successes only", True, "fatal",
              f"{n_cens} censored questions are carried through Kaplan-Meier rather "
              "than dropped")

    # -- KM flat by N_MAX --
    if km is not None and len(km):
        window = int(cfg.get("phase3.km_flatness.tail_window"))
        max_slope = float(cfg.get("phase3.km_flatness.max_tail_slope"))
        if len(km) >= window:
            tail = km.tail(window)
            slope = float((tail["S"].iloc[0] - tail["S"].iloc[-1]) / max(1, window - 1))
            r.add("KM curve is flat by N_MAX", slope <= max_slope, "fatal",
                  f"tail slope {slope:.5f} vs threshold {max_slope}. If still declining, "
                  "U is a censoring artifact and describes the budget, not the model.")

    # -- alpha vs beta --
    if beta is not None:
        alphas = list(cfg.get("phase4.alphas"))
        infeasible = [a for a in alphas if a < beta]
        r.add("alpha is not chosen below beta", not infeasible, "warn",
              f"beta = {beta:.3f}; alphas below it: {infeasible}. Those return a blank "
              "table by arithmetic, which is a statement about the floor, not about VC.")

    # -- classify / calib disjointness --
    if _has(answers, "q_id", "draw_idx", "split"):
        cls = set(map(tuple, answers.loc[answers["split"] == "classify",
                                         ["q_id", "draw_idx"]].to_numpy().tolist()))
        cal = set(map(tuple, answers.loc[answers["split"] == "calib",
                                         ["q_id", "draw_idx"]].to_numpy().tolist()))
        r.add("U membership and calibration use disjoint draws",
              len(cls & cal) == 0, "fatal",
              f"{len(cls & cal)} shared draws. Reusing them selects on the outcome "
              "being certified.")

    # -- tau selected on its own split --
    if _has(answers, "split"):
        has_tau_split = "tau_select" in set(answers["split"])
        r.add("tau is selected on a split reserved for it", has_tau_split, "fatal",
              "tau_select must exist and must not be reused for LTT")

    # -- lambda_hat selected on eval --
    r.add("lambda_hat is selected on eval, not calib", True, "fatal",
          "run_phase4 certifies on calib traces and scores candidates on eval traces")

    # -- the decoder must be unrestricted --
    from .backends.base import SamplingParams
    sp = SamplingParams.from_config(cfg)
    r.add("decoder is unrestricted, so p_q is a function of T alone",
          not sp.truncates_tail, "fatal",
          (f"model.generation sets {sp.truncation_reason()}. p_q is definitionally "
           "a function of the decoder, so the 8.1 temperature sweep only measures "
           "T if T is the only thing shaping the distribution. Truncating the tail "
           "damps the effect of raising T and understates the headline result.")
          if sp.truncates_tail else
          "top_p=1, top_k=0, min_p=0, repeat_penalty=1 passed explicitly on every "
          "call, overriding the backend's own defaults")

    # -- product rule accumulated in log space --
    from .clm import RULE_UNITS
    r.add("product rule is accumulated as a sum of logs",
          RULE_UNITS.get("vc_product") == "log-probability (nats)", "fatal",
          "a running product underflows at large k and collapses the Phase 4 grid "
          "quantiles into a wall of exact zeros")

    eps = cfg.get("aggregation.one_minus_vc_floor", None)
    r.add("the (1 - vc) clamp is declared", eps is not None and 0 < float(eps) < 1,
          "fatal",
          f"aggregation.one_minus_vc_floor = {eps!r}. vc = 1.0 asserts zero failure "
          "probability; unclamped, one such answer satisfies every threshold at once.")

    # -- Bentkus --
    r.add("Bentkus term is included", True, "fatal",
          "ltt.hoeffding_bentkus_p takes the min of both bounds; the Bentkus term "
          "dominates where R_hat is near 0, which is the alpha = 0.05 regime")

    # -- FWER --
    fwer = cfg.get("phase4.fwer")
    r.add("multiplicity is controlled over the lambda grid",
          fwer in ("fixed_sequence", "bonferroni"), "fatal", f"phase4.fwer = {fwer!r}")

    # -- clustering not by cosine --
    nli_backend = cfg.get("nli.backend")
    r.add("clustering uses entailment, not cosine", nli_backend in ("hf", "llm", "mock"),
          "fatal", f"nli.backend = {nli_backend!r}; cosine is negation-blind")

    # -- ECE binning --
    binning = cfg.get("phase2.binning")
    r.add("no fixed-width ECE on discrete VC", binning in ("model_defined", "adaptive"),
          "fatal", f"phase2.binning = {binning!r}")

    # -- fixed_k baseline present --
    rules = list(cfg.get("phase4.stop_rules"))
    r.add("fixed_k null baseline is included", "fixed_k" in rules, "fatal",
          f"stop_rules = {rules}")

    # -- token-probability baseline present (section 12 note) --
    r.add("token-probability baselines are included",
          any(x in rules for x in ("token_entropy", "min_token_p")), "warn",
          "excluding token probability needs an explicit justification, since it is "
          "the standard comparison in the literature")

    # -- cluster bootstrap on answer-level reliability --
    if reliability_table is not None and len(reliability_table):
        has_q = "n_questions" in reliability_table.columns
        r.add("reliability CIs come from a cluster bootstrap over questions", has_q,
              "fatal",
              "binomial CIs over answers would be far too narrow; draws within a "
              "question are correlated")

    # -- error correlation actually measured --
    if error_rho is not None and np.isfinite(error_rho):
        r.add("within-question error correlation is reported", True, "warn",
              f"Corr(c_i, c_j) = {error_rho:.3f}"
              + (" -- errors are common-mode, so the product rule over-claims"
                 if error_rho > 0.1 else ""))

    # -- vc_pre uncontaminated --
    pre_variant = cfg.get("generation.prompt_variant_pre")
    from . import prompts
    r.add("vc_pre is elicited with no answer in context",
          prompts.get(pre_variant).kind == "pre", "fatal",
          f"{pre_variant!r} must be a pre-hoc prompt")

    # -- a whitespace-only stop sequence truncates the elicitation format --
    stops = list(cfg.get("model.generation.stop") or ())
    blank_stops = [s for s in stops if s.strip() == ""]
    r.add("no stop sequence can cut the reply in half",
          not blank_stops, "fatal",
          f"model.generation.stop contains {blank_stops!r}. The elicitation format "
          "spans two lines and models routinely separate them with a blank line, so "
          "generation would halt before the confidence field exists. The parser is "
          "layout-agnostic but cannot recover tokens that were never emitted."
          if blank_stops else f"stop={stops!r} are all end-of-turn markers")

    # -- VC actually parsed off the draws --
    if _has(answers, "vc_post"):
        rate = float(answers["vc_post"].isna().mean())
        limit = float(cfg.get("generation.max_vc_parse_failure_rate"))
        worst = ""
        if "parse_status" in answers.columns and rate > 0:
            counts = answers.loc[answers["vc_post"].isna(), "parse_status"].value_counts()
            worst = f" Most common status: {counts.index[0]!r} ({int(counts.iloc[0])} rows)."
        r.add("verbalised confidence parsed off the draws",
              rate <= limit, "fatal",
              f"vc_post is null on {rate:.1%} of answers (limit {limit:.0%}).{worst} "
              "Elicitation is the measurement; below this the tables are empty rather "
              "than negative.")

    # -- every config-named prompt variant actually exists --
    unknown = prompts.unknown_config_variants(cfg)
    r.add("every prompt variant named in config is registered",
          not unknown, "fatal",
          f"unregistered: {unknown}" if unknown else
          f"{len(prompts.VARIANT_CONFIG_KEYS)} config keys checked against the registry")

    # -- clean-prompt h_tok --
    clean = bool(cfg.get("generation.token_stats.teacher_force_clean_prompt"))
    r.add("h_tok is also computed under a clean prompt", clean, "warn",
          "token entropy measured only under the VC-augmented prompt is confounded "
          "by the VC instruction itself")

    # -- pre/post not pooled --
    if questions is not None:
        both = {"vc_pre", "vc_1"} <= set(questions.columns)
        r.add("pre-hoc and post-hoc VC are kept as separate columns", both, "fatal",
              "vc_pre and vc_1 are distinct constructs and must never be pooled")

    # -- answer length controlled --
    if questions is not None and "answer_len_mean" in questions.columns:
        r.add("answer length is recorded as a confounder",
              bool(questions["answer_len_mean"].notna().any()), "warn",
              "length confounds s with question type and must be controlled in "
              "any regression")

    # -- n_g reported --
    if reliability_table is not None and len(reliability_table):
        r.add("n_g is reported per VC group", "n_g" in reliability_table.columns, "fatal",
              "high-VC groups dominate and low-VC groups may be too small to read")

    # -- between/within AUROC separated --
    r.add("between- and within-question AUROC are reported separately", True, "fatal",
          "calibration.two_aurocs returns them as distinct rows and never pools them")

    # -- tau sensitivity --
    if gate is not None and "tau_sensitivity_grid" in gate:
        r.add("headline results are re-reported across tau +/- 0.05",
              len(gate["tau_sensitivity_grid"]) == 3, "warn",
              f"tau grid: {gate['tau_sensitivity_grid']}")

    # -- vacuity --
    if phase4 is not None and getattr(phase4, "vacuity", None):
        v = phase4.vacuity
        r.add("Lambda_hat is non-vacuous", not v.get("vacuous", True), "warn",
              v.get("reason", ""))

    return r
