"""Phase 4 -- conformal language modelling: retention, stopping, risk.

A configuration is ``lambda = (lambda_qual, lambda_div, lambda_stop)``:

  retain  a_i if quality(a_i) >= lambda_qual AND min_j d(a_i, a_j) >= lambda_div
          over the already-retained a_j
  stop    when the rule's confidence statistic crosses lambda_stop

Diversity gating is not optional. Without it a mode-collapsed model fills the
set with forty paraphrases of one wrong answer and the set size stops carrying
any information.

The loss is ``1[no admissible answer in C(q)]``, bounded in [0,1] and monotone
in ``lambda_stop`` -- stopping later can only help. That monotonicity is what
licenses fixed-sequence testing in ``ltt.py``.

Risk is held constant by construction across rules; efficiency is the free
variable. That is what makes the comparison meaningful: same guarantee,
different price. ``fixed_k`` is mandatory, because if a VC rule cannot beat
"always draw exactly k", VC carries no usable information about when to stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .backends.base import pairwise_cosine_distance
from .config import Config
from .ltt import feasibility, run_ltt
from .stats import ONE_MINUS_VC_FLOOR, log_one_minus_vc, log_to_prob

#: rule -> (statistic name, comparison). "le" stops when the statistic falls to
#: lambda_stop; "ge" stops when it rises to it.
RULE_DIRECTION: dict[str, str] = {
    "vc_product": "le",
    "vc_max": "ge",
    "vc_first": "ge",
    "vc_prehoc": "ge",
    "token_entropy": "le",
    "min_token_p": "ge",
    "fixed_k": "ge",
}
# The sample-diversity signals (largest-cluster share, H_sem) are deliberately
# absent. A diversity statistic over one draw is not a low value, it is not a
# value: one draw is one cluster, so H_sem = 0 and share = 1, which are exactly
# the values a "stop when converged" rule treats as convergence. They would fire
# on the first draw at every grid point, making both rules a duplicate of
# fixed_k = 1 under a name that implies otherwise. Requiring a minimum draw
# count would change the efficiency they report, so they are reported
# descriptively instead -- see cluster.question_diversity and the 6.6 2x2.

#: Rules that commit to a budget without adapting to what the draws show.
NON_ADAPTIVE = ("vc_first", "vc_prehoc", "fixed_k")

#: Units of each rule's statistic, and therefore of its lambda_stop. Reported in
#: the headline table so a threshold in nats is never read as a probability.
RULE_UNITS: dict[str, str] = {
    "vc_product": "log-probability (nats)",
    "vc_max": "probability",
    "vc_first": "probability",
    "vc_prehoc": "probability",
    "token_entropy": "nats per token",
    "min_token_p": "probability",
    "fixed_k": "draws",
}


@dataclass
class QuestionTrace:
    """Everything Phase 4 needs about one question, precomputed once."""

    q_id: str
    dataset: str
    correct: np.ndarray        # [N] admissibility of each draw
    quality: np.ndarray        # [N] retention score
    vc_post: np.ndarray        # [N]
    h_tok: np.ndarray          # [N]
    min_token_p: np.ndarray    # [N]
    cluster_id: np.ndarray     # [N]
    dist: np.ndarray           # [N, N] pairwise semantic distance for duplicate gating
    vc_pre: float
    p_hat: float
    in_U: bool

    @property
    def n(self) -> int:
        return int(len(self.correct))


def build_traces(cfg: Config, answers: pd.DataFrame, questions: pd.DataFrame,
                 embedder=None) -> list[QuestionTrace]:
    from .judge import build_embedding_space

    quality_col = cfg.get("phase4.quality_score")
    if quality_col not in ("vc_post", "neg_s_anchor"):
        raise ValueError(f"phase4.quality_score must be vc_post or neg_s_anchor, "
                         f"got {quality_col!r}")

    space = build_embedding_space(cfg, list(answers["answer"].astype(str)),
                                  embedder=embedder)
    q_idx = questions.set_index("q_id")
    traces = []
    for q_id, g in answers.sort_values("draw_idx").groupby("q_id"):
        if q_id not in q_idx.index:
            continue
        qrow = q_idx.loc[q_id]
        texts = list(g["answer"].astype(str))
        emb = space.get(texts)
        vc = g["vc_post"].astype(float).fillna(0.0).to_numpy()
        if quality_col == "vc_post":
            quality = vc
        else:
            # Retention by agreement with the anchor biases toward mode collapse,
            # so this path is an ablation, never the default.
            quality = -g["s_anchor"].astype(float).fillna(0.0).to_numpy()
        traces.append(QuestionTrace(
            q_id=str(q_id), dataset=str(g["dataset"].iloc[0]),
            correct=g["correct"].astype(bool).to_numpy(),
            quality=quality, vc_post=vc,
            h_tok=g["h_tok_mean"].astype(float).fillna(np.inf).to_numpy(),
            min_token_p=g["min_token_p"].astype(float).fillna(0.0).to_numpy(),
            cluster_id=g["cluster_id"].astype("Int32").fillna(-1).astype(int).to_numpy(),
            dist=pairwise_cosine_distance(emb),
            vc_pre=float(qrow.get("vc_pre", np.nan)),
            p_hat=float(qrow.get("p_hat", np.nan)),
            in_U=bool(qrow.get("in_U", False)),
        ))
    return traces


# --------------------------------------------------------------------------
# Retention + running statistics
# --------------------------------------------------------------------------

def retention_path(trace: QuestionTrace, lambda_qual: float,
                   lambda_div: float) -> list[int]:
    """Indices retained, in draw order, under quality and duplicate gating."""
    retained: list[int] = []
    for i in range(trace.n):
        if trace.quality[i] < lambda_qual:
            continue
        if retained and float(np.min(trace.dist[i, retained])) < lambda_div:
            continue           # a paraphrase of something already held
        retained.append(i)
    return retained


def running_statistic(trace: QuestionTrace, rule: str,
                      retained: Sequence[int],
                      one_minus_vc_floor: float = ONE_MINUS_VC_FLOOR) -> np.ndarray:
    """The rule's statistic after each DRAW (not after each retention).

    Indexed by draw so the stopping decision is made with exactly the evidence
    available at that point in the sampling loop.
    """
    n = trace.n
    out = np.empty(n, dtype=np.float64)
    retained_set = list(retained)

    if rule == "vc_product":
        # Accumulated as a SUM OF LOGS: Lambda_k = sum_i log(1 - vc_i), over
        # diverse RETAINED answers only -- repeating a duplicate would otherwise
        # add the same claim in again and manufacture confidence.
        #
        # log is strictly increasing, so `prod <= t` and `sum log <= log t` cut
        # the sample space identically and the decision boundary is unchanged.
        # What changes is that the statistic stays representable at any k, and
        # that a stated confidence of exactly 1.0 contributes a bounded
        # log(floor) rather than annihilating the product (see stats.log_one_minus_vc).
        log_terms = log_one_minus_vc(trace.vc_post, one_minus_vc_floor)
        total = 0.0
        ptr = 0
        for k in range(n):
            while ptr < len(retained_set) and retained_set[ptr] <= k:
                total += log_terms[retained_set[ptr]]
                ptr += 1
            out[k] = total
    elif rule == "vc_max":
        out = np.maximum.accumulate(trace.vc_post)
    elif rule == "vc_first":
        out[:] = trace.vc_post[0] if n else np.nan
    elif rule == "vc_prehoc":
        out[:] = trace.vc_pre
    elif rule == "token_entropy":
        out = np.minimum.accumulate(trace.h_tok)
    elif rule == "min_token_p":
        out = np.maximum.accumulate(trace.min_token_p)
    elif rule == "fixed_k":
        out = np.arange(1, n + 1, dtype=np.float64)
    else:
        raise ValueError(f"unknown stop rule {rule!r}")
    return out


def stop_index(stat: np.ndarray, lambda_stop: float, direction: str) -> int:
    """First draw index (0-based) at which the rule stops; n-1 if it never does."""
    crossed = stat <= lambda_stop if direction == "le" else stat >= lambda_stop
    hits = np.flatnonzero(crossed)
    return int(hits[0]) if hits.size else int(len(stat) - 1)


def evaluate(trace: QuestionTrace, rule: str, lambda_qual: float, lambda_div: float,
             lambda_stop: float,
             one_minus_vc_floor: float = ONE_MINUS_VC_FLOOR) -> dict:
    retained = retention_path(trace, lambda_qual, lambda_div)
    stat = running_statistic(trace, rule, retained, one_minus_vc_floor)
    k = stop_index(stat, lambda_stop, RULE_DIRECTION[rule])
    kept = [i for i in retained if i <= k]
    # If gating retained nothing, the model still emitted answers; the set is
    # empty and the loss is 1. Silently falling back to the raw draws would
    # hide the cost of an over-strict lambda_qual.
    has_admissible = bool(np.any(trace.correct[kept])) if kept else False
    return {"q_id": trace.q_id, "dataset": trace.dataset, "draws": k + 1,
            "set_size": len(kept), "loss": 0.0 if has_admissible else 1.0,
            "p_hat": trace.p_hat, "in_U": trace.in_U}


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------

def _linspace(spec: dict) -> np.ndarray:
    return np.linspace(float(spec["start"]), float(spec["stop"]), int(spec["num"]))


def rule_stop_grid(rule: str, traces: Sequence[QuestionTrace], num: int,
                   n_max: int,
                   one_minus_vc_floor: float = ONE_MINUS_VC_FLOOR) -> np.ndarray:
    """Per-rule stopping grid.

    The statistics live on incompatible scales -- a product in [0,1], an entropy
    in nats, a draw count in integers -- so a single [0,1] grid would be
    meaningful for some rules and vacuous for others. Grids are therefore
    quantiles of the statistic as actually observed, which makes the rules
    comparable at equal grid resolution rather than at equal numeric range.

    For ``vc_product`` the statistic is a sum of logs, and taking quantiles of
    it rather than of the raw product is a real gain in resolution, not a
    relabelling: the product is astronomically skewed, so linear-space quantiles
    pile almost every grid point up against 0 and leave the interesting
    thresholds untested. In log space the same ten points spread evenly across
    orders of magnitude.

    Derived from CALIBRATION traces only; using eval here would leak.
    """
    if rule == "fixed_k":
        return np.arange(1, n_max + 1, dtype=np.float64)
    vals = []
    for t in traces:
        stat = running_statistic(t, rule, retention_path(t, -np.inf, -np.inf),
                                 one_minus_vc_floor)
        vals.append(stat[np.isfinite(stat)])
    if not vals:
        return np.linspace(0, 1, num)
    pool = np.concatenate(vals)
    if pool.size == 0:
        return np.linspace(0, 1, num)
    grid = np.unique(np.quantile(pool, np.linspace(0, 1, num)))
    return grid


def build_grid(cfg: Config, rule: str, traces: Sequence[QuestionTrace],
               n_max: int) -> list[tuple[float, float, float]]:
    qual = _linspace(cfg.section("phase4.grid.lambda_qual"))
    div = _linspace(cfg.section("phase4.grid.lambda_div"))
    stop = rule_stop_grid(rule, traces, int(cfg.get("phase4.grid.lambda_stop.num")),
                          n_max, float(cfg.get("aggregation.one_minus_vc_floor")))
    return [(float(a), float(b), float(c)) for a in qual for b in div for c in stop]


# --------------------------------------------------------------------------
# Risk over the grid
# --------------------------------------------------------------------------

def risk_table(traces: Sequence[QuestionTrace], rule: str,
               grid: Sequence[tuple[float, float, float]],
               one_minus_vc_floor: float = ONE_MINUS_VC_FLOOR) -> pd.DataFrame:
    """Empirical risk for every lambda, plus the efficiency columns."""
    rows = []
    # Retention depends only on (qual, div), so each path is computed once and
    # reused across every lambda_stop -- the grid is 10x10x10 and this is the
    # difference between minutes and hours.
    cache: dict[tuple[float, float], list[tuple[QuestionTrace, list[int], np.ndarray]]] = {}
    for (q, d, s) in grid:
        key = (q, d)
        if key not in cache:
            entries = []
            for t in traces:
                ret = retention_path(t, q, d)
                entries.append((t, ret,
                                running_statistic(t, rule, ret, one_minus_vc_floor)))
            cache[key] = entries
        losses, draws, sizes = [], [], []
        direction = RULE_DIRECTION[rule]
        for t, ret, stat in cache[key]:
            k = stop_index(stat, s, direction)
            kept = [i for i in ret if i <= k]
            losses.append(0.0 if (kept and bool(np.any(t.correct[kept]))) else 1.0)
            draws.append(k + 1)
            sizes.append(len(kept))
        n = len(losses)
        rows.append({"rule": rule, "lambda_qual": q, "lambda_div": d,
                     "lambda_stop": s, "lambda_stop_units": RULE_UNITS[rule],
                     "risk_hat": float(np.mean(losses)),
                     "mean_draws": float(np.mean(draws)),
                     "mean_set_size": float(np.mean(sizes)), "n": n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The headline table
# --------------------------------------------------------------------------

@dataclass
class PhaseFourResult:
    headline: pd.DataFrame
    per_rule: dict[str, pd.DataFrame] = field(default_factory=dict)
    feasibility: dict = field(default_factory=dict)
    vacuity: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def run_phase4(cfg: Config, calib_traces: Sequence[QuestionTrace],
               eval_traces: Sequence[QuestionTrace], *, beta: float,
               alpha: float, n_max: int) -> PhaseFourResult:
    """Certify on calib, SELECT on eval.

    Selecting ``lambda_hat`` on the calibration set would reuse the same data
    that produced the guarantee; the LTT guarantee holds for any selection from
    ``Lambda_hat``, but the efficiency numbers used to choose must come from
    held-out data.
    """
    delta = float(cfg.get("phase4.delta"))
    method = cfg.get("phase4.fwer")
    floor = float(cfg.get("aggregation.one_minus_vc_floor"))
    feas = feasibility(beta, alpha)
    result = PhaseFourResult(headline=pd.DataFrame(), feasibility=feas)

    if not feas["feasible"]:
        result.notes.append(feas["reason"])
        return result

    rows = []
    for rule in cfg.get("phase4.stop_rules"):
        grid = build_grid(cfg, rule, calib_traces, n_max)
        risks = risk_table(calib_traces, rule, grid, floor)
        ltt = run_ltt(risks, alpha=alpha, delta=delta, method=method)
        result.per_rule[rule] = ltt.table

        if ltt.is_empty:
            rows.append({"rule": rule, "certified": False, "n_valid": 0,
                         "lambda_stop_units": RULE_UNITS[rule],
                         "mean_draws": np.nan, "mean_set_size": np.nan,
                         "realized_risk": np.nan, "lambda_stop_hat": np.nan,
                         **discount_factor(rule, alpha, np.nan)})
            continue

        # Cheapest certified configuration, scored on EVAL.
        best, best_eval = None, None
        for _, cand in ltt.valid.iterrows():
            ev = risk_table(eval_traces, rule,
                            [(cand["lambda_qual"], cand["lambda_div"],
                              cand["lambda_stop"])], floor).iloc[0]
            if best_eval is None or ev["mean_draws"] < best_eval["mean_draws"]:
                best, best_eval = cand, ev

        lam_stop = float(best["lambda_stop"])
        rows.append({
            "rule": rule, "certified": True, "n_valid": int(len(ltt.valid)),
            "lambda_qual_hat": float(best["lambda_qual"]),
            "lambda_div_hat": float(best["lambda_div"]),
            "lambda_stop_hat": lam_stop,
            "lambda_stop_units": RULE_UNITS[rule],
            "mean_draws": float(best_eval["mean_draws"]),
            "mean_set_size": float(best_eval["mean_set_size"]),
            "realized_risk": float(best_eval["risk_hat"]),
            "calib_risk": float(best["risk_hat"]),
            **discount_factor(rule, alpha, lam_stop),
        })

    result.headline = pd.DataFrame(rows)
    result.vacuity = vacuity_check(result.headline, cfg, n_max)
    if result.vacuity["vacuous"]:
        result.notes.append(result.vacuity["reason"])
    return result


def discount_factor(rule: str, alpha: float, lambda_stop_hat: float) -> dict:
    """``c_discount = alpha / lambda_stop_hat``, for vc_product.

    ``lambda_stop`` for this rule is now a LOG-probability, so it is
    exponentiated back before dividing -- the discount is defined against the
    probability the model claims, and reporting ``alpha / (a threshold in nats)``
    would be a units error that still produces a plausible-looking number.

    The quotient is formed in log space and only then exponentiated:
    ``log c = log alpha - lambda_stop_hat``. A certified threshold far below
    ``alpha`` makes the linear form overflow to inf while the log form stays
    finite and readable, which is exactly the regime this measurement is for.

    If VC were truthful the certified threshold would land at alpha itself and
    the discount would be 1. Expect it well above: the model's self-reported
    failure probability has to reach alpha/c before 1-alpha coverage is actually
    achieved. Stability of this number across datasets and strata is the
    load-bearing check -- roughly constant means a fixed, fixable
    miscalibration; varying by an order of magnitude means no fixed correction
    exists.
    """
    nan = float("nan")
    if rule != "vc_product" or not np.isfinite(lambda_stop_hat):
        return {"c_discount": nan, "log10_c_discount": nan,
                "lambda_stop_hat_as_prob": nan}
    log_c = float(np.log(alpha) - lambda_stop_hat)
    return {
        "c_discount": log_to_prob(log_c),
        # Primary reporting form: survives a discount of 1e300 without becoming inf.
        "log10_c_discount": log_c / float(np.log(10.0)),
        "lambda_stop_hat_as_prob": log_to_prob(lambda_stop_hat),
    }


def vacuity_check(headline: pd.DataFrame, cfg: Config, n_max: int) -> dict:
    """Does any certified configuration actually stop early?

    If Lambda_hat only contains configurations that draw all N_MAX samples and
    retain everything, every method ties at the ceiling and the table says
    nothing at all.
    """
    if headline.empty or "mean_draws" not in headline.columns:
        return {"vacuous": True, "reason": "no certified configurations at any rule"}
    frac = float(cfg.get("phase4.vacuity.max_early_stop_fraction_of_n_max"))
    ceiling = frac * n_max
    early = headline["mean_draws"] < ceiling
    vacuous = not bool(early.any())
    return {
        "vacuous": vacuous,
        "n_rules_stopping_early": int(early.sum()),
        "ceiling_draws": ceiling,
        "reason": "" if not vacuous else (
            f"every certified configuration draws at least {ceiling:.1f} of {n_max} "
            "samples: all rules tie at the ceiling and the comparison is empty. "
            "Raise alpha, or report that no rule buys anything at this risk level."),
    }


def stratified_discount(cfg: Config, traces: Sequence[QuestionTrace], rule: str,
                        alpha: float, lambda_hat: tuple[float, float, float],
                        by: str = "dataset") -> pd.DataFrame:
    """Recompute realized risk and the discount within strata.

    The interesting subset is not the easy questions but those with low but
    nonzero p_hat, where a correct answer exists yet is rare -- that is where a
    stopping rule earns its keep, and where a flat VC costs real compute.
    """
    floor = float(cfg.get("aggregation.one_minus_vc_floor"))
    groups: dict[str, list[QuestionTrace]] = {}
    if by == "dataset":
        for t in traces:
            groups.setdefault(t.dataset, []).append(t)
    elif by == "p_hat":
        edges = list(cfg.get("phase4.strata.p_hat_edges"))
        for t in traces:
            if not np.isfinite(t.p_hat):
                continue
            i = int(np.clip(np.searchsorted(edges, t.p_hat, side="right") - 1,
                            0, len(edges) - 2))
            groups.setdefault(f"p_hat[{edges[i]},{edges[i + 1]})", []).append(t)
    else:
        raise ValueError(f"unknown stratifier {by!r}")

    rows = []
    for name, sub in sorted(groups.items()):
        if not sub:
            continue
        r = risk_table(sub, rule, [lambda_hat], floor).iloc[0]
        rows.append({"stratum_by": by, "stratum": name, "n": int(r["n"]),
                     "risk_hat": float(r["risk_hat"]),
                     "mean_draws": float(r["mean_draws"]),
                     "mean_set_size": float(r["mean_set_size"]),
                     **discount_factor(rule, alpha, lambda_hat[2])})
    return pd.DataFrame(rows)
