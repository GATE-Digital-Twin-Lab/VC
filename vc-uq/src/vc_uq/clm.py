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

**Almost every component of lambda is a number in [0, 1].** Each score is
rescaled to the unit interval before it is thresholded -- a length-normalised
likelihood, a cosine distance halved, a geometric-mean token probability, a
fraction of the budget -- so one search space serves them all and a certified
threshold is readable without a units table.

``vc_product`` is the deliberate exception. Its ``lambda_stop`` is a
log-probability in nats, on the same scale as the ``Lambda_k`` it is compared
against and the same scale 6.7 reports the product-rule gap on. The claim
compounds, so the thresholds that matter span orders of magnitude: as
probabilities they bunch against 0 and stop being readable, while -5 nats and
-20 nats are two legible numbers.

**Retention and stopping must use different signals.** ``quality`` defaults to
the length-normalised likelihood ``exp(mean_t log p_t)`` -- the geometric mean
per-token probability -- which is what conformal language modelling admits on.
VC appears only in the stopping rules,
which is the claim under test. Were VC to gate retention as well, every baseline
-- token entropy, min token probability, even ``fixed_k`` -- would be scored on a
set VC had already filtered, no VC-free arm would remain, and a difference in
draws could not be attributed to the stopping rule.

Draws whose statistics are missing are DROPPED, never imputed. A parse failure
is not a confidence of zero, and filling one in would put a number VC never
produced into the statistic the study is about. The count is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .backends.base import pairwise_cosine_distance
from .config import Config
from .ltt import feasibility, order_is_ascending, run_ltt
from .stats import ONE_MINUS_VC_FLOOR, log_one_minus_vc, log_to_prob

#: rule -> (statistic name, comparison). "le" stops when the statistic falls to
#: lambda_stop; "ge" stops when it rises to it.
RULE_DIRECTION: dict[str, str] = {
    "vc_product": "le",
    "vc_max": "ge",
    "token_entropy": "ge",
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

#: Every statistic Phase 4 reads off a draw. A draw missing any of them cannot
#: be scored by all rules on the same footing, so it is dropped rather than
#: filled: an unparsed VC is not a confidence of 0.0, and an answer that emitted
#: no tokens has no entropy rather than an entropy of zero.
REQUIRED_STATS = ("vc_post", "logp_mean", "h_tok_mean", "min_token_p")

#: Retention scores that are legitimate to admit on, ALL bounded in [0, 1] so
#: ``lambda_qual`` is a plain unit-interval search. ``vc_post`` is deliberately
#: not the default -- see the module docstring and the pitfall check.
QUALITY_SCORES = ("likelihood", "anchor_sim", "vc_post")

#: What a rule's ``lambda_stop`` MEANS. Every one is a number in [0, 1] except
#: ``vc_product``, whose threshold is in nats; this column is what the headline
#: table needs in order to be read, and the reason a threshold in nats is never
#: mistaken for a probability.
RULE_UNITS: dict[str, str] = {
    "vc_product": "log-probability (nats)",
    "vc_max": "probability",
    "token_entropy": "geometric mean token probability",
    "fixed_k": "|C(q)| as a fraction of N_MAX",
}

#: Rules whose running statistic AND threshold both live in log space. A
#: running product of (1 - vc) underflows float64 by around k = 160 at
#: vc = 0.99, and every question past that point becomes an indistinguishable
#: 0.0; thresholding in nats keeps both sides representable and legible at any k.
LOG_SPACE_RULES = ("vc_product",)


@dataclass
class QuestionTrace:
    """Everything Phase 4 needs about one question, precomputed once."""

    q_id: str
    dataset: str
    correct: np.ndarray        # [N] admissibility of each draw
    quality: np.ndarray        # [N] retention score -- NOT vc_post by default
    vc_post: np.ndarray        # [N]
    logp_mean: np.ndarray      # [N] length-normalised log-likelihood
    h_tok: np.ndarray          # [N]
    min_token_p: np.ndarray    # [N]
    cluster_id: np.ndarray     # [N]
    dist: np.ndarray           # [N, N] pairwise cosine distance / 2, so in [0, 1]
    vc_pre: float
    p_hat: float
    in_U: bool
    #: draws discarded for a missing statistic. `draws` counts positions in the
    #: RETAINED sequence, so mean_draws understates the sampling actually spent
    #: by about this fraction; it travels with the traces so the headline can
    #: say so.
    n_dropped: int = 0

    @property
    def n(self) -> int:
        return int(len(self.correct))


def build_traces(cfg: Config, answers: pd.DataFrame, questions: pd.DataFrame,
                 embedder=None) -> list[QuestionTrace]:
    from .judge import build_embedding_space

    quality_col = cfg.get("phase4.quality_score")
    if quality_col not in QUALITY_SCORES:
        raise ValueError(f"phase4.quality_score must be one of {QUALITY_SCORES}, "
                         f"got {quality_col!r}")

    space = build_embedding_space(cfg, list(answers["answer"].astype(str)),
                                  embedder=embedder)
    q_idx = questions.set_index("q_id")
    needed = [c for c in REQUIRED_STATS if c in answers.columns]
    if quality_col == "anchor_sim" and "s_anchor" in answers.columns:
        needed = needed + ["s_anchor"]
    traces = []
    for q_id, g in answers.sort_values("draw_idx").groupby("q_id"):
        if q_id not in q_idx.index:
            continue
        # Drop, never fill. A draw with an unparseable VC has no confidence, and
        # writing 0.0 in its place would feed a number the model never produced
        # into the statistic under test. Dropping costs the baselines that draw
        # too, which is the price of every rule seeing one identical sequence.
        usable = g[needed].notna().all(axis=1) if needed else pd.Series(True, index=g.index)
        n_dropped = int((~usable).sum())
        g = g[usable]
        if g.empty:
            continue
        qrow = q_idx.loc[q_id]
        texts = list(g["answer"].astype(str))
        emb = space.get(texts)
        vc = g["vc_post"].astype(float).to_numpy()
        logp = g["logp_mean"].astype(float).to_numpy()
        if quality_col == "likelihood":
            # p(y|x)^(1/T): the geometric mean per-token probability. Exponentiated
            # rather than left in nats so the score is a probability in [0, 1] and
            # lambda_qual reads directly -- "retain answers the model gave at least
            # this average per-token probability". The length normalisation is what
            # stops the threshold from being a proxy for answer length.
            quality = np.exp(logp)
        elif quality_col == "vc_post":
            quality = vc
        else:
            # cos(anchor, a_i) = 1 - s_anchor, clipped into [0, 1] so it shares
            # lambda_qual's scale. Retention by agreement with the anchor biases
            # toward mode collapse, so this path is an ablation, never the default.
            quality = np.clip(1.0 - g["s_anchor"].astype(float).to_numpy(), 0.0, 1.0)
        traces.append(QuestionTrace(
            q_id=str(q_id), dataset=str(g["dataset"].iloc[0]),
            correct=g["correct"].astype(bool).to_numpy(),
            quality=quality, vc_post=vc, logp_mean=logp,
            h_tok=g["h_tok_mean"].astype(float).to_numpy(),
            min_token_p=g["min_token_p"].astype(float).to_numpy(),
            cluster_id=g["cluster_id"].astype("Int32").fillna(-1).astype(int).to_numpy(),
            # Divided by 2, not clipped: cosine distance runs over [0, 2]
            # (0 identical, 1 orthogonal, 2 opposite) and lambda_div is searched
            # over [0, 1], so the whole range is scaled onto the unit interval.
            # Clipping would have collapsed every anti-correlated pair -- 18% of
            # them on the corpus this was measured on -- into one value.
            dist=pairwise_cosine_distance(emb) / 2.0,
            vc_pre=float(qrow.get("vc_pre", np.nan)),
            p_hat=float(qrow.get("p_hat", np.nan)),
            in_U=bool(qrow.get("in_U", False)),
            n_dropped=n_dropped,
        ))
    return traces


def trace_audit(traces: Sequence[QuestionTrace]) -> dict:
    """What dropping missing statistics cost, so mean_draws can be read right."""
    kept = int(sum(t.n for t in traces))
    dropped = int(sum(t.n_dropped for t in traces))
    return {
        "n_questions": len(traces),
        "n_draws_used": kept,
        "n_draws_dropped_missing_stats": dropped,
        "frac_draws_dropped": (dropped / (kept + dropped)) if (kept + dropped) else 0.0,
        "note": ("draws with a missing statistic are dropped, not imputed; `draws` "
                 "counts positions in the retained sequence, so mean_draws "
                 "understates sampling actually spent by about this fraction"),
    }


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


def _accumulate_over_retained(n: int, retained: Sequence[int],
                              values: np.ndarray, op: str,
                              init: float) -> np.ndarray:
    """Fold ``values`` over the retained answers seen so far, indexed by DRAW.

    Two indices are in play and confusing them is the bug this function exists to
    prevent. ``retained`` names positions in the draw sequence, and the result is
    indexed by draw -- so entry ``k`` is the fold over every answer that had been
    both drawn and admitted by the time draw ``k`` came back. That is exactly the
    evidence an online caller holds at that point in the sampling loop.

    ``init`` is what the statistic reads before anything has been admitted:
    ``-inf`` for a max and ``0`` for a sum or a count, so that a rule cannot fire
    on an empty set.
    """
    out = np.empty(n, dtype=np.float64)
    cur = float(init)
    ptr = 0
    order = list(retained)
    for k in range(n):
        while ptr < len(order) and order[ptr] <= k:
            v = float(values[order[ptr]]) if values is not None else 1.0
            cur = max(cur, v) if op == "max" else cur + v
            ptr += 1
        out[k] = cur
    return out


def running_statistic(trace: QuestionTrace, rule: str,
                      retained: Sequence[int],
                      one_minus_vc_floor: float = ONE_MINUS_VC_FLOOR) -> np.ndarray:
    """The rule's statistic after each DRAW, over the answers actually RETAINED.

    Every rule reads the set, not the raw draw stream. An answer the quality or
    diversity gate rejected is not in ``C(q)``, so it cannot be the evidence that
    ends sampling -- stopping because of an answer you then throw away would
    certify a set that never contained it.

    This is also what makes ``lambda_qual`` and ``lambda_div`` mean anything in
    the search. When the statistic ran over all draws, gating changed the set but
    never the draw count, so the two retention dimensions could not affect
    efficiency and were never selected. Now a stricter gate admits less, the
    statistic advances more slowly, and the rule draws for longer: a real
    trade-off between how many samples you spend and how clean a set you keep.
    """
    n = trace.n
    if n == 0:
        return np.empty(0, dtype=np.float64)

    if rule == "vc_product":
        # Accumulated as a SUM OF LOGS: Lambda_k = sum_i log(1 - vc_i) over
        # diverse retained answers -- repeating a duplicate would otherwise add
        # the same claim in again and manufacture confidence.
        #
        # log is strictly increasing, so `prod <= t` and `sum log <= log t` cut
        # the sample space identically and the decision boundary is unchanged.
        # What changes is that the statistic stays representable at any k, and
        # that a stated confidence of exactly 1.0 contributes a bounded
        # log(floor) rather than annihilating the product (see stats.log_one_minus_vc).
        return _accumulate_over_retained(
            n, retained, log_one_minus_vc(trace.vc_post, one_minus_vc_floor),
            op="sum", init=0.0)
    if rule == "vc_max":
        # The most confident answer IN THE SET so far.
        return _accumulate_over_retained(n, retained, trace.vc_post,
                                         op="max", init=-np.inf)
    if rule == "token_entropy":
        # exp(-H), the geometric mean token probability: the same ordering as the
        # entropy it comes from, but bounded in (0, 1] so lambda_stop is a
        # probability. "Lowest-entropy answer in the set so far" is therefore a
        # running max rather than a running min.
        return _accumulate_over_retained(n, retained, np.exp(-trace.h_tok),
                                         op="max", init=-np.inf)
    if rule == "fixed_k":
        # |C(q)|, not the draw index. The null baseline is "keep sampling until
        # the set holds k admissible-looking answers", so gating costs it draws
        # exactly as it costs every other rule -- which is what makes it a fair
        # floor rather than a rule playing a different game.
        return _accumulate_over_retained(n, retained, None, op="sum", init=0.0)
    raise ValueError(f"unknown stop rule {rule!r}")


def stop_threshold(rule: str, lambda_stop: float, n_max: int) -> float:
    """Map ``lambda_stop`` onto the scale its statistic lives on.

    The search space is the unit interval for every rule but ``vc_product``,
    whose threshold is already in nats; the statistics are on their own scales.
    Mapping the THRESHOLD rather than rescaling the statistic is the rule here:
    exponentiating ``Lambda_k`` to meet a probability threshold would reproduce
    the underflow wall of 6.7, where every question past large k becomes an
    indistinguishable 0.0.

    Every map is monotone increasing in ``lambda_stop``, which is what lets
    :func:`ltt.order_is_ascending` read the certification order off
    :data:`RULE_DIRECTION` in lambda space just as it did in statistic space.
    """
    lam = float(lambda_stop)
    if rule in LOG_SPACE_RULES:
        # Already in nats, on the statistic's own scale. Nothing to map.
        return lam
    if rule == "fixed_k":
        # lambda is |C(q)| as a fraction of the budget, so the threshold is a
        # count of retained answers.
        return lam * float(n_max)
    return lam


def stop_index(stat: np.ndarray, lambda_stop: float, direction: str) -> int:
    """First draw index (0-based) at which the rule stops; n-1 if it never does."""
    crossed = stat <= lambda_stop if direction == "le" else stat >= lambda_stop
    hits = np.flatnonzero(crossed)
    return int(hits[0]) if hits.size else int(len(stat) - 1)


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------

def quality_grid(num: int) -> np.ndarray:
    """Retention thresholds, over the unit interval.

    Every score in :data:`QUALITY_SCORES` is bounded in [0, 1] -- a
    length-normalised likelihood, a cosine similarity, a stated confidence -- so
    the threshold reads directly and is comparable across models rather than
    being a number on whatever scale the backend produced.
    """
    return np.linspace(0.0, 1.0, max(1, num))


def diversity_grid(num: int) -> np.ndarray:
    """Duplicate-gating thresholds, over the unit interval.

    Cosine distance runs over [0, 2] -- 0 identical, 1 orthogonal, 2 opposite --
    and the traces carry it divided by 2 (see :func:`build_traces`), so the
    whole range maps onto [0, 1] rather than being clipped. Clipping at 1 would
    have thrown away the anti-correlated pairs, which were 18% of them on the
    corpus this was measured on.
    """
    return np.linspace(0.0, 1.0, max(1, num))


def rule_stop_grid(rule: str, num: int, product_floor: float = 1e-6) -> np.ndarray:
    """Stopping thresholds: [0, 1] for every rule but ``vc_product``.

    ``vc_product``'s threshold is a log-probability in nats, evenly spaced from
    ``log(product_floor)`` up to 0 -- which is orders of magnitude in the claim
    it represents, because that is what a compounding product needs. Reporting it
    in nats keeps it legible (-5 and -20 rather than 6.7e-3 and 2.1e-9) and puts
    it on the same axis as the 6.7 product-rule gap.

    ``-inf`` is prepended for those rules. It means "never stop early", so it is
    the strictest setting available and the conservative end the fixed sequence
    must start from -- and unlike any finite threshold it stays reachable
    whatever ``N_MAX`` is. ``discount_factor`` returns NaN there, since a rule
    that never fires makes no claim to discount.
    """
    num = max(1, int(num))
    if rule in LOG_SPACE_RULES:
        floor = float(product_floor)
        if not (0.0 < floor < 1.0):
            raise ValueError("phase4.grid.lambda_stop.product_floor must be in (0, 1)")
        if num == 1:
            return np.array([-np.inf])
        return np.concatenate([[-np.inf],
                               np.linspace(float(np.log(floor)), 0.0, num - 1)])
    return np.linspace(0.0, 1.0, num)


def build_grid(cfg: Config, rule: str) -> list[tuple[float, float, float]]:
    """The full lambda grid. Every coordinate is in [0, 1].

    It depends on no data at all, which is the strongest position a search space
    can be in: nothing about the grid can leak from the calibration set into the
    certification.
    """
    qual = quality_grid(int(cfg.get("phase4.grid.lambda_qual.num")))
    div = diversity_grid(int(cfg.get("phase4.grid.lambda_div.num")))
    stop = rule_stop_grid(rule, int(cfg.get("phase4.grid.lambda_stop.num")),
                          float(cfg.get("phase4.grid.lambda_stop.product_floor")))
    return [(float(a), float(b), float(c)) for a in qual for b in div for c in stop]


# --------------------------------------------------------------------------
# Risk over the grid
# --------------------------------------------------------------------------

def risk_table(traces: Sequence[QuestionTrace], rule: str,
               grid: Sequence[tuple[float, float, float]],
               one_minus_vc_floor: float = ONE_MINUS_VC_FLOOR,
               n_max: int | None = None) -> pd.DataFrame:
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
        budget = n_max if n_max is not None else max((t.n for t in traces), default=1)
        # lambda_stop arrives in [0, 1] and is mapped onto the statistic's own
        # scale here, once per grid point rather than per trace.
        thresh = stop_threshold(rule, s, budget)
        for t, ret, stat in cache[key]:
            k = stop_index(stat, thresh, direction)
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
    #: rule -> the certification order and the level each family actually ran at
    ltt_meta: dict[str, dict] = field(default_factory=dict)
    #: what dropping draws with missing statistics cost, on calib and on eval
    trace_audit: dict = field(default_factory=dict)
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
    result.trace_audit = {"calib": trace_audit(calib_traces),
                          "eval": trace_audit(eval_traces),
                          "quality_score": cfg.get("phase4.quality_score")}
    dropped = result.trace_audit["calib"]["frac_draws_dropped"]
    if dropped > 0:
        result.notes.append(
            f"Phase 4 dropped {dropped:.1%} of calibration draws for a missing "
            "statistic (unparsed VC or an answer with no tokens). They are not "
            "imputed, so no invented confidence enters the comparison -- but "
            "`draws` counts positions in the retained sequence, so mean_draws "
            "understates the sampling actually spent by about that fraction.")

    if not feas["feasible"]:
        result.notes.append(feas["reason"])
        return result

    rows = []
    for rule in cfg.get("phase4.stop_rules"):
        grid = build_grid(cfg, rule)
        risks = risk_table(calib_traces, rule, grid, floor, n_max=n_max)
        # Which end of the lambda_stop grid is the conservative one is a property
        # of the rule, not of the data. A "le" rule stops when its statistic
        # FALLS to the threshold, so the strictest setting is the lowest one and
        # the sequence walks up; "ge" walks down. Get this backwards and the
        # sequence opens on its worst grid point, halts, and reports the rule as
        # uncertifiable -- which for vc_product is the study's headline result
        # arriving as a sort order.
        ascending = order_is_ascending(RULE_DIRECTION[rule])
        ltt = run_ltt(risks, alpha=alpha, delta=delta, method=method,
                      ascending=ascending)
        result.per_rule[rule] = ltt.table
        result.ltt_meta[rule] = {
            "ascending": ascending, "direction": RULE_DIRECTION[rule],
            "n_families": ltt.n_families, "delta": delta,
            "delta_family": ltt.delta_family, "n_valid": int(len(ltt.valid)),
        }
        if ltt.nonmonotone_families:
            result.notes.append(
                f"{rule}: risk is not monotone along the certification order in "
                f"{ltt.nonmonotone_families} of {ltt.n_families} families. Within a "
                "family a stricter threshold can only stop later and stopping later "
                "can only turn a loss of 1 into a 0, so this is exactly monotone in "
                "the sample -- a violation means the order or the statistic is wrong, "
                "not that the estimate is noisy.")

        if ltt.is_empty:
            rows.append({"rule": rule, "certified": False, "n_valid": 0,
                         "lambda_stop_units": RULE_UNITS[rule],
                         "n_families": ltt.n_families,
                         "delta_family": ltt.delta_family,
                         "mean_draws": np.nan, "mean_set_size": np.nan,
                         "realized_risk": np.nan, "lambda_stop_hat": np.nan,
                         **discount_factor(rule, alpha, np.nan)})
            continue

        # Cheapest certified configuration, scored on EVAL.
        best, best_eval = None, None
        for _, cand in ltt.valid.iterrows():
            ev = risk_table(eval_traces, rule,
                            [(cand["lambda_qual"], cand["lambda_div"],
                              cand["lambda_stop"])], floor, n_max=n_max).iloc[0]
            if best_eval is None or ev["mean_draws"] < best_eval["mean_draws"]:
                best, best_eval = cand, ev

        lam_stop = float(best["lambda_stop"])
        rows.append({
            "rule": rule, "certified": True, "n_valid": int(len(ltt.valid)),
            # delta_family, not delta: each (lambda_qual, lambda_div) cell is its
            # own fixed sequence, so the stated confidence is only honest once
            # the level is split across them.
            "n_families": ltt.n_families, "delta_family": ltt.delta_family,
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

    ``lambda_stop`` for this rule is a LOG-probability, so it is exponentiated
    back before dividing -- the discount is defined against the probability the
    model claims, and reporting ``alpha / (a threshold in nats)`` would be a
    units error that still produces a plausible-looking number.

    The quotient is formed in log space and only then exponentiated:
    ``log c = log alpha - lambda_stop_hat``. A certified threshold far below
    ``alpha`` makes the linear form overflow to inf while the log form stays
    finite and readable, which is exactly the regime this measurement is for.
    ``lambda_stop_hat = -inf`` ("never stop early") yields NaN: a rule that never
    fires has made no claim to discount.

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
