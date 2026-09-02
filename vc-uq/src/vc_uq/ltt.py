"""Learn Then Test: p-values and multiplicity control.

Split conformal does not apply to this problem. The risk is set-valued ("the
set contains at least one admissible answer"), the set is built adaptively as
draws arrive, and ``lambda`` is three-dimensional -- there is no single score
whose quantile solves it. LTT tests each configuration and keeps the ones that
survive.

Plain Hoeffding is not enough. At ``alpha = 0.05`` the empirical risk sits near
zero, and that is precisely where the Bentkus term dominates; using Hoeffding
alone would throw away most of the power exactly in the regime of interest.

Two things about the fixed-sequence procedure are easy to get wrong and neither
announces itself:

1. **The order must run from most conservative to least.** Fixed-sequence
   testing is valid under any PRE-SPECIFIED order, so a reversed sequence does
   not break the guarantee -- it destroys the power, silently. It halts on its
   first test and returns an empty ``Lambda_hat``, which reads as "this rule
   cannot be certified" rather than as "the grid was sorted backwards". Which
   end is conservative depends on the rule's comparison direction, so callers
   must pass ``ascending`` explicitly; there is no safe default.
2. **One sequence controls FWER at delta. G of them do not.** Running an
   independent sequence per ``(lambda_qual, lambda_div)`` cell at level delta
   gives a union bound of ``G * delta``, and G is 100 at the default grid. The
   families are Bonferroni-corrected here: fixed sequence within, ``delta / G``
   across. The correction lands on delta alone, so the ``lambda_stop``
   resolution stays free.

The order must never be derived from the observed risks. Data-dependent
ordering is exactly what fixed-sequence validity forbids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom


def h1(a: float, b: float) -> float:
    """KL between Bernoulli(a) and Bernoulli(b), in nats."""
    a = float(np.clip(a, 1e-12, 1 - 1e-12))
    b = float(np.clip(b, 1e-12, 1 - 1e-12))
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hoeffding_bentkus_p(r_hat: float, n: int, alpha: float) -> float:
    """p-value for H0: R(lambda) > alpha, from the two bounds combined.

    ``min`` of the two is valid because each is separately a valid p-value for
    the same null.
    """
    if n <= 0:
        return 1.0
    r_hat = float(np.clip(r_hat, 0.0, 1.0))
    hoeff = float(np.exp(-n * h1(min(r_hat, alpha), alpha)))
    bentkus = float(np.e * binom.cdf(np.ceil(n * r_hat), n, alpha))
    return float(min(1.0, hoeff, bentkus))


@dataclass
class LTTResult:
    table: pd.DataFrame          # every lambda with its risk, p-value and verdict
    valid: pd.DataFrame          # Lambda_hat
    alpha: float
    delta: float
    method: str
    n: int
    n_families: int = 1          # parallel sequences, Bonferroni-corrected
    delta_family: float = float("nan")   # the level each sequence actually ran at
    nonmonotone_families: int = 0        # see check_monotone_order

    @property
    def is_empty(self) -> bool:
        return len(self.valid) == 0


def order_is_ascending(direction: str) -> bool:
    """Which end of the ``lambda_stop`` grid is the conservative one.

    ``ge`` rules stop when the statistic RISES to lambda_stop, so a higher
    threshold is harder to trigger, stops later, retains more, and carries less
    risk -- the sequence starts at the top and walks down. ``le`` rules stop when
    the statistic FALLS to lambda_stop, so it is the LOWEST threshold that is
    strictest and the sequence walks up.

    This is a property of the rule, fixed before any data is seen. It is never
    inferred from the observed risks: a data-dependent order is precisely what
    fixed-sequence validity forbids.
    """
    if direction not in ("le", "ge"):
        raise ValueError(f"direction must be 'le' or 'ge', got {direction!r}")
    return direction == "le"


def check_monotone_order(risks: pd.DataFrame, *, order_col: str, ascending: bool,
                         group_cols: tuple[str, ...] = ()) -> int:
    """How many families have a risk that is NOT non-decreasing along the order.

    Within a family the retention path is fixed, so a stricter threshold can only
    stop later, and stopping later can only turn a loss of 1 into a 0. Risk is
    therefore exactly monotone in the sample, not merely in expectation -- which
    makes any violation a defect rather than noise, and most often a reversed
    ``ascending``. Reported, never acted on: reordering to fix it would make the
    order data-dependent and void the guarantee it was protecting.
    """
    keys = list(group_cols)
    groups = risks.groupby(keys) if keys else [((), risks)]
    bad = 0
    for _, g in groups:
        seq = g.sort_values(order_col, ascending=ascending)["risk_hat"]
        if not seq.is_monotonic_increasing:
            bad += 1
    return bad


def fixed_sequence_test(risks: pd.DataFrame, *, alpha: float, delta: float,
                        ascending: bool,
                        order_col: str = "lambda_stop",
                        group_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Walk each family from most conservative to least; halt at the first
    non-rejection, and split ``delta`` across the families.

    Within one sequence there is no multiplicity penalty, and that comes from
    the monotonicity of R in ``lambda_stop``: stopping later can only lower the
    risk. Across families there is one. Each ``(lambda_qual, lambda_div)`` cell
    is its own sequence, so running all of them at ``delta`` would give a union
    bound of ``G * delta`` -- 100x the stated confidence at the default grid --
    and the claim that any selection from ``Lambda_hat`` is safe would not hold.
    Bonferroni across families is the cheap fix: it costs a factor of G on delta
    only, and delta enters the p-value comparison rather than the sequence, so
    the ``lambda_stop`` resolution is still free.

    ``ascending`` has no default on purpose. Which end of the grid is
    conservative depends on the rule's direction, and a reversed sequence does
    not raise -- it halts on its first test and returns nothing, which reads as
    "this rule is not certifiable". Use :func:`order_is_ascending`.
    """
    out = []
    keys = list(group_cols)
    groups = list(risks.groupby(keys)) if keys else [((), risks)]
    n_families = max(1, len(groups))
    delta_family = delta / n_families
    for _, g in groups:
        g = g.sort_values(order_col, ascending=ascending).copy()
        g["p_value"] = [hoeffding_bentkus_p(r, int(n), alpha)
                        for r, n in zip(g["risk_hat"], g["n"])]
        rejected = []
        halted = False
        for p in g["p_value"]:
            if halted or not (p < delta_family):
                rejected.append(False)
                halted = True
            else:
                rejected.append(True)
        g["rejected"] = rejected
        g["stopped_here"] = (~g["rejected"]).cumsum() == 1
        g["n_families"] = n_families
        g["delta_family"] = delta_family
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else risks.assign(
        p_value=[], rejected=[], n_families=n_families, delta_family=delta_family)


def bonferroni_test(risks: pd.DataFrame, *, alpha: float, delta: float) -> pd.DataFrame:
    m = max(1, len(risks))
    g = risks.copy()
    g["p_value"] = [hoeffding_bentkus_p(r, int(n), alpha)
                    for r, n in zip(g["risk_hat"], g["n"])]
    g["rejected"] = g["p_value"] < (delta / m)
    return g


def run_ltt(risks: pd.DataFrame, *, alpha: float, delta: float, method: str,
            ascending: bool, order_col: str = "lambda_stop",
            group_cols: tuple[str, ...] = ("lambda_qual", "lambda_div")) -> LTTResult:
    """``Lambda_hat``, at a confidence that accounts for every test run.

    The guarantee is ``P(R(lambda) <= alpha for all lambda in Lambda_hat)
    >= 1 - delta`` -- which is what licenses choosing the cheapest configuration
    afterwards, on held-out data, without paying again.

    ``ascending`` is required, and comes from the rule's comparison direction via
    :func:`order_is_ascending`. It is not optional and it is not inferable: pass
    the wrong one and the rule silently reports as uncertifiable.
    """
    if risks.empty:
        return LTTResult(risks, risks, alpha, delta, method, 0)
    if method == "fixed_sequence":
        table = fixed_sequence_test(risks, alpha=alpha, delta=delta,
                                    ascending=ascending, order_col=order_col,
                                    group_cols=group_cols)
        n_families = int(table["n_families"].iloc[0]) if len(table) else 1
        delta_family = float(table["delta_family"].iloc[0]) if len(table) else delta
        nonmono = check_monotone_order(risks, order_col=order_col,
                                       ascending=ascending, group_cols=group_cols)
    elif method == "bonferroni":
        table = bonferroni_test(risks, alpha=alpha, delta=delta)
        n_families = 1
        delta_family = delta / max(1, len(risks))
        nonmono = 0
    else:
        raise ValueError(f"unknown fwer method {method!r}")
    valid = table[table["rejected"]].copy()
    n = int(risks["n"].max()) if "n" in risks.columns else 0
    return LTTResult(table=table, valid=valid, alpha=alpha, delta=delta,
                     method=method, n=n, n_families=n_families,
                     delta_family=delta_family, nonmonotone_families=nonmono)


def feasibility(beta: float, alpha: float) -> dict:
    """R(lambda) = beta*1 + (1-beta)*R_ans(lambda) >= beta.

    If beta > alpha then NO lambda is certifiable and Phase 4 must return a
    blank table. That is not "VC failed" -- it is "no stopping rule of any kind
    could have succeeded", and every baseline fails identically.
    """
    feasible = bool(beta <= alpha)
    return {
        "beta": float(beta), "alpha": float(alpha), "feasible": feasible,
        "reason": "" if feasible else (
            f"beta = {beta:.3f} exceeds alpha = {alpha:.3f}. The risk floor is beta, "
            "so no configuration can be certified: Lambda_hat is empty by "
            "arithmetic, not by any property of the stopping signal. Either raise "
            "alpha above beta or restrict to A (and state that you conditioned on "
            "solvability, which is unavailable at deployment)."),
    }


def min_certifiable_alpha(beta: float) -> float:
    return float(beta)
