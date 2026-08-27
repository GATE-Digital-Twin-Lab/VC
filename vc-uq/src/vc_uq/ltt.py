"""Learn Then Test: p-values and multiplicity control.

Split conformal does not apply to this problem. The risk is set-valued ("the
set contains at least one admissible answer"), the set is built adaptively as
draws arrive, and ``lambda`` is three-dimensional -- there is no single score
whose quantile solves it. LTT tests each configuration and keeps the ones that
survive.

Plain Hoeffding is not enough. At ``alpha = 0.05`` the empirical risk sits near
zero, and that is precisely where the Bentkus term dominates; using Hoeffding
alone would throw away most of the power exactly in the regime of interest.
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

    @property
    def is_empty(self) -> bool:
        return len(self.valid) == 0


def fixed_sequence_test(risks: pd.DataFrame, *, alpha: float, delta: float,
                        order_col: str = "lambda_stop",
                        ascending: bool = False,
                        group_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Walk from most conservative to least; halt at the first non-rejection.

    Valid with NO multiplicity penalty, and the validity comes precisely from
    the monotonicity of R in ``lambda_stop``: stopping later can only lower the
    risk. Bonferroni over a 1000-point grid would be valid too but would spend
    most of the power on configurations nobody would select.
    """
    out = []
    keys = list(group_cols)
    groups = risks.groupby(keys) if keys else [((), risks)]
    for _, g in groups:
        g = g.sort_values(order_col, ascending=ascending).copy()
        g["p_value"] = [hoeffding_bentkus_p(r, int(n), alpha)
                        for r, n in zip(g["risk_hat"], g["n"])]
        rejected = []
        halted = False
        for p in g["p_value"]:
            if halted or not (p < delta):
                rejected.append(False)
                halted = True
            else:
                rejected.append(True)
        g["rejected"] = rejected
        g["stopped_here"] = (~g["rejected"]).cumsum() == 1
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else risks.assign(
        p_value=[], rejected=[])


def bonferroni_test(risks: pd.DataFrame, *, alpha: float, delta: float) -> pd.DataFrame:
    m = max(1, len(risks))
    g = risks.copy()
    g["p_value"] = [hoeffding_bentkus_p(r, int(n), alpha)
                    for r, n in zip(g["risk_hat"], g["n"])]
    g["rejected"] = g["p_value"] < (delta / m)
    return g


def run_ltt(risks: pd.DataFrame, *, alpha: float, delta: float, method: str,
            order_col: str = "lambda_stop",
            group_cols: tuple[str, ...] = ("lambda_qual", "lambda_div")) -> LTTResult:
    """``Lambda_hat = {lambda : p_lambda < delta}``.

    The guarantee is ``P(R(lambda_hat) <= alpha) >= 1 - delta`` for ANY selection
    from ``Lambda_hat`` -- which is what licenses choosing the cheapest
    configuration afterwards.
    """
    if risks.empty:
        return LTTResult(risks, risks, alpha, delta, method, 0)
    if method == "fixed_sequence":
        table = fixed_sequence_test(risks, alpha=alpha, delta=delta,
                                    order_col=order_col, group_cols=group_cols)
    elif method == "bonferroni":
        table = bonferroni_test(risks, alpha=alpha, delta=delta)
    else:
        raise ValueError(f"unknown fwer method {method!r}")
    valid = table[table["rejected"]].copy()
    n = int(risks["n"].max()) if "n" in risks.columns else 0
    return LTTResult(table=table, valid=valid, alpha=alpha, delta=delta,
                     method=method, n=n)


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
