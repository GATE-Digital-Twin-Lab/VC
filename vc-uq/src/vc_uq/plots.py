"""Figures. Matplotlib only, no styling dependencies."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402


def _new(cfg: Config):
    try:
        plt.style.use(cfg.get("plots.style"))
    except (OSError, ValueError):
        pass
    return plt.subplots(figsize=tuple(cfg.get("plots.figsize")))


def _save(fig, path: Path, cfg: Config) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=int(cfg.get("plots.dpi")))
    plt.close(fig)
    return path


def km_curve(km: pd.DataFrame, path: Path, cfg: Config, beta: float | None = None) -> Path:
    fig, ax = _new(cfg)
    ax.step(km["k"], km["S"], where="post")
    ax.fill_between(km["k"], km["lo"], km["hi"], step="post", alpha=0.2)
    if beta is not None:
        ax.axhline(beta, ls="--", lw=1)
        ax.annotate(f"beta = {beta:.3f}", (km["k"].iloc[-1], beta),
                    ha="right", va="bottom", fontsize=9)
    ax.set_xlabel("draws k")
    ax.set_ylabel("S(k) = P(no correct answer in k draws)")
    ax.set_ylim(0, 1)
    ax.set_title("Kaplan-Meier: time to first admissible answer")
    return _save(fig, path, cfg)


def hazard_curve(hz: pd.DataFrame, path: Path, cfg: Config) -> Path:
    fig, ax = _new(cfg)
    ax.errorbar(hz["k"], hz["hazard"],
                yerr=[hz["hazard"] - hz["lo"], hz["hi"] - hz["hazard"]],
                marker="o", capsize=3)
    ax.set_xlabel("draw k")
    ax.set_ylabel("h(k) = P(correct at k | all previous wrong)")
    ax.set_title("Hazard: flat means conditionally i.i.d., declining means heterogeneity")
    ax.set_ylim(bottom=0)
    return _save(fig, path, cfg)


def reliability_diagram(rel: pd.DataFrame, path: Path, cfg: Config,
                        title: str = "Reliability (answer-level, model-defined bins)") -> Path:
    fig, ax = _new(cfg)
    sub = rel[rel["reportable"]] if "reportable" in rel.columns else rel
    sub = sub.sort_values("v_g")
    # VC piles up at the top of the scale and p_hat_g can be exactly 0, so the
    # groups that matter most sit ON the edge of the unit square. Limits of
    # exactly [0, 1] draw them half-clipped behind the axes; pad past the edge
    # and mark the edge itself instead.
    pad = 0.04
    for edge in (0.0, 1.0):
        ax.axhline(edge, color="black", lw=0.8, zorder=1)
        ax.axvline(edge, color="black", lw=0.8, zorder=1)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", zorder=1)
    ax.errorbar(sub["v_g"], sub["p_hat_g"],
                yerr=[sub["p_hat_g"] - sub["lo"], sub["hi"] - sub["p_hat_g"]],
                marker="o", ls="none", capsize=3, zorder=3)
    # Group sizes go in a table, not beside each point: neighbouring groups can
    # be 0.01 apart (0.99 vs 1.0), where point labels overprint each other or
    # land on the wrong marker. Top-left is empty unless a group is
    # underconfident.
    rows = [f"{'v_g':>5} {'n_g':>7} {'p_hat':>5}"]
    rows += [f"{r['v_g']:>5.3f} {int(r['n_g']):>7,} {r['p_hat_g']:>5.2f}"
             for _, r in sub.iterrows()]
    ax.text(0.03, 0.97, "\n".join(rows), transform=ax.transAxes, va="top",
            ha="left", family="monospace", fontsize=7, zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("stated confidence v_g")
    ax.set_ylabel("observed correctness p_hat_g")
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-pad, 1 + pad)
    ax.set_title(title)
    return _save(fig, path, cfg)


def vc_histogram(hist: pd.DataFrame, path: Path, cfg: Config, col: str) -> Path:
    fig, ax = _new(cfg)
    # Same reason as the reliability diagram: the mass sits at 1.0, and a bar
    # centred on the axis limit is drawn half-clipped.
    pad = 0.04
    for edge in (0.0, 1.0):
        ax.axvline(edge, color="black", lw=0.8, zorder=1)
    # Narrow enough that adjacent support points (0.99, 1.0) do not overlap.
    ax.bar(hist["value"], hist["n"], width=0.008, zorder=3)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_xlim(-pad, 1 + pad)
    ax.set_title(f"{col}: support and group sizes")
    return _save(fig, path, cfg)


def product_rule(curve: pd.DataFrame, path: Path, cfg: Config) -> Path:
    """Log-log claimed vs observed, one curve per k. The signature is divergence."""
    fig, ax = _new(cfg)
    floor = float(cfg.get("phase3.product_rule.log_floor"))
    for k, g in curve.groupby("k"):
        # Sorted by the log-space statistic, which is what the rule accumulates;
        # exp() is applied only for the axis, and the floor only guards the plot.
        g = g.sort_values("log_claimed")
        ax.plot(np.exp(np.clip(g["log_claimed"], np.log(floor), 0.0)),
                g["observed"].clip(lower=floor), marker="o", label=f"k = {k}")
    lims = [floor, 1.0]
    ax.plot(lims, lims, ls="--", lw=1, color="grey", label="honest aggregation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("claimed P(all k wrong) = exp(sum log(1 - vc))")
    ax.set_ylabel("observed frequency of all k wrong")
    ax.set_title("Product rule: the gap widens with k")
    ax.legend(fontsize=8)
    return _save(fig, path, cfg)


def temperature_sweep(per_q: pd.DataFrame, path: Path, cfg: Config) -> Path:
    fig, ax = _new(cfg)
    agg = per_q.groupby("temperature").agg(p_hat=("p_hat", "mean"),
                                           vc=("vc_bar", "mean")).reset_index()
    ax.plot(agg["temperature"], agg["p_hat"], marker="o", label="p_hat (moves with T)")
    ax.plot(agg["temperature"], agg["vc"], marker="s",
            label="VC (has no argument for T)")
    ax.set_xlabel("temperature")
    ax.set_ylabel("value")
    ax.set_ylim(0, 1)
    ax.set_title("Decoder non-invariance: p_q is a function of T, VC is not")
    ax.legend(fontsize=8)
    return _save(fig, path, cfg)


def budget_collapse(questions: pd.DataFrame, path: Path, cfg: Config,
                    vc_col: str = "vc_pre") -> Path:
    """VC's implied budget against the budget actually needed."""
    from .survival import predicted_budget
    alpha = float(cfg.get("phase3.budget_alpha"))
    sub = questions.dropna(subset=[vc_col]).copy()
    sub["n_hat"] = sub[vc_col].map(lambda v: predicted_budget(v, alpha))
    sub["observed"] = sub["K_q"].where(sub["K_q"] > 0, np.nan)
    fig, ax = _new(cfg)
    ax.scatter(sub["n_hat"], sub["observed"].fillna(sub["n_draws"].max() + 5),
               alpha=0.4, s=14)
    ax.axhline(sub["n_draws"].max() + 5, ls=":", lw=1, color="red")
    ax.annotate("never correct (censored)", (ax.get_xlim()[0], sub["n_draws"].max() + 5),
                fontsize=8, va="bottom")
    ax.set_xlabel(f"N_hat implied by {vc_col} at alpha = {alpha}")
    ax.set_ylabel("observed draws to first correct answer")
    ax.set_title("Dynamic-range collapse, in units of compute")
    return _save(fig, path, cfg)
