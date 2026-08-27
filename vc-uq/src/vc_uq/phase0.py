"""Phase 0 -- instrument validation. This is a GATE.

Nothing downstream is interpretable if the correctness criterion cannot tell
right answers from wrong ones. A flat result from a blunt instrument is not
evidence of anything; it is the instrument.

Run on ``split == tau_select`` only. Selecting ``tau`` on the same split that
later carries the LTT calibration would void the guarantee.

Four outputs:
  1. a stratified labelling sheet (across datasets AND across the e_cos range --
     a random sample is almost all easy and says nothing about the boundary)
  2. a tau sweep with Cohen's kappa against the labels, selecting tau_star
  3. AUROC of the criterion; below the gate, the primary criterion flips to NLI
  4. a null band on mismatched pairs -- if the null overlaps the observed band,
     the metric cannot resolve the effect and any flat result downstream is
     uninformative rather than negative
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .stats import NAN as NAN_, auroc, cohens_kappa
from .store import Store


@dataclass
class GateResult:
    passed: bool
    kappa: float
    auroc: float
    tau_star: float
    primary: str
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "kappa": self.kappa, "auroc": self.auroc,
                "tau_star": self.tau_star, "primary": self.primary,
                "reasons": self.reasons}


def build_label_sheet(cfg: Config, answers: pd.DataFrame,
                      questions: pd.DataFrame) -> pd.DataFrame:
    """Stratified across datasets and across e_cos deciles."""
    n = int(cfg.get("phase0.n_hand_label"))
    n_bins = int(cfg.get("phase0.stratify_bins"))
    sub = answers[answers["split"] == "tau_select"].dropna(subset=["e_cos"]).copy()
    if sub.empty:
        raise ValueError("no scored answers in the tau_select split")

    sub["e_bin"] = sub.groupby("dataset")["e_cos"].transform(
        lambda s: pd.qcut(s.rank(method="first"), min(n_bins, max(1, s.nunique())),
                          labels=False, duplicates="drop"))
    strata = sub.groupby(["dataset", "e_bin"], dropna=False)
    per_stratum = max(1, n // max(1, strata.ngroups))
    rng = np.random.default_rng(int(cfg.get("run.seed")))

    picks = []
    for _, g in strata:
        take = min(len(g), per_stratum)
        picks.append(g.sample(take, random_state=int(rng.integers(2**31 - 1))))
    sheet = pd.concat(picks, ignore_index=True)
    if len(sheet) > n:
        sheet = sheet.sample(n, random_state=int(cfg.get("run.seed")))

    a_star = questions.set_index("q_id")["a_star"]
    question_text = questions.set_index("q_id")["question"]
    return pd.DataFrame({
        "q_id": sheet["q_id"].to_numpy(),
        "draw_idx": sheet["draw_idx"].to_numpy(),
        "dataset": sheet["dataset"].to_numpy(),
        "question": sheet["q_id"].map(question_text).to_numpy(),
        "answer": sheet["answer"].to_numpy(),
        "a_star": sheet["q_id"].map(a_star).to_numpy(),
        "e_cos": sheet["e_cos"].to_numpy(),
        "correct_human": pd.Series([pd.NA] * len(sheet), dtype="boolean"),
    })


def simulate_human_labels(sheet: pd.DataFrame, answers: pd.DataFrame,
                          *, oracle_col: str = "correct_oracle",
                          error_rate: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Stand-in labels for smoke runs against the simulated backend.

    Real runs must fill ``correct_human`` by hand. This exists so the gate logic
    itself can be exercised; it is never a substitute for labelling, and the
    gate report records which of the two produced the labels.
    """
    if oracle_col not in answers.columns:
        raise ValueError(f"{oracle_col} is absent; simulated labels need a known truth")
    key = answers.set_index(["q_id", "draw_idx"])[oracle_col]
    rng = np.random.default_rng(seed)
    truth = [bool(key.loc[(q, d)]) for q, d in zip(sheet["q_id"], sheet["draw_idx"])]
    flips = rng.random(len(truth)) < error_rate
    out = sheet.copy()
    out["correct_human"] = [bool(t) ^ bool(f) for t, f in zip(truth, flips)]
    out["label_source"] = "simulated"
    return out


def tau_sweep(labelled: pd.DataFrame, cfg: Config,
              score_col: str = "e_cos") -> pd.DataFrame:
    grid = cfg.section("phase0.tau_grid")
    taus = np.arange(grid["start"], grid["stop"] + 1e-9, grid["step"])
    y = labelled["correct_human"].astype(bool).to_numpy()
    s = labelled[score_col].to_numpy(dtype=float)
    rows = []
    for tau in taus:
        pred = s <= tau
        rows.append({"tau": float(tau), "kappa": cohens_kappa(pred, y),
                     "accuracy": float((pred == y).mean()),
                     "predicted_positive_rate": float(pred.mean())})
    return pd.DataFrame(rows)


def null_band(cfg: Config, answers: pd.DataFrame, questions: pd.DataFrame,
              embedder=None, matched_scores=None) -> dict:
    """Distribution of e_cos on MISMATCHED pairs (a_i, a_star_j), i != j.

    Same null is run for s_anchor using anchors borrowed from other questions.
    If the null overlaps the observed band, the metric cannot resolve the
    effects being looked for, and a flat downstream result is uninformative.
    """
    from .backends.base import cosine_distance
    from .judge import build_embedding_space

    n_pairs = int(cfg.get("phase0.null_band.n_mismatched_pairs"))
    sub = answers[answers["split"] == "tau_select"].dropna(subset=["e_cos"])
    if sub.empty:
        raise ValueError("no scored answers in tau_select")
    q_idx = questions.set_index("q_id")
    texts = list(sub["answer"].astype(str)) + list(q_idx["a_star"].astype(str))
    if "anchor" in q_idx.columns:
        texts += [t for t in q_idx["anchor"].dropna().astype(str)]
    space = build_embedding_space(cfg, texts, embedder=embedder)

    rng = np.random.default_rng(int(cfg.get("run.seed")))
    ans = sub["answer"].astype(str).to_numpy()
    ans_q = sub["q_id"].to_numpy()
    other_q = q_idx.index.to_numpy()

    i = rng.integers(0, len(ans), size=n_pairs)
    j = rng.integers(0, len(other_q), size=n_pairs)
    keep = ans_q[i] != other_q[j]
    i, j = i[keep], j[keep]

    e_null = cosine_distance(space.get(ans[i]),
                             space.get(q_idx["a_star"].astype(str).to_numpy()[j]))
    out = {
        "e_cos_observed_mean": float(sub["e_cos"].mean()),
        "e_cos_observed_p05": float(sub["e_cos"].quantile(0.05)),
        "e_cos_observed_p95": float(sub["e_cos"].quantile(0.95)),
        "e_cos_null_mean": float(np.mean(e_null)),
        "e_cos_null_p05": float(np.quantile(e_null, 0.05)),
        "e_cos_null_p95": float(np.quantile(e_null, 0.95)),
        "n_null_pairs": int(len(e_null)),
    }
    # Separation of the observed band from the null. Pooling ALL observed pairs
    # here would understate the instrument: a wrong answer is a mismatched pair
    # in every respect that the embedding can see, so a model with a high error
    # rate would look like a blunt metric. The instrument's resolving power is
    # the separation of KNOWN-MATCHED pairs from the null; the pooled number is
    # kept alongside it, but the gate reads the matched one.
    def _sep(observed: np.ndarray) -> float:
        scores = np.concatenate([observed, e_null])
        labels = np.concatenate([np.ones(len(observed), bool),
                                 np.zeros(len(e_null), bool)])
        return float(1.0 - auroc(scores, labels))

    out["separation_auroc_all_pairs"] = _sep(sub["e_cos"].to_numpy(dtype=float))
    if matched_scores is not None and len(matched_scores):
        matched = np.asarray(matched_scores, dtype=float)
        matched = matched[~np.isnan(matched)]
        out["n_matched"] = int(len(matched))
        out["e_cos_matched_mean"] = float(np.mean(matched)) if len(matched) else NAN_
        out["separation_auroc"] = _sep(matched) if len(matched) else NAN_
        out["separation_basis"] = "known-correct pairs vs mismatched null"
    else:
        out["separation_auroc"] = out["separation_auroc_all_pairs"]
        out["separation_basis"] = ("all observed pairs vs mismatched null "
                                   "(no labels supplied; conflates model error "
                                   "with instrument error)")

    if "s_anchor" in sub.columns and "anchor" in q_idx.columns:
        anchors = q_idx["anchor"].astype(str).to_numpy()
        s_null = cosine_distance(space.get(anchors[j % len(anchors)]), space.get(ans[i]))
        out["s_anchor_observed_mean"] = float(sub["s_anchor"].mean())
        out["s_anchor_null_mean"] = float(np.mean(s_null))
        s_scores = np.concatenate([sub["s_anchor"].to_numpy(dtype=float), s_null])
        out["s_anchor_separation_auroc"] = float(
            1.0 - auroc(s_scores, np.concatenate([np.ones(len(sub), bool),
                                                  np.zeros(len(s_null), bool)])))
    return out


def run_gate(cfg: Config, store: Store, labelled: pd.DataFrame,
             answers: pd.DataFrame, questions: pd.DataFrame,
             embedder=None) -> GateResult:
    y = labelled["correct_human"].astype(bool).to_numpy()
    reasons: list[str] = []

    sweep_cos = tau_sweep(labelled, cfg, "e_cos")
    best_cos = sweep_cos.loc[sweep_cos["kappa"].idxmax()]
    # e_cos is a DISTANCE, so a low value indicates correctness: negate for AUROC.
    auroc_cos = auroc(-labelled["e_cos"].to_numpy(dtype=float), y)

    auroc_nli = float("nan")
    if "correct_nli" in labelled.columns and labelled["correct_nli"].notna().any():
        auroc_nli = auroc(labelled["correct_nli"].astype(float).to_numpy(), y)

    kappa_min = float(cfg.get("phase0.gate.kappa_min"))
    auroc_min = float(cfg.get("phase0.gate.auroc_min"))

    primary = "cos"
    kappa = float(best_cos["kappa"])
    au = float(auroc_cos)
    tau_star = float(best_cos["tau"])

    if au < auroc_min:
        reasons.append(
            f"AUROC(e_cos) = {au:.3f} < {auroc_min}: cosine is too blunt to judge "
            "correctness. Switching the primary criterion to NLI bidirectional "
            "entailment; cosine is retained as an ablation.")
        primary = "nli"
        if not np.isnan(auroc_nli):
            au = auroc_nli
            kappa = cohens_kappa(labelled["correct_nli"].astype(bool).to_numpy(), y)
        else:
            reasons.append("correct_nli is not populated; run judge_nli before the gate.")

    matched = labelled.loc[labelled["correct_human"].astype(bool), "e_cos"]
    nb = null_band(cfg, answers, questions, embedder=embedder,
                   matched_scores=matched.to_numpy(dtype=float))
    sep_min = float(cfg.get("phase0.null_band.overlap_min_auroc"))
    if nb["separation_auroc"] < sep_min:
        reasons.append(
            f"null band overlaps the observed band (separation AUROC "
            f"{nb['separation_auroc']:.3f} < {sep_min}). A flat downstream result "
            "would be uninformative, not negative.")

    passed = (kappa >= kappa_min) and (au >= auroc_min) and \
             (nb["separation_auroc"] >= sep_min)
    if kappa < kappa_min:
        reasons.append(f"kappa = {kappa:.3f} < {kappa_min}")
    if au < auroc_min:
        reasons.append(f"AUROC({primary}) = {au:.3f} < {auroc_min} even after fallback")

    sens = float(cfg.get("phase0.tau_sensitivity"))
    store.write_table("phase0_tau_sweep", sweep_cos)
    store.write_json("phase0_gate", {
        **GateResult(passed, kappa, au, tau_star, primary, reasons).as_dict(),
        "auroc_cos": float(auroc_cos), "auroc_nli": auroc_nli,
        "null_band": nb,
        "tau_sensitivity_grid": [tau_star - sens, tau_star, tau_star + sens],
        "n_labelled": int(len(labelled)),
        "label_source": (labelled["label_source"].iloc[0]
                         if "label_source" in labelled.columns else "human"),
    })
    return GateResult(passed, kappa, au, tau_star, primary, reasons)


def tau_sensitivity_values(cfg: Config, tau_star: float) -> list[float]:
    """Every headline result is re-reported at these three taus (protocol 3.5)."""
    s = float(cfg.get("phase0.tau_sensitivity"))
    return [round(tau_star - s, 6), round(tau_star, 6), round(tau_star + s, 6)]
