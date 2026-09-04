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
from .store import PhaseStore, Store  # noqa: F401  (PhaseStore used in annotations)

# A hand label identifies ONE answer, and (q_id, draw_idx) stopped identifying
# one the moment generation grew a second pass (protocol 6.4): draw 3 of the
# classification pass and draw 3 of the downstream pass are different text with
# the same pair. Keying without draw_set would attach one human verdict to both,
# silently, and tau_star is selected against exactly these labels.
LABEL_KEY = ["q_id", "draw_set", "draw_idx"]



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

    if bool(cfg.get("phase0.labels.unique_answers")):
        # One labelling decision per DISTINCT (question, answer). N_MAX draws of
        # one question repeat themselves constantly -- a question the model is
        # sure about emits the same string 8 times out of 40 -- and those rows
        # are not independent evidence about the criterion: identical text gets
        # identical e_cos and identical human verdict. Labelling them repeatedly
        # spends the human budget on rows that cannot disagree, and then kappa
        # and AUROC are computed as though they were separate observations, so
        # the gate reports more precision than the labels contain.
        #
        # Deduplicated BEFORE the deciles are cut, so the strata span distinct
        # pairs rather than distinct draws.
        sub = (sub.sort_values(["q_id", "draw_set", "draw_idx"])
               .drop_duplicates(subset=["q_id", "answer"], keep="first"))

    sub["e_bin"] = sub.groupby("dataset")["e_cos"].transform(
        lambda s: pd.qcut(s.rank(method="first"), min(n_bins, max(1, s.nunique())),
                          labels=False, duplicates="drop"))
    sub["stratum"] = (sub["dataset"].astype(str) + "|e"
                      + sub["e_bin"].astype("Int64").astype(str))
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
    if bool(cfg.get("phase0.labels.shuffle_sheet")):
        # The concat above is in stratum order, and it only overflows n (and so
        # only gets shuffled) by accident. Labelling the first half of an
        # unshuffled sheet would cover the low-e_cos bins of the first dataset
        # and nothing else, so tau_star would be selected against pairs that are
        # all easy -- with a flattering kappa, because the boundary was never
        # sampled. Shuffling makes partial labelling degrade gracefully.
        sheet = sheet.sample(frac=1.0, random_state=int(cfg.get("run.seed")))
    sheet = sheet.reset_index(drop=True)

    a_star = questions.set_index("q_id")["a_star"]
    question_text = questions.set_index("q_id")["question"]
    return pd.DataFrame({
        "q_id": sheet["q_id"].astype(str).to_numpy(),
        "draw_set": sheet["draw_set"].astype(str).to_numpy(),
        "draw_idx": sheet["draw_idx"].astype(int).to_numpy(),
        "dataset": sheet["dataset"].to_numpy(),
        "stratum": sheet["stratum"].to_numpy(),
        "question": sheet["q_id"].map(question_text).to_numpy(),
        "answer": sheet["answer"].to_numpy(),
        "a_star": sheet["q_id"].map(a_star).to_numpy(),
        "e_cos": sheet["e_cos"].to_numpy(),
        "correct_human": pd.Series([pd.NA] * len(sheet), dtype="boolean"),
    })


# --------------------------------------------------------------------------
# Reading hand labels back in
# --------------------------------------------------------------------------

_TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0", "correct"}
_FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0", "incorrect"}
_BLANK_TOKENS = {"", "na", "nan", "none", "<na>", "null", "?"}


def _parse_label(value) -> object:
    if isinstance(value, bool):
        return value
    if value is None:
        return pd.NA
    if isinstance(value, float) and np.isnan(value):
        return pd.NA
    token = str(value).strip().lower()
    if token in _BLANK_TOKENS:
        return pd.NA
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return pd.NA


class LabelCoverageError(ValueError):
    """The labelled subset does not cover the stratification.

    Carries the coverage ``report`` so the caller can persist the per-stratum
    table before failing -- being told which strata still need labelling is the
    whole point of refusing.
    """

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}


def load_hand_labels(cfg: Config, sheet: pd.DataFrame,
                     path) -> tuple[pd.DataFrame, dict]:
    """Merge a filled-in label CSV onto a FRESHLY BUILT sheet.

    Only ``correct_human`` is taken from the file. Everything else -- ``e_cos``
    above all -- is recomputed, because ``e_cos`` is not a stable property of a
    pair: the embedding space is mean-centred over whichever texts were scored
    together, so adding questions or changing the embedding backend moves it.
    Trusting a stale column would select ``tau_star`` against numbers that no
    longer exist, and then apply that threshold to numbers that do. Nothing
    would raise; every correctness label downstream would simply be wrong.

    Returns the labelled subset and a coverage report. Unlabelled rows are
    dropped rather than guessed, but a stratum with NO labelled pair is refused:
    overall coverage would hide exactly that hole, and the whole point of
    stratifying was to sample the boundary rather than the easy mass.
    """
    raw = pd.read_csv(path)
    missing = {*LABEL_KEY, "correct_human"} - set(raw.columns)
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {sorted(missing)}. Fill in the sheet that "
            "`vc_uq gate` wrote rather than building a CSV by hand -- draw_set is "
            "part of the key, and without it a label cannot be matched to the "
            "answer it was written for.")

    parsed = raw["correct_human"].map(_parse_label)
    # fillna BEFORE the membership test: the string accessors propagate NA, so
    # an empty cell would come through as pd.NA, and `pd.NA not in {...}` is
    # True -- which would report every deliberately blank row as an
    # unrecognised value on any partially labelled sheet.
    tokens = (raw["correct_human"].astype("string").str.strip().str.lower()
              .fillna(""))
    n_unrecognised = int((parsed.isna() & ~tokens.isin(list(_BLANK_TOKENS))).sum())

    csv = pd.DataFrame({
        "q_id": raw["q_id"].astype(str),
        "draw_set": raw["draw_set"].astype(str),
        "draw_idx": pd.to_numeric(raw["draw_idx"], errors="coerce").astype("Int64"),
        "correct_human": parsed,
    }).dropna(subset=["draw_idx"])
    csv["draw_idx"] = csv["draw_idx"].astype(int)
    n_duplicate_keys = int(csv.duplicated(subset=LABEL_KEY).sum())
    csv = csv.drop_duplicates(subset=LABEL_KEY, keep="last")
    labelled_csv = csv[csv["correct_human"].notna()]

    base = sheet.drop(columns=["correct_human"]).copy()
    base["q_id"] = base["q_id"].astype(str)
    base["draw_set"] = base["draw_set"].astype(str)
    base["draw_idx"] = base["draw_idx"].astype(int)
    merged = base.merge(labelled_csv, on=LABEL_KEY, how="left")

    base_keys = set(zip(*(base[c] for c in LABEL_KEY)))
    unjoined = [f"{q}#{s}#{d}" for q, s, d in zip(*(labelled_csv[c] for c in LABEL_KEY))
                if (q, s, d) not in base_keys]

    per_stratum = (merged.assign(_lab=merged["correct_human"].notna())
                   .groupby("stratum")
                   .agg(n_in_sheet=("q_id", "size"), n_labelled=("_lab", "sum"))
                   .reset_index())
    per_stratum["coverage"] = per_stratum["n_labelled"] / per_stratum["n_in_sheet"]

    min_per = int(cfg.get("phase0.labels.min_per_stratum"))
    thin = per_stratum[per_stratum["n_labelled"] < min_per]

    out = merged[merged["correct_human"].notna()].copy()
    out["correct_human"] = out["correct_human"].astype(bool)
    out["label_source"] = "human"

    coverage = float(len(out) / len(base)) if len(base) else 0.0
    report = {
        "path": str(path),
        "n_sheet_rows": int(len(base)),
        "n_csv_rows": int(len(raw)),
        "n_labelled_in_csv": int(len(labelled_csv)),
        "n_joined": int(len(out)),
        "n_unjoined": len(unjoined),
        "unjoined_examples": unjoined[:10],
        "n_unrecognised_values": n_unrecognised,
        "n_duplicate_keys": n_duplicate_keys,
        "coverage": coverage,
        "min_coverage": float(cfg.get("phase0.labels.min_coverage")),
        "coverage_ok": coverage >= float(cfg.get("phase0.labels.min_coverage")),
        "n_strata": int(len(per_stratum)),
        "n_strata_below_min": int(len(thin)),
        "per_stratum": per_stratum.to_dict("records"),
        "warnings": [],
    }
    if unjoined:
        report["warnings"].append(
            f"{len(unjoined)} labelled CSV rows matched no pair in the current sheet. "
            "The split assignment or the sampled strata have moved since the sheet "
            "was written (run.seed, dataset sizes, or stratify_bins).")
    if n_unrecognised:
        report["warnings"].append(
            f"{n_unrecognised} correct_human values were not recognised as "
            "true/false and were treated as unlabelled.")
    if not report["coverage_ok"]:
        report["warnings"].append(
            f"only {coverage:.0%} of the sheet is labelled "
            f"(below phase0.labels.min_coverage).")

    if len(thin):
        raise LabelCoverageError(
            f"{len(thin)} of {len(per_stratum)} strata have fewer than {min_per} "
            f"labelled pair(s): {list(thin['stratum'])[:8]}. tau_star would be "
            "selected against an unrepresentative slice of the e_cos range, which "
            "is the failure the stratification exists to prevent. Label at least "
            "one pair in each stratum, or lower phase0.labels.min_per_stratum "
            "deliberately. The full per-stratum table is in "
            "tables/phase0_label_coverage.json.",
            report=report)

    return out, report


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
    key = answers.set_index(LABEL_KEY)[oracle_col]
    rng = np.random.default_rng(seed)
    truth = [bool(key.loc[k]) for k in zip(*(sheet[c] for c in LABEL_KEY))]
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
        # An anchor is the medoid of a question's OWN draws, so it exists only
        # for questions that were actually generated. Staged generation
        # (`generate --splits tau_select`) leaves the rest of the question table
        # anchorless, and the previous indexing borrowed by position from the
        # full table -- reaching a NaN and failing to embed it. Sample from the
        # anchored questions instead, and re-derive the mismatch condition
        # against THAT pool rather than reusing `j`, which indexes the other one.
        anchored = q_idx["anchor"].dropna().astype(str)
        if len(anchored):
            a_ids = anchored.index.to_numpy()
            a_txt = anchored.to_numpy()
            ia = rng.integers(0, len(ans), size=n_pairs)
            ja = rng.integers(0, len(a_txt), size=n_pairs)
            keep_a = ans_q[ia] != a_ids[ja]     # a borrowed anchor, never its own
            ia, ja = ia[keep_a], ja[keep_a]
            s_null = cosine_distance(space.get(a_txt[ja]), space.get(ans[ia]))
            out["s_anchor_observed_mean"] = float(sub["s_anchor"].mean())
            out["s_anchor_null_mean"] = float(np.mean(s_null))
            out["n_s_anchor_null_pairs"] = int(len(s_null))
            out["n_anchored_questions"] = int(len(anchored))
            s_scores = np.concatenate([sub["s_anchor"].to_numpy(dtype=float), s_null])
            out["s_anchor_separation_auroc"] = float(
                1.0 - auroc(s_scores, np.concatenate([np.ones(len(sub), bool),
                                                      np.zeros(len(s_null), bool)])))
    return out


def run_gate(cfg: Config, store: "Store | PhaseStore", labelled: pd.DataFrame,
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
    store.write_table("tau_sweep", sweep_cos)
    store.write_json("gate", {
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
