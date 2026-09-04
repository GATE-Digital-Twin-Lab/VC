"""Correctness criterion (``e``) and label-free score (``s``).

These are different functions and must never be merged (protocol section 0):

  ``e(a_i) = 1 - cos(emb(a_i), emb(a_star))``
      ``a_star`` is the REFERENCE and ``a_i`` the argument. On a new question
      there is nothing to sweep, so ``e`` cannot be evaluated at test time at
      all. It is a correctness criterion, confined to evaluation.

  ``s(q, a) = 1 - cos(emb(anchor(q)), emb(a))``
      ``anchor(q)`` is the reference and ``a`` is swept. At calibration you
      evaluate at ``a = a_star``; at test you sweep candidates. Same operation,
      no label. A valid nonconformity score.

Using ``e`` as a score would be undefined at deployment; thresholding the same
function for both set membership and correctness would make every retained
answer correct by construction and drive the loss identically to zero. The
circularity guard below refuses that configuration outright.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd

from .backends import build_embedder, build_nli
from .backends.base import cosine_distance, pairwise_cosine_distance
from .config import Config


# --------------------------------------------------------------------------
# Answer normalisation
# --------------------------------------------------------------------------

_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+")

# Letters Unicode treats as their own, not as a base plus a combining mark, so
# NFKD leaves them alone: "Coeur" and "Coeur" with the ligature never meet
# without an explicit map. Standard ASCII-folding pairs, the same set a search
# index would use.
#
# Inert on the corpus this was written against -- the only ligatures present sit
# in model answers to fabricated questions whose gold is "No such entity
# exists", so no PAIR is unified by them. Included because the rule is correct
# in general and the corpus is not fixed, not because it changed a number here.
_LIGATURES = {
    "\u0153": "oe", "\u0152": "oe",   # oe
    "\u00e6": "ae", "\u00c6": "ae",   # ae
    "\u00f8": "o", "\u00d8": "o",     # o with stroke
    "\u00df": "ss",                    # sharp s
    "\u0111": "d", "\u0110": "d",     # d with stroke
    "\u0142": "l", "\u0141": "l",     # l with stroke
    "\u00fe": "th", "\u00de": "th",   # thorn
    "\u00f0": "d", "\u00d0": "d",     # eth
}
_LIGATURE_TABLE = str.maketrans(_LIGATURES)


def fold_accents(text) -> str:
    """Drop diacritics, keeping the letters underneath.

    "La Boheme" and "La Boheme" with the grave, "Nastase" and "Nastase" with the
    breve, "Comaneci" and "Comaneci" -- the criterion scored the accented and
    unaccented spellings of these as DIFFERENT answers, and one of them
    ("Le Carre" for "John Le Carre") landed at e_cos = 1.058, further apart than
    two unrelated strings.

    Decomposition-and-drop rather than "delete anything non-ASCII": stripping
    the high range outright would erase a non-Latin answer down to the empty
    string, and two unrelated answers that both became "" would then be
    identical -- a false positive in the correctness criterion, which is the one
    error this study cannot absorb. Non-Latin scripts carry no combining marks
    here and pass through untouched.
    """
    out = str(text).translate(_LIGATURE_TABLE)
    out = unicodedata.normalize("NFKD", out)
    return "".join(c for c in out if not unicodedata.combining(c))

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

# Word forms are GENERATED from num2words rather than hand-listed, and the table
# is matched on the whole normalised string only.
#
# The opposite direction -- parsing arbitrary text with word2number -- was tried
# and rejected. On this corpus it accepts 8 of 4,323 distinct answers as
# numbers, and 5 of those are wrong: "Four Weddings and a Funeral" -> 4,
# "Marine One" -> 1, "Seven Samurai" -> 7, "Million Dollar Baby" -> 1000000.
# Collapsing film titles to digits would merge unrelated answers into one
# cluster and make them correct for each other. A generated exact-match table
# cannot do that: it only fires on a string that IS a number word.
#
# Through 2100 so that YEARS are covered -- a trivia corpus answers "1969" and
# "nineteen sixty-nine" interchangeably, and years are the most common numeric
# answer in it. Built once and cached; the cost is measured in milliseconds.
_NUMERAL_MAX = 2100


@lru_cache(maxsize=1)
def _numeral_table() -> dict[str, str]:
    """{normalised word form: digit string} for 0.._NUMERAL_MAX, plus round
    magnitudes above it. Built once, lazily -- num2words is an optional import
    and the analysis phases must not require it at module load."""
    try:
        from num2words import num2words
    except ImportError:  # pragma: no cover - environment dependent
        return {}
    values = list(range(_NUMERAL_MAX + 1)) + [
        5000, 10000, 100000, 1000000, 1000000000]
    table = {}
    for v in values:
        # Squashed the same way an answer is, so "twenty-one" and "twenty one"
        # reach the table as one key. Hyphens become spaces rather than being
        # deleted, or num2words' "twenty-one" would never match a typed
        # "twenty one".
        table.setdefault(_squash(num2words(v)), str(v))
        # Years are SPOKEN differently from cardinals -- 1969 is "nineteen
        # sixty-nine", not "one thousand nine hundred and sixty-nine" -- and the
        # spoken form is the one a model writes. num2words carries both.
        if 1000 <= v <= _NUMERAL_MAX:
            try:
                table.setdefault(_squash(num2words(v, to="year")), str(v))
            except (NotImplementedError, TypeError, OverflowError):
                pass
    return table


def _squash(text: str) -> str:
    out = fold_accents(str(text)).lower().strip().replace("-", " ")
    out = _PUNCT_RE.sub("", out)
    return _WS_RE.sub(" ", out).strip()


def normalise_answer(text, *, case: bool = True, punctuation: bool = True,
                     articles: bool = True, numerals: bool = True,
                     accents: bool = True) -> str:
    """Strip formatting the correctness criterion should never have counted.

    The embedding criterion scores ``"7"`` against a gold ``"Seven"`` at
    ``e_cos = 0.84`` and calls it wrong -- and bidirectional NLI agrees, so the
    error is not caught by switching criteria. Measured on the eval split, a
    tenth to a third of the questions that look like the model is INCONSISTENT
    (some draws right, some wrong) are really one answer written two ways.

    Applied to the text BEFORE it is embedded, not patched onto the distance
    afterwards: ``e_cos`` is then a distance between the things being compared
    rather than between their spellings, and ``s_anchor``, the medoid anchor and
    the null band all inherit the same treatment for free.

    The raw ``answer`` column is never modified. Generation output is
    append-only; this is a scoring-time view of it.
    """
    out = str(text).strip()
    if case:
        out = out.lower()
    if accents:
        # Before the punctuation pass, so any compatibility decomposition it
        # produces (a ligature splitting, a fraction becoming digits and a
        # fraction slash) is then cleaned up by the same rules as everything
        # else rather than surviving as a stray symbol.
        out = fold_accents(out)
    if punctuation:
        # Hyphen to space, everything else dropped: "twenty-one" and "twenty
        # one" are one answer, but "St. Martin's" must become "st martins"
        # rather than "st martin s".
        out = _WS_RE.sub(" ", _PUNCT_RE.sub("", out.replace("-", " "))).strip()
    else:
        out = _WS_RE.sub(" ", out).strip()
    if articles:
        out = _ARTICLE_RE.sub("", out)
    if numerals:
        # Whole-string match only. A substring rule would turn "Seven Samurai"
        # into "7 Samurai".
        out = _numeral_table().get(out, out)
    return out


def normaliser(cfg: Config):
    """The configured normalisation as a single callable.

    Returns ``str`` unchanged when disabled, so callers never branch and the
    "off" path is exactly the old behaviour rather than an approximation of it.
    """
    if not bool(cfg.get("judge.normalise.enabled", True)):
        return lambda t: str(t)
    opts = dict(case=bool(cfg.get("judge.normalise.case", True)),
                punctuation=bool(cfg.get("judge.normalise.punctuation", True)),
                articles=bool(cfg.get("judge.normalise.articles", True)),
                numerals=bool(cfg.get("judge.normalise.numerals", True)),
                accents=bool(cfg.get("judge.normalise.accents", True)))
    return lambda t: normalise_answer(t, **opts)


def token_subset_equivalent(answer: str, reference: str) -> bool:
    """True when one answer is the other with tokens dropped or added.

    This rescues ``"Orton"`` for a gold ``"Joe Orton"`` and ``"Jones"`` for
    ``"James Jones"`` -- surname-only replies that cosine puts at 0.63 and NLI
    also rejects. It is NOT a normalisation and cannot be folded into the
    embedding: it is a claim about a PAIR, so it is applied as an override on
    the criterion.

    It is off by default and deliberately kept on its own flag, because the same
    rule accepts ``"Bridge"`` for ``"Bridge Over Troubled Water"``, which is an
    incomplete answer rather than a differently-spelled one. Enabling it trades
    a known false-negative class for a known false-positive class; report which
    was used.
    """
    a, b = str(answer).strip(), str(reference).strip()
    if not a or not b:
        return False
    if a == b:
        return True
    at, bt = a.split(), b.split()
    if set(at) and set(at).issubset(set(bt)):
        return True
    if set(bt) and set(bt).issubset(set(at)):
        return True
    return a.startswith(b) or b.startswith(a)


def equivalence_mask(cfg: Config, answers, references) -> np.ndarray:
    """Pairwise equivalence overrides, as a boolean array.

    Normalisation is applied first, so the override only has to carry the cases
    normalisation cannot: token-level differences rather than spelling ones.
    """
    norm = normaliser(cfg)
    pairs = [(norm(a), norm(r)) for a, r in zip(answers, references)]
    if bool(cfg.get("judge.equivalence.token_subset", False)):
        return np.array([token_subset_equivalent(a, r) for a, r in pairs],
                        dtype=bool)
    # With normalisation on, exact equality after normalisation already shows up
    # as e_cos = 0 and needs no override; the array is all-False rather than a
    # special case, so callers do not branch.
    return np.zeros(len(pairs), dtype=bool)


def correct_by_cosine(answers: pd.DataFrame, tau: float, *,
                      column: str = "e_cos",
                      equiv_column: str = "answer_equivalent") -> pd.Series:
    """The cosine correctness predicate, in ONE place.

    ``tau_sweep`` selects ``tau_star`` against this and every downstream phase
    applies it. If the two disagreed -- the sweep on the distance alone, the
    phases on the distance OR an equivalence override -- ``tau_star`` would be
    chosen to optimise a rule nothing afterwards uses.
    """
    base = answers[column] <= float(tau)
    if equiv_column in answers.columns:
        base = base | answers[equiv_column].astype("boolean").fillna(False)
    return base.astype("boolean")


class CircularityError(ValueError):
    """Set membership and correctness were defined by the same function."""


def guard_against_circularity(membership_score: str, correctness_score: str) -> None:
    if membership_score == correctness_score:
        raise CircularityError(
            f"membership and correctness both thresholded on {membership_score!r}. "
            "Every retained answer would be correct by construction and the risk "
            "would be identically zero. Membership must use s_anchor (label-free); "
            "correctness must use e_cos or NLI (a_star-anchored)."
        )


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

@dataclass
class EmbeddingSpace:
    """Embeddings for one split, with anisotropy correction applied jointly.

    Correction is fitted on the split as a whole rather than per question:
    unrelated English sentences sit at cosine 0.3-0.6 and a fixed domain can
    compress everything into 0.75-0.95, so the common component has to be
    removed from the same population that both ``e`` and ``s`` are computed in.
    """

    vectors: dict[str, np.ndarray]
    dim: int
    mean_centered: bool
    whitened: bool
    # Applied on the way IN (when the space is built) and on the way OUT (every
    # lookup), so callers keep passing raw answer text and cannot desynchronise
    # the two. Identity when normalisation is off.
    normalise: Callable[[str], str] = str

    def key(self, text) -> str:
        return self.normalise(str(text))

    def get(self, texts) -> np.ndarray:
        return np.vstack([self.vectors[self.key(t)] for t in texts])


def build_embedding_space(cfg: Config, texts, embedder=None) -> EmbeddingSpace:
    embedder = embedder if embedder is not None else build_embedder(cfg)
    norm = normaliser(cfg)
    # Deduplicated AFTER normalising: "7" and "Seven" collapse to one vector, so
    # the embedder is called once for them and they are at distance 0 by
    # construction rather than by luck.
    uniq = list(dict.fromkeys(norm(t) for t in texts))
    mat = np.asarray(embedder.embed(uniq), dtype=np.float64)
    mean_center = bool(cfg.get("embedding.postprocess.mean_center"))
    whiten = bool(cfg.get("embedding.postprocess.whiten"))

    if mean_center:
        mat = mat - mat.mean(axis=0, keepdims=True)
    if whiten:
        cov = np.cov(mat, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-8, None)
        mat = mat @ vecs @ np.diag(vals ** -0.5) @ vecs.T
    return EmbeddingSpace(vectors=dict(zip(uniq, mat)), dim=mat.shape[1],
                          mean_centered=mean_center, whitened=whiten,
                          normalise=norm)


# --------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------

def medoid_anchor(answers: list[str], space: EmbeddingSpace) -> str:
    """Medoid of the N draws -- deterministic given the draws.

    Preferred over a greedy decode because the radius around the medoid *is*
    semantic dispersion, so ``s`` doubles as the diversity measure used in 6.6
    instead of being computed separately.
    """
    if not answers:
        raise ValueError("cannot build an anchor from zero draws")
    if len(answers) == 1:
        return answers[0]
    mat = space.get(answers)
    d = pairwise_cosine_distance(mat)
    return answers[int(np.argmin(d.sum(axis=1)))]


def greedy_anchor(lm, cfg: Config, question_row: pd.Series) -> str:
    """T=0 decode. Also deterministic, and independent of the draw set."""
    from . import prompts
    from .generate import _mock_meta, derive_seed
    spec = prompts.get(cfg.get("generation.prompt_variant_clean"))
    msgs = spec.build(question=question_row["question"])
    if cfg.get("model.backend") == "mock":
        msgs = _mock_meta(msgs, {"qid": question_row["q_id"],
                                 "ds": question_row["dataset"],
                                 "variant": spec.variant,
                                 "kind": "post"})
    from .backends.base import SamplingParams
    params = SamplingParams.from_config(
        cfg, temperature=float(cfg.get("anchor.greedy_temperature")),
        max_tokens=int(cfg.get("anchor.greedy_max_tokens")))
    gen = lm.generate(msgs, params=params,
                      seed=derive_seed(int(cfg.get("run.seed")), "anchor",
                                       question_row["q_id"]))
    from .parsing import parse_answer_and_vc
    return parse_answer_and_vc(gen.text).answer


def anchor_group_keys(answers: pd.DataFrame) -> list[str]:
    """Anchors are per DRAW SET, not per question.

    A question carries two independent sets of draws -- the classification pass
    and the fresh downstream pass (protocol 6.4). The medoid of one is not the
    medoid of the other, and an online caller only ever sees the set it is
    running on, so pooling them would score every draw against an anchor that no
    deployed rule could have computed.
    """
    return ["q_id", "draw_set"] if "draw_set" in answers.columns else ["q_id"]


def build_anchors(cfg: Config, answers: pd.DataFrame, questions: pd.DataFrame,
                  space: EmbeddingSpace, lm=None) -> pd.Series:
    """One anchor per (question, draw set), built IDENTICALLY at calib and test."""
    method = cfg.get("anchor.method")
    keys = anchor_group_keys(answers)
    if method == "medoid":
        return (answers.sort_values("draw_idx").groupby(keys, sort=False)["answer"]
                .apply(lambda s: medoid_anchor(list(s), space)))
    if method == "greedy":
        if lm is None:
            from .backends import build_lm
            lm = build_lm(cfg)
        idx = questions.set_index("q_id")
        base = {q: greedy_anchor(lm, cfg, idx.loc[q]) for q in idx.index}
        if len(keys) == 1:
            return pd.Series(base)
        # A greedy decode does not depend on the draws, so both passes get the
        # same anchor -- but it is still indexed by pass, so every caller can map
        # rows the same way whichever method is configured.
        combos = answers[keys].drop_duplicates()
        return pd.Series([base.get(q) for q in combos["q_id"]],
                         index=pd.MultiIndex.from_frame(combos))
    raise ValueError(f"unknown anchor.method {method!r}; must be medoid or greedy")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_answers(cfg: Config, answers: pd.DataFrame, questions: pd.DataFrame,
                  *, embedder=None, lm=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Populate ``e_cos``, ``s_anchor``, ``s_anchor_rank`` and the anchor column."""
    q_idx = questions.set_index("q_id")
    texts = list(answers["answer"].astype(str)) + list(q_idx["a_star"].astype(str))
    space = build_embedding_space(cfg, texts, embedder=embedder)

    anchors = build_anchors(cfg, answers, questions, space, lm=lm)
    # Through space.key, not raw text: with normalisation on, every raw string
    # would look absent and the space would be rebuilt on every call.
    missing = [t for t in anchors.values if space.key(t) not in space.vectors]
    if missing:
        space = build_embedding_space(cfg, texts + missing, embedder=embedder)

    out = answers.copy()
    keys = anchor_group_keys(out)
    amap = anchors.to_dict()
    a_star_texts = out["q_id"].map(q_idx["a_star"].astype(str))
    if len(keys) == 1:
        anchor_texts = out["q_id"].map(amap)
    else:
        anchor_texts = pd.Series(
            [amap.get(k) for k in zip(*(out[c] for c in keys))], index=out.index)

    emb_a = space.get(out["answer"].astype(str))
    emb_star = space.get(a_star_texts)
    emb_anchor = space.get(anchor_texts)

    out["e_cos"] = cosine_distance(emb_a, emb_star)
    out["s_anchor"] = cosine_distance(emb_anchor, emb_a)
    # A property of the (answer, a_star) PAIR, so it cannot live in the
    # embedding and is carried as its own column. Written even when the rule is
    # disabled -- an all-False column keeps every consumer branch-free, and its
    # presence in the parquet records that the question was asked.
    out["answer_equivalent"] = equivalence_mask(
        cfg, out["answer"].astype(str), a_star_texts)

    if bool(cfg.get("embedding.postprocess.rank_transform_s")):
        # Only the ordering of s matters downstream, and ranking within a split
        # restores dynamic range that anisotropy compresses away. Ranked within
        # the draw set too: the two passes are scored against different anchors,
        # so pooling them would rank each row against distances it was never
        # comparable to.
        rank_keys = ["split", "draw_set"] if "draw_set" in out.columns else ["split"]
        out["s_anchor_rank"] = out.groupby(rank_keys)["s_anchor"].rank(pct=True)
    else:
        out["s_anchor_rank"] = out["s_anchor"]

    questions_out = questions.copy()
    if len(keys) == 1:
        questions_out["anchor"] = questions_out["q_id"].map(amap)
    else:
        # The question table holds ONE anchor per question for reporting: the
        # classify pass's, since that is the only draw set every question has
        # and the one the Phase 0 dispersion diagnostics describe. Per-row
        # scoring above already used each row's own pass.
        cls = {q: t for (q, ds), t in amap.items() if ds == "classify"}
        questions_out["anchor"] = questions_out["q_id"].map(cls)
    return out, questions_out


def apply_tau(answers: pd.DataFrame, tau: float,
              column: str = "e_cos") -> pd.DataFrame:
    out = answers.copy()
    out["correct_cos"] = correct_by_cosine(out, tau, column=column)
    return out


def judge_nli(cfg: Config, answers: pd.DataFrame, questions: pd.DataFrame,
              nli=None) -> pd.DataFrame:
    """Bidirectional entailment against ``a_star``.

    Bidirectional, not one-way: "Paris" entails "a city in France" but the two
    are not the same answer, and a one-way test would accept the weaker string.
    """
    nli = nli if nli is not None else build_nli(cfg)
    thr = float(cfg.get("nli.entail_threshold"))
    q_idx = questions.set_index("q_id")["a_star"].astype(str)
    out = answers.copy()
    ans = out["answer"].astype(str).tolist()
    refs = [q_idx.get(q, "") for q in out["q_id"]]
    n = len(ans)

    # Every (answer, a_star) pair here is independent -- unlike the greedy loop
    # in cluster_answers, there is no order to preserve -- so the whole column
    # goes to the judge in one call and entailment_probs chunks it. At batch
    # size 1 an encoder head wastes almost all of a GPU, and this is 2N forward
    # passes over the full corpus.
    batch = getattr(nli, "entailment_probs", None)
    if batch is not None:
        probs = np.asarray(list(batch(ans + refs, refs + ans)), dtype=float)
        fwd, bwd = probs[:n], probs[n:]
    else:
        fwd = np.array([nli.entailment_prob(a, r) for a, r in zip(ans, refs)])
        bwd = np.array([nli.entailment_prob(r, a) for a, r in zip(ans, refs)])

    entailed = (np.minimum(fwd, bwd) >= thr) if n else np.zeros(0, dtype=bool)
    if n and "answer_equivalent" in out.columns:
        # Same override as the cosine criterion. "Orton" for "Joe Orton" is not
        # a bidirectional entailment and NLI rejects it too, so leaving this out
        # would make the two criteria disagree about a pair on which they have
        # no actual disagreement.
        entailed = entailed | out["answer_equivalent"].astype("boolean").fillna(False).to_numpy()
    out["correct_nli"] = entailed if n else []
    return out


def correctness_column(cfg: Config) -> str:
    primary = cfg.get("judge.primary")
    if primary not in ("cos", "nli"):
        raise ValueError(f"judge.primary must be cos or nli, got {primary!r}")
    return "correct_cos" if primary == "cos" else "correct_nli"


def attach_correct(cfg: Config, answers: pd.DataFrame) -> pd.DataFrame:
    """Materialise the single ``correct`` column every phase reads.

    A row whose criterion is missing is forced to False, because the downstream
    phases need a plain bool. That is a real assumption, not a formality: an
    UNDEFINED criterion is being recorded as a WRONG answer, which pushes p_hat
    down and beta up. It is reachable -- ``e_cos <= tau`` on a nullable Float64
    column yields pd.NA, and a failed parse still produces an empty answer that
    gets embedded -- so the affected rows are marked in ``correct_undefined``
    rather than absorbed, and a pitfall check reports the count.
    """
    col = correctness_column(cfg)
    if col not in answers.columns or answers[col].isna().all():
        raise ValueError(
            f"{col} is empty. Phase 0 must select tau_star (or flip judge.primary "
            "to nli) before any downstream phase can define correctness."
        )
    out = answers.copy()
    undefined = out[col].isna()
    out["correct_undefined"] = undefined.to_numpy(dtype=bool)
    out["correct"] = out[col].astype("boolean").fillna(False).astype(bool)
    return out
