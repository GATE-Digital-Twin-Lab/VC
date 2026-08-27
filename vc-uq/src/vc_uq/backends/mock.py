"""A simulated model with a KNOWN ground truth.

This is not a stand-in for results; it is the test harness for the analysis
code. Every phase downstream of generation is a pure function of the cached
tables, so a generator whose latent ``p_q``, error correlation, and VC
miscalibration are all set by hand lets each estimator be checked against the
value it is supposed to recover.

The defaults simulate the world the protocol predicts:

  * ``p_q`` varies widely across questions and MOVES WITH TEMPERATURE, while VC
    does not depend on T at all (the 8.1 non-invariance).
  * VC is drawn per QUESTION with near-zero within-question spread, so
    within-question AUROC sits at 0.5 (the 5.3/5.5 prediction).
  * Errors are common-mode: a wrong draw usually repeats the question's dominant
    wrong belief, so Corr(c_i, c_j) > 0 and the product rule over-claims (5.7/6.7).
  * The fabricated dataset has ``p_q = 0`` exactly, giving a non-empty U whose
    membership is known rather than inferred.

Every one of those is a knob. Setting ``vc_informativeness = 1.0`` and
``error_correlation = 0.0`` produces a world where VC *is* a good estimator --
which the tests use to confirm the analysis reports a positive result when one
exists, rather than always reporting failure.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .base import Generation, SamplingParams, TokenStats

SEM_TAG = re.compile(r"\[\[sem:([^\]]+)\]\]")

#: Discrete support VC is quantised onto -- the protocol expects 3-5 values in [0.7, 1].
DEFAULT_VC_SUPPORT = (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)


def _u01(*parts: object) -> float:
    """Deterministic uniform in [0,1) from arbitrary keys."""
    h = hashlib.blake2b(("::".join(str(p) for p in parts)).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big") / 2**64


def _rng(*parts: object) -> np.random.Generator:
    h = hashlib.blake2b(("::".join(str(p) for p in parts)).encode("utf-8"), digest_size=8)
    return np.random.default_rng(int.from_bytes(h.digest(), "big"))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class MockWorld:
    """Latent per-question truth. Deterministic in ``q_id``."""

    world_seed: int = 7
    t_ref: float = 0.8
    temp_sensitivity: float = 1.6      # how hard p_q moves with T (8.1 effect size)
    diversity_temp_sensitivity: float = 0.5
    p_q_beta: tuple[float, float] = (0.8, 0.9)   # Beta prior on baseline p_q
    error_correlation: float = 0.75    # P(wrong draw takes the DOMINANT wrong belief)
    low_diversity_share: float = 0.5   # fraction of questions that are mode-collapsed
    n_wrong_pool: int = 6
    vc_support: tuple[float, ...] = DEFAULT_VC_SUPPORT
    vc_informativeness: float = 0.35   # 0 = VC ignores p_q entirely, 1 = VC == p_q
    vc_floor: float = 0.7              # VC rarely goes below this -- the range collapse
    vc_within_sd: float = 0.01         # ~0 => VC is a property of the prompt
    #: How much vc_post reflects the correctness of the ANSWER just produced,
    #: rather than of the question. 0 reproduces the protocol's prediction that
    #: post-hoc VC is not reading its own output (within-question AUROC ~ 0.5);
    #: 1 gives a VC that genuinely does, which is what the analysis must be able
    #: to detect if it is to report a positive result when one exists.
    vc_reads_answer: float = 0.0
    vc_answer_signal: tuple[float, float] = (0.25, 0.95)   # (wrong, correct)
    vc_prompt_offsets: dict[str, float] | None = None
    scale_bias: dict[str, float] | None = None
    sycophancy_drop: float = 0.35
    forced_decode_shift: float = 0.5   # how much an injected VC moves p_q (8.5)
    prehoc_noise: float = 0.08         # vc_pre repeat-to-repeat instability (8.6c)
    prehoc_blind_to_fabricated: bool = True

    def baseline_p(self, q_id: str, dataset: str) -> float:
        if dataset == "fabricated":
            return 0.0            # p_q = 0 by construction
        a, b = self.p_q_beta
        return float(_rng(self.world_seed, "p", q_id).beta(a, b))

    def p_at(self, q_id: str, dataset: str, temperature: float,
             injected_vc: float | None = None) -> float:
        p0 = self.baseline_p(q_id, dataset)
        if p0 <= 0.0:
            return 0.0
        x = _logit(p0) - self.temp_sensitivity * (temperature - self.t_ref)
        if injected_vc is not None:
            # A confidence token in context perturbs the output distribution.
            x += self.forced_decode_shift * (injected_vc - 0.5) * 2.0
        return _sigmoid(x)

    def is_low_diversity(self, q_id: str) -> bool:
        return _u01(self.world_seed, "div", q_id) < self.low_diversity_share

    def n_modes(self, q_id: str, temperature: float) -> int:
        base = 1 if self.is_low_diversity(q_id) else self.n_wrong_pool
        extra = self.diversity_temp_sensitivity * max(0.0, temperature - self.t_ref)
        return int(max(1, round(base * (1.0 + extra))))

    def vc_level(self, q_id: str, dataset: str, prompt_variant: str = "vc_post_v1",
                 scale: str = "unit") -> float:
        """VC has no argument for temperature -- that is the point of 8.1."""
        p0 = self.baseline_p(q_id, dataset)
        noise = _u01(self.world_seed, "vc", q_id)
        raw = self.vc_informativeness * p0 + (1 - self.vc_informativeness) * noise
        raw = self.vc_floor + (1.0 - self.vc_floor) * raw
        offsets = self.vc_prompt_offsets or {}
        raw += offsets.get(prompt_variant, 0.0)
        raw += (self.scale_bias or {}).get(scale, 0.0)
        return float(np.clip(raw, 0.0, 1.0))

    def quantise_vc(self, value: float) -> float:
        support = np.asarray(self.vc_support, dtype=float)
        return float(support[int(np.argmin(np.abs(support - value)))])


class MockLM:
    """Sampling generator over :class:`MockWorld`."""

    name = "mock-lm"
    full_vocab_logits = True

    def __init__(self, world: MockWorld | None = None, vocab_size: int = 32000):
        self.world = world or MockWorld()
        self.vocab_size = vocab_size

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _field(messages: Sequence[dict], key: str) -> str | None:
        for m in messages:
            if m.get("role") == "user":
                mt = re.search(rf"{key}:\s*(.+)", m.get("content", ""), re.IGNORECASE)
                if mt:
                    return mt.group(1).strip()
        return None

    @staticmethod
    def _meta(messages: Sequence[dict]) -> dict[str, str]:
        """Mock-only side channel: q_id/dataset are carried in a system note."""
        meta: dict[str, str] = {}
        for m in messages:
            for k, v in re.findall(r"\[\[(\w+):([^\]]*)\]\]", m.get("content", "")):
                meta[k] = v
        return meta

    def _answer_text(self, q_id: str, dataset: str, is_correct: bool, mode: int) -> str:
        sem = f"{q_id}|gold" if is_correct else f"{q_id}|wrong{mode}"
        label = "the documented value" if is_correct else f"alternative reading {mode}"
        return f"{label} for {q_id} [[sem:{sem}]]"

    def _token_stats(self, n_tokens: int, *, concentration: float, key: str) -> TokenStats:
        rng = _rng(key)
        # Low entropy on mode-collapsed output: fluent, consistent, and possibly wrong.
        scale = float(np.clip(1.6 - 1.4 * concentration, 0.05, 3.0))
        ent = np.abs(rng.normal(scale, 0.25 * scale, size=n_tokens))
        probs = np.clip(np.exp(-ent) * rng.uniform(0.7, 1.0, size=n_tokens), 1e-6, 1.0)
        return TokenStats(entropies=ent, logprobs=np.log(probs), chosen_probs=probs)

    # -- interface ---------------------------------------------------------
    def generate(self, messages: Sequence[dict], *, params: SamplingParams,
                 seed: int) -> Generation:
        temperature, max_tokens = params.temperature, params.max_tokens
        # The simulated world responds to T only. Truncation knobs are recorded
        # so a test can assert they were threaded through, but they do not shape
        # the mock distribution -- pretending otherwise would invent an effect.
        self.last_params = params
        meta = self._meta(messages)
        q_id = meta.get("qid", self._field(messages, "Question") or "q?")
        dataset = meta.get("ds", "unknown")
        variant = meta.get("variant", "vc_post_v1")
        scale = meta.get("scale", "unit")
        kind = meta.get("kind", "post")
        injected = meta.get("injected")
        injected_vc = float(injected) if injected not in (None, "") else None
        w = self.world
        rng = _rng(w.world_seed, "draw", q_id, seed, temperature, variant, injected)

        if kind == "pre":
            return self._generate_pre(q_id, dataset, variant, seed, rng)

        p = w.p_at(q_id, dataset, temperature, injected_vc)
        is_correct = bool(rng.random() < p)
        if is_correct:
            mode, concentration = 0, 0.8
        else:
            n_modes = w.n_modes(q_id, temperature)
            # Common-mode failure: a wrong draw usually repeats the dominant belief.
            if rng.random() < w.error_correlation or n_modes == 1:
                mode = 1
            else:
                mode = int(rng.integers(1, n_modes + 1))
            concentration = 0.9 if w.is_low_diversity(q_id) else 0.35

        answer = self._answer_text(q_id, dataset, is_correct, mode)

        vc = w.vc_level(q_id, dataset, variant, scale)
        if w.vc_reads_answer > 0:
            wrong_v, right_v = w.vc_answer_signal
            answer_signal = right_v if is_correct else wrong_v
            vc = (1 - w.vc_reads_answer) * vc + w.vc_reads_answer * answer_signal
        if w.vc_within_sd > 0:
            vc = float(np.clip(vc + rng.normal(0.0, w.vc_within_sd), 0.0, 1.0))
        vc = w.quantise_vc(vc)
        if self._has_challenge(messages):
            vc = float(np.clip(vc - w.sycophancy_drop, 0.0, 1.0))

        text = f"Answer: {answer}\nConfidence: {self._render_vc(vc, scale)}"
        stats = self._token_stats(max(4, min(max_tokens, 24)),
                                  concentration=concentration,
                                  key=f"{q_id}:{seed}:{temperature}:{variant}")
        return Generation(text=text, stats=stats)

    def _generate_pre(self, q_id: str, dataset: str, variant: str, seed: int,
                      rng: np.random.Generator) -> Generation:
        w = self.world
        base = w.vc_level(q_id, dataset, "vc_post_v1")
        if dataset == "fabricated" and not w.prehoc_blind_to_fabricated:
            base = min(base, 0.2)     # a metacognitive model would drop here
        value = float(np.clip(base + rng.normal(0.0, w.prehoc_noise), 0.0, 1.0))
        value = w.quantise_vc(value)
        stats = self._token_stats(6, concentration=0.7, key=f"pre:{q_id}:{seed}")
        return Generation(text=f"Confidence: {value:g}", stats=stats)

    @staticmethod
    def _has_challenge(messages: Sequence[dict]) -> bool:
        return any("don't think that's right" in m.get("content", "").lower()
                   for m in messages)

    @staticmethod
    def _render_vc(vc: float, scale: str) -> str:
        if scale == "percent":
            return f"{vc * 100:g}%"
        if scale == "outof10":
            return f"{int(round(vc * 10))}"
        if scale == "verbal":
            from ..parsing import VERBAL_SCALE
            label = min(VERBAL_SCALE, key=lambda k: abs(VERBAL_SCALE[k] - vc))
            return label
        return f"{vc:g}"

    def teacher_force(self, messages: Sequence[dict], continuation: str) -> TokenStats:
        meta = self._meta(messages)
        key = f"tf:{meta.get('qid', '')}:{continuation}:{meta.get('variant', '')}"
        # A clean prompt is less peaked than the VC-augmented one; that gap is
        # exactly the confound section 4 asks to be measured.
        return self._token_stats(max(4, min(24, len(continuation.split()) + 2)),
                                 concentration=0.45, key=key)


class MockEmbedder:
    """Deterministic embeddings with a controllable signal-to-noise ratio.

    Strings carrying the same ``[[sem:...]]`` tag share a latent vector; the
    remainder of the string contributes noise. ``noise`` therefore sets how good
    the *instrument* is, independently of how good the model is -- which is what
    Phase 0 exists to measure.
    """

    name = "mock-embed"

    def __init__(self, dim: int = 64, noise: float = 0.35, seed: int = 11):
        self.dim = dim
        self.noise = noise
        self.seed = seed

    def _sem_vec(self, sem: str) -> np.ndarray:
        v = _rng(self.seed, "sem", sem).normal(size=self.dim)
        return v / np.linalg.norm(v)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            m = SEM_TAG.search(t or "")
            sem = m.group(1) if m else (t or "")
            base = self._sem_vec(sem)
            jitter = _rng(self.seed, "noise", t).normal(size=self.dim)
            jitter /= np.linalg.norm(jitter)
            v = base + self.noise * jitter
            # Anisotropy: real embeddings share a large common component, which
            # is what compresses cosine into a narrow band (Phase 0 note).
            v = v + 0.6 * self._sem_vec("__anisotropy__")
            out[i] = v / np.linalg.norm(v)
        return out


class MockNLI:
    """Entailment from the semantic tag, with configurable judge error."""

    name = "mock-nli"

    def __init__(self, error_rate: float = 0.05, seed: int = 13):
        self.error_rate = error_rate
        self.seed = seed

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        a = SEM_TAG.search(premise or "")
        b = SEM_TAG.search(hypothesis or "")
        same = (a.group(1) == b.group(1)) if (a and b) else (premise.strip() == hypothesis.strip())
        flip = _u01(self.seed, "nli", premise, hypothesis) < self.error_rate
        if flip:
            same = not same
        return 0.95 if same else 0.05
