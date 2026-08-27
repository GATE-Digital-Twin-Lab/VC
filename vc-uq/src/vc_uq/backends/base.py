"""Backend interfaces.

Three capabilities are needed and they are kept separate because they are
served by different models:

  LMBackend        -- sampling + full next-token distributions + teacher forcing
  EmbeddingBackend -- vectors for e_cos (criterion) and s_anchor (score)
  NLIBackend       -- bidirectional entailment for clustering and correct_nli

The token-statistics contract is the demanding one. ``h_tok`` is defined over
the FULL next-token distribution (protocol section 0), which rules out any
transport that returns only a truncated top-k. Where a backend can only supply
top-k it must say so via ``full_vocab_logits = False`` so the analysis can label
the column as an estimate rather than the quantity in the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class TokenStats:
    """Per-position statistics for one generated (or forced) sequence."""

    entropies: np.ndarray        # [n_tokens] entropy of the full next-token distribution
    logprobs: np.ndarray         # [n_tokens] log p of the chosen token
    chosen_probs: np.ndarray     # [n_tokens] p of the chosen token
    full_vocab: bool = True      # False => entropies are truncated-top-k estimates

    def __post_init__(self) -> None:
        n = len(self.entropies)
        if not (len(self.logprobs) == len(self.chosen_probs) == n):
            raise ValueError("token statistic arrays must be the same length")

    @property
    def n_tokens(self) -> int:
        return int(len(self.entropies))

    def summary(self, prefix: str = "") -> dict[str, float | int | None]:
        if self.n_tokens == 0:
            keys = ["h_tok_mean", "h_tok_max", "logp_mean", "min_token_p", "n_tokens"]
            return {f"{prefix}{k}": None for k in keys}
        return {
            f"{prefix}h_tok_mean": float(np.mean(self.entropies)),
            f"{prefix}h_tok_max": float(np.max(self.entropies)),
            f"{prefix}logp_mean": float(np.mean(self.logprobs)),
            f"{prefix}min_token_p": float(np.min(self.chosen_probs)),
            f"{prefix}n_tokens": int(self.n_tokens),
        }


@dataclass
class Generation:
    text: str
    stats: TokenStats
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass(frozen=True)
class SamplingParams:
    """The complete decoder specification for one call.

    Bundled rather than passed as loose keyword arguments because the truncation
    knobs are load-bearing and easy to forget. ``p_q`` is *definitionally* a
    function of the decoder (protocol 8.1), and the temperature sweep is only a
    measurement of ``T`` if ``T`` is the only thing that varies. Backend defaults
    are not neutral -- llama.cpp ships ``top_k=40, top_p=0.95, min_p=0.05,
    repeat_penalty=1.1`` -- so leaving these unset would silently truncate the
    tail, damp the effect of raising ``T``, and understate the headline result.

    The defaults here are the unrestricted decoder the protocol requires.
    """

    temperature: float
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = 0            # 0 = disabled
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    stop: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, cfg, *, temperature: float | None = None,
                    max_tokens: int | None = None) -> "SamplingParams":
        gen = cfg.section("model.generation")
        return cls(
            temperature=float(temperature if temperature is not None
                              else cfg.get("generation.temperature")),
            max_tokens=int(max_tokens if max_tokens is not None
                           else gen["max_tokens"]),
            top_p=float(gen["top_p"]), top_k=int(gen["top_k"]),
            min_p=float(gen["min_p"]),
            repeat_penalty=float(gen["repeat_penalty"]),
            stop=tuple(gen.get("stop") or ()),
        )

    def replace(self, **kw) -> "SamplingParams":
        from dataclasses import replace as _replace
        return _replace(self, **kw)

    @property
    def truncates_tail(self) -> bool:
        """True if anything other than temperature is shaping the distribution."""
        return not (self.top_p >= 1.0 and self.top_k <= 0
                    and self.min_p <= 0.0 and self.repeat_penalty == 1.0)

    def truncation_reason(self) -> str:
        bad = []
        if self.top_p < 1.0:
            bad.append(f"top_p={self.top_p}")
        if self.top_k > 0:
            bad.append(f"top_k={self.top_k}")
        if self.min_p > 0.0:
            bad.append(f"min_p={self.min_p}")
        if self.repeat_penalty != 1.0:
            bad.append(f"repeat_penalty={self.repeat_penalty}")
        return ", ".join(bad)


@runtime_checkable
class LMBackend(Protocol):
    name: str
    full_vocab_logits: bool

    def generate(self, messages: Sequence[dict], *, params: SamplingParams,
                 seed: int) -> Generation: ...

    def teacher_force(self, messages: Sequence[dict], continuation: str) -> TokenStats:
        """Token statistics for ``continuation`` under ``messages``.

        Used to recover h_tok under a clean, non-VC-augmented prompt for an
        answer that was produced under the augmented one.
        """
        ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class NLIBackend(Protocol):
    name: str

    def entailment_prob(self, premise: str, hypothesis: str) -> float: ...


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - cosine similarity, row-wise. Shapes broadcast as (n, d) x (n, d)."""
    a = np.atleast_2d(a).astype(np.float64)
    b = np.atleast_2d(b).astype(np.float64)
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    sim = np.sum((a / na) * (b / nb), axis=1)
    return 1.0 - sim


def pairwise_cosine_distance(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x).astype(np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    xn = x / n
    return np.clip(1.0 - xn @ xn.T, 0.0, 2.0)


def entropy_from_logits(logits: np.ndarray) -> float:
    """Shannon entropy (nats) of softmax(logits) over the full vocabulary."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))


def entropy_bounds_from_topk(probs: Sequence[float], vocab_size: int) -> tuple[float, float]:
    """Entropy bounds when only the top-k probabilities are observable.

    Lower bound puts all residual mass on a single unseen token; upper bound
    spreads it uniformly over the remaining vocabulary. Reported as an interval
    so a truncated transport never masquerades as a full-vocab measurement.
    """
    p = np.asarray(list(probs), dtype=np.float64)
    p = p[p > 0]
    head = float(-np.sum(p * np.log(p)))
    rest = max(0.0, 1.0 - float(p.sum()))
    n_rest = max(1, vocab_size - len(p))
    if rest <= 0:
        return head, head
    lower = head - rest * np.log(rest)
    upper = head - rest * np.log(rest / n_rest)
    return float(lower), float(upper)
