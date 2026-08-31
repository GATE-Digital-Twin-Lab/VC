"""Backend construction from config."""

from __future__ import annotations

from ..config import Config
from .base import (EmbeddingBackend, Generation, LMBackend, NLIBackend,
                   SamplingParams, TokenStats, cosine_distance,
                   entropy_from_logits, pairwise_cosine_distance)

__all__ = [
    "Generation", "TokenStats", "SamplingParams", "LMBackend",
    "EmbeddingBackend", "NLIBackend",
    "cosine_distance", "pairwise_cosine_distance", "entropy_from_logits",
    "build_lm", "build_embedder", "build_nli",
]


def _require_mock_model(cfg: Config, what: str, key: str) -> None:
    """The mock embedder and mock NLI can only score mock output.

    Both work by reading a hidden ``[[sem:...]]`` tag that the simulated world
    writes into every answer, and that ``datasets.build_questions`` appends to
    ``a_star`` -- but only when ``model.backend`` is mock. Pair one of them with
    a real model and the tag is absent everywhere: they fall back to hashing raw
    text, every similarity becomes noise, and the Phase 0 gate fails while
    blaming the instrument. Nothing crashes, so refuse the combination here.
    """
    backend = cfg.get("model.backend")
    if backend != "mock":
        raise ValueError(
            f"{key}=mock cannot be used with model.backend={backend!r}. The mock "
            f"{what} scores by reading the simulated world's [[sem:...]] tag, which "
            "real model output does not carry, so every similarity it returns would "
            "be noise. Use a real embedding/NLI backend, or set model.backend=mock.")


def build_lm(cfg: Config):
    backend = cfg.get("model.backend")
    if backend == "mock":
        from .mock import MockLM, MockWorld
        return MockLM(MockWorld(**cfg.get("model.mock", {})))
    if backend == "llamacpp":
        from .llamacpp import LlamaCppConfig, LlamaCppLM
        return LlamaCppLM(LlamaCppConfig.from_dict(cfg.section("model.llamacpp")),
                          name=cfg.get("model.name"))
    raise ValueError(f"unknown model.backend {backend!r}")


def build_embedder(cfg: Config):
    backend = cfg.get("embedding.backend")
    if backend == "mock":
        _require_mock_model(cfg, "embedder", "embedding.backend")
        from .mock import MockEmbedder
        return MockEmbedder(dim=cfg.get("embedding.dim"),
                            **cfg.get("embedding.mock", {}))
    if backend == "llamacpp":
        from .llamacpp import LlamaCppConfig, LlamaCppEmbedder
        return LlamaCppEmbedder(LlamaCppConfig.from_dict(cfg.section("embedding.llamacpp")),
                                name=cfg.get("embedding.name"))
    if backend == "sentence_transformers":
        from .hf_nli import STEmbedder
        return STEmbedder(**cfg.section("embedding.sentence_transformers"))
    raise ValueError(f"unknown embedding.backend {backend!r}")


def build_nli(cfg: Config, lm=None):
    backend = cfg.get("nli.backend")
    if backend == "mock":
        _require_mock_model(cfg, "NLI judge", "nli.backend")
        from .mock import MockNLI
        return MockNLI(**cfg.get("nli.mock", {}))
    if backend == "hf":
        from .hf_nli import HFNLI
        return HFNLI(**cfg.section("nli.hf"))
    if backend == "llm":
        from .llamacpp import LlamaCppNLI
        return LlamaCppNLI(lm if lm is not None else build_lm(cfg),
                           max_tokens=cfg.get("nli.llm.max_tokens"))
    raise ValueError(f"unknown nli.backend {backend!r}")
