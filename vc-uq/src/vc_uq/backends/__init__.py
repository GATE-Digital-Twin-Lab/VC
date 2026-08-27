"""Backend construction from config."""

from __future__ import annotations

from ..config import Config
from .base import (EmbeddingBackend, Generation, LMBackend, NLIBackend,
                   SamplingParams, TokenStats, cosine_distance,
                   entropy_bounds_from_topk, entropy_from_logits,
                   pairwise_cosine_distance)

__all__ = [
    "Generation", "TokenStats", "SamplingParams", "LMBackend",
    "EmbeddingBackend", "NLIBackend",
    "cosine_distance", "pairwise_cosine_distance", "entropy_from_logits",
    "entropy_bounds_from_topk", "build_lm", "build_embedder", "build_nli",
]


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
        from .mock import MockEmbedder
        return MockEmbedder(dim=cfg.get("embedding.dim"),
                            **cfg.get("embedding.mock", {}))
    if backend == "llamacpp":
        from .llamacpp import LlamaCppConfig, LlamaCppEmbedder
        return LlamaCppEmbedder(LlamaCppConfig.from_dict(cfg.section("embedding.llamacpp")),
                                name=cfg.get("embedding.name"))
    if backend == "sentence_transformers":
        from .hf_nli import STEmbedder
        return STEmbedder(cfg.get("embedding.sentence_transformers.name"))
    raise ValueError(f"unknown embedding.backend {backend!r}")


def build_nli(cfg: Config, lm=None):
    backend = cfg.get("nli.backend")
    if backend == "mock":
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
