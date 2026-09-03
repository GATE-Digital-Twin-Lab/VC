"""Encoder NLI head (DeBERTa-MNLI by default).

Clustering must use bidirectional entailment, never cosine -- cosine is blind to
negation, and "Paris" vs "not Paris" scoring ~0.9 would merge a right and a
wrong answer into one cluster and destroy the diversity 2x2 (protocol 6.6).

Imported lazily so the package works without torch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class HFNLI:
    def __init__(self, name: str = "microsoft/deberta-large-mnli",
                 batch_size: int = 16, device: str = "cuda"):
        try:
            import torch
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "transformers + torch are required for the hf NLI backend; "
                "use nli.backend: llm to route entailment through llama.cpp instead."
            ) from exc

        self.name = name
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForSequenceClassification.from_pretrained(name)
        self._model.to(self.device).eval()
        labels = {v.lower(): k for k, v in self._model.config.id2label.items()}
        if "entailment" not in labels:
            raise ValueError(f"{name} does not expose an 'entailment' label: {labels}")
        self._entail_idx = labels["entailment"]

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        return float(self.entailment_probs([premise], [hypothesis])[0])

    def entailment_probs(self, premises: Sequence[str],
                         hypotheses: Sequence[str]) -> np.ndarray:
        import torch

        out = np.empty(len(premises), dtype=np.float64)
        for start in range(0, len(premises), self.batch_size):
            sl = slice(start, start + self.batch_size)
            enc = self._tok(list(premises[sl]), list(hypotheses[sl]),
                            return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, self._entail_idx]
            out[sl] = probs.detach().cpu().numpy()
        return out


class STEmbedder:
    """sentence-transformers embeddings, as an alternative to the GGUF path."""

    def __init__(self, name: str = "sentence-transformers/all-mpnet-base-v2",
                 batch_size: int = 32, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("sentence-transformers is not installed") from exc
        self.name = name
        self.batch_size = batch_size
        # Explicit device, because the default lands on cuda:0 -- which on this
        # rig is the card holding the generator's GGUF. Two large models on one
        # card is an OOM in the middle of a run, not a slowdown.
        self.device = device
        self._model = SentenceTransformer(name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), convert_to_numpy=True,
                                             batch_size=self.batch_size),
                          dtype=np.float64)
