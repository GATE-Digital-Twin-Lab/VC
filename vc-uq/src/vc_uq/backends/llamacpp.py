"""llama.cpp backend, in-process via ``llama-cpp-python``.

The in-process binding is used rather than ``llama-server`` for one reason: the
HTTP API returns only the top-N token probabilities, and ``h_tok`` is defined
over the FULL next-token distribution (protocol section 0). Truncated top-k
would turn the token-level baseline family into an estimate with an
uncontrolled downward bias exactly where the distribution is flat -- which is
the regime the baseline is meant to detect. In-process, ``scores`` exposes the
whole logit vector.

Teacher forcing (section 4) also needs full logits over a sequence the model did
not itself produce, which the server cannot do at all.

The import is deferred so the rest of the package -- and the entire test suite
against the mock backend -- runs without llama-cpp-python installed.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .base import Generation, TokenStats


def _require_llama_cpp():
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "llama-cpp-python is required for the llamacpp backend.\n"
            "  CUDA build:  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python\n"
            "Install it on the generation machine; analysis phases do not need it."
        ) from exc
    return Llama


@dataclass
class LlamaCppConfig:
    repo_id: str | None = None
    filename: str | None = None
    model_path: str | None = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_batch: int = 512
    main_gpu: int = 0
    tensor_split: list[float] | None = None
    flash_attn: bool = True
    verbose: bool = False
    chat_format: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LlamaCppConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class LlamaCppLM:
    """Sampling + exact token statistics from a GGUF model."""

    full_vocab_logits = True

    def __init__(self, cfg: LlamaCppConfig, *, name: str = "llamacpp"):
        Llama = _require_llama_cpp()
        self.cfg = cfg
        self.name = name
        kwargs: dict[str, Any] = dict(
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            n_batch=cfg.n_batch,
            main_gpu=cfg.main_gpu,
            tensor_split=cfg.tensor_split,
            flash_attn=cfg.flash_attn,
            verbose=cfg.verbose,
            # Required for teacher forcing: keeps logits for every position, not
            # just the last one.
            logits_all=True,
        )
        if cfg.chat_format:
            kwargs["chat_format"] = cfg.chat_format
        if cfg.model_path:
            self._llm = Llama(model_path=cfg.model_path, **kwargs)
        elif cfg.repo_id and cfg.filename:
            self._llm = Llama.from_pretrained(repo_id=cfg.repo_id, filename=cfg.filename,
                                              **kwargs)
        else:
            raise ValueError("llamacpp backend needs model_path, or repo_id + filename")

    # -- prompt handling ---------------------------------------------------
    def _render(self, messages: Sequence[dict]) -> str:
        """Apply the GGUF's own chat template.

        Going through the model's template rather than hand-concatenating keeps
        the prompt identical between the sampling pass and the teacher-forced
        clean-prompt pass, which is the only way the two h_tok columns are
        comparable.
        """
        handler = self._llm.chat_handler or getattr(self._llm, "_chat_handler", None)
        fmt = getattr(handler, "to_chat_completion_prompt", None)
        if fmt is not None:  # pragma: no cover - depends on binding version
            return fmt(list(messages))
        from llama_cpp.llama_chat_format import format_llama3  # type: ignore
        with contextlib.suppress(Exception):
            return format_llama3(messages=list(messages)).prompt
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def _tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)

    # -- interface ---------------------------------------------------------
    def generate(self, messages: Sequence[dict], *, temperature: float, seed: int,
                 max_tokens: int = 128, stop: Sequence[str] | None = None) -> Generation:
        self._llm.set_seed(int(seed))
        out = self._llm.create_chat_completion(
            messages=list(messages),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            stop=list(stop) if stop else None,
            seed=int(seed),
        )
        choice = out["choices"][0]
        text = choice["message"]["content"] or ""
        finish = choice.get("finish_reason", "stop")
        # Statistics come from a teacher-forced pass over the produced text under
        # the SAME prompt, so entropies are full-vocab rather than post-sampling.
        stats = self.teacher_force(messages, text)
        return Generation(text=text, stats=stats, finish_reason=finish)

    def teacher_force(self, messages: Sequence[dict], continuation: str) -> TokenStats:
        prompt = self._render(messages)
        prompt_ids = self._tokenize(prompt, add_bos=True)
        cont_ids = self._tokenize(continuation, add_bos=False)
        if not cont_ids:
            empty = np.zeros(0)
            return TokenStats(entropies=empty, logprobs=empty, chosen_probs=empty)

        self._llm.reset()
        self._llm.eval(prompt_ids + cont_ids)
        scores = np.asarray(self._llm.scores, dtype=np.float64)
        n_eval = len(prompt_ids) + len(cont_ids)
        # Position i predicts token i+1, so the distribution scoring the first
        # continuation token sits at index len(prompt_ids) - 1.
        start = len(prompt_ids) - 1
        rows = scores[start:n_eval - 1]
        if rows.shape[0] != len(cont_ids):  # pragma: no cover - binding drift
            raise RuntimeError(
                f"expected {len(cont_ids)} logit rows, got {rows.shape[0]}; "
                "the binding may not have been built with logits_all"
            )

        ent = np.empty(len(cont_ids))
        logp = np.empty(len(cont_ids))
        cp = np.empty(len(cont_ids))
        for i, tok in enumerate(cont_ids):
            z = rows[i] - rows[i].max()
            p = np.exp(z)
            p /= p.sum()
            nz = p > 0
            ent[i] = -np.sum(p[nz] * np.log(p[nz]))
            cp[i] = max(float(p[tok]), 1e-12)
            logp[i] = np.log(cp[i])
        return TokenStats(entropies=ent, logprobs=logp, chosen_probs=cp)


class LlamaCppEmbedder:
    """GGUF embedding model (e.g. nomic-embed) through the same binding."""

    def __init__(self, cfg: LlamaCppConfig, *, name: str = "llamacpp-embed"):
        Llama = _require_llama_cpp()
        self.name = name
        kwargs = dict(n_ctx=cfg.n_ctx, n_gpu_layers=cfg.n_gpu_layers,
                      embedding=True, verbose=cfg.verbose)
        if cfg.model_path:
            self._llm = Llama(model_path=cfg.model_path, **kwargs)
        elif cfg.repo_id and cfg.filename:
            self._llm = Llama.from_pretrained(repo_id=cfg.repo_id, filename=cfg.filename,
                                              **kwargs)
        else:
            raise ValueError("llamacpp embedder needs model_path, or repo_id + filename")
        self.dim = int(self._llm.n_embd())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._llm.create_embedding(list(texts))
        rows = [np.asarray(d["embedding"], dtype=np.float64) for d in vecs["data"]]
        rows = [r.mean(axis=0) if r.ndim == 2 else r for r in rows]   # pooled or per-token
        return np.vstack(rows)


class LlamaCppNLI:
    """Prompted entailment judge, for when no NLI encoder is available.

    Weaker than a trained MNLI head and reported as such; the encoder path in
    ``backends/hf_nli.py`` is preferred wherever transformers can be installed.
    """

    def __init__(self, lm: LlamaCppLM, *, max_tokens: int = 8, seed: int = 0):
        self.lm = lm
        self.name = f"{lm.name}-nli"
        self.max_tokens = max_tokens
        self.seed = seed

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        from ..prompts import build_nli
        gen = self.lm.generate(build_nli(premise, hypothesis), temperature=0.0,
                               seed=self.seed, max_tokens=self.max_tokens)
        text = gen.text.strip().lower()
        if text.startswith("entail"):
            return 1.0
        if text.startswith("contradict"):
            return 0.0
        return 0.5
