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

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .base import Generation, SamplingParams, TokenStats


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


def _jinja_formatter(template: str, *, bos_token: str, eos_token: str):
    """The model's own Jinja chat template, wrapped by llama-cpp-python.

    Split out so the resolution ORDER below can be tested without
    llama-cpp-python installed.
    """
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter  # type: ignore

    formatter = Jinja2ChatFormatter(template=template, bos_token=bos_token,
                                    eos_token=eos_token)
    return lambda msgs: formatter(messages=list(msgs)).prompt


def _named_formatter(name: str):
    """A chat format requested explicitly via ``model.llamacpp.chat_format``."""
    from llama_cpp import llama_chat_format  # type: ignore

    flat = name.lower().replace("-", "").replace("_", "")
    snake = name.lower().replace("-", "_")
    for candidate in (f"format_{flat}", f"format_{snake}"):
        fn = getattr(llama_chat_format, candidate, None)
        if fn is not None:
            return lambda msgs: fn(messages=list(msgs)).prompt
    return None


def _special_token(llm, which: str) -> str:
    tok = llm.token_bos() if which == "bos" else llm.token_eos()
    getter = getattr(getattr(llm, "_model", None), "token_get_text", None)
    if getter is not None:
        return str(getter(tok))
    return llm.detokenize([tok]).decode("utf-8", errors="replace")


def resolve_chat_formatter(llm, *, chat_format: str | None = None):
    """Return ``(render(messages) -> str, source)``, or raise.

    The refusal at the end is the point. An earlier version fell back to
    ``format_llama3`` and then to hand-concatenated "role: content" text,
    neither of which raises. With a non-Llama GGUF the first emits Llama-3
    control tokens the model has never seen in that arrangement and the second
    emits no control tokens at all -- so the run completes, the tables fill, and
    every token statistic describes the wrong context. Since the token-entropy
    family is what VC is being COMPARED AGAINST, a silently degraded baseline
    flatters VC. Better to stop and say so.
    """
    handler = getattr(llm, "chat_handler", None) or getattr(llm, "_chat_handler", None)
    fmt = getattr(handler, "to_chat_completion_prompt", None)
    if fmt is not None:
        return (lambda msgs: fmt(list(msgs))), "chat_handler.to_chat_completion_prompt"

    template = (getattr(llm, "metadata", None) or {}).get("tokenizer.chat_template")
    if template:
        return (_jinja_formatter(template,
                                 bos_token=_special_token(llm, "bos"),
                                 eos_token=_special_token(llm, "eos")),
                "gguf tokenizer.chat_template")

    if chat_format:
        named = _named_formatter(chat_format)
        if named is not None:
            return named, f"chat_format={chat_format!r}"
        raise RuntimeError(
            f"model.llamacpp.chat_format={chat_format!r} matches no formatter in "
            "llama_cpp.llama_chat_format. Use a name that does, or load a GGUF "
            "that carries its own tokenizer.chat_template.")

    raise RuntimeError(
        "cannot reconstruct this model's chat prompt: its chat handler exposes no "
        "to_chat_completion_prompt and the GGUF carries no tokenizer.chat_template. "
        "teacher_force must rebuild exactly the string create_chat_completion used, "
        "so guessing a template would compute every token statistic under the wrong "
        "context. Set model.llamacpp.chat_format to the format this model was "
        "trained with.")


class LlamaCppLM:
    """Sampling + exact token statistics from a GGUF model."""

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

        # Resolved eagerly: an unrenderable prompt must surface now, not after
        # the first hours of generation.
        self._format_prompt, self.prompt_template_source = resolve_chat_formatter(
            self._llm, chat_format=cfg.chat_format)

    # -- prompt handling ---------------------------------------------------
    def _render(self, messages: Sequence[dict]) -> str:
        """Apply the loaded model's own chat template.

        ``create_chat_completion`` formats the messages internally and
        ``teacher_force`` has to reconstruct EXACTLY that string: a different
        prompt is different conditioning, so every h_tok, logp_mean and
        min_token_p would describe a context the answer was never produced in.
        The formatter is resolved once at construction by
        :func:`resolve_chat_formatter`, which refuses to guess.
        """
        return self._format_prompt(list(messages))

    def _tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)

    # -- interface ---------------------------------------------------------
    def generate(self, messages: Sequence[dict], *, params: SamplingParams,
                 seed: int) -> Generation:
        self._llm.set_seed(int(seed))
        # Every truncation knob is passed EXPLICITLY. llama.cpp's defaults are
        # top_k=40, top_p=0.95, min_p=0.05, repeat_penalty=1.1, so omitting them
        # would sample from a truncated, repetition-penalised distribution while
        # the config claims the decoder is unrestricted. That damps the effect of
        # raising T and understates the 8.1 result.
        out = self._llm.create_chat_completion(
            messages=list(messages),
            temperature=float(params.temperature),
            max_tokens=int(params.max_tokens),
            stop=list(params.stop) if params.stop else None,
            top_p=float(params.top_p),
            top_k=int(params.top_k),
            min_p=float(params.min_p),
            repeat_penalty=float(params.repeat_penalty),
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
        n_eval = len(prompt_ids) + len(cont_ids)
        # Position i predicts token i+1, so the distribution scoring the first
        # continuation token sits at index len(prompt_ids) - 1.
        start = len(prompt_ids) - 1
        # Slice BEFORE converting dtype. ``scores`` is [n_ctx, n_vocab] float32:
        # 2.1 GB at n_ctx=4096 for Llama-3.1's 128k vocabulary. Converting the
        # whole buffer to float64 would allocate 4.2 GB on EVERY draw in order
        # to read a few dozen rows, and there are two such passes per draw.
        rows = np.asarray(self._llm.scores[start:n_eval - 1], dtype=np.float64)
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
        self.n_calls = 0
        self.n_unparsed = 0

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        """P(premise entails hypothesis), from a three-way verdict.

        The judge emits a LABEL, not a distribution, so the only honest mapping
        is 1.0 for entailment and 0.0 for the two non-entailment labels. An
        earlier version returned 0.5 for neutral, which was worse than merely
        vague: ``cluster.cluster_answers`` merges when ``min(fwd, bwd) >=
        nli.entail_threshold`` and that threshold is 0.5, so every neutral
        verdict silently merged two distinct answers into one cluster --
        deflating H_sem and inflating largest_cluster_share, which are exactly
        the quantities the diversity 2x2 (6.6) and the semantic-entropy baseline
        are built from.

        Unparseable output is a failure, not a verdict. It scores 0.0 -- the
        conservative direction, since refusing to merge leaves the answers
        distinguishable -- and is counted so the rate can be reported rather
        than absorbed.
        """
        from ..prompts import build_nli
        # Greedy and unrestricted: an entailment verdict must not vary run to run.
        gen = self.lm.generate(
            build_nli(premise, hypothesis),
            params=SamplingParams(temperature=0.0, max_tokens=self.max_tokens),
            seed=self.seed)
        text = gen.text.strip().lower()
        self.n_calls += 1
        if text.startswith("entail"):
            return 1.0
        if text.startswith("contradict") or text.startswith("neutral"):
            return 0.0
        self.n_unparsed += 1
        return 0.0

    def audit(self) -> dict[str, float | int]:
        """Verdict-parse failure rate, reported alongside the clustering."""
        return {"judge": self.name, "n_calls": self.n_calls,
                "n_unparsed": self.n_unparsed,
                "unparsed_rate": (self.n_unparsed / self.n_calls) if self.n_calls else 0.0}
