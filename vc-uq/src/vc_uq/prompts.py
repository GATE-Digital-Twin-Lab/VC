"""Elicitation prompts.

Pre-hoc and post-hoc VC are *different constructs* and are kept strictly
separate (protocol section 4):

  vc_pre  -- "can you answer this?", asked with the question only and NO answer
             in context. Contaminating this with an answer destroys the construct.
  vc_post -- confidence in a specific produced answer, elicited in the same call
             that produces it.

``answer_clean_v1`` carries no confidence instruction at all; it exists so token
entropy can be measured under a non-augmented prompt by teacher forcing, since
VC elicitation itself changes the next-token distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

Scale = Literal["unit", "percent", "verbal", "outof10"]

#: Ordered ladder for verbal confidence, midpoints of ten equal bins. No prompt
#: asks for a word -- every elicitation asks for a number in [0, 1] -- but models
#: answer "fairly confident" anyway, and ``parsing`` maps such an answer onto the
#: ladder rather than discarding it. Kept here with the other elicitation
#: vocabulary. Insertion order is the ladder.
VERBAL_SCALE: dict[str, float] = {
    "impossible": 0.00,
    "doubtful": 0.10,
    "unlikely": 0.20,
    "uncertain": 0.30,
    "even": 0.50,
    "likely": 0.65,
    "probable": 0.75,
    "confident": 0.85,
    "highly confident": 0.95,
    "certain": 1.00,
}

Message = dict[str, str]


@dataclass(frozen=True)
class PromptSpec:
    variant: str
    kind: Literal["post", "pre", "clean", "nli"]
    scale: Scale
    build: Callable[..., list[Message]]
    elicits_vc: bool = True
    notes: str = ""


REGISTRY: dict[str, PromptSpec] = {}


def register(spec: PromptSpec) -> PromptSpec:
    if spec.variant in REGISTRY:
        raise ValueError(f"duplicate prompt variant {spec.variant!r}")
    REGISTRY[spec.variant] = spec
    return spec


def get(variant: str) -> PromptSpec:
    if variant not in REGISTRY:
        raise KeyError(f"unknown prompt variant {variant!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[variant]


def variants(kind: str | None = None) -> list[str]:
    return sorted(v for v, s in REGISTRY.items() if kind is None or s.kind == kind)


#: Every config key that names a prompt variant.
VARIANT_CONFIG_KEYS: tuple[str, ...] = (
    "generation.prompt_variant_post",
    "generation.prompt_variant_pre",
    "generation.prompt_variant_clean",
    "phase5.paraphrase.variants",
    "nli.llm.prompt_variant",
)


def unknown_config_variants(cfg) -> dict[str, list[str]]:
    """Config-named variants that are not registered, keyed by config path."""
    out: dict[str, list[str]] = {}
    for key in VARIANT_CONFIG_KEYS:
        value = cfg.get(key, None)
        if value is None:
            continue
        names = list(value) if isinstance(value, (list, tuple)) else [value]
        missing = [str(n) for n in names if n and str(n) not in REGISTRY]
        if missing:
            out[key] = missing
    return out


def check_config_variants(cfg) -> None:
    """Fail on a mistyped variant name BEFORE any generation happens.

    Without this, ``phase5.paraphrase.variants: [vc_post_v11]`` raises a KeyError
    from inside Phase 5 -- after generation, the gate, survival and CLM have all
    run. On a real model that is hours of GPU time spent before the typo
    surfaces.
    """
    bad = unknown_config_variants(cfg)
    if bad:
        detail = "; ".join(f"{k}: {v}" for k, v in bad.items())
        raise KeyError(
            f"unknown prompt variant(s) named in config -- {detail}. "
            f"Registered: {sorted(REGISTRY)}")


# --------------------------------------------------------------------------
# Post-hoc: answer and confidence in one call.
# The ten variants are semantically equivalent paraphrases (protocol 8.2) --
# per-question variance across them is test-retest reliability.
# --------------------------------------------------------------------------

_POST_BODIES: dict[str, str] = {
    "vc_post_v1": (
        "Answer the question. Then state your confidence that your answer is correct.\n"
        "Reply in exactly this format:\n"
        "Answer: <your answer>\n"
        "Confidence: <a number between 0 and 1>"
    ),
    "vc_post_v2": (
        "Give your answer to the question below, followed by how confident you are "
        "that it is right.\n"
        "Format:\n"
        "Answer: <your answer>\n"
        "Confidence: <number from 0 to 1>"
    ),
    "vc_post_v3": (
        "Respond with your best answer and the probability that it is correct.\n"
        "Answer: <your answer>\n"
        "Confidence: <probability between 0 and 1>"
    ),
    "vc_post_v4": (
        "First answer the question. Then report, as a number in [0, 1], how likely "
        "it is that the answer you just gave is correct.\n"
        "Answer: <your answer>\n"
        "Confidence: <number>"
    ),
    "vc_post_v5": (
        "Answer the following. Afterwards, rate your certainty in that specific "
        "answer on a 0 to 1 scale.\n"
        "Answer: <your answer>\n"
        "Confidence: <0-1>"
    ),
    "vc_post_v6": (
        "Provide an answer, then a confidence score between 0 and 1 reflecting how "
        "sure you are of it.\n"
        "Answer: <your answer>\n"
        "Confidence: <score>"
    ),
    "vc_post_v7": (
        "Answer the question below. Then say how much you would trust that answer, "
        "expressed as a number between 0 and 1.\n"
        "Answer: <your answer>\n"
        "Confidence: <number>"
    ),
    "vc_post_v8": (
        "Write your answer. Then estimate the chance that it is the correct one, "
        "as a decimal between 0 and 1.\n"
        "Answer: <your answer>\n"
        "Confidence: <decimal>"
    ),
    "vc_post_v9": (
        "Answer, then quantify your belief that the answer is correct using a "
        "number in the range 0 to 1.\n"
        "Answer: <your answer>\n"
        "Confidence: <number in [0,1]>"
    ),
    "vc_post_v10": (
        "Reply with an answer to the question and, on the next line, your "
        "self-assessed probability of correctness between 0 and 1.\n"
        "Answer: <your answer>\n"
        "Confidence: <probability>"
    ),
}


def _post_builder(body: str) -> Callable[..., list[Message]]:
    def build(question: str, **_: object) -> list[Message]:
        return [
            {"role": "system", "content": body},
            {"role": "user", "content": f"Question: {question}"},
        ]
    return build


for _variant, _body in _POST_BODIES.items():
    register(PromptSpec(variant=_variant, kind="post", scale="unit",
                        build=_post_builder(_body)))


# --------------------------------------------------------------------------
# Pre-hoc: question only, no answer generated, no answer in context.
# --------------------------------------------------------------------------

_PRE_BODY = (
    "You will be shown a question. Do NOT answer it. State only how confident you "
    "are that you could answer it correctly if asked.\n"
    "Reply in exactly this format:\n"
    "Confidence: <a number between 0 and 1>"
)


def _pre_build(question: str, **_: object) -> list[Message]:
    return [
        {"role": "system", "content": _PRE_BODY},
        {"role": "user", "content": f"Question: {question}"},
    ]


register(PromptSpec(variant="vc_pre_v1", kind="pre", scale="unit", build=_pre_build,
                    notes="prospective feeling-of-knowing; no answer in context"))


# --------------------------------------------------------------------------
# Clean prompt: no confidence instruction. Used to teacher-force the same answer
# tokens and recover unconfounded token entropy (protocol section 4).
# --------------------------------------------------------------------------

_CLEAN_BODY = "Answer the question concisely."


def _clean_build(question: str, **_: object) -> list[Message]:
    return [
        {"role": "system", "content": _CLEAN_BODY},
        {"role": "user", "content": f"Question: {question}"},
    ]


register(PromptSpec(variant="answer_clean_v1", kind="clean", scale="unit",
                    build=_clean_build, elicits_vc=False,
                    notes="no VC instruction; teacher-forcing target for clean h_tok"))


# --------------------------------------------------------------------------
# NLI judging
# --------------------------------------------------------------------------

def build_nli(premise: str, hypothesis: str) -> list[Message]:
    """LLM-judged entailment, used when no NLI encoder is available."""
    return [
        {"role": "system", "content": (
            "Decide whether the premise entails the hypothesis in the context of the "
            "same question. Reply with exactly one word: entailment, neutral, or "
            "contradiction."
        )},
        {"role": "user", "content": f"Premise: {premise}\nHypothesis: {hypothesis}"},
    ]


register(PromptSpec(variant="nli_bidirectional_v1", kind="nli", scale="unit",
                    build=lambda premise, hypothesis, **_: build_nli(premise, hypothesis),
                    elicits_vc=False))
