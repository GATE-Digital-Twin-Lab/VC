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
# Scale reframings (protocol 8.3). A real quantity is invariant to its
# reporting scale; these ask the same thing four ways.
# --------------------------------------------------------------------------

_SCALE_BODIES: dict[str, tuple[Scale, str]] = {
    "scale_unit": ("unit", _POST_BODIES["vc_post_v1"]),
    "scale_percent": ("percent", (
        "Answer the question. Then state your confidence that your answer is correct "
        "as a percentage.\n"
        "Answer: <your answer>\n"
        "Confidence: <a percentage between 0% and 100%>"
    )),
    "scale_verbal": ("verbal", (
        "Answer the question. Then state your confidence using exactly one of these "
        "words: impossible, doubtful, unlikely, uncertain, even, likely, probable, "
        "confident, highly confident, certain.\n"
        "Answer: <your answer>\n"
        "Confidence: <word>"
    )),
    "scale_outof10": ("outof10", (
        "Answer the question. Then say: out of 10 independent attempts at this "
        "question, how many would you get right?\n"
        "Answer: <your answer>\n"
        "Confidence: <a whole number from 0 to 10>"
    )),
}

for _variant, (_scale, _body) in _SCALE_BODIES.items():
    register(PromptSpec(variant=_variant, kind="post", scale=_scale,
                        build=_post_builder(_body),
                        notes="scale reframing probe (8.3)"))


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
# Intervention prompts (Phase 5).
# --------------------------------------------------------------------------

def build_sycophancy(question: str, answer: str, challenge: str,
                     variant: str = "vc_post_v1") -> list[Message]:
    """Re-elicit VC on an UNCHANGED answer after conversational pushback (8.4)."""
    spec = get(variant)
    msgs = list(spec.build(question=question))
    msgs += [
        {"role": "assistant", "content": f"Answer: {answer}"},
        {"role": "user", "content": (
            f"{challenge} Please restate your confidence in the same answer.\n"
            "Confidence: <number between 0 and 1>"
        )},
    ]
    return msgs


def build_forced_decode(question: str, injected_vc: float) -> list[Message]:
    """Inject a confidence value into context, THEN sample an answer (8.5).

    If the output distribution moves, VC is a control signal that perturbs what
    it claims to passively measure.
    """
    return [
        {"role": "system", "content": "Answer the question concisely."},
        {"role": "user", "content": f"Question: {question}"},
        {"role": "assistant", "content": f"confidence: {injected_vc:g}\nAnswer:"},
    ]


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
