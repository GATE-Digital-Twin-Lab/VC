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

Message = dict[str, str]


@dataclass(frozen=True)
class PromptSpec:
    variant: str
    kind: Literal["post", "pre", "clean", "nli"]
    build: Callable[..., list[Message]]
    elicits_vc: bool = True
    notes: str = ""
    #: How the confidence value is written: "unit" is a decimal in [0, 1],
    #: "digit10" is one digit 0-9 read as d / 9 (see parsing.digit_to_vc).
    #: The parser is told the scale; it never guesses it from the reply.
    vc_scale: Literal["unit", "digit10"] = "unit"


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
    register(PromptSpec(variant=_variant, kind="post",
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


register(PromptSpec(variant="vc_pre_v1", kind="pre", build=_pre_build,
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


register(PromptSpec(variant="answer_clean_v1", kind="clean",
                    build=_clean_build, elicits_vc=False,
                    notes="no VC instruction; teacher-forcing target for clean h_tok"))


# --------------------------------------------------------------------------
# Short-answer arm.
#
# e_cos compares an answer to a_star, and a_star is a short reference string
# ("Adam Smith"). An unconstrained model answers a trivia question with a
# sentence or a paragraph, and the cosine distance then reflects how much extra
# text was produced as much as whether the fact was right: measured against the
# hand labels, e_cos among HUMAN-CORRECT answers rises from 0.07 at under ten
# tokens to 0.87 past sixty, correlating 0.80 with length. That is the length
# confounder of protocol section 4 landing directly on the correctness
# criterion, so a gate computed on it would be reporting verbosity.
#
# Constraining the answer's FORM is not the same as constraining the decoder:
# top_p, top_k, min_p and repeat_penalty are untouched, so p_q remains a
# property of the sampling distribution and the temperature sweep still
# measures T.
#
# The clean variant carries the SAME brevity instruction and differs from the
# post variant only in the confidence request. That is the whole point of the
# clean pass -- it isolates what VC elicitation does to token entropy -- so a
# clean prompt that did not also ask for brevity would differ in two ways at
# once and h_tok_mean_clean would stop being a control.
# --------------------------------------------------------------------------

_SHORT_RULE = (
    "Answer with the shortest possible answer: a single word, name, number or "
    "date where possible, and never more than a short phrase. Give the answer "
    "only -- no sentence, no explanation, no restatement of the question. If no "
    "answer exists, reply exactly: no such entity."
)

register(PromptSpec(
    variant="vc_post_short_v1", kind="post",
    build=_post_builder(
        _SHORT_RULE + "\nThen state your confidence that your answer is correct.\n"
        "Reply in exactly this format:\n"
        "Answer: <your answer>\n"
        "Confidence: <a number between 0 and 1>"),
    notes="short-form answers; removes the length confound in e_cos"))


def _clean_short_build(question: str, **_: object) -> list[Message]:
    return [
        {"role": "system", "content": _SHORT_RULE},
        {"role": "user", "content": f"Question: {question}"},
    ]


register(PromptSpec(variant="answer_clean_short_v1", kind="clean",
                    build=_clean_short_build, elicits_vc=False,
                    notes="short-form clean prompt; matches vc_post_short_v1 "
                          "except for the confidence request"))


# --------------------------------------------------------------------------
# Digit arm: confidence as ONE digit, 0-9.
#
# A decimal in [0, 1] spans several tokens, and the model settles on a handful
# of values -- 0.95 and 1.0 carry nearly every post-hoc answer in the decimal
# arm. One digit is one token, and there are exactly ten values. The wording is
# the short arm's, with only the scale changed ("from 0 to 9"): it does not
# explain what the digits mean. The analysis reads digit d as d / 9 (0 -> 0.0,
# 9 -> 1.0; see parsing.digit_to_vc), so the ends of the scale are the ends of
# [0, 1] -- a convention of the analysis, not something the model was told. answer_clean_short_v1 is still the matching clean prompt,
# since it differs only in carrying no confidence request.
# --------------------------------------------------------------------------

register(PromptSpec(
    variant="vc_post_short_digit_v1", kind="post", vc_scale="digit10",
    build=_post_builder(
        _SHORT_RULE + "\nThen state your confidence that your answer is correct, "
        "from 0 to 9.\n"
        "Reply in exactly this format:\n"
        "Answer: <your answer>\n"
        "Confidence: <0-9>"),
    notes="short-form answers; confidence as one digit 0-9"))

_PRE_DIGIT_BODY = (
    "You will be shown a question. Do NOT answer it. State only how confident you "
    "are that you could answer it correctly if asked, from 0 to 9.\n"
    "Reply in exactly this format:\n"
    "Confidence: <0-9>"
)

def _pre_digit_build(question: str, **_: object) -> list[Message]:
    return [
        {"role": "system", "content": _PRE_DIGIT_BODY},
        {"role": "user", "content": f"Question: {question}"},
    ]


register(PromptSpec(variant="vc_pre_digit_v1", kind="pre", vc_scale="digit10",
                    build=_pre_digit_build,
                    notes="pre-hoc; confidence as one digit 0-9; otherwise worded "
                          "as vc_pre_v1"))


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


register(PromptSpec(variant="nli_bidirectional_v1", kind="nli",
                    build=lambda premise, hypothesis, **_: build_nli(premise, hypothesis),
                    elicits_vc=False))
