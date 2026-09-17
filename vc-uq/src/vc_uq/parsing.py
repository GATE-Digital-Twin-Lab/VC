"""Parsing model output into (answer, VC).

The raw string is always retained. Parse failure is not swept up: if VC cannot
be reliably elicited, that is a finding about VC, not a bug to be papered over
with a default value (protocol section 4). A failed parse yields ``None``, never
a number.

Two rules follow from that, and both are narrower than they used to be:

**The confidence must be a plain number in [0, 1].** Word ladders are gone. They
were matched as substrings, so "not confident" scored 0.85 and "not certain"
scored 1.00 -- an inverted reading marked ``ok``, in the direction that makes VC
look worse calibrated than it is. Negation-blindness is the same failure that
rules cosine out of clustering (see ``cluster``); it has no place in the
measurement either. A worded reply is now ``unparseable_value`` and shows up in
the audit.

**The number must follow a Confidence label.** There was a fallback that read a
bare number off the last line, meant for the slip where a model writes the value
on its own line. When no confidence line existed at all, the last line was the
ANSWER -- so ``Answer: 1`` was read as confidence 1.0, marked ``ok``. A trivia
answer became a confidence score. Fabricating a value is exactly what this
module exists not to do.

**The scale comes from the prompt, never from the reply.** A ``unit`` prompt asks
for a decimal in [0, 1]; a ``digit10`` prompt asks for one digit 0-9, mapped to
d / 9 by :func:`digit_to_vc` so the scale's ends stay its ends. That is the
scale the model was asked for, not a rescue: a ``digit10`` reply of "85" or
"0.9" is still ``out_of_range``, exactly as "85" is under ``unit``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass



# Instruct models decorate. The fields are located by NAME, wherever they fall:
# any preamble, any number of blank lines, trailing chatter and markdown around
# the label are all tolerated, because none of them change what was said. What
# cannot be recovered here is text the model never emitted -- see the note on
# stop sequences in config/default.yaml.
_LEAD = r"[\s>*_#\-]*"          # bullets, block quotes, bold/italic markers
_MK = r"[*_`\s]*"               # markup hugging the colon
_ANSWER_RE = re.compile(rf"^{_LEAD}answer{_MK}[:\-]{_MK}(.+?)\s*$",
                        re.IGNORECASE | re.MULTILINE)
_CONF_LINE_RE = re.compile(rf"^{_LEAD}(?:confidence|certainty|probability)",
                           re.IGNORECASE)
#: The value is whatever token follows the label; ``parse_vc_value`` decides
#: whether it is a number. Capturing loosely and validating strictly keeps the
#: failure legible -- "Confidence: high" becomes unparseable_value rather than
#: no_confidence_field, so the audit distinguishes "did not answer in the
#: format" from "did not produce a confidence line at all".
_CONF_RE = re.compile(
    rf"(?:confidence|certainty|probability){_MK}[:\-]?{_MK}(?P<value>[^\s,;]+)",
    re.IGNORECASE,
)
#: A plain decimal. No percent signs, no words, no thousands separators.
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
#: Stripped from the ends of a captured answer so "**Lisbon**" scores as
#: "Lisbon". The answer text feeds e_cos against a_star, so leftover markup is
#: noise in the correctness criterion rather than a cosmetic issue.
_MARKUP_CHARS = "*_`~ "
#: A leading bullet or block-quote marker, which must be followed by whitespace
#: so that an answer like "-40 degrees" keeps its sign.
_BULLET_RE = re.compile(r"^\s*(?:[-*+>#]+\s+)+")
#: Exactly one digit, the whole of a ``digit10`` reply.
_DIGIT_RE = re.compile(r"^\d$")

#: Confidence scales a prompt can ask for (``prompts.PromptSpec.vc_scale``).
UNIT = "unit"
DIGIT10 = "digit10"


def digit_to_vc(digit: int) -> float:
    """Digit d on the 0-9 scale as d / 9: 0 -> 0.0, 9 -> 1.0.

    The prompt says only "from 0 to 9", so its ends are read as the ends of
    [0, 1]. A 9 is therefore a claim of certainty; the product rule already
    clamps that (see aggregation.one_minus_vc_floor), as it does for 1.0 on the
    unit scale. Tables and plots show the digit itself (vc * 9).
    """
    return int(digit) / 9.0


def _strip_markup(text: str) -> str:
    return _BULLET_RE.sub("", text).strip().strip(_MARKUP_CHARS).strip()


class ParseFailure(str):
    """Reason codes for the parse audit."""


NO_CONFIDENCE_FIELD = ParseFailure("no_confidence_field")
UNPARSEABLE_VALUE = ParseFailure("unparseable_value")
OUT_OF_RANGE = ParseFailure("out_of_range")
NO_ANSWER_FIELD = ParseFailure("no_answer_field")
OK = ParseFailure("ok")


@dataclass(frozen=True)
class Parsed:
    answer: str
    vc: float | None
    raw: str
    status: str

    @property
    def ok(self) -> bool:
        return self.status == OK


def parse_vc_value(text: str, scale: str = UNIT) -> tuple[float | None, str]:
    """Read a confidence token as a float in [0, 1]. Returns (value, status).

    ``scale`` is what the prompt asked for: ``unit`` (a decimal in [0, 1]) or
    ``digit10`` (one digit 0-9, returned as d / 9).

    Strict on purpose. Anything that is not a plain decimal in range returns
    ``None`` with a status, because every lenient reading this function could
    make is a guess about what the model meant, and a guess recorded as ``ok``
    is indistinguishable downstream from a real measurement.

    A percentage is singled out from other junk: it is a recognisable number on
    a scale the model was not asked for, which is a different finding from
    unintelligible output, and rescaling it would launder a format failure into
    a valid reading.
    """
    token = text.strip().rstrip(".,;:").strip()
    if not token:
        return None, UNPARSEABLE_VALUE
    if token.endswith("%"):
        return None, OUT_OF_RANGE
    if not _NUMBER_RE.match(token):
        return None, UNPARSEABLE_VALUE
    if scale == DIGIT10:
        # One digit and nothing else. "10", "85" or "0.9" is a number on a scale
        # the model was not asked for, reported as such rather than rescaled.
        if not _DIGIT_RE.match(token):
            return None, OUT_OF_RANGE
        return digit_to_vc(int(token)), OK
    if scale != UNIT:
        raise ValueError(f"unknown confidence scale {scale!r}")
    value = float(token)
    if not (0.0 <= value <= 1.0):
        return None, OUT_OF_RANGE
    return value, OK


def parse_answer_and_vc(text: str, scale: str = UNIT) -> Parsed:
    raw = text if text is not None else ""

    m_ans = _ANSWER_RE.search(raw)
    if m_ans is not None:
        answer = _strip_markup(m_ans.group(1))
        answer_status = OK
    else:
        # Fall back to the first non-empty line that is not the confidence line.
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        lines = [ln for ln in lines if not _CONF_LINE_RE.match(ln)]
        answer = _strip_markup(lines[0]) if lines else ""
        answer_status = OK if lines else NO_ANSWER_FIELD

    # No bare-number fallback: the value must follow a Confidence label. The
    # last line of a reply with no confidence line is the ANSWER, and reading a
    # digit out of it turned "Answer: 1" into confidence 1.0.
    m_conf = _CONF_RE.search(raw)
    if m_conf is None:
        return Parsed(answer=answer, vc=None, raw=raw, status=NO_CONFIDENCE_FIELD)

    value, status = parse_vc_value(m_conf.group("value"), scale)
    if answer_status != OK:
        status = NO_ANSWER_FIELD if status == OK else status
    return Parsed(answer=answer, vc=value, raw=raw, status=status)


def parse_vc_only(text: str, scale: str = UNIT) -> Parsed:
    """Pre-hoc elicitation: there is no answer to extract, by construction."""
    raw = text if text is not None else ""
    m_conf = _CONF_RE.search(raw)
    if m_conf is None:
        return Parsed(answer="", vc=None, raw=raw, status=NO_CONFIDENCE_FIELD)
    value, status = parse_vc_value(m_conf.group("value"), scale)
    return Parsed(answer="", vc=value, raw=raw, status=status)


def parse_audit_from_status(statuses) -> dict[str, float | int]:
    """Same audit, computed from a stored ``parse_status`` column.

    Needed because a resumed run generates only the rows it is missing, so the
    audit has to be derived from the whole returned table -- cached rows
    included -- rather than from the Parsed objects this invocation happened to
    create. Otherwise the reported failure rate would depend on how much of the
    job had already been done.
    """
    counts: dict[str, int] = {}
    for s in statuses:
        if s is None or s != s:      # NaN / pd.NA
            s = "unknown"
        counts[str(s)] = counts.get(str(s), 0) + 1
    n = sum(counts.values())
    if n == 0:
        return {"n": 0, "failure_rate": 0.0}
    failures = n - counts.get(OK, 0)
    out: dict[str, float | int] = {"n": n, "n_failed": failures,
                                   "failure_rate": failures / n}
    for status, c in sorted(counts.items()):
        out[f"status__{status}"] = c
    return out


def parse_audit(parsed: list[Parsed]) -> dict[str, float | int]:
    """Parse-failure rates, reported alongside every VC result."""
    return parse_audit_from_status(p.status for p in parsed)
