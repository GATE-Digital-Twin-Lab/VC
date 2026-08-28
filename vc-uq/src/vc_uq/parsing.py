"""Parsing model output into (answer, VC).

The raw string is always retained. Parse failure is not swept up: if VC cannot
be reliably elicited, that is a finding about VC, not a bug to be papered over
with a default value (protocol section 4). A failed parse yields ``None``, never
0.5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .prompts import VERBAL_SCALE, Scale  # noqa: F401  -- re-exported

_ANSWER_RE = re.compile(r"^\s*answer\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_CONF_RE = re.compile(
    r"(?:confidence|certainty|probability)\s*[:\-]?\s*"
    r"(?P<value>[0-9]*\.?[0-9]+\s*%?|[A-Za-z][A-Za-z ]{2,24})",
    re.IGNORECASE,
)
_BARE_NUM_RE = re.compile(r"(?<![\w.])(?P<value>[01](?:\.\d+)?|0?\.\d+|\d{1,3}\s*%)(?![\w.])")

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


def parse_vc_value(text: str, scale: Scale = "unit") -> tuple[float | None, str]:
    """Map a confidence token onto [0, 1]. Returns (value, status)."""
    token = text.strip().strip(".,;").lower()
    if not token:
        return None, UNPARSEABLE_VALUE

    if scale == "verbal" or not re.search(r"\d", token):
        # Longest label first so "highly confident" wins over "confident".
        for label in sorted(VERBAL_SCALE, key=len, reverse=True):
            if label in token:
                return VERBAL_SCALE[label], OK
        return None, UNPARSEABLE_VALUE

    is_percent = "%" in token
    num = re.search(r"[0-9]*\.?[0-9]+", token)
    if num is None:
        return None, UNPARSEABLE_VALUE
    value = float(num.group())

    if scale == "percent" or is_percent:
        value = value / 100.0
    elif scale == "outof10":
        value = value / 10.0
    elif scale == "unit" and value > 1.0:
        # The model answered on a scale it was not asked for. Rescaling here
        # would launder a failure to follow the elicitation format into a valid
        # reading, so it is recorded as out of range and shows up in the parse
        # audit instead.
        if value <= 100.0:
            return None, OUT_OF_RANGE
        return None, UNPARSEABLE_VALUE

    if not (0.0 <= value <= 1.0):
        return None, OUT_OF_RANGE
    return value, OK


def parse_answer_and_vc(text: str, scale: Scale = "unit",
                        *, require_answer: bool = True) -> Parsed:
    raw = text if text is not None else ""

    m_ans = _ANSWER_RE.search(raw)
    if m_ans is not None:
        answer = m_ans.group(1).strip()
        answer_status = OK
    else:
        # Fall back to the first non-empty line that is not the confidence line.
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        lines = [ln for ln in lines if not _CONF_RE.match(ln)]
        answer = lines[0] if lines else ""
        answer_status = OK if lines else NO_ANSWER_FIELD

    m_conf = _CONF_RE.search(raw)
    if m_conf is None:
        # A bare trailing number is a common format slip; accept it but keep the
        # distinction visible via the status field.
        tail = raw.strip().splitlines()[-1] if raw.strip() else ""
        m_bare = _BARE_NUM_RE.search(tail)
        if m_bare is None:
            return Parsed(answer=answer, vc=None, raw=raw, status=NO_CONFIDENCE_FIELD)
        value, status = parse_vc_value(m_bare.group("value"), scale)
        return Parsed(answer=answer, vc=value, raw=raw,
                      status=status if status != OK else OK)

    value, status = parse_vc_value(m_conf.group("value"), scale)
    if require_answer and answer_status != OK:
        status = NO_ANSWER_FIELD if status == OK else status
    return Parsed(answer=answer, vc=value, raw=raw, status=status)


def parse_vc_only(text: str, scale: Scale = "unit") -> Parsed:
    """Pre-hoc elicitation: there is no answer to extract, by construction."""
    raw = text if text is not None else ""
    m_conf = _CONF_RE.search(raw)
    if m_conf is None:
        m_bare = _BARE_NUM_RE.search(raw)
        if m_bare is None:
            return Parsed(answer="", vc=None, raw=raw, status=NO_CONFIDENCE_FIELD)
        value, status = parse_vc_value(m_bare.group("value"), scale)
        return Parsed(answer="", vc=value, raw=raw, status=status)
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
    n = len(parsed)
    if n == 0:
        return {"n": 0, "failure_rate": 0.0}
    counts: dict[str, int] = {}
    for p in parsed:
        counts[p.status] = counts.get(p.status, 0) + 1
    failures = n - counts.get(OK, 0)
    out: dict[str, float | int] = {
        "n": n,
        "n_failed": failures,
        "failure_rate": failures / n,
    }
    for status, c in sorted(counts.items()):
        out[f"status__{status}"] = c
    return out
