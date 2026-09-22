"""Protected spans: the parts of a text a rewrite has to carry over unchanged.

A rewrite may move words around. It may not drop a URL, a code span, a number
or an identifier. Those spans are extracted from the source and then looked for
in the rewrite byte for byte: a span survives only when the exact characters
are present and are not glued into a longer word, number or URL, so ``42`` in
the rewrite never passes for a ``4`` in the source. Repeated spans are counted,
not merged.

Extraction reads identifiers widely -- file names, dotted paths, versions,
ticket keys, hashes and digit-letter mixes such as ``200ms`` stay whole -- and
never overlaps: a URL inside a code span belongs to that code span.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from voicefit.profile import SCHEMA_VERSION

__all__ = [
    "PROTECTED_KINDS",
    "ProtectedSpan",
    "SpanCheck",
    "SpanReport",
    "extract_spans",
    "verify_spans",
]

PROTECTED_KINDS: tuple[str, ...] = ("code", "url", "identifier", "number")

_ABBREVIATIONS = frozenset({"a.m", "e.g", "et.al", "i.e", "p.m"})
_URL_TRAIL = ".,;:!?)]}>\"'"

_CODE = (
    r"(?P<code_fence>```[^\n`]*\n(?P<fence_body>.*?)```)"
    r"|(?P<code_inline>`(?P<inline_body>[^`\n]+)`)"
)

_URL = (
    r"(?:https?|ftp)://[^\s<>\"'`\\]+"
    r"|www\.[^\s<>\"'`\\]+"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
)

_IDENTIFIER = (
    r"[A-Za-z][A-Za-z0-9]*-\d+(?:[.-][A-Za-z0-9]+)*"
    r"|[A-Za-z_][A-Za-z0-9_]*(?:[._][A-Za-z0-9_]+)+"
    r"|[A-Za-z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+"
    r"|[A-Za-z]+\d+(?:\.\d+)*[A-Za-z0-9]*"
    r"|\d+(?:\.\d+){2,}"
    r"|\d+[A-Za-z]+[A-Za-z0-9]*"
)

_NUMBER = (
    r"\d{4}-\d{2}-\d{2}"
    r"|(?<![\w.])[$\u20ac\u00a3]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|(?<![\w.])[$\u20ac\u00a3]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
)

_SPANS = re.compile(
    f"{_CODE}|(?P<url>{_URL})|(?P<identifier>{_IDENTIFIER})|(?P<number>{_NUMBER})",
    re.DOTALL,
)

_WORD_CHAR = re.compile(r"[A-Za-z0-9_]")
_URL_CHAR = re.compile(r"[A-Za-z0-9_/?#=&%~+@:-]")


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _preview(text: str, limit: int = 48) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "\u2026"


def _wanted_kinds(kinds: Iterable[str] | None) -> frozenset[str]:
    if kinds is None:
        return frozenset(PROTECTED_KINDS)
    if isinstance(kinds, str):
        raise TypeError("kinds expects an iterable of kind names, not a single string")
    wanted = frozenset(str(kind) for kind in kinds)
    unknown = wanted - set(PROTECTED_KINDS)
    if unknown:
        raise ValueError(f"unknown span kinds: {', '.join(sorted(unknown))}")
    return wanted


def _span_bounds(text: str, match: re.Match[str]) -> tuple[str, int, int] | None:
    if match.group("code_fence") is not None:
        start, end = _trim(text, match.start("fence_body"), match.end("fence_body"))
        return ("code", start, end) if end > start else None
    if match.group("code_inline") is not None:
        start, end = _trim(text, match.start("inline_body"), match.end("inline_body"))
        return ("code", start, end) if end > start else None
    if match.group("url") is not None:
        start, end = match.span("url")
        while end > start and text[end - 1] in _URL_TRAIL:
            end -= 1
        return ("url", start, end) if end > start else None
    if match.group("identifier") is not None:
        start, end = match.span("identifier")
        if text[start:end].lower() in _ABBREVIATIONS:
            return None
        return ("identifier", start, end)
    start, end = match.span("number")
    return ("number", start, end)


def _glued(haystack: str, start: int, end: int, kind: str) -> bool:
    pattern = _URL_CHAR if kind == "url" else _WORD_CHAR
    before = haystack[start - 1] if start else ""
    after = haystack[end] if end < len(haystack) else ""
    if before and pattern.match(before):
        return True
    if after and pattern.match(after):
        return True
    if before in {".", ","} and start >= 2 and haystack[start - 2].isdigit():
        return True
    if after in {".", ","} and end + 1 < len(haystack) and haystack[end + 1].isdigit():
        return True
    return False


def _occurrences(haystack: str, needle: str, kind: str) -> int:
    hits = 0
    index = haystack.find(needle)
    while index != -1:
        if not _glued(haystack, index, index + len(needle), kind):
            hits += 1
        index = haystack.find(needle, index + 1)
    return hits


@dataclass(frozen=True, kw_only=True)
class ProtectedSpan:
    """One stretch of source text that a rewrite must reproduce exactly."""

    kind: str
    text: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True, kw_only=True)
class SpanCheck:
    """How often one protected text was expected, and how often it was found."""

    kind: str
    text: str
    expected: int
    found: int

    @property
    def survived(self) -> bool:
        return self.found >= self.expected

    @property
    def missing(self) -> int:
        return max(self.expected - self.found, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "expected": self.expected,
            "found": self.found,
            "missing": self.missing,
            "survived": self.survived,
        }


@dataclass(frozen=True, kw_only=True)
class SpanReport:
    """What a rewrite kept of the protected spans of its source."""

    spans: tuple[ProtectedSpan, ...]
    checks: tuple[SpanCheck, ...]

    def check(self, text: str) -> SpanCheck:
        for check in self.checks:
            if check.text == text:
                return check
        raise KeyError(text)

    @property
    def lost(self) -> tuple[SpanCheck, ...]:
        return tuple(check for check in self.checks if not check.survived)

    @property
    def intact(self) -> bool:
        return not self.lost

    def report(self) -> str:
        """Render one line per protected text, in order of first appearance."""
        lines = [f"{'kind':<12}{'want':>6}{'found':>7}  {'flag':<6}span"]
        for check in self.checks:
            flag = "ok" if check.survived else "lost"
            lines.append(
                f"{check.kind:<12}{check.expected:>6}{check.found:>7}  "
                f"{flag:<6}{_preview(check.text)}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "intact": self.intact,
            "span_count": len(self.spans),
            "lost": [check.text for check in self.lost],
            "checks": [check.to_dict() for check in self.checks],
            "spans": [span.to_dict() for span in self.spans],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def extract_spans(
    text: str, *, kinds: Iterable[str] | None = None
) -> tuple[ProtectedSpan, ...]:
    """Return the protected spans of ``text``, in order and without overlap.

    ``kinds`` narrows the result to names from :data:`PROTECTED_KINDS`. It
    filters what is reported, not what is scanned: a URL inside a code span
    stays part of that code span either way.
    """
    if not isinstance(text, str):
        raise TypeError("extract_spans expects one string")

    wanted = _wanted_kinds(kinds)
    spans: list[ProtectedSpan] = []
    for match in _SPANS.finditer(text):
        bounds = _span_bounds(text, match)
        if bounds is None:
            continue
        kind, start, end = bounds
        if kind not in wanted:
            continue
        spans.append(
            ProtectedSpan(kind=kind, text=text[start:end], start=start, end=end)
        )
    return tuple(spans)


def verify_spans(
    source: str, rewrite: str, *, kinds: Iterable[str] | None = None
) -> SpanReport:
    """Check that every protected span of ``source`` is still in ``rewrite``.

    Identical spans are grouped by their text and counted, so a number used
    twice in the source has to appear twice in the rewrite.
    """
    if not isinstance(rewrite, str):
        raise TypeError("verify_spans expects one rewritten string")

    spans = extract_spans(source, kinds=kinds)
    order: list[str] = []
    kind_of: dict[str, str] = {}
    expected: dict[str, int] = {}
    for span in spans:
        if span.text not in expected:
            order.append(span.text)
            kind_of[span.text] = span.kind
            expected[span.text] = 0
        expected[span.text] += 1

    checks = tuple(
        SpanCheck(
            kind=kind_of[text],
            text=text,
            expected=expected[text],
            found=_occurrences(rewrite, text, kind_of[text]),
        )
        for text in order
    )
    return SpanReport(spans=spans, checks=checks)
