"""Rewrite a text through a caller-supplied model, under checks.

The loop is short on purpose: build one instruction from the target profile and
the constraints, hand it to the callable, measure what comes back against the
protected spans of the source and against the profile itself, and on failure
ask once more with the failures spelled out. When the retry fails as well, the
original text is returned, so a caller never silently ships a rewrite that lost
a number or drifted off the profile.

The model is any callable that takes the instruction and returns the rewritten
text. voicefit does not talk to a provider itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from voicefit.distance import DEFAULT_TOLERANCE, ProfileDistance, compare_text
from voicefit.profile import SCHEMA_VERSION, StyleProfile
from voicefit.spans import ProtectedSpan, SpanReport, extract_spans, verify_spans

__all__ = [
    "Model",
    "RewriteAttempt",
    "RewriteConstraints",
    "RewriteResult",
    "build_instruction",
    "rewrite",
]

Model = Callable[[str], str]

_MAX_ATTEMPTS = 2


@dataclass(frozen=True, kw_only=True)
class RewriteConstraints:
    """What a rewrite has to respect, beyond the profile itself.

    ``tolerance`` is read in standardised units, as in
    :func:`~voicefit.distance.compare_text`. ``kinds`` narrows the protected
    spans, ``hedges`` must be the list the profile was built with, ``scales``
    overrides per-axis scales and ``notes`` are extra rules for the model.
    """

    tolerance: float = DEFAULT_TOLERANCE
    kinds: Sequence[str] | None = None
    hedges: Sequence[str] | None = None
    scales: Mapping[str, float] | None = None
    notes: Sequence[str] = ()


def _share(value: float) -> str:
    return "none" if value <= 0 else f"about {value * 100:.0f}%"


def _profile_lines(profile: StyleProfile) -> list[str]:
    marks = ", ".join(
        f"{name.replace('_', ' ')} {rate:.0f}"
        for name, rate in sorted(profile.punctuation_per_1000_words.items())
        if rate >= 0.5
    )
    return [
        f"- sentences of about {profile.sentence_length_mean:.1f} words, varying by"
        f" about {profile.sentence_length_stdev:.1f}",
        f"- words of about {profile.word_length_mean:.1f} characters",
        f"- questions in {_share(profile.question_share)} of the sentences",
        f"- exclamations in {_share(profile.exclamation_share)} of the sentences",
        f"- hedging in {_share(profile.hedge_sentence_share)} of the sentences, about"
        f" {profile.hedge_rate * 1000:.0f} markers per 1000 words",
        f"- paragraphs of about {profile.sentences_per_paragraph_mean:.1f} sentences"
        f" and about {profile.words_per_paragraph_mean:.0f} words",
        f"- punctuation per 1000 words: {marks}"
        if marks
        else "- punctuation used sparingly",
    ]


def _protected_texts(spans: Sequence[ProtectedSpan]) -> list[str]:
    texts: list[str] = []
    for span in spans:
        if span.text not in texts:
            texts.append(span.text)
    return texts


def _rule_lines(
    spans: Sequence[ProtectedSpan], constraints: RewriteConstraints
) -> list[str]:
    lines = [
        "- Keep every fact, number, link and identifier the text states.",
        "- Add nothing the text does not say, and drop nothing it does.",
    ]
    protected = _protected_texts(spans)
    if protected:
        listed = ", ".join(f'"{text}"' for text in protected)
        lines.append(
            f"- Reproduce these character for character, as often as they appear:"
            f" {listed}."
        )
    lines.append("- Keep the paragraph breaks of the original.")
    lines.append("- Vary sentence openings and do not reuse a phrasing twice.")
    lines.append("- Return the rewritten text only, with no commentary.")
    lines.extend(f"- {note}" for note in constraints.notes)
    return lines


def build_instruction(
    text: str,
    profile: StyleProfile,
    *,
    constraints: RewriteConstraints | None = None,
    failures: Sequence[str] = (),
    previous: str = "",
) -> str:
    """Render the instruction for one attempt.

    The profile is spelled out axis by axis, the protected spans are listed
    verbatim, and on a retry the ``failures`` of the previous attempt are quoted
    as they were measured, with ``previous`` attached for reference.
    """
    if not isinstance(text, str):
        raise TypeError("build_instruction expects one source string")

    limits = constraints if constraints is not None else RewriteConstraints()
    spans = extract_spans(text, kinds=limits.kinds)
    blocks = [
        "Rewrite the text below in the target style. Keep the meaning, the facts "
        "and the order of the argument.",
        "Target style, measured from the author's own corpus:\n"
        + "\n".join(_profile_lines(profile)),
        "Rules:\n" + "\n".join(_rule_lines(spans, limits)),
    ]
    if failures:
        blocks.append(
            "The previous attempt failed these checks:\n"
            + "\n".join(f"- {failure}" for failure in failures)
        )
        if previous.strip():
            blocks.append(f"Previous attempt:\n{previous.strip()}")
        blocks.append("Rewrite the original text again and fix every listed failure.")
    blocks.append(f"Text:\n{text.strip()}")
    return "\n\n".join(blocks)


def _failures(spans: SpanReport, distance: ProfileDistance | None) -> list[str]:
    failures = [
        f'protected {check.kind} "{check.text}" appears {check.found} times in the'
        f" rewrite, expected {check.expected}"
        for check in spans.lost
    ]
    if distance is None:
        return failures
    failures.extend(
        f"{axis.axis} is {axis.direction} the target: {axis.value:.3g} against"
        f" {axis.target:.3g} {axis.unit} (deviation {axis.deviation:+.2f},"
        f" tolerance {axis.tolerance:.2f})"
        for axis in distance.off_profile
    )
    return failures


@dataclass(frozen=True, kw_only=True)
class RewriteAttempt:
    """One call to the model, with the instruction sent and the checks it faced."""

    number: int
    instruction: str
    output: str
    spans: SpanReport
    distance: ProfileDistance | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "passed": self.passed,
            "failures": list(self.failures),
            "instruction": self.instruction,
            "output": self.output,
            "spans": self.spans.to_dict(),
            "distance": None if self.distance is None else self.distance.to_dict(),
        }


@dataclass(frozen=True, kw_only=True)
class RewriteResult:
    """The text a caller should use, and every attempt that led to it."""

    text: str
    source: str
    attempts: tuple[RewriteAttempt, ...]

    @property
    def accepted(self) -> bool:
        return self.attempts[-1].passed

    @property
    def failures(self) -> tuple[str, ...]:
        return self.attempts[-1].failures

    def report(self) -> str:
        """Render each attempt with its failures, then what was returned."""
        lines: list[str] = []
        for attempt in self.attempts:
            flag = "ok" if attempt.passed else "failed"
            lines.append(f"attempt {attempt.number}: {flag}")
            lines.extend(f"  - {failure}" for failure in attempt.failures)
        lines.append(
            "result: rewrite accepted" if self.accepted else "result: original returned"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "accepted": self.accepted,
            "attempt_count": len(self.attempts),
            "failures": list(self.failures),
            "text": self.text,
            "source": self.source,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _measure(
    *,
    number: int,
    instruction: str,
    output: str,
    source: str,
    profile: StyleProfile,
    constraints: RewriteConstraints,
) -> RewriteAttempt:
    spans = verify_spans(source, output, kinds=constraints.kinds)
    distance: ProfileDistance | None = None
    failures: list[str] = []
    if not output.strip():
        failures.append("the model returned an empty rewrite")
    else:
        try:
            distance = compare_text(
                output,
                profile,
                tolerance=constraints.tolerance,
                hedges=constraints.hedges,
                scales=constraints.scales,
            )
        except ValueError:
            failures.append("the rewrite has no measurable sentence")
    failures.extend(_failures(spans, distance))
    return RewriteAttempt(
        number=number,
        instruction=instruction,
        output=output,
        spans=spans,
        distance=distance,
        failures=tuple(failures),
    )


def rewrite(
    text: str,
    profile: StyleProfile,
    model: Model,
    *,
    constraints: RewriteConstraints | None = None,
) -> RewriteResult:
    """Rewrite ``text`` towards ``profile`` through ``model``, then check it.

    A failing attempt is retried once with its failures listed in the new
    instruction. When the retry fails too, ``RewriteResult.text`` is the
    original ``text`` and ``RewriteResult.accepted`` is ``False``: the loop
    never hands back a rewrite that did not pass the checks.
    """
    if not isinstance(text, str):
        raise TypeError("rewrite expects one source string")

    limits = constraints if constraints is not None else RewriteConstraints()
    attempts: list[RewriteAttempt] = []
    failures: tuple[str, ...] = ()
    previous = ""
    for number in range(1, _MAX_ATTEMPTS + 1):
        instruction = build_instruction(
            text, profile, constraints=limits, failures=failures, previous=previous
        )
        output = model(instruction)
        if not isinstance(output, str):
            raise TypeError("the model must return the rewritten text as a string")
        attempt = _measure(
            number=number,
            instruction=instruction,
            output=output,
            source=text,
            profile=profile,
            constraints=limits,
        )
        attempts.append(attempt)
        if attempt.passed:
            break
        failures = attempt.failures
        previous = output

    last = attempts[-1]
    return RewriteResult(
        text=last.output if last.passed else text,
        source=text,
        attempts=tuple(attempts),
    )
