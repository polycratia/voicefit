"""Measure how far a text sits from a style profile, axis by axis.

Every axis of a :class:`~voicefit.profile.StyleProfile` is compared in its own
units, then standardised by a scale that says how much drift on that axis is
worth noticing. The result keeps the per-axis numbers: target, measured value,
raw difference, scale and standardised deviation. Summaries such as
``max_deviation`` are derived from those rows and never replace them, so a
caller can always see which axis moved and by how much.

The scale for an axis is the largest of: a spread already measured in the
profile (sentence length uses its own standard deviation), a relative fraction
of the target, and an absolute floor. The floors keep near-zero targets from
turning a rounding difference into a huge deviation. Callers who know better
can pass their own scales.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

from voicefit.profile import SCHEMA_VERSION, StyleProfile, build_profile

__all__ = [
    "AXES",
    "AxisDeviation",
    "AxisSpec",
    "DEFAULT_TOLERANCE",
    "PUNCTUATION_AXIS",
    "ProfileDistance",
    "compare_profiles",
    "compare_text",
    "compare_texts",
]

DEFAULT_TOLERANCE = 1.0


def _round(value: float) -> float:
    return round(float(value), 6)


@dataclass(frozen=True, kw_only=True)
class AxisSpec:
    """One comparable axis and the scale that gives its drift a meaning."""

    name: str
    unit: str
    relative: float
    floor: float
    spread: str | None = None

    def scale_for(self, profile: StyleProfile, target: float) -> float:
        measured = float(getattr(profile, self.spread)) if self.spread else 0.0
        return max(measured, self.relative * abs(target), self.floor)


AXES: tuple[AxisSpec, ...] = (
    AxisSpec(
        name="sentence_length_mean",
        unit="words",
        relative=0.1,
        floor=1.0,
        spread="sentence_length_stdev",
    ),
    AxisSpec(name="sentence_length_stdev", unit="words", relative=0.25, floor=1.0),
    AxisSpec(name="word_length_mean", unit="characters", relative=0.05, floor=0.2),
    AxisSpec(
        name="question_share", unit="share of sentences", relative=0.25, floor=0.05
    ),
    AxisSpec(
        name="exclamation_share", unit="share of sentences", relative=0.25, floor=0.05
    ),
    AxisSpec(name="hedge_rate", unit="per word", relative=0.25, floor=0.005),
    AxisSpec(
        name="hedge_sentence_share",
        unit="share of sentences",
        relative=0.25,
        floor=0.05,
    ),
    AxisSpec(
        name="sentences_per_paragraph_mean",
        unit="sentences",
        relative=0.25,
        floor=0.5,
    ),
    AxisSpec(
        name="words_per_paragraph_mean", unit="words", relative=0.25, floor=10.0
    ),
)

PUNCTUATION_AXIS = AxisSpec(
    name="punctuation", unit="per 1000 words", relative=0.5, floor=2.0
)


@dataclass(frozen=True, kw_only=True)
class AxisDeviation:
    """How far one axis of a text sits from the profile it is measured against."""

    axis: str
    unit: str
    target: float
    value: float
    difference: float
    scale: float
    deviation: float
    tolerance: float

    @property
    def within_tolerance(self) -> bool:
        return abs(self.deviation) <= self.tolerance

    @property
    def direction(self) -> str:
        if self.difference > 0:
            return "above"
        if self.difference < 0:
            return "below"
        return "on"

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "unit": self.unit,
            "target": self.target,
            "value": self.value,
            "difference": self.difference,
            "scale": self.scale,
            "deviation": self.deviation,
            "direction": self.direction,
            "within_tolerance": self.within_tolerance,
        }


@dataclass(frozen=True, kw_only=True)
class ProfileDistance:
    """A per-axis distance report between a measured text and a profile."""

    tolerance: float
    axes: tuple[AxisDeviation, ...]
    sample: StyleProfile

    def axis(self, name: str) -> AxisDeviation:
        for axis in self.axes:
            if axis.axis == name:
                return axis
        raise KeyError(name)

    @property
    def off_profile(self) -> tuple[AxisDeviation, ...]:
        return tuple(axis for axis in self.axes if not axis.within_tolerance)

    @property
    def matches(self) -> bool:
        return not self.off_profile

    @property
    def max_deviation(self) -> float:
        return _round(max(abs(axis.deviation) for axis in self.axes))

    @property
    def mean_absolute_deviation(self) -> float:
        total = sum(abs(axis.deviation) for axis in self.axes)
        return _round(total / len(self.axes))

    @property
    def worst(self) -> AxisDeviation:
        return max(self.axes, key=lambda axis: abs(axis.deviation))

    def report(self) -> str:
        """Render one line per axis, widest deviation last in the flag column."""
        lines = [
            f"{'axis':<30}{'target':>10}{'text':>10}{'diff':>10}{'dev':>8}  flag"
        ]
        for axis in self.axes:
            flag = "ok" if axis.within_tolerance else "off"
            lines.append(
                f"{axis.axis:<30}{axis.target:>10.3f}{axis.value:>10.3f}"
                f"{axis.difference:>+10.3f}{axis.deviation:>+8.2f}  {flag}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "tolerance": self.tolerance,
            "matches": self.matches,
            "max_deviation": self.max_deviation,
            "mean_absolute_deviation": self.mean_absolute_deviation,
            "off_profile": [axis.axis for axis in self.off_profile],
            "axes": [axis.to_dict() for axis in self.axes],
            "sample": self.sample.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _pairs(
    sample: StyleProfile, profile: StyleProfile
) -> Iterator[tuple[AxisSpec, float, float]]:
    for spec in AXES:
        yield spec, float(getattr(profile, spec.name)), float(getattr(sample, spec.name))

    targets = profile.punctuation_per_1000_words
    values = sample.punctuation_per_1000_words
    for name in sorted(set(targets) | set(values)):
        spec = replace(PUNCTUATION_AXIS, name=f"punctuation.{name}")
        yield spec, float(targets.get(name, 0.0)), float(values.get(name, 0.0))


def compare_profiles(
    sample: StyleProfile,
    profile: StyleProfile,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    scales: Mapping[str, float] | None = None,
) -> ProfileDistance:
    """Compare a measured profile against a target profile, axis by axis.

    ``tolerance`` is read in standardised units: an axis is flagged once its
    deviation exceeds it. ``scales`` replaces the scale of named axes, and an
    unknown axis name or a non-positive scale is an error rather than a
    silently ignored argument.
    """
    if tolerance < 0:
        raise ValueError("tolerance must not be negative")

    overrides = {str(name): float(scale) for name, scale in (scales or {}).items()}
    axes: list[AxisDeviation] = []
    for spec, target, value in _pairs(sample, profile):
        scale = overrides.pop(spec.name, None)
        if scale is None:
            scale = spec.scale_for(profile, target)
        elif scale <= 0:
            raise ValueError(f"scale for axis {spec.name!r} must be positive")
        difference = value - target
        axes.append(
            AxisDeviation(
                axis=spec.name,
                unit=spec.unit,
                target=_round(target),
                value=_round(value),
                difference=_round(difference),
                scale=_round(scale),
                deviation=_round(difference / scale),
                tolerance=float(tolerance),
            )
        )

    if overrides:
        unknown = ", ".join(sorted(overrides))
        raise ValueError(f"unknown axes in scales: {unknown}")

    return ProfileDistance(
        tolerance=float(tolerance), axes=tuple(axes), sample=sample
    )


def compare_texts(
    texts: Iterable[str],
    profile: StyleProfile,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    hedges: Iterable[str] | None = None,
    scales: Mapping[str, float] | None = None,
) -> ProfileDistance:
    """Measure a corpus and report its distance from ``profile``.

    Pass the same ``hedges`` list the profile was built with, otherwise the
    hedging axes compare two different definitions of a hedge.
    """
    sample = build_profile(texts, hedges=hedges)
    return compare_profiles(sample, profile, tolerance=tolerance, scales=scales)


def compare_text(
    text: str,
    profile: StyleProfile,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    hedges: Iterable[str] | None = None,
    scales: Mapping[str, float] | None = None,
) -> ProfileDistance:
    """Measure a single text and report its distance from ``profile``."""
    if not isinstance(text, str):
        raise TypeError("compare_text expects one string; use compare_texts for a corpus")
    return compare_texts(
        [text], profile, tolerance=tolerance, hedges=hedges, scales=scales
    )
