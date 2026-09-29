"""Keep recent outputs per profile and refuse to repeat their shapes.

A voice is more than its averages. A run of texts that all open the same way
and all land on the same closing phrase reads as a template even when every
axis of the profile is on target. This module keeps the recent outputs of a
profile and measures two shapes of a candidate against them: the opening words
of its first sentence and the closing words of its last. A candidate is refused
once one of its shapes already carries more than the configured share of the
remembered window.

The window is kept per profile, under a caller-supplied name or under a key
derived from the profile itself, so two voices never share a history.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from voicefit.profile import (
    SCHEMA_VERSION,
    StyleProfile,
    split_paragraphs,
    split_sentences,
    tokenize_words,
)

__all__ = [
    "DEFAULT_CAPACITY",
    "DEFAULT_ENDING_WORDS",
    "DEFAULT_MAX_SHARE",
    "DEFAULT_OPENING_WORDS",
    "OutputMemory",
    "PATTERN_KINDS",
    "PatternUse",
    "RepetitionReport",
    "profile_key",
]

DEFAULT_CAPACITY = 20
DEFAULT_MAX_SHARE = 0.34
DEFAULT_OPENING_WORDS = 4
DEFAULT_ENDING_WORDS = 4

PATTERN_KINDS: tuple[str, ...] = ("opening", "ending")


def profile_key(profile: StyleProfile) -> str:
    """Return a stable key for a profile, so its window follows its numbers."""
    payload = json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_key(key: str | StyleProfile) -> str:
    if isinstance(key, StyleProfile):
        return profile_key(key)
    if not isinstance(key, str):
        raise TypeError("a memory key is a string or a StyleProfile")
    if not key.strip():
        raise ValueError("a memory key must not be empty")
    return key


def _sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in split_paragraphs(text):
        sentences.extend(split_sentences(paragraph))
    return sentences


def _phrase(tokens: list[str], size: int, *, tail: bool = False) -> str:
    chosen = tokens[-size:] if tail else tokens[:size]
    return " ".join(token.lower() for token in chosen)


@dataclass(frozen=True, kw_only=True)
class PatternUse:
    """How much of a remembered window one opening or ending already carries."""

    kind: str
    pattern: str
    count: int
    window: int
    share: float
    max_share: float

    @property
    def repeats(self) -> bool:
        return self.share > self.max_share

    def describe(self) -> str:
        return (
            f'the {self.kind} "{self.pattern}" already carries {self.count} of the'
            f" {self.window} recent outputs (share {self.share:.2f}, limit"
            f" {self.max_share:.2f})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "pattern": self.pattern,
            "count": self.count,
            "window": self.window,
            "share": self.share,
            "max_share": self.max_share,
            "repeats": self.repeats,
        }


@dataclass(frozen=True, kw_only=True)
class RepetitionReport:
    """What a candidate output reuses from the recent outputs of one profile."""

    key: str
    uses: tuple[PatternUse, ...]

    def use(self, kind: str) -> PatternUse:
        for use in self.uses:
            if use.kind == kind:
                return use
        raise KeyError(kind)

    @property
    def repeats(self) -> tuple[PatternUse, ...]:
        return tuple(use for use in self.uses if use.repeats)

    @property
    def fresh(self) -> bool:
        return not self.repeats

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(use.describe() for use in self.repeats)

    def report(self) -> str:
        """Render one line per shape, with the share of the window it carries."""
        lines = [
            f"{'shape':<10}{'used':>6}{'window':>8}{'share':>8}{'limit':>8}  "
            f"{'flag':<6}pattern"
        ]
        for use in self.uses:
            flag = "over" if use.repeats else "ok"
            lines.append(
                f"{use.kind:<10}{use.count:>6}{use.window:>8}{use.share:>8.2f}"
                f"{use.max_share:>8.2f}  {flag:<6}{use.pattern}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "key": self.key,
            "fresh": self.fresh,
            "repeats": [use.kind for use in self.repeats],
            "uses": [use.to_dict() for use in self.uses],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class OutputMemory:
    """The recent outputs of each profile, and the shapes they used up.

    ``capacity`` is the length of the window per profile, ``max_share`` the
    share of that window a single opening or ending may carry, and
    ``opening_words`` and ``ending_words`` how many words each shape is read
    from.
    """

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_CAPACITY,
        max_share: float = DEFAULT_MAX_SHARE,
        opening_words: int = DEFAULT_OPENING_WORDS,
        ending_words: int = DEFAULT_ENDING_WORDS,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least one output")
        if not 0.0 <= max_share <= 1.0:
            raise ValueError("max_share must sit between 0 and 1")
        if opening_words < 1 or ending_words < 1:
            raise ValueError("a pattern must be at least one word long")
        self.capacity = int(capacity)
        self.max_share = float(max_share)
        self.opening_words = int(opening_words)
        self.ending_words = int(ending_words)
        self._outputs: dict[str, deque[str]] = {}

    def __len__(self) -> int:
        return sum(len(outputs) for outputs in self._outputs.values())

    def keys(self) -> tuple[str, ...]:
        return tuple(self._outputs)

    def recent(self, key: str | StyleProfile) -> tuple[str, ...]:
        """Return the remembered outputs of ``key``, oldest first."""
        return tuple(self._outputs.get(_resolve_key(key), ()))

    def patterns(self, text: str) -> dict[str, str]:
        """Read the opening and the ending of a text, lowercased and tokenised."""
        if not isinstance(text, str):
            raise TypeError("patterns expects one string")
        sentences = _sentences(text)
        if not sentences:
            return {"opening": "", "ending": ""}
        return {
            "opening": _phrase(tokenize_words(sentences[0]), self.opening_words),
            "ending": _phrase(
                tokenize_words(sentences[-1]), self.ending_words, tail=True
            ),
        }

    def check(self, text: str, key: str | StyleProfile) -> RepetitionReport:
        """Measure the shapes of ``text`` against the window of ``key``.

        The candidate itself is not counted, so the first output of a profile is
        always fresh and a shape is judged only against what was kept before it.
        """
        name = _resolve_key(key)
        patterns = self.patterns(text)
        history = self._history(name)
        window = len(history)
        uses = [
            self._use(
                kind,
                patterns[kind],
                sum(1 for seen in history if seen[kind] == patterns[kind]),
                window,
            )
            for kind in PATTERN_KINDS
            if patterns[kind]
        ]
        return RepetitionReport(key=name, uses=tuple(uses))

    def remember(self, text: str, key: str | StyleProfile) -> None:
        """Add an output to the window of ``key``, dropping the oldest if full."""
        name = _resolve_key(key)
        if not self.patterns(text)["opening"]:
            raise ValueError("cannot remember a text without a measurable opening")
        self._outputs.setdefault(name, deque(maxlen=self.capacity)).append(text)

    def overused(self, key: str | StyleProfile) -> tuple[PatternUse, ...]:
        """Return the shapes a new output must avoid to stay under the limit."""
        name = _resolve_key(key)
        history = self._history(name)
        window = len(history)
        uses: list[PatternUse] = []
        for kind in PATTERN_KINDS:
            counts: dict[str, int] = {}
            for seen in history:
                if seen[kind]:
                    counts[seen[kind]] = counts.get(seen[kind], 0) + 1
            for pattern, count in counts.items():
                use = self._use(kind, pattern, count, window)
                if use.repeats:
                    uses.append(use)
        return tuple(sorted(uses, key=lambda use: (use.kind, -use.count, use.pattern)))

    def forget(self, key: str | StyleProfile) -> None:
        self._outputs.pop(_resolve_key(key), None)

    def clear(self) -> None:
        self._outputs.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "capacity": self.capacity,
            "max_share": self.max_share,
            "opening_words": self.opening_words,
            "ending_words": self.ending_words,
            "outputs": {key: list(outputs) for key, outputs in self._outputs.items()},
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OutputMemory:
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported memory schema version: {version!r}")
        memory = cls(
            capacity=int(data["capacity"]),
            max_share=float(data["max_share"]),
            opening_words=int(data["opening_words"]),
            ending_words=int(data["ending_words"]),
        )
        for key, outputs in data["outputs"].items():
            for text in outputs:
                memory.remember(str(text), str(key))
        return memory

    @classmethod
    def from_json(cls, payload: str) -> OutputMemory:
        return cls.from_dict(json.loads(payload))

    def _history(self, name: str) -> list[dict[str, str]]:
        return [self.patterns(text) for text in self._outputs.get(name, ())]

    def _use(self, kind: str, pattern: str, count: int, window: int) -> PatternUse:
        share = round(count / window, 6) if window else 0.0
        return PatternUse(
            kind=kind,
            pattern=pattern,
            count=count,
            window=window,
            share=share,
            max_share=self.max_share,
        )
