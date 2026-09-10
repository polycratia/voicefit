"""Measure a style profile from an author's own corpus.

A profile is a handful of numbers that serialise to JSON and can later be
compared against a rewrite: sentence length and its spread, hedging, the share
of questions, punctuation habits and paragraph shape.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_HEDGES",
    "SCHEMA_VERSION",
    "StyleProfile",
    "build_profile",
    "split_paragraphs",
    "split_sentences",
    "tokenize_words",
]

SCHEMA_VERSION = 1

DEFAULT_HEDGES: frozenset[str] = frozenset(
    {
        "a bit",
        "a little",
        "apparently",
        "approximately",
        "arguably",
        "as a rule",
        "could",
        "fairly",
        "for the most part",
        "generally",
        "i believe",
        "i think",
        "in general",
        "in most cases",
        "kind of",
        "largely",
        "likely",
        "mainly",
        "may",
        "maybe",
        "might",
        "more or less",
        "mostly",
        "not necessarily",
        "often",
        "or so",
        "perhaps",
        "possibly",
        "presumably",
        "probably",
        "quite",
        "rather",
        "relatively",
        "roughly",
        "seem",
        "seemed",
        "seems",
        "sometimes",
        "somewhat",
        "sort of",
        "suggest",
        "suggests",
        "tend to",
        "tends to",
        "to some extent",
        "typically",
        "usually",
        "we believe",
        "we think",
    }
)

_ABBREVIATIONS = frozenset(
    {
        "al",
        "approx",
        "cf",
        "co",
        "dr",
        "e.g",
        "eg",
        "etc",
        "fig",
        "i.e",
        "ie",
        "inc",
        "jr",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "sr",
        "st",
        "vs",
    }
)

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_SENTENCE_END = re.compile(r"([.!?\u2026]+)([)\"'\u201d\u2019\]]*)(\s+|$)")
_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.'\u2019-]*)\.$")
_TERMINAL_RUN = re.compile(r"[.!?\u2026]+$")
_WORD = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+(?:['\u2019-][^\W\d_]+)*")
_CLOSERS = " \t\n\"'\u201d\u2019)]\u00bb"

_PUNCTUATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "comma": re.compile(r","),
    "semicolon": re.compile(r";"),
    "colon": re.compile(r":"),
    "question_mark": re.compile(r"\?"),
    "exclamation_mark": re.compile(r"!"),
    "dash": re.compile(r"[\u2013\u2014]|(?<=\s)-(?=\s)"),
    "ellipsis": re.compile(r"\.{3}|\u2026"),
    "parenthesis": re.compile(r"[()]"),
    "quote": re.compile(r"[\"\u201c\u201d\u00ab\u00bb]"),
}


def split_paragraphs(text: str) -> list[str]:
    """Split a text into paragraphs on blank lines."""
    return [block.strip() for block in _PARAGRAPH_BREAK.split(text) if block.strip()]


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, keeping common abbreviations intact."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if (
            match.group(1) == "."
            and not match.group(2)
            and _is_abbreviation(text[: match.end(1)])
        ):
            continue
        sentence = text[start : match.end(2)].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def tokenize_words(text: str) -> list[str]:
    """Return the word tokens of a text, keeping contractions and hyphens."""
    return _WORD.findall(text)


def _is_abbreviation(chunk: str) -> bool:
    match = _TRAILING_WORD.search(chunk)
    if match is None:
        return False
    word = match.group(1).lower().rstrip(".")
    return len(word) == 1 or word in _ABBREVIATIONS


def _terminal_run(sentence: str) -> str:
    match = _TERMINAL_RUN.search(sentence.rstrip(_CLOSERS))
    return match.group(0) if match else ""


def _hedge_ngrams(phrases: Iterable[str]) -> frozenset[tuple[str, ...]]:
    ngrams = (tuple(token.lower() for token in tokenize_words(phrase)) for phrase in phrases)
    return frozenset(ngram for ngram in ngrams if ngram)


def _count_hedges(
    tokens: list[str], ngrams: frozenset[tuple[str, ...]], max_length: int
) -> int:
    hits = 0
    index = 0
    total = len(tokens)
    while index < total:
        for size in range(min(max_length, total - index), 0, -1):
            if tuple(tokens[index : index + size]) in ngrams:
                hits += 1
                index += size
                break
        else:
            index += 1
    return hits


def _round(value: float) -> float:
    return round(float(value), 6)


@dataclass(frozen=True, kw_only=True)
class StyleProfile:
    """Measurable style axes of a corpus.

    Lengths are counted in word tokens; shares are fractions of the sentence
    count; ``hedge_rate`` is hedge markers per word; punctuation is counted per
    1000 words.
    """

    text_count: int
    paragraph_count: int
    sentence_count: int
    word_count: int
    sentence_length_mean: float
    sentence_length_variance: float
    sentence_length_stdev: float
    word_length_mean: float
    question_share: float
    exclamation_share: float
    hedge_rate: float
    hedge_sentence_share: float
    punctuation_per_1000_words: dict[str, float]
    sentences_per_paragraph_mean: float
    words_per_paragraph_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "text_count": self.text_count,
            "paragraph_count": self.paragraph_count,
            "sentence_count": self.sentence_count,
            "word_count": self.word_count,
            "sentence_length_mean": self.sentence_length_mean,
            "sentence_length_variance": self.sentence_length_variance,
            "sentence_length_stdev": self.sentence_length_stdev,
            "word_length_mean": self.word_length_mean,
            "question_share": self.question_share,
            "exclamation_share": self.exclamation_share,
            "hedge_rate": self.hedge_rate,
            "hedge_sentence_share": self.hedge_sentence_share,
            "punctuation_per_1000_words": dict(self.punctuation_per_1000_words),
            "sentences_per_paragraph_mean": self.sentences_per_paragraph_mean,
            "words_per_paragraph_mean": self.words_per_paragraph_mean,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StyleProfile:
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported profile schema version: {version!r}")
        return cls(
            text_count=int(data["text_count"]),
            paragraph_count=int(data["paragraph_count"]),
            sentence_count=int(data["sentence_count"]),
            word_count=int(data["word_count"]),
            sentence_length_mean=float(data["sentence_length_mean"]),
            sentence_length_variance=float(data["sentence_length_variance"]),
            sentence_length_stdev=float(data["sentence_length_stdev"]),
            word_length_mean=float(data["word_length_mean"]),
            question_share=float(data["question_share"]),
            exclamation_share=float(data["exclamation_share"]),
            hedge_rate=float(data["hedge_rate"]),
            hedge_sentence_share=float(data["hedge_sentence_share"]),
            punctuation_per_1000_words={
                str(name): float(rate)
                for name, rate in data["punctuation_per_1000_words"].items()
            },
            sentences_per_paragraph_mean=float(data["sentences_per_paragraph_mean"]),
            words_per_paragraph_mean=float(data["words_per_paragraph_mean"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> StyleProfile:
        return cls.from_dict(json.loads(payload))


def build_profile(
    texts: Iterable[str], *, hedges: Iterable[str] | None = None
) -> StyleProfile:
    """Compute a style profile from an author's texts.

    Raises ``ValueError`` when the corpus yields no sentences or no words, so a
    caller never receives a profile built from nothing.
    """
    if isinstance(texts, str):
        raise TypeError("build_profile expects an iterable of texts, not a single string")

    hedge_ngrams = _hedge_ngrams(DEFAULT_HEDGES if hedges is None else hedges)
    max_hedge_length = max((len(ngram) for ngram in hedge_ngrams), default=1)

    corpus = list(texts)
    paragraphs: list[list[str]] = []
    for text in corpus:
        for paragraph in split_paragraphs(text):
            sentences = split_sentences(paragraph)
            if sentences:
                paragraphs.append(sentences)

    sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
    if not sentences:
        raise ValueError("corpus contains no sentences")

    sentence_words = [tokenize_words(sentence) for sentence in sentences]
    lengths = [len(words) for words in sentence_words]
    word_count = sum(lengths)
    if word_count == 0:
        raise ValueError("corpus contains no words")

    hedge_hits = 0
    hedge_sentences = 0
    for words in sentence_words:
        hits = _count_hedges(
            [word.lower() for word in words], hedge_ngrams, max_hedge_length
        )
        hedge_hits += hits
        hedge_sentences += 1 if hits else 0

    runs = [_terminal_run(sentence) for sentence in sentences]
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)

    joined = " ".join(sentences)
    punctuation = {
        name: _round(len(pattern.findall(joined)) * 1000 / word_count)
        for name, pattern in _PUNCTUATION_PATTERNS.items()
    }

    variance = statistics.pvariance(lengths)
    character_count = sum(len(word) for words in sentence_words for word in words)

    return StyleProfile(
        text_count=len(corpus),
        paragraph_count=paragraph_count,
        sentence_count=sentence_count,
        word_count=word_count,
        sentence_length_mean=_round(word_count / sentence_count),
        sentence_length_variance=_round(variance),
        sentence_length_stdev=_round(math.sqrt(variance)),
        word_length_mean=_round(character_count / word_count),
        question_share=_round(sum("?" in run for run in runs) / sentence_count),
        exclamation_share=_round(sum("!" in run for run in runs) / sentence_count),
        hedge_rate=_round(hedge_hits / word_count),
        hedge_sentence_share=_round(hedge_sentences / sentence_count),
        punctuation_per_1000_words=punctuation,
        sentences_per_paragraph_mean=_round(sentence_count / paragraph_count),
        words_per_paragraph_mean=_round(word_count / paragraph_count),
    )
