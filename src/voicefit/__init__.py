"""Fit text to a measured style profile."""

from voicefit.distance import (
    AXES,
    DEFAULT_TOLERANCE,
    PUNCTUATION_AXIS,
    AxisDeviation,
    AxisSpec,
    ProfileDistance,
    compare_profiles,
    compare_text,
    compare_texts,
)
from voicefit.profile import (
    DEFAULT_HEDGES,
    SCHEMA_VERSION,
    StyleProfile,
    build_profile,
    split_paragraphs,
    split_sentences,
    tokenize_words,
)

__all__ = [
    "AXES",
    "AxisDeviation",
    "AxisSpec",
    "DEFAULT_HEDGES",
    "DEFAULT_TOLERANCE",
    "PUNCTUATION_AXIS",
    "ProfileDistance",
    "SCHEMA_VERSION",
    "StyleProfile",
    "__version__",
    "build_profile",
    "compare_profiles",
    "compare_text",
    "compare_texts",
    "split_paragraphs",
    "split_sentences",
    "tokenize_words",
]

__version__ = "0.1.0"
