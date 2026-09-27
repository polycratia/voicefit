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
from voicefit.rewrite import (
    Model,
    RewriteAttempt,
    RewriteConstraints,
    RewriteResult,
    build_instruction,
    rewrite,
)
from voicefit.spans import (
    PROTECTED_KINDS,
    ProtectedSpan,
    SpanCheck,
    SpanReport,
    extract_spans,
    verify_spans,
)

__all__ = [
    "AXES",
    "AxisDeviation",
    "AxisSpec",
    "DEFAULT_HEDGES",
    "DEFAULT_TOLERANCE",
    "Model",
    "PROTECTED_KINDS",
    "PUNCTUATION_AXIS",
    "ProfileDistance",
    "ProtectedSpan",
    "RewriteAttempt",
    "RewriteConstraints",
    "RewriteResult",
    "SCHEMA_VERSION",
    "SpanCheck",
    "SpanReport",
    "StyleProfile",
    "__version__",
    "build_instruction",
    "build_profile",
    "compare_profiles",
    "compare_text",
    "compare_texts",
    "extract_spans",
    "rewrite",
    "split_paragraphs",
    "split_sentences",
    "tokenize_words",
    "verify_spans",
]

__version__ = "0.1.0"
