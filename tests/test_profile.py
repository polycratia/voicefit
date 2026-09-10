from __future__ import annotations

import json

import pytest

from voicefit import (
    SCHEMA_VERSION,
    StyleProfile,
    build_profile,
    split_sentences,
    tokenize_words,
)


def test_split_sentences_keeps_abbreviations_intact():
    text = "Dr. Adams reviewed the patch. It landed on Tuesday."
    assert split_sentences(text) == [
        "Dr. Adams reviewed the patch.",
        "It landed on Tuesday.",
    ]


def test_tokenize_words_keeps_contractions_numbers_and_hyphens():
    assert tokenize_words("It's a 3.5 well-known case.") == [
        "It's",
        "a",
        "3.5",
        "well-known",
        "case",
    ]


def test_sentence_length_mean_and_variance():
    profile = build_profile(["One two three. Four five."])
    assert profile.sentence_count == 2
    assert profile.word_count == 5
    assert profile.sentence_length_mean == pytest.approx(2.5)
    assert profile.sentence_length_variance == pytest.approx(0.25)
    assert profile.sentence_length_stdev == pytest.approx(0.5)


def test_question_share_counts_terminal_marks():
    profile = build_profile(["Why now? Because. Ready?"])
    assert profile.sentence_count == 3
    assert profile.question_share == pytest.approx(2 / 3, abs=1e-6)
    assert profile.exclamation_share == pytest.approx(0.0)


def test_hedge_rate_and_hedge_sentence_share():
    profile = build_profile(["This might work.", "It shipped."])
    assert profile.word_count == 5
    assert profile.hedge_rate == pytest.approx(0.2)
    assert profile.hedge_sentence_share == pytest.approx(0.5)


def test_multiword_hedges_are_matched():
    profile = build_profile(["It is sort of ready."])
    assert profile.word_count == 5
    assert profile.hedge_rate == pytest.approx(0.2)


def test_custom_hedge_list_replaces_the_default():
    profile = build_profile(["It shipped fine."], hedges=["fine"])
    assert profile.hedge_rate == pytest.approx(1 / 3, abs=1e-6)


def test_punctuation_rates_are_per_1000_words():
    profile = build_profile(["Yes, and no; maybe."])
    assert profile.word_count == 4
    assert profile.punctuation_per_1000_words["comma"] == pytest.approx(250.0)
    assert profile.punctuation_per_1000_words["semicolon"] == pytest.approx(250.0)
    assert profile.punctuation_per_1000_words["colon"] == pytest.approx(0.0)


def test_paragraph_metrics():
    profile = build_profile(["Alpha one. Beta two.\n\nGamma three."])
    assert profile.paragraph_count == 2
    assert profile.sentence_count == 3
    assert profile.sentences_per_paragraph_mean == pytest.approx(1.5)
    assert profile.words_per_paragraph_mean == pytest.approx(3.0)


def test_profile_survives_a_json_round_trip():
    profile = build_profile(
        [
            "The migration ran clean. No rollback was needed.",
            "Numbers first, then the story?\n\nIt might be that simple.",
        ]
    )
    payload = profile.to_json()
    assert json.loads(payload)["schema_version"] == SCHEMA_VERSION
    assert StyleProfile.from_json(payload) == profile


def test_from_dict_rejects_an_unknown_schema_version():
    data = build_profile(["A single sentence here."]).to_dict()
    data["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError):
        StyleProfile.from_dict(data)


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        build_profile([])
    with pytest.raises(ValueError):
        build_profile(["..."])


def test_a_single_string_is_not_a_corpus():
    with pytest.raises(TypeError):
        build_profile("One two three.")
