from __future__ import annotations

import json

import pytest

from voicefit import (
    SCHEMA_VERSION,
    OutputMemory,
    build_profile,
    profile_key,
)

CLEAN = "The migration ran clean. Nothing else broke."
SHUT = "The migration ran clean. The rollback stayed shut."
NUMBERS = "Numbers first, then the story. It shipped."
BETA = "Beta two landed late. Nobody noticed."
KEY = "handbook"


def test_patterns_read_the_first_and_the_last_sentence():
    memory = OutputMemory()
    assert memory.patterns(CLEAN) == {
        "opening": "the migration ran clean",
        "ending": "nothing else broke",
    }


def test_pattern_length_is_configurable():
    memory = OutputMemory(opening_words=2, ending_words=2)
    assert memory.patterns(CLEAN) == {
        "opening": "the migration",
        "ending": "else broke",
    }


def test_the_first_output_of_a_profile_is_fresh():
    memory = OutputMemory()
    report = memory.check(CLEAN, KEY)
    assert report.fresh
    assert report.key == KEY
    assert report.use("opening").window == 0
    assert report.use("opening").share == pytest.approx(0.0)


def test_repeating_an_opening_in_a_small_window_is_refused():
    memory = OutputMemory()
    memory.remember(CLEAN, KEY)
    report = memory.check(SHUT, KEY)
    assert not report.fresh
    assert [use.kind for use in report.repeats] == ["opening"]
    use = report.use("opening")
    assert use.pattern == "the migration ran clean"
    assert use.count == 1
    assert use.window == 1
    assert use.share == pytest.approx(1.0)
    assert memory.check(NUMBERS, KEY).fresh


def test_a_share_under_the_limit_is_allowed():
    memory = OutputMemory(max_share=0.34)
    for text in (CLEAN, NUMBERS, BETA):
        memory.remember(text, KEY)
    report = memory.check(SHUT, KEY)
    assert report.fresh
    assert report.use("opening").share == pytest.approx(1 / 3, abs=1e-6)
    memory.remember(SHUT, KEY)
    later = memory.check(CLEAN, KEY)
    assert not later.fresh
    assert later.use("opening").share == pytest.approx(0.5)


def test_capacity_forgets_the_oldest_output():
    memory = OutputMemory(capacity=2)
    for text in (CLEAN, NUMBERS, BETA):
        memory.remember(text, KEY)
    assert memory.recent(KEY) == (NUMBERS, BETA)
    assert len(memory) == 2
    assert memory.check(SHUT, KEY).fresh


def test_each_profile_keeps_its_own_window():
    memory = OutputMemory()
    memory.remember(CLEAN, "blog")
    assert not memory.check(SHUT, "blog").fresh
    assert memory.check(SHUT, "handbook").fresh
    assert memory.keys() == ("blog",)


def test_a_profile_can_stand_in_for_its_key():
    profile = build_profile([CLEAN])
    memory = OutputMemory()
    memory.remember(CLEAN, profile)
    assert memory.recent(profile_key(profile)) == (CLEAN,)
    assert not memory.check(SHUT, profile).fresh


def test_overused_lists_what_a_new_output_must_avoid():
    memory = OutputMemory()
    memory.remember(CLEAN, KEY)
    assert {(use.kind, use.pattern) for use in memory.overused(KEY)} == {
        ("opening", "the migration ran clean"),
        ("ending", "nothing else broke"),
    }
    assert memory.overused("empty") == ()


def test_a_window_can_be_forgotten():
    memory = OutputMemory()
    memory.remember(CLEAN, KEY)
    memory.forget(KEY)
    assert memory.recent(KEY) == ()
    assert memory.check(SHUT, KEY).fresh
    memory.remember(CLEAN, KEY)
    memory.clear()
    assert len(memory) == 0


def test_the_report_flags_the_shape_that_went_over():
    memory = OutputMemory()
    memory.remember(CLEAN, KEY)
    report = memory.check(SHUT, KEY).report()
    assert "opening" in report
    assert "the migration ran clean" in report
    assert "over" in report


def test_a_repetition_report_serialises_to_json():
    memory = OutputMemory()
    memory.remember(CLEAN, KEY)
    result = memory.check(SHUT, KEY)
    payload = json.loads(result.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["fresh"] is False
    assert payload["repeats"] == ["opening"]
    assert [use["kind"] for use in payload["uses"]] == ["opening", "ending"]
    assert result.failures == tuple(use.describe() for use in result.repeats)
    assert "opening" in result.failures[0]


def test_a_memory_survives_a_json_round_trip():
    memory = OutputMemory(capacity=3, max_share=0.5, opening_words=3, ending_words=2)
    memory.remember(CLEAN, KEY)
    restored = OutputMemory.from_json(memory.to_json())
    assert restored.capacity == 3
    assert restored.max_share == pytest.approx(0.5)
    assert restored.opening_words == 3
    assert restored.ending_words == 2
    assert restored.recent(KEY) == (CLEAN,)
    assert not restored.check(SHUT, KEY).fresh


def test_from_dict_rejects_an_unknown_schema_version():
    data = OutputMemory().to_dict()
    data["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError):
        OutputMemory.from_dict(data)


def test_settings_keys_and_texts_are_validated():
    with pytest.raises(ValueError):
        OutputMemory(capacity=0)
    with pytest.raises(ValueError):
        OutputMemory(max_share=1.5)
    with pytest.raises(ValueError):
        OutputMemory(opening_words=0)
    memory = OutputMemory()
    with pytest.raises(TypeError):
        memory.check(CLEAN, 7)
    with pytest.raises(ValueError):
        memory.check(CLEAN, "  ")
    with pytest.raises(TypeError):
        memory.patterns([CLEAN])
    with pytest.raises(ValueError):
        memory.remember("   ", KEY)
    with pytest.raises(KeyError):
        memory.check(CLEAN, KEY).use("middle")
