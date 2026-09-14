from __future__ import annotations

import json

import pytest

from voicefit import (
    AXES,
    SCHEMA_VERSION,
    build_profile,
    compare_profiles,
    compare_text,
    compare_texts,
)

TWO_SENTENCES = "One two three four. One two three four."
EIGHT_WORDS = "One two three four five six seven eight."


def test_a_text_matches_the_profile_it_came_from():
    texts = ["Alpha one two. Beta three four.\n\nGamma five six."]
    profile = build_profile(texts)
    result = compare_texts(texts, profile)
    assert result.matches
    assert result.off_profile == ()
    assert result.max_deviation == pytest.approx(0.0)
    assert all(axis.direction == "on" for axis in result.axes)


def test_longer_sentences_deviate_above_the_profile():
    profile = build_profile([TWO_SENTENCES])
    result = compare_text(EIGHT_WORDS, profile)
    axis = result.axis("sentence_length_mean")
    assert axis.target == pytest.approx(4.0)
    assert axis.value == pytest.approx(8.0)
    assert axis.difference == pytest.approx(4.0)
    assert axis.scale == pytest.approx(1.0)
    assert axis.deviation == pytest.approx(4.0)
    assert axis.direction == "above"
    assert not axis.within_tolerance
    assert not result.matches
    assert result.worst.axis == "sentence_length_mean"
    assert "sentence_length_mean" in [off.axis for off in result.off_profile]


def test_fewer_sentences_per_paragraph_deviate_below_the_profile():
    profile = build_profile([TWO_SENTENCES])
    axis = compare_text(EIGHT_WORDS, profile).axis("sentences_per_paragraph_mean")
    assert axis.target == pytest.approx(2.0)
    assert axis.value == pytest.approx(1.0)
    assert axis.deviation == pytest.approx(-2.0)
    assert axis.direction == "below"


def test_a_wider_tolerance_accepts_a_wider_drift():
    profile = build_profile([TWO_SENTENCES])
    assert not compare_text(EIGHT_WORDS, profile).matches
    assert compare_text(EIGHT_WORDS, profile, tolerance=5.0).matches


def test_a_scale_override_rescales_one_axis():
    profile = build_profile([TWO_SENTENCES])
    result = compare_text(
        EIGHT_WORDS, profile, scales={"sentence_length_mean": 2.0}
    )
    axis = result.axis("sentence_length_mean")
    assert axis.scale == pytest.approx(2.0)
    assert axis.deviation == pytest.approx(2.0)


def test_unknown_or_impossible_scales_are_rejected():
    profile = build_profile([TWO_SENTENCES])
    with pytest.raises(ValueError):
        compare_text(EIGHT_WORDS, profile, scales={"sentence_lenght_mean": 2.0})
    with pytest.raises(ValueError):
        compare_text(EIGHT_WORDS, profile, scales={"sentence_length_mean": 0.0})


def test_punctuation_is_reported_per_mark():
    profile = build_profile(["One two three four."])
    axis = compare_text("Yes, and no.", profile).axis("punctuation.comma")
    assert axis.unit == "per 1000 words"
    assert axis.target == pytest.approx(0.0)
    assert axis.value == pytest.approx(1000 / 3, abs=1e-3)
    assert axis.scale == pytest.approx(2.0)
    assert axis.deviation > 0


def test_every_declared_axis_is_reported_with_its_unit():
    profile = build_profile(["One two three."])
    result = compare_text("One two three.", profile)
    for spec in AXES:
        assert result.axis(spec.name).unit == spec.unit


def test_axis_lookup_rejects_an_unknown_name():
    profile = build_profile(["One two three."])
    with pytest.raises(KeyError):
        compare_text("One two three.", profile).axis("mood")


def test_mean_absolute_deviation_averages_the_axes():
    profile = build_profile([TWO_SENTENCES])
    result = compare_text(EIGHT_WORDS, profile)
    expected = sum(abs(axis.deviation) for axis in result.axes) / len(result.axes)
    assert result.mean_absolute_deviation == pytest.approx(expected, abs=1e-6)


def test_two_profiles_can_be_compared_directly():
    profile = build_profile([TWO_SENTENCES])
    sample = build_profile([EIGHT_WORDS])
    result = compare_profiles(sample, profile)
    assert result.sample == sample
    assert result.axis("sentence_length_mean").deviation == pytest.approx(4.0)


def test_report_names_every_axis_and_flags_the_ones_off_profile():
    profile = build_profile([TWO_SENTENCES])
    result = compare_text(EIGHT_WORDS, profile)
    report = result.report()
    for axis in result.axes:
        assert axis.axis in report
    assert "off" in report


def test_distance_serialises_to_json_with_every_axis():
    profile = build_profile(["Alpha one. Beta two.\n\nGamma three."])
    result = compare_text("Alpha one, beta two; gamma three?", profile)
    payload = json.loads(result.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["tolerance"] == pytest.approx(result.tolerance)
    assert [axis["axis"] for axis in payload["axes"]] == [
        axis.axis for axis in result.axes
    ]
    assert set(payload["off_profile"]) == {axis.axis for axis in result.off_profile}
    assert payload["sample"]["sentence_count"] == 1


def test_a_custom_hedge_list_is_used_for_the_sample_too():
    profile = build_profile(["It shipped fine."], hedges=["fine"])
    result = compare_text("It shipped fine.", profile, hedges=["fine"])
    assert result.axis("hedge_rate").deviation == pytest.approx(0.0)


def test_compare_text_wants_a_single_string():
    profile = build_profile(["One two three."])
    with pytest.raises(TypeError):
        compare_text(["One two three."], profile)
