from __future__ import annotations

import json

import pytest

from voicefit import (
    RewriteConstraints,
    SCHEMA_VERSION,
    build_instruction,
    build_profile,
    rewrite,
)

SOURCE = (
    "The migration ran clean and needed no rollback. "
    "We bumped the client to v2.1.0.\n\n"
    "Notes live at https://example.com/runbook."
)
NOTHING_KEPT = "Nothing at all survived the rewrite."
DRIFTED = (
    "The migration ran clean and needed no rollback at all, and we then bumped "
    "the client to v2.1.0 while the notes live at https://example.com/runbook."
)


def test_the_instruction_carries_the_profile_the_spans_and_the_text():
    profile = build_profile([SOURCE])
    instruction = build_instruction(SOURCE, profile)
    assert f"{profile.sentence_length_mean:.1f}" in instruction
    assert '"v2.1.0"' in instruction
    assert '"https://example.com/runbook"' in instruction
    assert SOURCE.strip() in instruction


def test_notes_are_appended_to_the_rules():
    profile = build_profile([SOURCE])
    instruction = build_instruction(
        SOURCE, profile, constraints=RewriteConstraints(notes=["Keep the headings."])
    )
    assert "- Keep the headings." in instruction


def test_a_passing_rewrite_is_returned_after_one_call():
    profile = build_profile([SOURCE])
    seen: list[str] = []

    def model(instruction: str) -> str:
        seen.append(instruction)
        return SOURCE

    result = rewrite(SOURCE, profile, model)
    assert result.accepted
    assert result.text == SOURCE
    assert result.failures == ()
    assert len(result.attempts) == 1
    assert seen == [build_instruction(SOURCE, profile)]
    assert result.attempts[0].spans.intact
    assert result.attempts[0].distance is not None
    assert result.attempts[0].distance.matches


def test_a_failed_attempt_is_retried_once_with_the_failures_listed():
    profile = build_profile([SOURCE])
    outputs = iter([NOTHING_KEPT, SOURCE])
    seen: list[str] = []

    def model(instruction: str) -> str:
        seen.append(instruction)
        return next(outputs)

    result = rewrite(SOURCE, profile, model)
    assert result.accepted
    assert result.text == SOURCE
    assert len(result.attempts) == 2
    assert not result.attempts[0].passed
    assert "failed these checks" in seen[1]
    assert "Previous attempt" in seen[1]
    assert all(failure in seen[1] for failure in result.attempts[0].failures)


def test_a_rewrite_that_keeps_failing_gives_back_the_original():
    profile = build_profile([SOURCE])
    calls: list[str] = []

    def model(instruction: str) -> str:
        calls.append(instruction)
        return NOTHING_KEPT

    result = rewrite(SOURCE, profile, model)
    assert not result.accepted
    assert result.text == SOURCE
    assert len(calls) == 2
    assert len(result.attempts) == 2
    assert result.failures
    assert any("v2.1.0" in failure for failure in result.failures)


def test_an_empty_rewrite_is_a_failure():
    profile = build_profile([SOURCE])
    result = rewrite(SOURCE, profile, lambda instruction: "   ")
    assert not result.accepted
    assert result.text == SOURCE
    assert any("empty" in failure for failure in result.failures)
    assert result.attempts[0].distance is None


def test_a_wider_tolerance_accepts_a_drifted_rewrite():
    profile = build_profile([SOURCE])

    def model(instruction: str) -> str:
        return DRIFTED

    assert not rewrite(SOURCE, profile, model).accepted
    lenient = rewrite(
        SOURCE, profile, model, constraints=RewriteConstraints(tolerance=100.0)
    )
    assert lenient.accepted
    assert lenient.text == DRIFTED


def test_constraints_can_narrow_what_is_protected():
    profile = build_profile([SOURCE])
    output = "Notes live at https://example.com/runbook."

    def model(instruction: str) -> str:
        return output

    strict = rewrite(
        SOURCE, profile, model, constraints=RewriteConstraints(tolerance=100.0)
    )
    assert not strict.accepted
    loose = rewrite(
        SOURCE,
        profile,
        model,
        constraints=RewriteConstraints(tolerance=100.0, kinds=["url"]),
    )
    assert loose.accepted
    assert loose.text == output


def test_a_custom_hedge_list_is_used_for_the_checks():
    profile = build_profile([SOURCE], hedges=["clean"])
    constraints = RewriteConstraints(hedges=["clean"], tolerance=100.0)
    result = rewrite(SOURCE, profile, lambda instruction: SOURCE, constraints=constraints)
    assert result.accepted
    distance = result.attempts[0].distance
    assert distance is not None
    assert distance.axis("hedge_rate").deviation == pytest.approx(0.0)


def test_report_lists_each_attempt_and_the_outcome():
    profile = build_profile([SOURCE])
    report = rewrite(SOURCE, profile, lambda instruction: NOTHING_KEPT).report()
    assert "attempt 1" in report
    assert "attempt 2" in report
    assert "original" in report


def test_the_result_serialises_to_json():
    profile = build_profile([SOURCE])
    result = rewrite(SOURCE, profile, lambda instruction: NOTHING_KEPT)
    payload = json.loads(result.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["accepted"] is False
    assert payload["text"] == SOURCE
    assert payload["attempt_count"] == 2
    assert payload["failures"] == list(result.failures)
    assert [attempt["number"] for attempt in payload["attempts"]] == [1, 2]
    assert payload["attempts"][0]["spans"]["intact"] is False


def test_wrong_types_are_rejected():
    profile = build_profile([SOURCE])
    with pytest.raises(TypeError):
        rewrite([SOURCE], profile, lambda instruction: SOURCE)
    with pytest.raises(TypeError):
        rewrite(SOURCE, profile, lambda instruction: None)
