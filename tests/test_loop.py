"""Deterministic tests of the rewrite loop against a scripted model.

Every output the model hands back is written down here, so each check of the
loop, the retry and the fallback to the original are exercised without a
provider and without randomness.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from voicefit import (
    OutputMemory,
    RewriteConstraints,
    StyleProfile,
    build_instruction,
    build_profile,
    rewrite,
)

SOURCE = (
    "The migration ran clean and needed no rollback. "
    "We bumped the client to v2.1.0.\n\n"
    "Notes live at https://example.com/runbook."
)
# The same sentences in another order: every axis measures the same, the
# opening and the ending do not.
SHUFFLED = (
    "Notes live at https://example.com/runbook.\n\n"
    "We bumped the client to v2.1.0. "
    "The migration ran clean and needed no rollback."
)
DRIFTED = (
    "The migration ran clean and needed no rollback at all, and we then bumped "
    "the client to v2.1.0 while the notes live at https://example.com/runbook."
)
NOTHING_KEPT = "Nothing at all survived the rewrite."
NO_WORDS = "..."
BLANK = "   "
COUNTED = "Retry 3 times, then 3 more, and log it with `make audit`."
COUNTED_ONCE = "Retry 3 times with `make audit`."

LOOSE = RewriteConstraints(tolerance=100.0)
KEY = "handbook"
LOST_IDENTIFIER = (
    'protected identifier "v2.1.0" appears 0 times in the rewrite, expected 1'
)
LOST_URL = (
    'protected url "https://example.com/runbook" appears 0 times in the rewrite,'
    " expected 1"
)


class FakeModel:
    """A model that returns scripted outputs and keeps the instructions it saw.

    The last output is returned again once the script runs out, so one value
    stands for a model that keeps making the same mistake.
    """

    def __init__(self, *outputs: Any) -> None:
        if not outputs:
            raise ValueError("a fake model needs at least one scripted output")
        self._outputs = outputs
        self.instructions: list[str] = []

    def __call__(self, instruction: str) -> Any:
        self.instructions.append(instruction)
        return self._outputs[min(len(self.instructions), len(self._outputs)) - 1]

    @property
    def calls(self) -> int:
        return len(self.instructions)


@pytest.fixture
def profile() -> StyleProfile:
    return build_profile([SOURCE])


def test_a_rewrite_that_passes_every_check_is_returned_after_one_call(profile):
    model = FakeModel(SOURCE)
    result = rewrite(SOURCE, profile, model)
    assert result.accepted
    assert result.text == SOURCE
    assert result.source == SOURCE
    assert result.failures == ()
    assert model.calls == 1
    assert model.instructions == [build_instruction(SOURCE, profile)]
    attempt = result.attempts[0]
    assert attempt.number == 1
    assert attempt.passed
    assert attempt.instruction == model.instructions[0]
    assert attempt.output == SOURCE
    assert attempt.spans.intact
    assert attempt.distance is not None
    assert attempt.distance.matches
    assert attempt.repetition is None


def test_the_reordered_rewrite_measures_exactly_like_its_source(profile):
    result = rewrite(SOURCE, profile, FakeModel(SHUFFLED))
    assert result.accepted
    assert result.text == SHUFFLED
    distance = result.attempts[0].distance
    assert distance is not None
    assert distance.max_deviation == pytest.approx(0.0)


def test_a_lost_span_is_named_with_the_counts_that_were_wanted(profile):
    result = rewrite(SOURCE, profile, FakeModel(NOTHING_KEPT), constraints=LOOSE)
    assert not result.accepted
    assert result.text == SOURCE
    assert result.failures == (LOST_IDENTIFIER, LOST_URL)
    spans = result.attempts[0].spans
    assert spans.check("v2.1.0").missing == 1
    assert spans.check("https://example.com/runbook").found == 0


def test_a_span_used_twice_in_the_source_must_survive_twice():
    counted = build_profile([COUNTED])
    result = rewrite(COUNTED, counted, FakeModel(COUNTED_ONCE), constraints=LOOSE)
    assert not result.accepted
    assert result.text == COUNTED
    assert result.failures == (
        'protected number "3" appears 1 times in the rewrite, expected 2',
    )
    assert result.attempts[0].spans.check("make audit").survived


def test_an_empty_rewrite_fails_before_it_is_measured(profile):
    model = FakeModel(BLANK)
    result = rewrite(SOURCE, profile, model, constraints=LOOSE)
    assert not result.accepted
    assert result.text == SOURCE
    attempt = result.attempts[0]
    assert attempt.failures[0] == "the model returned an empty rewrite"
    assert attempt.distance is None
    assert attempt.repetition is None
    assert model.calls == 2


def test_a_rewrite_without_a_measurable_sentence_is_refused(profile):
    result = rewrite(SOURCE, profile, FakeModel(NO_WORDS), constraints=LOOSE)
    assert not result.accepted
    assert result.text == SOURCE
    attempt = result.attempts[0]
    assert attempt.failures[0] == "the rewrite has no measurable sentence"
    assert attempt.distance is None


def test_an_axis_past_the_tolerance_fails_the_attempt(profile):
    model = FakeModel(DRIFTED)
    result = rewrite(SOURCE, profile, model)
    assert not result.accepted
    assert result.text == SOURCE
    assert model.calls == 2
    assert result.attempts[0].spans.intact
    assert any(
        failure.startswith("sentence_length_mean is above the target")
        for failure in result.failures
    )


def test_the_same_drift_passes_under_a_wider_tolerance(profile):
    result = rewrite(SOURCE, profile, FakeModel(DRIFTED), constraints=LOOSE)
    assert result.accepted
    assert result.text == DRIFTED
    assert result.failures == ()


def test_a_repeated_shape_fails_like_a_lost_span(profile):
    memory = OutputMemory()
    memory.remember(SOURCE, KEY)
    result = rewrite(SOURCE, profile, FakeModel(SOURCE), memory=memory, key=KEY)
    assert not result.accepted
    assert result.text == SOURCE
    repetition = result.attempts[0].repetition
    assert repetition is not None
    assert not repetition.fresh
    assert [use.kind for use in repetition.repeats] == ["opening", "ending"]
    assert result.failures == repetition.failures
    assert memory.recent(KEY) == (SOURCE,)


def test_the_shapes_to_avoid_are_named_in_every_instruction(profile):
    memory = OutputMemory()
    memory.remember(SOURCE, KEY)
    model = FakeModel(SOURCE)
    rewrite(SOURCE, profile, model, memory=memory, key=KEY)
    assert model.calls == 2
    for instruction in model.instructions:
        assert '- Do not open with: "the migration ran clean".' in instruction
        assert '- Do not end with: "https example com runbook".' in instruction


def test_a_failed_attempt_is_retried_once_with_its_failures_quoted(profile):
    model = FakeModel(NOTHING_KEPT, SHUFFLED)
    result = rewrite(SOURCE, profile, model)
    assert result.accepted
    assert result.text == SHUFFLED
    assert model.calls == 2
    first, second = result.attempts
    assert not first.passed
    assert second.passed
    assert first.output == NOTHING_KEPT
    assert second.output == SHUFFLED
    retry = model.instructions[1]
    assert "The previous attempt failed these checks:" in retry
    assert all(failure in retry for failure in first.failures)
    assert NOTHING_KEPT in retry
    assert retry == build_instruction(
        SOURCE, profile, failures=first.failures, previous=NOTHING_KEPT
    )


def test_the_loop_gives_back_the_original_after_the_second_failure(profile):
    model = FakeModel(NOTHING_KEPT)
    result = rewrite(SOURCE, profile, model)
    assert not result.accepted
    assert result.text == SOURCE
    assert model.calls == 2
    assert [attempt.number for attempt in result.attempts] == [1, 2]
    assert not any(attempt.passed for attempt in result.attempts)
    assert result.failures == result.attempts[-1].failures


def test_only_an_accepted_rewrite_enters_the_memory(profile):
    memory = OutputMemory()
    failing = rewrite(SOURCE, profile, FakeModel(NOTHING_KEPT), memory=memory, key=KEY)
    assert not failing.accepted
    assert memory.recent(KEY) == ()

    retried = rewrite(
        SOURCE, profile, FakeModel(NOTHING_KEPT, SHUFFLED), memory=memory, key=KEY
    )
    assert retried.accepted
    assert memory.recent(KEY) == (SHUFFLED,)
    assert len(memory) == 1


def test_the_report_and_the_json_record_the_fallback(profile):
    result = rewrite(SOURCE, profile, FakeModel(NOTHING_KEPT))
    report = result.report()
    assert "attempt 1: failed" in report
    assert "attempt 2: failed" in report
    assert "result: original returned" in report

    payload = json.loads(result.to_json())
    assert payload["accepted"] is False
    assert payload["attempt_count"] == 2
    assert payload["text"] == SOURCE
    assert payload["source"] == SOURCE
    assert payload["failures"] == list(result.failures)
    assert [attempt["number"] for attempt in payload["attempts"]] == [1, 2]
    assert payload["attempts"][0]["spans"]["intact"] is False
    assert payload["attempts"][0]["distance"]["matches"] is False


def test_the_report_of_an_accepted_retry_names_the_attempt_that_passed(profile):
    report = rewrite(SOURCE, profile, FakeModel(NOTHING_KEPT, SHUFFLED)).report()
    assert "attempt 1: failed" in report
    assert "attempt 2: ok" in report
    assert "result: rewrite accepted" in report


def test_a_model_that_does_not_return_text_stops_the_loop(profile):
    model = FakeModel(None)
    with pytest.raises(TypeError):
        rewrite(SOURCE, profile, model)
    assert model.calls == 1


def test_two_runs_of_the_same_script_produce_the_same_report(profile):
    first = rewrite(SOURCE, profile, FakeModel(NOTHING_KEPT, DRIFTED))
    second = rewrite(SOURCE, profile, FakeModel(NOTHING_KEPT, DRIFTED))
    assert not first.accepted
    assert first.to_dict() == second.to_dict()
