"""Edge cases and failure modes of the spans, the rewrite loop and the CLI.

These checks sit next to the per-module suites: they cover the edges -- an
empty text, a number glued to a thousands separator, a model command that
cannot be run -- rather than the behaviour each module advertises.
"""

from __future__ import annotations

import io
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest

from voicefit import (
    OutputMemory,
    RewriteConstraints,
    StyleProfile,
    build_instruction,
    build_profile,
    extract_spans,
    profile_key,
    rewrite,
    verify_spans,
)
from voicefit.cli import main

SOURCE = (
    "The migration ran clean and needed no rollback. "
    "We bumped the client to v2.1.0.\n\n"
    "Notes live at https://example.com/runbook."
)
NOTHING_KEPT = "Nothing at all survived the rewrite."
NO_WORDS = "..."
INVENTED = (
    "The migration ran clean and needed no rollback. "
    "We bumped the client to v2.1.0 after 4 retries.\n\n"
    "Notes live at https://example.com/runbook."
)
SHORT = "One two three four. One two three four."
LONG = "One two three four five six seven eight."
CLI_SOURCE = (
    "The migration ran clean and needed no rollback. "
    "We bumped the client to v2.1.0."
)

LOOSE = RewriteConstraints(tolerance=100.0)
LOST_IDENTIFIER = (
    'protected identifier "v2.1.0" appears 0 times in the rewrite, expected 1'
)
LOST_URL = (
    'protected url "https://example.com/runbook" appears 0 times in the rewrite,'
    " expected 1"
)
ADDED_NUMBER = 'added number "4" appears 1 times in the rewrite, expected 0'


class Scripted:
    """A model returning written-down outputs, repeating the last one."""

    def __init__(self, *outputs: Any) -> None:
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


def test_an_empty_text_has_no_spans_and_is_trivially_intact():
    assert extract_spans("") == ()
    result = verify_spans("", "")
    assert result.checks == ()
    assert result.added == ()
    assert result.intact


def test_an_empty_source_reports_every_span_of_the_rewrite_as_added():
    result = verify_spans("", "Retry 3 times.")
    assert not result.intact
    assert result.checks == ()
    assert [check.text for check in result.added] == ["3"]


def test_an_empty_kind_list_protects_nothing():
    source = "Retry 3 times at https://example.com."
    assert extract_spans(source, kinds=[]) == ()
    result = verify_spans(source, "Retry a few times, or 5.", kinds=[])
    assert result.checks == ()
    assert result.added == ()
    assert result.intact


def test_kinds_may_be_any_iterable_but_not_a_single_string():
    spans = extract_spans("Retry 3 times.", kinds=iter(["number"]))
    assert [span.text for span in spans] == ["3"]
    with pytest.raises(TypeError):
        extract_spans("Retry 3 times.", kinds="number")


def test_a_parenthesised_url_keeps_only_its_address():
    spans = extract_spans("Notes (https://example.com/a) are short.")
    assert [(span.kind, span.text) for span in spans] == [
        ("url", "https://example.com/a")
    ]


def test_an_email_address_is_protected_like_a_url():
    spans = extract_spans("Write to hey@example.com, please.")
    assert [(span.kind, span.text) for span in spans] == [("url", "hey@example.com")]
    result = verify_spans("Write to hey@example.com.", "Write to the team.")
    assert [check.text for check in result.lost] == ["hey@example.com"]


def test_an_empty_code_span_is_not_a_span():
    assert extract_spans("Run ` ` or:\n\n```sh\n```\n") == ()


def test_a_clock_time_abbreviation_is_not_an_identifier():
    spans = extract_spans("Ship at 5 p.m. tomorrow.")
    assert [(span.kind, span.text) for span in spans] == [("number", "5")]


def test_a_currency_sign_belongs_to_the_number_it_marks():
    spans = extract_spans("It costs $1,000 today.")
    assert [(span.kind, span.text) for span in spans] == [("number", "$1,000")]
    result = verify_spans("It costs $1,000 today.", "It costs 1,000 today.")
    assert not result.intact
    assert result.check("$1,000").missing == 1
    assert result.added == ()


def test_a_number_glued_to_a_thousands_separator_does_not_count():
    result = verify_spans("We saw 1 error.", "We saw 1,500 errors.")
    assert result.check("1").expected == 1
    assert result.check("1").found == 0
    assert [check.text for check in result.added] == ["1,500"]


def test_a_changed_capital_counts_as_lost_and_as_invented():
    result = verify_spans("Bump to v2.1.0.", "Bump to V2.1.0.")
    assert not result.intact
    assert result.check("v2.1.0").found == 0
    assert [check.text for check in result.added] == ["V2.1.0"]


def test_a_long_span_is_shortened_in_the_report():
    code = "deploy --target production --wait 300 --retries 3 --verbose"
    result = verify_spans(f"Run `{code}` now.", "Run it now.")
    assert [check.text for check in result.lost] == [code]
    assert "\u2026" in result.report()


def test_build_instruction_wants_a_source_string(profile):
    with pytest.raises(TypeError):
        build_instruction([SOURCE], profile)


def test_a_blank_previous_attempt_is_not_quoted_back(profile):
    instruction = build_instruction(
        SOURCE, profile, failures=("some check failed",), previous="   "
    )
    assert "The previous attempt failed these checks:" in instruction
    assert "- some check failed" in instruction
    assert "Previous attempt:" not in instruction
    assert instruction.endswith(f"Text:\n{SOURCE.strip()}")


def test_a_share_of_zero_is_spelled_out_as_none(profile):
    instruction = build_instruction(SOURCE, profile)
    assert "- questions in none of the sentences" in instruction
    assert "- exclamations in none of the sentences" in instruction


def test_notes_are_listed_after_the_standing_rules(profile):
    instruction = build_instruction(
        SOURCE,
        profile,
        constraints=RewriteConstraints(notes=("Keep the headings.", "Keep the list.")),
    )
    assert (
        "- Return the rewritten text only, with no commentary.\n"
        "- Keep the headings.\n"
        "- Keep the list."
    ) in instruction


def test_protecting_nothing_accepts_a_rewrite_that_kept_nothing(profile):
    result = rewrite(
        SOURCE,
        profile,
        Scripted(NOTHING_KEPT),
        constraints=RewriteConstraints(tolerance=100.0, kinds=[]),
    )
    assert result.accepted
    assert result.text == NOTHING_KEPT
    assert result.attempts[0].spans.checks == ()
    assert result.attempts[0].spans.added == ()
    assert "Reproduce these" not in result.attempts[0].instruction


def test_an_unmeasurable_rewrite_still_reports_the_spans_it_lost(profile):
    attempt = rewrite(SOURCE, profile, Scripted(NO_WORDS), constraints=LOOSE).attempts[0]
    assert attempt.distance is None
    assert attempt.failures == (
        "the rewrite has no measurable sentence",
        LOST_IDENTIFIER,
        LOST_URL,
    )


def test_only_the_last_attempt_decides_the_reported_failures(profile):
    model = Scripted(NOTHING_KEPT, INVENTED)
    result = rewrite(SOURCE, profile, model, constraints=LOOSE)
    assert not result.accepted
    assert result.text == SOURCE
    assert model.calls == 2
    assert result.attempts[0].failures == (LOST_IDENTIFIER, LOST_URL)
    assert result.failures == (ADDED_NUMBER,)


def test_a_model_that_stops_returning_text_on_the_retry_raises(profile):
    model = Scripted(NOTHING_KEPT, None)
    with pytest.raises(TypeError):
        rewrite(SOURCE, profile, model)
    assert model.calls == 2


def test_an_empty_memory_names_no_shape_to_avoid(profile):
    memory = OutputMemory()
    model = Scripted(SOURCE)
    result = rewrite(SOURCE, profile, model, memory=memory, key="handbook")
    assert result.accepted
    assert "Do not open with" not in model.instructions[0]
    assert "Do not end with" not in model.instructions[0]
    repetition = result.attempts[0].repetition
    assert repetition is not None
    assert repetition.fresh
    assert memory.recent("handbook") == (SOURCE,)


def test_the_report_indents_the_failures_of_each_attempt(profile):
    report = rewrite(
        SOURCE, profile, Scripted(NOTHING_KEPT), constraints=LOOSE
    ).report()
    lines = report.splitlines()
    assert lines[0] == "attempt 1: failed"
    assert f"  - {LOST_IDENTIFIER}" in lines
    assert lines[-1] == "result: original returned"


def test_the_json_can_be_written_on_one_line(profile):
    payload = rewrite(SOURCE, profile, Scripted(SOURCE)).to_json(indent=None)
    assert "\n" not in payload
    assert json.loads(payload)["accepted"] is True


def run(argv: list[str], *, stdin: str = "") -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    code = main(argv, stdin=io.StringIO(stdin), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def profile_file(tmp_path: Path, texts: list[str]) -> str:
    path = tmp_path / "profile.json"
    path.write_text(build_profile(texts).to_json(), encoding="utf-8")
    return str(path)


def model_command(
    tmp_path: Path, text: str, *, status: int = 0, noise: str = ""
) -> str:
    script = tmp_path / "model.py"
    script.write_text(
        "import sys\n"
        "sys.stdin.read()\n"
        f"sys.stderr.write({noise!r})\n"
        f"sys.stdout.write({text!r})\n"
        f"sys.exit({status})\n",
        encoding="utf-8",
    )
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def test_the_model_stderr_is_quoted_when_the_command_fails(tmp_path):
    code, out, err = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [CLI_SOURCE]),
            "--model",
            model_command(tmp_path, "", status=1, noise="out of\n quota\n"),
        ],
        stdin=CLI_SOURCE,
    )
    assert code == 2
    assert out == ""
    assert "out of quota" in err


def test_a_model_command_that_cannot_be_run_is_reported(tmp_path):
    code, _, err = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [CLI_SOURCE]),
            "--model",
            str(tmp_path / "missing-model"),
        ],
        stdin=CLI_SOURCE,
    )
    assert code == 2
    assert "cannot run the model command" in err


def test_an_empty_model_command_is_a_usage_error(tmp_path):
    code, _, err = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [CLI_SOURCE]),
            "--model",
            "   ",
        ],
        stdin=CLI_SOURCE,
    )
    assert code == 2
    assert "--model" in err


def test_a_dry_run_never_reaches_the_model(tmp_path):
    code, out, _ = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [CLI_SOURCE]),
            "--dry-run",
            "--model",
            model_command(tmp_path, "", status=3),
        ],
        stdin=CLI_SOURCE,
    )
    assert code == 0
    assert '"v2.1.0"' in json.loads(out)["instruction"]


def test_an_unknown_span_kind_is_refused_by_the_parser(tmp_path, capsys):
    with pytest.raises(SystemExit) as refused:
        main(
            [
                "fit",
                "--profile",
                profile_file(tmp_path, [CLI_SOURCE]),
                "--dry-run",
                "--kind",
                "urls",
            ]
        )
    assert refused.value.code == 2
    assert "--kind" in capsys.readouterr().err


def test_a_negative_tolerance_is_reported(tmp_path):
    code, out, err = run(
        ["diff", "--profile", profile_file(tmp_path, [SHORT]), "--tolerance", "-1"],
        stdin=SHORT,
    )
    assert code == 2
    assert out == ""
    assert "tolerance" in err


def test_a_scale_without_a_value_is_rejected(tmp_path):
    code, _, err = run(
        [
            "diff",
            "--profile",
            profile_file(tmp_path, [SHORT]),
            "--scale",
            "sentence_length_mean=",
        ],
        stdin=SHORT,
    )
    assert code == 2
    assert "AXIS=N" in err


def test_a_scale_may_be_spelled_with_spaces(tmp_path):
    code, out, _ = run(
        [
            "diff",
            "--profile",
            profile_file(tmp_path, [SHORT]),
            "--scale",
            " sentence_length_mean = 4 ",
        ],
        stdin=LONG,
    )
    assert code == 1
    axes = {axis["axis"]: axis for axis in json.loads(out)["axes"]}
    assert axes["sentence_length_mean"]["scale"] == pytest.approx(4.0)


def test_a_report_that_cannot_be_written_is_reported(tmp_path):
    code, out, err = run(
        ["measure", "-o", str(tmp_path / "missing" / "out.json")], stdin=SHORT
    )
    assert code == 2
    assert out == ""
    assert "cannot write" in err


def test_compact_prints_the_report_on_one_line():
    code, out, _ = run(["measure", "--compact"], stdin=SHORT)
    assert code == 0
    assert out.strip().count("\n") == 0
    assert json.loads(out)["sentence_count"] == 2


def test_stdin_cannot_serve_as_the_profile_and_the_text_at_once():
    code, _, err = run(
        ["fit", "--profile", "-", "--dry-run"],
        stdin=build_profile([CLI_SOURCE]).to_json(),
    )
    assert code == 2
    assert "stdin" in err


def test_a_hedge_list_cannot_share_stdin_with_the_corpus():
    code, _, err = run(["measure", "--hedges", "-"], stdin="clean\n")
    assert code == 2
    assert "stdin" in err


def test_measure_mixes_stdin_with_a_file(tmp_path):
    first = tmp_path / "a.txt"
    first.write_text(LONG, encoding="utf-8")
    code, out, _ = run(["measure", "-", str(first)], stdin=SHORT)
    assert code == 0
    assert json.loads(out) == build_profile([SHORT, LONG]).to_dict()


def test_the_memory_file_is_written_even_when_the_rewrite_is_refused(tmp_path):
    memory_path = tmp_path / "memory.json"
    code, out, _ = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [CLI_SOURCE]),
            "--model",
            model_command(tmp_path, NOTHING_KEPT),
            "--memory",
            str(memory_path),
            "--key",
            "handbook",
        ],
        stdin=CLI_SOURCE,
    )
    assert code == 1
    assert json.loads(out)["accepted"] is False
    restored = OutputMemory.from_json(memory_path.read_text(encoding="utf-8"))
    assert restored.recent("handbook") == ()


def test_a_memory_without_a_key_is_filed_under_the_profile(tmp_path):
    memory_path = tmp_path / "memory.json"
    code, _, _ = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [CLI_SOURCE]),
            "--model",
            model_command(tmp_path, CLI_SOURCE),
            "--memory",
            str(memory_path),
        ],
        stdin=CLI_SOURCE,
    )
    assert code == 0
    restored = OutputMemory.from_json(memory_path.read_text(encoding="utf-8"))
    assert restored.keys() == (profile_key(build_profile([CLI_SOURCE])),)


def test_a_memory_that_is_a_directory_is_reported(tmp_path):
    code, _, err = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [CLI_SOURCE]),
            "--dry-run",
            "--memory",
            str(tmp_path),
        ],
        stdin=CLI_SOURCE,
    )
    assert code == 2
    assert "cannot read" in err


def test_a_dry_run_does_not_create_the_memory_file(tmp_path):
    memory_path = tmp_path / "memory.json"
    code, out, _ = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [CLI_SOURCE]),
            "--dry-run",
            "--memory",
            str(memory_path),
        ],
        stdin=CLI_SOURCE,
    )
    assert code == 0
    assert json.loads(out)["avoid"] == []
    assert not memory_path.exists()
