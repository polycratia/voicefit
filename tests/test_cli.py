from __future__ import annotations

import io
import json
import shlex
import sys
from pathlib import Path

import pytest

from voicefit import SCHEMA_VERSION, OutputMemory, __version__, build_profile
from voicefit.cli import main

CORPUS = "The migration ran clean. No rollback was needed."
STORY = "Numbers first, then the story."
SHORT = "One two three four. One two three four."
LONG = "One two three four five six seven eight."
SOURCE = (
    "The migration ran clean and needed no rollback. "
    "We bumped the client to v2.1.0."
)
NOTHING_KEPT = "Nothing at all survived the rewrite."


def run(argv: list[str], *, stdin: str = "") -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    code = main(argv, stdin=io.StringIO(stdin), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def profile_file(tmp_path: Path, texts: list[str]) -> str:
    path = tmp_path / "profile.json"
    path.write_text(build_profile(texts).to_json(), encoding="utf-8")
    return str(path)


def model_command(tmp_path: Path, text: str, *, status: int = 0) -> str:
    script = tmp_path / "model.py"
    script.write_text(
        "import sys\n"
        "sys.stdin.read()\n"
        f"sys.stdout.write({text!r})\n"
        f"sys.exit({status})\n",
        encoding="utf-8",
    )
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def axis_of(payload: dict, name: str) -> dict:
    for axis in payload["axes"]:
        if axis["axis"] == name:
            return axis
    raise AssertionError(f"no axis {name} in the report")


def test_measure_reads_stdin_and_prints_a_profile():
    code, out, err = run(["measure"], stdin=CORPUS)
    payload = json.loads(out)
    assert code == 0
    assert err == ""
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["sentence_count"] == 2


def test_measure_reads_several_files_as_one_corpus(tmp_path):
    first = tmp_path / "a.txt"
    first.write_text(CORPUS, encoding="utf-8")
    second = tmp_path / "b.txt"
    second.write_text(STORY, encoding="utf-8")
    code, out, _ = run(["measure", str(first), str(second)])
    assert code == 0
    assert json.loads(out) == build_profile([CORPUS, STORY]).to_dict()


def test_measure_takes_a_custom_hedge_list(tmp_path):
    hedges = tmp_path / "hedges.txt"
    hedges.write_text("clean\n\nrollback\n", encoding="utf-8")
    code, out, _ = run(["measure", "--hedges", str(hedges)], stdin=CORPUS)
    assert code == 0
    assert json.loads(out)["hedge_rate"] == pytest.approx(
        build_profile([CORPUS], hedges=["clean", "rollback"]).hedge_rate
    )


def test_an_empty_hedge_file_is_a_usage_error(tmp_path):
    hedges = tmp_path / "hedges.txt"
    hedges.write_text("\n\n", encoding="utf-8")
    code, out, err = run(["measure", "--hedges", str(hedges)], stdin=CORPUS)
    assert code == 2
    assert out == ""
    assert "hedge" in err


def test_a_corpus_without_words_is_a_usage_error():
    code, out, err = run(["measure"], stdin="...")
    assert code == 2
    assert out == ""
    assert err.startswith("voicefit:")


def test_a_missing_file_is_reported(tmp_path):
    code, _, err = run(["measure", str(tmp_path / "nope.txt")])
    assert code == 2
    assert "nope.txt" in err


def test_the_report_can_be_written_to_a_file_on_one_line(tmp_path):
    target = tmp_path / "out.json"
    code, out, _ = run(["measure", "-o", str(target), "--compact"], stdin=CORPUS)
    written = target.read_text(encoding="utf-8")
    assert code == 0
    assert out == ""
    assert json.loads(written)["sentence_count"] == 2
    assert written.strip().count("\n") == 0


def test_diff_reports_a_text_on_profile(tmp_path):
    code, out, _ = run(
        ["diff", "--profile", profile_file(tmp_path, [SHORT])], stdin=SHORT
    )
    payload = json.loads(out)
    assert code == 0
    assert payload["matches"] is True
    assert payload["off_profile"] == []


def test_diff_exits_one_when_an_axis_is_off_profile(tmp_path):
    code, out, _ = run(
        ["diff", "--profile", profile_file(tmp_path, [SHORT])], stdin=LONG
    )
    payload = json.loads(out)
    assert code == 1
    assert payload["matches"] is False
    assert "sentence_length_mean" in payload["off_profile"]


def test_a_wider_tolerance_accepts_the_drift(tmp_path):
    code, out, _ = run(
        ["diff", "--profile", profile_file(tmp_path, [SHORT]), "--tolerance", "100"],
        stdin=LONG,
    )
    assert code == 0
    assert json.loads(out)["matches"] is True


def test_a_scale_override_rescales_one_axis(tmp_path):
    _, out, _ = run(
        [
            "diff",
            "--profile",
            profile_file(tmp_path, [SHORT]),
            "--scale",
            "sentence_length_mean=4",
        ],
        stdin=LONG,
    )
    axis = axis_of(json.loads(out), "sentence_length_mean")
    assert axis["scale"] == pytest.approx(4.0)
    assert axis["deviation"] == pytest.approx(1.0)


def test_a_malformed_or_unknown_scale_is_rejected(tmp_path):
    profile = profile_file(tmp_path, [SHORT])
    code, _, err = run(
        ["diff", "--profile", profile, "--scale", "sentence_length_mean"], stdin=SHORT
    )
    assert code == 2
    assert "AXIS=N" in err
    code, _, err = run(["diff", "--profile", profile, "--scale", "mood=2"], stdin=SHORT)
    assert code == 2
    assert "mood" in err


def test_a_profile_can_come_from_stdin_but_only_once(tmp_path):
    draft = tmp_path / "draft.txt"
    draft.write_text(SHORT, encoding="utf-8")
    code, out, _ = run(
        ["diff", str(draft), "--profile", "-"], stdin=build_profile([SHORT]).to_json()
    )
    assert code == 0
    assert json.loads(out)["matches"] is True

    code, _, err = run(
        ["diff", "--profile", "-"], stdin=build_profile([SHORT]).to_json()
    )
    assert code == 2
    assert "stdin" in err


def test_a_file_that_is_not_a_profile_is_reported(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    code, _, err = run(["diff", "--profile", str(bad)], stdin=SHORT)
    assert code == 2
    assert "bad.json" in err


def test_fit_dry_run_prints_the_instruction(tmp_path):
    code, out, _ = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [SOURCE]),
            "--dry-run",
            "--note",
            "Keep the headings.",
        ],
        stdin=SOURCE,
    )
    payload = json.loads(out)
    assert code == 0
    assert payload["schema_version"] == SCHEMA_VERSION
    assert '"v2.1.0"' in payload["instruction"]
    assert "- Keep the headings." in payload["instruction"]
    assert payload["avoid"] == []


def test_fit_can_narrow_the_protected_kinds(tmp_path):
    profile = profile_file(tmp_path, [SOURCE])
    _, out, _ = run(["fit", "--profile", profile, "--dry-run"], stdin=SOURCE)
    assert "Reproduce these" in json.loads(out)["instruction"]
    _, out, _ = run(
        ["fit", "--profile", profile, "--dry-run", "--kind", "url"], stdin=SOURCE
    )
    assert "Reproduce these" not in json.loads(out)["instruction"]


def test_fit_needs_a_model_or_a_dry_run(tmp_path):
    code, _, err = run(
        ["fit", "--profile", profile_file(tmp_path, [SOURCE])], stdin=SOURCE
    )
    assert code == 2
    assert "--model" in err


def test_fit_runs_a_model_command_and_accepts_the_rewrite(tmp_path):
    code, out, _ = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [SOURCE]),
            "--model",
            model_command(tmp_path, SOURCE),
        ],
        stdin=SOURCE,
    )
    payload = json.loads(out)
    assert code == 0
    assert payload["accepted"] is True
    assert payload["text"] == SOURCE


def test_fit_exits_one_when_the_rewrite_loses_a_span(tmp_path):
    code, out, _ = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [SOURCE]),
            "--model",
            model_command(tmp_path, NOTHING_KEPT),
        ],
        stdin=SOURCE,
    )
    payload = json.loads(out)
    assert code == 1
    assert payload["accepted"] is False
    assert payload["text"] == SOURCE
    assert any("v2.1.0" in failure for failure in payload["failures"])


def test_a_failing_model_command_is_reported(tmp_path):
    code, out, err = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [SOURCE]),
            "--model",
            model_command(tmp_path, "", status=3),
        ],
        stdin=SOURCE,
    )
    assert code == 2
    assert out == ""
    assert "model command" in err


def test_fit_keeps_a_memory_window_between_runs(tmp_path):
    profile = profile_file(tmp_path, [SOURCE])
    command = model_command(tmp_path, SOURCE)
    memory_path = tmp_path / "memory.json"
    argv = [
        "fit",
        "--profile",
        profile,
        "--model",
        command,
        "--memory",
        str(memory_path),
        "--key",
        "handbook",
    ]

    code, _, _ = run(argv, stdin=SOURCE)
    assert code == 0
    restored = OutputMemory.from_json(memory_path.read_text(encoding="utf-8"))
    assert restored.recent("handbook") == (SOURCE,)

    code, out, _ = run(argv, stdin=SOURCE)
    payload = json.loads(out)
    assert code == 1
    assert payload["accepted"] is False
    assert any("opening" in failure for failure in payload["failures"])

    code, out, _ = run(
        [
            "fit",
            "--profile",
            profile,
            "--dry-run",
            "--memory",
            str(memory_path),
            "--key",
            "handbook",
        ],
        stdin=SOURCE,
    )
    payload = json.loads(out)
    assert code == 0
    assert "Do not open with" in payload["instruction"]
    assert [use["kind"] for use in payload["avoid"]] == ["ending", "opening"]


def test_a_file_that_is_not_a_memory_is_reported(tmp_path):
    memory_path = tmp_path / "memory.json"
    memory_path.write_text("{}", encoding="utf-8")
    code, _, err = run(
        [
            "fit",
            "--profile",
            profile_file(tmp_path, [SOURCE]),
            "--dry-run",
            "--memory",
            str(memory_path),
        ],
        stdin=SOURCE,
    )
    assert code == 2
    assert "memory.json" in err


def test_the_version_and_unknown_commands_exit_through_argparse(capsys):
    with pytest.raises(SystemExit) as version:
        main(["--version"])
    assert version.value.code == 0
    assert __version__ in capsys.readouterr().out

    with pytest.raises(SystemExit) as unknown:
        main(["frobnicate"])
    assert unknown.value.code == 2
