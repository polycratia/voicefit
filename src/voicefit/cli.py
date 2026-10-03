"""Command line entry points: measure a corpus, diff a text, fit a rewrite.

Each command reads files or standard input and prints one JSON report -- the
same dictionaries the library returns, so a shell pipeline sees what a Python
caller sees. ``measure`` builds a profile from a corpus, ``diff`` reports the
per-axis distance of a text from a stored profile, and ``fit`` runs the rewrite
loop with a model command, handing the instruction to its standard input and
reading the rewrite from its standard output.

Exit codes keep a judgement apart from a failure: 0 when the text is on profile
or the rewrite was accepted, 1 when it was not, and 2 when the command could
not run at all.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from voicefit import __version__
from voicefit.distance import DEFAULT_TOLERANCE, compare_texts
from voicefit.memory import OutputMemory
from voicefit.profile import SCHEMA_VERSION, StyleProfile, build_profile
from voicefit.rewrite import Model, RewriteConstraints, build_instruction
from voicefit.rewrite import rewrite as rewrite_text
from voicefit.spans import PROTECTED_KINDS

__all__ = ["CommandError", "build_parser", "main"]

PROGRAM = "voicefit"
STDIN = "-"


class CommandError(Exception):
    """A command could not run: a missing file, a bad argument, a failing model."""


class _Reader:
    """Read a path or standard input, and refuse to read standard input twice."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._stdin_used = False

    def read(self, path: str) -> str:
        if path != STDIN:
            try:
                return Path(path).read_text(encoding="utf-8")
            except OSError as error:
                raise CommandError(
                    f"cannot read {path}: {error.strerror or error}"
                ) from error
        if self._stdin_used:
            raise CommandError(
                "stdin can only be read once; pass the other input as a file"
            )
        self._stdin_used = True
        return self._stream.read()

    def read_all(self, paths: Sequence[str]) -> list[str]:
        return [self.read(path) for path in (list(paths) or [STDIN])]


def _hedges(args: argparse.Namespace, reader: _Reader) -> list[str] | None:
    if args.hedges is None:
        return None
    phrases = [line.strip() for line in reader.read(args.hedges).splitlines()]
    hedges = [phrase for phrase in phrases if phrase]
    if not hedges:
        raise CommandError(f"no hedge phrases found in {args.hedges}")
    return hedges


def _scales(args: argparse.Namespace) -> dict[str, float] | None:
    if not args.scale:
        return None
    scales: dict[str, float] = {}
    for item in args.scale:
        name, separator, value = item.partition("=")
        if not separator or not name.strip() or not value.strip():
            raise CommandError(f"--scale expects AXIS=N, got {item!r}")
        try:
            scales[name.strip()] = float(value)
        except ValueError:
            raise CommandError(
                f"--scale value for {name.strip()!r} is not a number: {value!r}"
            ) from None
    return scales


def _profile(args: argparse.Namespace, reader: _Reader) -> StyleProfile:
    payload = reader.read(args.profile)
    try:
        return StyleProfile.from_json(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise CommandError(f"{args.profile} is not a style profile: {error}") from error


def _memory(args: argparse.Namespace) -> OutputMemory | None:
    if args.memory is None:
        return None
    path = Path(args.memory)
    if not path.exists():
        return OutputMemory()
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CommandError(
            f"cannot read {args.memory}: {error.strerror or error}"
        ) from error
    try:
        return OutputMemory.from_json(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise CommandError(f"{args.memory} is not an output memory: {error}") from error


def _save_memory(path: str, memory: OutputMemory) -> None:
    try:
        Path(path).write_text(memory.to_json(), encoding="utf-8")
    except OSError as error:
        raise CommandError(f"cannot write {path}: {error.strerror or error}") from error


def _command_model(command: str) -> Model:
    argv = shlex.split(command)
    if not argv:
        raise CommandError("--model needs a command to run")

    def model(instruction: str) -> str:
        try:
            done = subprocess.run(
                argv, input=instruction, capture_output=True, text=True, check=False
            )
        except OSError as error:
            raise CommandError(f"cannot run the model command: {error}") from error
        if done.returncode != 0:
            detail = " ".join(done.stderr.split())
            message = f"the model command exited with {done.returncode}"
            raise CommandError(f"{message}: {detail}" if detail else message)
        return done.stdout

    return model


def _run_measure(
    args: argparse.Namespace, reader: _Reader
) -> tuple[dict[str, Any], int]:
    hedges = _hedges(args, reader)
    profile = build_profile(reader.read_all(args.paths), hedges=hedges)
    return profile.to_dict(), 0


def _run_diff(args: argparse.Namespace, reader: _Reader) -> tuple[dict[str, Any], int]:
    profile = _profile(args, reader)
    hedges = _hedges(args, reader)
    distance = compare_texts(
        reader.read_all(args.paths),
        profile,
        tolerance=args.tolerance,
        hedges=hedges,
        scales=_scales(args),
    )
    return distance.to_dict(), 0 if distance.matches else 1


def _run_fit(args: argparse.Namespace, reader: _Reader) -> tuple[dict[str, Any], int]:
    if args.model is None and not args.dry_run:
        raise CommandError(
            "fit needs --model CMD, or --dry-run to print the instruction"
        )
    profile = _profile(args, reader)
    hedges = _hedges(args, reader)
    text = reader.read(args.path)
    constraints = RewriteConstraints(
        tolerance=args.tolerance,
        kinds=args.kind or None,
        hedges=hedges,
        scales=_scales(args),
        notes=tuple(args.note),
    )
    memory = _memory(args)

    if args.dry_run:
        target: str | StyleProfile = args.key if args.key is not None else profile
        avoid = memory.overused(target) if memory is not None else ()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "instruction": build_instruction(
                text, profile, constraints=constraints, avoid=avoid
            ),
            "avoid": [use.to_dict() for use in avoid],
        }
        return payload, 0

    result = rewrite_text(
        text,
        profile,
        _command_model(args.model),
        constraints=constraints,
        memory=memory,
        key=args.key,
    )
    if memory is not None:
        _save_memory(args.memory, memory)
    return result.to_dict(), 0 if result.accepted else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser, so the whole interface is readable in one place."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="write the JSON report to PATH instead of stdout",
    )
    common.add_argument(
        "--compact", action="store_true", help="print the JSON on a single line"
    )
    common.add_argument(
        "--hedges",
        metavar="PATH",
        help="hedge phrases, one per line, replacing the default list",
    )

    against = argparse.ArgumentParser(add_help=False)
    against.add_argument(
        "--profile",
        metavar="PATH",
        required=True,
        help="the style profile to measure against",
    )
    against.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        metavar="N",
        help=f"standardised deviation an axis may reach (default {DEFAULT_TOLERANCE})",
    )
    against.add_argument(
        "--scale",
        action="append",
        default=[],
        metavar="AXIS=N",
        help="replace the scale of one axis; may be repeated",
    )

    parser = argparse.ArgumentParser(
        prog=PROGRAM, description="Fit text to a measured style profile."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    measure = commands.add_parser(
        "measure",
        parents=[common],
        help="build a style profile from a corpus",
        description="Measure a style profile from an author's own texts.",
    )
    measure.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="corpus files, or - for stdin (the default)",
    )
    measure.set_defaults(run=_run_measure)

    diff = commands.add_parser(
        "diff",
        parents=[common, against],
        help="report how far a text sits from a profile",
        description="Report the per-axis distance of a text from a stored profile.",
    )
    diff.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="texts to measure, or - for stdin (the default)",
    )
    diff.set_defaults(run=_run_diff)

    fit = commands.add_parser(
        "fit",
        parents=[common, against],
        help="rewrite a text towards a profile through a model command",
        description="Rewrite a text towards a profile, then check what came back.",
    )
    fit.add_argument(
        "path",
        nargs="?",
        default=STDIN,
        metavar="PATH",
        help="the text to rewrite, or - for stdin (the default)",
    )
    fit.add_argument(
        "--model",
        metavar="CMD",
        help="command reading the instruction on stdin and writing the rewrite"
        " on stdout",
    )
    fit.add_argument(
        "--dry-run",
        action="store_true",
        help="print the instruction instead of calling a model",
    )
    fit.add_argument(
        "--kind",
        action="append",
        default=[],
        choices=PROTECTED_KINDS,
        help="protect only these span kinds; may be repeated",
    )
    fit.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="TEXT",
        help="extra rule for the model; may be repeated",
    )
    fit.add_argument(
        "--memory",
        metavar="PATH",
        help="output memory to read and update, created when missing",
    )
    fit.add_argument(
        "--key",
        metavar="NAME",
        help="memory key for this voice (default: the profile itself)",
    )
    fit.set_defaults(run=_run_fit)
    return parser


def _emit(payload: dict[str, Any], args: argparse.Namespace, stdout: TextIO) -> None:
    text = json.dumps(payload, indent=None if args.compact else 2)
    if args.output is None:
        print(text, file=stdout)
        return
    try:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    except OSError as error:
        raise CommandError(
            f"cannot write {args.output}: {error.strerror or error}"
        ) from error


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one command and return its exit code: 0 pass, 1 fail, 2 cannot run."""
    args = build_parser().parse_args(argv)
    reader = _Reader(sys.stdin if stdin is None else stdin)
    try:
        payload, code = args.run(args, reader)
        _emit(payload, args, sys.stdout if stdout is None else stdout)
    except (CommandError, ValueError) as error:
        print(f"{PROGRAM}: {error}", file=sys.stderr if stderr is None else stderr)
        return 2
    return code
