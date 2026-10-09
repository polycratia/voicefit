# voicefit

Fit text to a measured style profile.

voicefit builds a style profile from an author's own corpus, rewrites text
through a caller-supplied model under measurable constraints, verifies that
facts, numbers, links and identifiers survive the rewrite, and keeps output
from repeating itself.

It is a tool for a team's own voice. It is not a tool for evading detectors.

## What it is for

Teams that write together drift apart. A handbook written over two years has
three voices in it, a changelog reads like whoever was on duty, and the only
advice a reviewer can give is "make it sound more like us", which is not
something a writer can act on. voicefit turns that sentence into numbers taken
from the corpus the team already has:

- bring a draft into the voice of a docs set, a handbook or a changelog that
  already exists;
- hand a new writer a measurement instead of an impression;
- rewrite through a model without losing the versions, numbers, links and
  ticket keys that make the text true;
- keep a run of rewrites from settling into one opening and one closing phrase.

The profile is built from texts you pass in, so the target is a voice the team
already owns and can point at.

## What it is not for

voicefit is not a tool for evading detectors, and it is not built to help text
pass as something it is not. That is a non-goal rather than a missing feature:
the design works against it.

- The target is always a corpus you supply. There is no "sound human" mode, no
  classifier in the loop, and nothing that scores how machine-like a text
  reads.
- The axes are plain style measurements: sentence length and its spread,
  hedging, questions, punctuation habits, paragraph shape. They are chosen to
  describe how a team writes, and tuning them against a detector would mean
  chasing a target that changes without notice and says nothing about a voice.
- Every judgement is reported axis by axis, in that axis's own units, with the
  scale it was standardised by. A tool for evasion wants one opaque number and
  an output nobody re-reads; voicefit wants the opposite.
- A rewrite that drops a number or a link is refused even when it sits
  perfectly on profile. Evasion tolerates drift in meaning as long as the
  output passes. Here meaning comes first and the voice second.
- Nothing is hidden from the author: the instruction sent to the model, the
  answer that came back and every check are kept in the result and serialise to
  JSON.

Imitating a writer who did not hand over their corpus is outside this as well.
Build the profile from texts the team wrote, and store it next to them.

## The checks and why they are there

A rewrite is only useful if you can say what it kept. Three checks run on every
attempt, and an attempt passes only when all three do.

| Check | What it catches | On failure |
| --- | --- | --- |
| Protected spans | a lost or altered number, link, code span or identifier | the attempt fails and the missing spans are quoted back to the model |
| Profile distance | prose that drifted off the measured voice | the attempt fails and the off-profile axes are quoted with their numbers |
| Output memory | a run of texts that all open and close the same way | the attempt fails and the overused shapes are named as shapes to avoid |

Spans are checked because a style rewrite has no business touching facts. A
model asked to shorten sentences will cheerfully turn `v2.1.0` into "the latest
version" and 3 replicas into "a few". The check is literal and counted, so the
author does not have to proofread for it.

Distance is checked per axis because "close enough" is not a number. One axis
sitting far off is a different problem from every axis sitting slightly off,
and a single score cannot tell you which of the two happened.

Memory is checked because a text can be on target on every axis and still read
as a template. The averages say nothing about the fact that the last six
outputs all began with the same four words.

A failing attempt is retried once with its failures spelled out. If the retry
fails too, the original text is returned and `accepted` is `False`: the loop
never hands back a rewrite that did not pass, so a silent regression is not one
of the outcomes.

## Status

Pre-alpha: profiling, distance reporting, protected spans, the rewrite loop,
output memory and the command line work. The public API is not stable.

## Requirements

- Python 3.11+
- No runtime dependencies (standard library only)

## Install

```bash
pip install -e .
```

## Usage

```python
from voicefit import build_profile

profile = build_profile([
    "The migration ran clean. No rollback was needed.",
    "Numbers first, then the story.\n\nIt might be that simple.",
])

print(profile.sentence_length_mean, profile.sentence_length_variance)
print(profile.hedge_rate, profile.question_share)
print(profile.to_json())
```

The profile measures sentence length (mean and variance), hedging, the share
of questions and exclamations, punctuation habits per 1000 words, and
paragraph shape. `StyleProfile.to_json()` and `StyleProfile.from_json()` round
trip it, so a profile can be stored next to the corpus it came from.

## Distance from a text to a profile

```python
from voicefit import compare_text

result = compare_text(
    "It could be argued that the migration, on the whole, ran fairly clean.",
    profile,
)

print(result.report())
for axis in result.off_profile:
    print(axis.axis, axis.direction, axis.difference, axis.deviation)
```

Each axis is compared in its own units and then standardised, so the report
carries the target, the measured value, the raw difference, the scale used and
the standardised deviation. `tolerance` decides which axes are flagged, and
`max_deviation` and `mean_absolute_deviation` are derived from the same rows
rather than standing in for them: there is no single opaque score.

The scale of an axis is the largest of a spread already in the profile, a
fraction of the target and an absolute floor, which stops a near-zero target
from exaggerating a small difference. Pass `scales={"hedge_rate": 0.01}` to
set your own. Use `compare_texts` for a corpus, or `compare_profiles` when the
measurement already exists. When the profile was built with a custom hedge
list, pass the same list to the comparison.

## Protected spans

```python
from voicefit import extract_spans, verify_spans

for span in extract_spans("Deploy 3 replicas behind https://api.example.com/v2."):
    print(span.kind, span.text, span.start, span.end)

result = verify_spans(
    "Deploy 3 replicas behind https://api.example.com/v2.",
    "Deploy several replicas behind the gateway.",
)

print(result.intact)
print(result.report())
for check in result.lost:
    print(check.kind, check.text, check.expected, check.found)
```

`extract_spans` pulls URLs, code spans (inline and fenced), numbers and
identifiers out of a text, in order and without overlap: a URL inside a code
span belongs to that code span. `verify_spans` takes those spans from the
source and looks for each one in the rewrite byte for byte. A span counts as
surviving only when the exact characters are there and are not glued into a
longer word, number or URL, so a `42` in the rewrite never passes for a `4` in
the source. Repeated spans are counted: two mentions of `3` must still be two.

Identifiers are read widely, so file names, dotted paths, versions such as
`v1.2.3`, ticket keys such as `ENG-421`, hashes and digit-letter mixes such as
`200ms` stay whole. Pass `kinds=["url", "number"]` to protect less, and use
`SpanReport.to_json()` to store the result next to the rewrite it judged.

## The rewrite loop

```python
from voicefit import RewriteConstraints, rewrite

def model(instruction: str) -> str:
    return my_llm(instruction)  # any callable, any provider

result = rewrite(
    "The migration ran clean and needed no rollback.",
    profile,
    model,
    constraints=RewriteConstraints(tolerance=1.5, notes=["Keep the headings."]),
)

print(result.accepted)
print(result.text)
print(result.report())
```

`rewrite` builds the instruction from the profile and the constraints, calls
the model, then checks what came back: every protected span of the source has
to survive, and every axis has to sit within `tolerance`. A failing attempt is
retried once, with the measured failures quoted in the new instruction and the
previous answer attached. If the retry fails too, `result.text` is the original
text and `result.accepted` is `False`: the loop never hands back a rewrite that
did not pass. Each attempt keeps its instruction, its output, its `SpanReport`
and its `ProfileDistance`, and `RewriteResult.to_json()` stores the lot.

`RewriteConstraints` carries the tolerance, the span `kinds` to protect, the
`hedges` list the profile was built with, per-axis `scales`, and free-form
`notes` appended to the rules. Call `build_instruction` on its own to read the
prompt before wiring a model in.

## Memory against self-repetition

```python
from voicefit import OutputMemory

memory = OutputMemory(capacity=20, max_share=0.34)

result = rewrite(source, profile, model, memory=memory, key="handbook")

print(memory.recent("handbook"))
print(memory.check(result.text, "handbook").report())
for use in memory.overused("handbook"):
    print(use.kind, use.pattern, use.count, use.share)
```

A run of texts can be on target on every axis and still read as a template
because each one opens the same way and lands on the same closing phrase.
`OutputMemory` keeps the recent outputs of a profile and measures two shapes of
a candidate against them: the opening words of its first sentence and the
closing words of its last. A shape is refused once it already carries more than
`max_share` of the window. `capacity` is how many outputs are kept, and
`opening_words` and `ending_words` how many words each shape is read from.

Windows are kept per key, so two voices never share a history. Pass `key` to
name one yourself, or leave it out and the profile stands in for it, keyed by
its own numbers through `profile_key`.

Inside the loop the overused shapes are named in the instruction as openings
and endings to avoid, a repeated shape fails an attempt the way a lost number
does, and only an accepted output is remembered. Drive the model yourself and
`OutputMemory.check`, `OutputMemory.remember` and
`build_instruction(..., avoid=memory.overused(key))` do the same work in the
open. `OutputMemory.to_json()` and `OutputMemory.from_json()` round trip the
whole window, so a voice keeps its history between runs.

## Command line

```bash
voicefit measure corpus/*.md > profile.json
voicefit diff --profile profile.json draft.md
voicefit fit --profile profile.json --model "my-llm --quiet" draft.md
```

Every command prints one JSON report, and the report is the same dictionary the
library returns, so a shell pipeline sees what a Python caller sees. `measure`
builds a profile from a corpus. `diff` reports the per-axis distance of a text
from a stored profile. `fit` runs the rewrite loop against a model command,
handing the instruction to its standard input and reading the rewrite from its
standard output, so any program that answers a prompt can be the model.

`-` stands for standard input and is what a command reads when no path is
given, so `cat draft.md | voicefit diff --profile profile.json` works as well.
It can be read only once per run: pass the other input as a file. `-o PATH`
writes the report to a file instead of stdout, `--compact` puts it on a single
line, and `--hedges PATH` replaces the default hedge list with one phrase per
line.

`diff` and `fit` share `--tolerance N` and `--scale AXIS=N`. `fit` adds `--kind`
to protect fewer span kinds, `--note` to append a rule to the instruction,
`--memory PATH` for a window that is read before the run and written back after
it, `--key NAME` to name that window, and `--dry-run` to print the instruction
and the shapes to avoid instead of calling a model.

The exit code keeps a judgement apart from a failure: 0 when the text is on
profile or the rewrite was accepted, 1 when it is not, and 2 when the command
could not run at all. Without installing the script, `python -m voicefit` takes
the same arguments.

## Tests

```bash
python -m pytest
```

The suite runs on fixed texts and a fake model whose outputs are written down
in the test itself, so every check, the retry and the fallback to the original
are deterministic and no provider is called.

## License

MIT. See [LICENSE](LICENSE).

Maintained by [polycratia](https://polycratia.com).
