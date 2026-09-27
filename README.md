# voicefit

Fit text to a measured style profile.

voicefit builds a style profile from an author's own corpus, rewrites text
through a caller-supplied model under measurable constraints, verifies that
facts, numbers, links and identifiers survive the rewrite, and keeps output
from repeating itself.

It is a tool for a team's own voice. It is not a tool for evading detectors.

## Status

Pre-alpha: profiling, distance reporting, protected spans and the rewrite loop
work. The public API is not stable.

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

## Tests

```bash
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).

Maintained by [polycratia](https://polycratia.com).
