# voicefit

Fit text to a measured style profile.

voicefit builds a style profile from an author's own corpus, rewrites text
through a caller-supplied model under measurable constraints, verifies that
facts, numbers, links and identifiers survive the rewrite, and keeps output
from repeating itself.

It is a tool for a team's own voice. It is not a tool for evading detectors.

## Status

Pre-alpha: profiling and distance reporting work, the rewrite and verification
stages are not implemented yet. The public API is not stable.

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

## Tests

```bash
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).

Maintained by [polycratia](https://polycratia.com).
