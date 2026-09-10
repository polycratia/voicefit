# voicefit

Fit text to a measured style profile.

voicefit builds a style profile from an author's own corpus, rewrites text
through a caller-supplied model under measurable constraints, verifies that
facts, numbers, links and identifiers survive the rewrite, and keeps output
from repeating itself.

It is a tool for a team's own voice. It is not a tool for evading detectors.

## Status

Pre-alpha: profiling works, the rewrite and verification stages are not
implemented yet. The public API is not stable.

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

## Tests

```bash
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).

Maintained by [polycratia](https://polycratia.com).
