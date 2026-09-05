# voicefit

Fit text to a measured style profile.

voicefit builds a style profile from an author's own corpus, rewrites text
through a caller-supplied model under measurable constraints, verifies that
facts, numbers, links and identifiers survive the rewrite, and keeps output
from repeating itself.

It is a tool for a team's own voice. It is not a tool for evading detectors.

## Status

Pre-alpha: this commit is the package skeleton only, no logic yet.

## Requirements

- Python 3.11+
- No runtime dependencies (standard library only)

## Install

```bash
pip install -e .
```

## Usage

```python
import voicefit

print(voicefit.__version__)
```

## Tests

```bash
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).

Maintained by [polycratia](https://polycratia.com).
