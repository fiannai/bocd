# Contributing to bocd

Thanks for your interest in bocd! We welcome issues, feedback, and discussion.

## Reporting Bugs

Open an issue using the **Bug Report** template. Include:

- What you did (minimal code to reproduce)
- What you expected
- What actually happened
- Python version, OS, and `bocd` version (`pip show bocd`)

## Requesting Features

Open an issue using the **Feature Request** template. Describe the use case — what you're trying to accomplish and why the current API doesn't support it.

## Code Contributions

The API is still stabilizing (v0.x). Before writing code, **open an issue first** to discuss the change. This avoids wasted effort on approaches that don't fit the project direction.

If you have a bug fix with a failing test, that's always welcome as a PR — just keep it focused on the fix.

## Development Setup

```bash
git clone https://github.com/fiannai/bocd.git
cd bocd
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest                                    # full suite
pytest tests/test_models.py -v            # one file
pytest --cov=bocd --cov-fail-under=95     # with coverage
```

### Linting and Type Checking

```bash
ruff check src/ tests/                    # lint
ruff format --check src/ tests/           # format check
mypy src/                                 # type check (strict)
```

All three must pass before merging. CI runs these on Python 3.11/3.12/3.13 across Linux, macOS, and Windows.

## Style

- Follow existing code patterns
- No new dependencies without discussion
- Tests required for all changes (TDD preferred)
- Keep PRs focused — one concern per PR
