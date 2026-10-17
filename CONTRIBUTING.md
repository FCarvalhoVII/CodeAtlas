# Contributing to CodeAtlas

CodeAtlas is in early development. Small, tested changes that advance the current roadmap milestone are preferred.

## Development setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Activate the virtual environment using the command appropriate for your shell, then run:

```bash
ruff check .
ruff format --check .
mypy src tests
pytest
```

## Change requirements

- Write source code, tests, documentation, configuration, and user-facing messages in English.
- Add behavior-focused tests for every change.
- Do not require live AI services in the default test suite.
- Update public documentation when behavior or configuration changes.
- Never commit credentials, local indexes, or source copied from indexed private repositories.

## Pull requests

Keep pull requests focused on one coherent change. Describe the motivation, implementation tradeoffs, tests performed, and any known limitations.

