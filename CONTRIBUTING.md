# Contributing to CodeAtlas

CodeAtlas accepts focused, tested changes that preserve its public contracts, security boundaries, and grounded-citation guarantees.

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
mypy src tests benchmarks
pytest
```

Build the distributions and run the offline benchmark when changing packaging, indexing, storage, or retrieval:

```bash
python -m build
python benchmarks/benchmark_local.py
```

## Change requirements

- Write source code, tests, documentation, configuration, and user-facing messages in English.
- Add behavior-focused tests for every change.
- Do not require live AI services in the default test suite.
- Update public documentation when behavior or configuration changes.
- Never commit credentials, local indexes, or source copied from indexed private repositories.

## Pull requests

Keep pull requests focused on one coherent change. Describe the motivation, implementation tradeoffs, tests performed, and any known limitations.

## Release process

Release preparation follows [RELEASE.md](RELEASE.md). Do not tag a release until the full Linux/macOS CI matrix and clean-wheel smoke test pass. Update the version and changelog together, inspect distribution contents, and verify the published wheel in a fresh environment.

