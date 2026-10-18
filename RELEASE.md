# Release checklist

Use this checklist to publish `v1.0.0`. The repository uses SQLite and has no external storage service, so Docker Compose is unnecessary.

## Prepare

- Confirm `pyproject.toml` and `codeatlas.__version__` contain the release version.
- Update `CHANGELOG.md` with the release date and user-visible changes.
- Confirm README examples and provider privacy statements match runtime behavior.
- Run the local benchmark and review material regressions against `benchmarks/baseline.json`.
- Ensure no credential, local `codeatlas.toml`, `.env`, index database, or private fixture is tracked.

## Validate

```bash
ruff check .
ruff format --check .
mypy src tests benchmarks
pytest -W error
python -m build
```

Install the wheel in a new environment and exercise the offline workflow:

```bash
python -m venv /tmp/codeatlas-release
/tmp/codeatlas-release/bin/python -m pip install dist/codeatlas-1.0.0-py3-none-any.whl
/tmp/codeatlas-release/bin/codeatlas --help
```

- Confirm the Linux Python 3.11–3.13 and macOS Python 3.13 CI jobs pass.
- Confirm the package smoke job builds both distributions and indexes its fixture.
- Inspect wheel and source archive contents before publication.

## Publish

- Create the signed or annotated `v1.0.0` tag from the validated commit.
- Build distributions from the tag in a clean checkout.
- Publish the artifacts to the chosen package index.
- Create release notes from `CHANGELOG.md` and attach checksums.
- Verify installation from the published index in a new environment.
