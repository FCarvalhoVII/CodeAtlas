import json
from pathlib import Path

from typer.testing import CliRunner

from codeatlas.cli.app import ExitCode, app
from codeatlas.core.exceptions import EmbeddingProviderError
from codeatlas.indexing.scanner import resolve_repository
from codeatlas.providers.fake import FakeEmbeddingProvider, FakeLLMProvider
from codeatlas.storage.sqlite import SQLiteVectorStore

runner = CliRunner()


def write_config(tmp_path: Path, content: str = "") -> Path:
    config = tmp_path / "codeatlas.toml"
    storage = (tmp_path / ".codeatlas").as_posix()
    config.write_text(f'[index]\nstorage_path = "{storage}"\n{content}', encoding="utf-8")
    return config


def write_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "project"
    repository.mkdir()
    (repository / "main.py").write_text(
        'def answer():\n    """Return the canonical answer."""\n    return 42\n',
        encoding="utf-8",
    )
    return repository


def test_cli_help_lists_public_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("index", "search", "ask", "status", "clear"):
        assert command in result.stdout


def test_cli_maps_invalid_repository_missing_index_and_usage(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    repository = write_repository(tmp_path)

    invalid = runner.invoke(app, ["index", str(tmp_path / "missing"), "--config", str(config)])
    missing = runner.invoke(
        app,
        ["status", "--repository", str(repository), "--config", str(config)],
    )
    invalid_query = runner.invoke(
        app,
        ["search", " ", "--repository", str(repository), "--config", str(config)],
    )

    assert invalid.exit_code == ExitCode.REPOSITORY
    assert "Cannot resolve repository" in invalid.output
    assert missing.exit_code == ExitCode.INDEX
    assert "No index found" in missing.output
    assert invalid_query.exit_code == ExitCode.USAGE


def test_cli_reports_missing_credentials(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.delenv("CODEATLAS_OPENAI_API_KEY", raising=False)  # type: ignore[attr-defined]
    config = write_config(
        tmp_path,
        '\n[embedding]\nprovider = "openai"\nmodel = "text-embedding-test"\n',
    )
    repository = write_repository(tmp_path)

    result = runner.invoke(app, ["index", str(repository), "--config", str(config)])

    assert result.exit_code == ExitCode.CONFIGURATION
    assert "CODEATLAS_OPENAI_API_KEY" in result.output


def test_cli_maps_provider_failures(tmp_path: Path, monkeypatch: object) -> None:
    class FailingEmbeddingProvider(FakeEmbeddingProvider):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise EmbeddingProviderError("Embedding service unavailable")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "codeatlas.indexing.factory.create_embedding_provider",
        lambda settings: FailingEmbeddingProvider(),
    )
    config = write_config(tmp_path)
    repository = write_repository(tmp_path)

    result = runner.invoke(app, ["index", str(repository), "--config", str(config)])

    assert result.exit_code == ExitCode.PROVIDER
    assert "Embedding service unavailable" in result.output


def test_cli_end_to_end_index_search_ask_status_and_clear(
    tmp_path: Path, monkeypatch: object
) -> None:
    config = write_config(tmp_path)
    repository = write_repository(tmp_path)
    common = ["--repository", str(repository), "--config", str(config)]

    indexed = runner.invoke(app, ["index", str(repository), "--config", str(config)])
    searched = runner.invoke(app, ["search", "canonical answer", *common])
    identity = resolve_repository(repository)
    store = SQLiteVectorStore(tmp_path / ".codeatlas" / "index.sqlite3")
    snapshot = store.load(identity.repository_id)
    assert snapshot is not None
    evidence_id = snapshot.records[0].chunk.chunk_id
    model_response = json.dumps(
        {
            "answer": "The answer function returns 42.",
            "citation_ids": [evidence_id],
            "confidence": "high",
            "limitations": [],
            "inferences": [],
            "insufficient_context": False,
        }
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "codeatlas.answering.factory.create_llm_provider",
        lambda settings: FakeLLMProvider(model_response),
    )
    asked = runner.invoke(app, ["ask", "What does answer return?", *common])
    status = runner.invoke(app, ["status", *common])
    cancelled = runner.invoke(app, ["clear", *common], input="n\n")
    assert cancelled.exit_code == 0
    assert "cancelled" in cancelled.output
    assert store.load(identity.repository_id) is not None
    cleared = runner.invoke(app, ["clear", *common, "--yes"])
    missing = runner.invoke(app, ["status", *common])

    assert indexed.exit_code == 0
    assert "Indexed" in indexed.output
    assert searched.exit_code == 0
    assert "main.py:1-3" in searched.output
    assert asked.exit_code == 0
    assert "The answer function returns 42." in asked.output
    assert "main.py:1-3" in asked.output
    assert status.exit_code == 0
    assert "Chunks:" in status.output
    assert cleared.exit_code == 0
    assert "Removed index" in cleared.output
    assert missing.exit_code == ExitCode.INDEX
    assert store.load(identity.repository_id) is None
