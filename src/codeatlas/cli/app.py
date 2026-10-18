"""Stable command-line entry points for the CodeAtlas workflow."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import IntEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from codeatlas.answering.factory import create_answer_service
from codeatlas.core.config import CodeAtlasSettings, load_settings
from codeatlas.core.exceptions import (
    CodeAtlasError,
    ConfigurationError,
    IndexNotFoundError,
    InvalidRepositoryError,
    ProviderError,
    StorageError,
)
from codeatlas.core.logging import configure_logging
from codeatlas.core.models import SourceReference
from codeatlas.indexing.factory import create_indexing_service
from codeatlas.indexing.scanner import resolve_repository
from codeatlas.retrieval.factory import create_search_service
from codeatlas.storage.sqlite import SQLiteVectorStore


class ExitCode(IntEnum):
    """Stable process exit codes for expected failures."""

    ERROR = 1
    USAGE = 2
    CONFIGURATION = 3
    REPOSITORY = 4
    INDEX = 5
    PROVIDER = 6


app = typer.Typer(
    name="codeatlas",
    help="Index, search, and ask grounded questions about a local codebase.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        help="TOML configuration file. Defaults to ./codeatlas.toml when present.",
        dir_okay=False,
        readable=True,
    ),
]
RepositoryOption = Annotated[
    Path,
    typer.Option(
        "--repository",
        "-r",
        help="Repository whose index should be used.",
        file_okay=False,
        resolve_path=False,
    ),
]


def _settings(config: Path | None) -> CodeAtlasSettings:
    settings = load_settings(config)
    sensitive_values = (
        (settings.openai_api_key.get_secret_value(),) if settings.openai_api_key else ()
    )
    configure_logging(
        settings.logging.level,
        json_output=settings.logging.json_output,
        sensitive_values=sensitive_values,
    )
    return settings


def _store(settings: CodeAtlasSettings) -> SQLiteVectorStore:
    return SQLiteVectorStore(settings.index.storage_path / "index.sqlite3")


def _source(reference: SourceReference) -> str:
    location = f"{reference.file_path}:{reference.start_line}-{reference.end_line}"
    return f"{location} ({reference.symbol_name})" if reference.symbol_name else location


def _abort(error: Exception) -> None:
    if isinstance(error, ConfigurationError):
        code = ExitCode.CONFIGURATION
    elif isinstance(error, InvalidRepositoryError):
        code = ExitCode.REPOSITORY
    elif isinstance(error, ProviderError):
        code = ExitCode.PROVIDER
    elif isinstance(error, StorageError):
        code = ExitCode.INDEX
    elif isinstance(error, ValidationError):
        code = ExitCode.USAGE
    else:
        code = ExitCode.ERROR
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=int(code))


def _run(action: Callable[[], None]) -> None:
    try:
        action()
    except (CodeAtlasError, ValidationError) as error:
        _abort(error)


@app.command("index")
def index_command(
    repository_path: Annotated[
        Path,
        typer.Argument(help="Local repository directory to index.", resolve_path=False),
    ],
    config: ConfigOption = None,
) -> None:
    """Build or incrementally update a repository index."""

    def action() -> None:
        settings = _settings(config)
        identity = resolve_repository(repository_path)
        typer.echo(f"Indexing {identity.root_path}...")
        service = create_indexing_service(settings)
        manifest = asyncio.run(service.index(identity.root_path))
        stats = manifest.statistics
        typer.echo(
            f"Indexed {manifest.chunk_count} chunks from {stats.scanned_files} files "
            f"({stats.embedded_chunks} embedded, {stats.unchanged_files} unchanged, "
            f"{stats.deleted_files} deleted, {stats.skipped_files} skipped)."
        )
        typer.echo(f"Repository: {manifest.repository.root_path}")
        typer.echo(f"Generation: {manifest.generation}")

    _run(action)


@app.command("search")
def search_command(
    query: Annotated[str, typer.Argument(help="Natural-language repository search query.")],
    repository: RepositoryOption = Path("."),
    config: ConfigOption = None,
    top_k: Annotated[
        int | None,
        typer.Option("--top-k", min=1, max=100, help="Maximum number of results."),
    ] = None,
) -> None:
    """Retrieve relevant source evidence without invoking an LLM."""

    def action() -> None:
        settings = _settings(config)
        identity = resolve_repository(repository)
        response = asyncio.run(
            create_search_service(settings).search_text(
                identity.repository_id,
                query,
                top_k=top_k,
            )
        )
        if not response.results:
            typer.echo("No matching evidence found.")
            return
        for position, result in enumerate(response.results, start=1):
            typer.echo(f"{position}. [{result.score:.3f}] {_source(result.reference)}")
            typer.echo(result.snippet)
            if position != len(response.results):
                typer.echo()

    _run(action)


@app.command("ask")
def ask_command(
    question: Annotated[str, typer.Argument(help="Question to answer from repository evidence.")],
    repository: RepositoryOption = Path("."),
    config: ConfigOption = None,
    top_k: Annotated[
        int | None,
        typer.Option("--top-k", min=1, max=100, help="Maximum retrieved evidence items."),
    ] = None,
) -> None:
    """Answer a question and validate every returned source citation."""

    def action() -> None:
        settings = _settings(config)
        identity = resolve_repository(repository)
        answer = asyncio.run(
            create_answer_service(settings).answer_text(
                identity.repository_id,
                question,
                top_k=top_k,
            )
        )
        typer.echo(answer.answer)
        typer.echo(f"Confidence: {answer.confidence}")
        if answer.citations:
            typer.echo("Sources:")
            for citation in answer.citations:
                typer.echo(f"- {_source(citation.reference)}")
        if answer.inferences:
            typer.echo("Inferences:")
            for inference in answer.inferences:
                typer.echo(f"- {inference}")
        if answer.limitations:
            typer.echo("Limitations:")
            for limitation in answer.limitations:
                typer.echo(f"- {limitation}")

    _run(action)


@app.command("status")
def status_command(
    repository: RepositoryOption = Path("."),
    config: ConfigOption = None,
) -> None:
    """Show the persisted index status for a repository."""

    def action() -> None:
        settings = _settings(config)
        identity = resolve_repository(repository)
        snapshot = _store(settings).load(identity.repository_id)
        if snapshot is None:
            raise IndexNotFoundError(f"No index found for repository: {identity.root_path}")
        manifest = snapshot.manifest
        typer.echo(f"Repository: {manifest.repository.root_path}")
        typer.echo(f"Generation: {manifest.generation}")
        typer.echo(f"Updated: {manifest.updated_at.isoformat()}")
        typer.echo(f"Files: {len(manifest.files)}")
        typer.echo(f"Chunks: {manifest.chunk_count}")
        typer.echo(
            f"Embedding: {manifest.embedding.provider}/{manifest.embedding.model} "
            f"({manifest.vector_dimensions or 0} dimensions)"
        )

    _run(action)


@app.command("clear")
def clear_command(
    repository: RepositoryOption = Path("."),
    config: ConfigOption = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Remove the index without an interactive prompt."),
    ] = False,
) -> None:
    """Remove the persisted index for one repository."""

    def action() -> None:
        settings = _settings(config)
        identity = resolve_repository(repository)
        store = _store(settings)
        snapshot = store.load(identity.repository_id)
        if snapshot is None:
            raise IndexNotFoundError(f"No index found for repository: {identity.root_path}")
        if not yes:
            confirmed = typer.confirm(f"Remove the index for {identity.root_path}?")
            if not confirmed:
                typer.echo("Index removal cancelled.")
                return
        store.delete(identity.repository_id, expected_generation=snapshot.manifest.generation)
        typer.echo(f"Removed index for {identity.root_path}.")

    _run(action)


def main() -> None:
    """Run the CodeAtlas CLI."""

    app()
