from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from codeatlas.core.exceptions import (
    EmbeddingProviderError,
    IncompatibleIndexError,
    IndexNotFoundError,
)
from codeatlas.core.models import (
    CodeChunk,
    IndexedFile,
    RepositoryIdentity,
    SearchResult,
    SourceReference,
    SymbolType,
)
from codeatlas.providers.base import EmbeddingProvider
from codeatlas.retrieval.factory import create_search_service
from codeatlas.retrieval.models import RetrievalQuery
from codeatlas.retrieval.semantic import SemanticRetriever
from codeatlas.retrieval.service import SearchService
from codeatlas.storage.models import (
    EmbeddedChunk,
    EmbeddingProfile,
    IndexManifest,
    IndexSnapshot,
    IndexStatistics,
)
from codeatlas.storage.sqlite import SQLiteVectorStore

_CONTROLLED_PROFILE = EmbeddingProfile(provider="controlled", model="test", dimensions=2)


class ControlledProvider:
    def __init__(
        self,
        vector: list[float] | None = None,
        profile: EmbeddingProfile = _CONTROLLED_PROFILE,
    ) -> None:
        self.vector = vector or [1.0, 0.0]
        self.profile = profile
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [list(self.vector) for _ in texts]


def make_chunk(
    chunk_id: str,
    path: str,
    symbol: str,
    start: int,
    end: int,
) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        repository_id="repo-a",
        language="python",
        content=f"# {symbol}\n" + "pass\n" * (end - start),
        reference=SourceReference(
            file_path=path,
            start_line=start,
            end_line=end,
            symbol_name=symbol,
        ),
        symbol_type=SymbolType.FUNCTION,
    )


def save_index(store: SQLiteVectorStore, root: Path) -> IndexSnapshot:
    files = (
        IndexedFile(
            repository_id="repo-a",
            file_path="src/auth.py",
            language="python",
            size_bytes=100,
            modified_at=datetime.now(UTC),
            content_hash="a" * 64,
        ),
        IndexedFile(
            repository_id="repo-a",
            file_path="src/payment.py",
            language="python",
            size_bytes=100,
            modified_at=datetime.now(UTC),
            content_hash="b" * 64,
        ),
    )
    records = (
        EmbeddedChunk(
            chunk=make_chunk("auth", "src/auth.py", "auth.authenticate", 10, 20),
            vector=(1.0, 0.0),
        ),
        EmbeddedChunk(
            chunk=make_chunk("auth-parent", "src/auth.py", "auth.AuthService", 5, 22),
            vector=(0.99, 0.1),
        ),
        EmbeddedChunk(
            chunk=make_chunk("payment", "src/payment.py", "payment.charge", 30, 40),
            vector=(0.0, 1.0),
        ),
    )
    manifest = IndexManifest(
        generation="generation-a",
        repository=RepositoryIdentity(repository_id="repo-a", root_path=root, name="repo"),
        embedding=_CONTROLLED_PROFILE,
        pipeline_id="test-pipeline",
        vector_dimensions=2,
        files=files,
        chunk_count=len(records),
        updated_at=datetime.now(UTC),
        statistics=IndexStatistics(
            scanned_files=2,
            unchanged_files=0,
            updated_files=2,
            deleted_files=0,
            embedded_chunks=3,
            skipped_files=0,
        ),
    )
    snapshot = IndexSnapshot(manifest=manifest, records=records)
    store.save(snapshot, None)
    return snapshot


def test_search_orders_filters_deduplicates_and_returns_stored_evidence(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "index.db")
    snapshot = save_index(store, tmp_path.resolve())
    provider = ControlledProvider()
    service = SearchService(provider, store)

    response = asyncio.run(
        service.search(
            RetrievalQuery(
                repository_id="repo-a",
                text="Where is authenticate implemented?",
                top_k=2,
                min_score=0.5,
            )
        )
    )

    assert response.index_generation == "generation-a"
    assert [result.chunk_id for result in response.results] == ["auth"]
    assert provider.calls == [["Where is authenticate implemented?"]]
    stored = {record.chunk.chunk_id: record.chunk for record in snapshot.records}
    for result in response.results:
        chunk = stored[result.chunk_id]
        assert result.repository_id == "repo-a"
        assert result.reference == chunk.reference
        assert result.snippet == chunk.content
        assert result.reference.file_path in {file.file_path for file in snapshot.manifest.files}


def test_top_k_threshold_and_empty_index(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "index.db")
    save_index(store, tmp_path.resolve())
    service = SearchService(ControlledProvider(), store)

    limited = asyncio.run(service.search_text("repo-a", "unmatched", top_k=1))
    assert len(limited.results) == 1
    filtered = asyncio.run(service.search_text("repo-a", "authenticate auth.py", min_score=0.99))
    assert [result.chunk_id for result in filtered.results] == ["auth"]

    empty_manifest = IndexManifest(
        generation="empty",
        repository=RepositoryIdentity(
            repository_id="empty-repo", root_path=tmp_path.resolve(), name="empty"
        ),
        embedding=_CONTROLLED_PROFILE,
        pipeline_id="pipeline",
        vector_dimensions=None,
        files=(),
        chunk_count=0,
        updated_at=datetime.now(UTC),
        statistics=IndexStatistics(
            scanned_files=0,
            unchanged_files=0,
            updated_files=0,
            deleted_files=0,
            embedded_chunks=0,
            skipped_files=0,
        ),
    )
    store.save(IndexSnapshot(manifest=empty_manifest, records=()), None)
    provider = ControlledProvider()
    empty = asyncio.run(SearchService(provider, store).search_text("empty-repo", "anything"))
    assert empty.results == ()
    assert provider.calls == []


def test_missing_repository_and_embedding_profile_are_rejected(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "index.db")
    save_index(store, tmp_path.resolve())
    with pytest.raises(IndexNotFoundError):
        asyncio.run(SearchService(ControlledProvider(), store).search_text("missing", "query"))

    incompatible = ControlledProvider(
        profile=EmbeddingProfile(provider="controlled", model="other", dimensions=2)
    )
    with pytest.raises(IncompatibleIndexError, match="profile"):
        asyncio.run(SearchService(incompatible, store).search_text("repo-a", "query"))
    assert incompatible.calls == []


@pytest.mark.parametrize("vectors", [[], [[0.0, 0.0]], [[1.0]], [[1.0, 0.0], [0.0, 1.0]]])
def test_invalid_query_embeddings_are_domain_errors(
    tmp_path: Path, vectors: list[list[float]]
) -> None:
    store = SQLiteVectorStore(tmp_path / "index.db")
    save_index(store, tmp_path.resolve())

    class BadProvider(ControlledProvider):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return vectors

    with pytest.raises(EmbeddingProviderError):
        asyncio.run(SearchService(BadProvider(), store).search_text("repo-a", "query"))


def test_generation_boundary_and_repository_isolation(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "index.db")
    snapshot = save_index(store, tmp_path.resolve())
    retriever = SemanticRetriever(store)
    query = RetrievalQuery(repository_id="repo-a", text="query")
    with pytest.raises(IncompatibleIndexError, match="changed"):
        retriever.retrieve(query, [1.0, 0.0], expected_generation="stale")
    assert (
        store.search(
            "other-repository",
            [1.0, 0.0],
            expected_generation=snapshot.manifest.generation,
        )
        == ()
    )


def test_service_rejects_foreign_evidence_from_custom_retriever(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "index.db")
    save_index(store, tmp_path.resolve())
    foreign_chunk = make_chunk("foreign", "src/auth.py", "auth.foreign", 50, 55).model_copy(
        update={"repository_id": "repo-b"}
    )

    class ForeignRetriever:
        def retrieve(
            self,
            query: RetrievalQuery,
            vector: list[float],
            *,
            expected_generation: str,
        ) -> tuple[SearchResult, ...]:
            return (SearchResult(chunk=foreign_chunk, score=1, snippet=foreign_chunk.content),)

    service = SearchService(ControlledProvider(), store, retriever=ForeignRetriever())
    with pytest.raises(IncompatibleIndexError, match="another repository"):
        asyncio.run(service.search_text("repo-a", "query"))


def test_factory_applies_configured_search_defaults(tmp_path: Path) -> None:
    from codeatlas.core.config import CodeAtlasSettings, EmbeddingSettings, IndexSettings

    settings = CodeAtlasSettings(
        embedding=EmbeddingSettings(dimensions=2),
        index=IndexSettings(
            storage_path=tmp_path,
            top_k=1,
            min_score=0.99,
            candidate_multiplier=2,
        ),
    )
    store = SQLiteVectorStore(tmp_path / "index.sqlite3")
    # The factory uses the fake profile, so persist a matching empty index.
    manifest = IndexManifest(
        generation="empty",
        repository=RepositoryIdentity(
            repository_id="empty", root_path=tmp_path.resolve(), name="empty"
        ),
        embedding=EmbeddingProfile(provider="fake", model="deterministic", dimensions=2),
        pipeline_id="pipeline",
        vector_dimensions=None,
        files=(),
        chunk_count=0,
        updated_at=datetime.now(UTC),
        statistics=IndexStatistics(
            scanned_files=0,
            unchanged_files=0,
            updated_files=0,
            deleted_files=0,
            embedded_chunks=0,
            skipped_files=0,
        ),
    )
    store.save(IndexSnapshot(manifest=manifest, records=()), None)
    service = create_search_service(settings)
    response = asyncio.run(service.search_text("empty", "query"))
    assert response.results == ()
    assert service.default_top_k == 1
    assert service.default_min_score == 0.99


def test_request_validation_and_retriever_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(repository_id="repo", text="   ")
    with pytest.raises(ValueError, match="candidate_multiplier"):
        SemanticRetriever(SQLiteVectorStore(tmp_path / "index.db"), 0)
    with pytest.raises(ValidationError):
        SearchService(
            ControlledProvider(), SQLiteVectorStore(tmp_path / "other.db"), default_top_k=0
        )


def test_protocol_accepts_controlled_provider() -> None:
    provider: EmbeddingProvider = ControlledProvider()
    assert provider.profile.model == "test"
