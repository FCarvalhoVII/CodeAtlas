import asyncio
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from codeatlas.core.config import CodeAtlasSettings, IndexSettings
from codeatlas.core.exceptions import CorruptedIndexError, EmbeddingProviderError, StorageError
from codeatlas.indexing.factory import create_indexing_service
from codeatlas.indexing.scanner import RepositoryScanner, ScanOptions
from codeatlas.indexing.service import IndexingService
from codeatlas.parsing.chunking import SemanticChunker
from codeatlas.parsing.service import RepositoryPreparer
from codeatlas.providers.fake import FakeEmbeddingProvider
from codeatlas.storage.sqlite import SQLiteVectorStore


def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def first():\n    return 1\n", encoding="utf-8")
    (root / "b.py").write_text("def second():\n    return 2\n", encoding="utf-8")
    return root


def test_incremental_add_change_delete_and_skip(tmp_path: Path) -> None:
    root = repo(tmp_path)
    provider = FakeEmbeddingProvider()
    store = SQLiteVectorStore(tmp_path / "index.sqlite3")
    service = IndexingService(provider, store, batch_size=1)
    first = asyncio.run(service.index(root))
    assert first.statistics.updated_files == 2
    assert first.statistics.embedded_chunks == 2
    old = store.load(first.repository.repository_id)
    assert old is not None
    provider.calls.clear()
    unchanged = asyncio.run(service.index(root))
    assert unchanged.statistics.unchanged_files == 2
    assert provider.calls == []
    (root / "a.py").write_text("def first():\n    return 10\n", encoding="utf-8")
    (root / "b.py").unlink()
    (root / "new.py").write_text("VALUE = 3\n", encoding="utf-8")
    final = asyncio.run(service.index(root))
    assert final.statistics.updated_files == 2
    assert final.statistics.deleted_files == 1
    snapshot = store.load(final.repository.repository_id)
    assert snapshot is not None
    assert {r.chunk.reference.file_path for r in snapshot.records} == {"a.py", "new.py"}
    assert all(r.chunk.content != "return 1\n" for r in snapshot.records)
    assert not {r.chunk.chunk_id for r in old.records} & {
        r.chunk.chunk_id for r in snapshot.records
    }


def test_reopen_in_fresh_process_and_round_trip(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = tmp_path / "index.sqlite3"
    store = SQLiteVectorStore(path)
    manifest = asyncio.run(IndexingService(FakeEmbeddingProvider(), store).index(root))
    snapshot = store.load(manifest.repository.repository_id)
    reopened = SQLiteVectorStore(path).load(manifest.repository.repository_id)
    assert snapshot == reopened
    script = (
        "from pathlib import Path\n"
        "from codeatlas.storage.sqlite import SQLiteVectorStore\n"
        f"store = SQLiteVectorStore(Path({str(path)!r}))\n"
        f"snapshot = store.load({manifest.repository.repository_id!r})\n"
        "assert snapshot is not None\n"
        "print(snapshot.manifest.chunk_count)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == str(manifest.chunk_count)
    assert snapshot is not None
    for record in snapshot.records:
        ref = record.chunk.reference
        lines = (root / ref.file_path).read_text(encoding="utf-8").splitlines(keepends=True)
        assert record.chunk.content == "".join(lines[ref.start_line - 1 : ref.end_line])


class InterruptedProvider(FakeEmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.calls:
            raise EmbeddingProviderError("simulated interruption")
        return await super().embed(texts)


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_batches_never_publish_partial_index(tmp_path: Path, existing: bool) -> None:
    root = repo(tmp_path)
    store = SQLiteVectorStore(tmp_path / "index.sqlite3")
    repository_id = RepositoryScanner().scan(root).repository.repository_id
    if existing:
        asyncio.run(IndexingService(FakeEmbeddingProvider(), store).index(root))
        (root / "a.py").write_text("X = 3\n", encoding="utf-8")
        (root / "b.py").write_text("Y = 4\n", encoding="utf-8")
    before = store.load(repository_id)
    with pytest.raises(EmbeddingProviderError, match="interruption"):
        asyncio.run(IndexingService(InterruptedProvider(), store, batch_size=1).index(root))
    assert store.load(repository_id) == before


@pytest.mark.parametrize("bad", [[], [[0.0]], [[float("nan")]], [[1.0], [1.0, 2.0]]])
def test_invalid_vectors_do_not_publish(tmp_path: Path, bad: list[list[float]]) -> None:
    class BadProvider(FakeEmbeddingProvider):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return bad

    root = repo(tmp_path)
    store = SQLiteVectorStore(tmp_path / "index.sqlite3")
    with pytest.raises(EmbeddingProviderError):
        asyncio.run(IndexingService(BadProvider(), store).index(root))
    repository_id = RepositoryScanner().scan(root).repository.repository_id
    assert store.load(repository_id) is None


def test_dimension_profile_mismatch_is_rejected(tmp_path: Path) -> None:
    class BadProvider(FakeEmbeddingProvider):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

    with pytest.raises(EmbeddingProviderError, match="profile"):
        asyncio.run(
            IndexingService(BadProvider(), SQLiteVectorStore(tmp_path / "index.db")).index(
                repo(tmp_path)
            )
        )


def test_embedding_and_chunk_configuration_changes_rebuild(tmp_path: Path) -> None:
    root = repo(tmp_path)
    store = SQLiteVectorStore(tmp_path / "index.sqlite3")
    first = asyncio.run(IndexingService(FakeEmbeddingProvider(), store).index(root))
    provider = FakeEmbeddingProvider(dimensions=4)
    changed = asyncio.run(IndexingService(provider, store).index(root))
    assert changed.statistics.unchanged_files == 0
    assert changed.vector_dimensions == 4
    assert len(provider.calls) == 1
    provider.calls.clear()
    chunker = RepositoryPreparer(chunker=SemanticChunker(max_lines=1))
    rechunked = asyncio.run(IndexingService(provider, store, preparer=chunker).index(root))
    assert rechunked.pipeline_id != first.pipeline_id
    assert rechunked.statistics.embedded_chunks == 4
    assert provider.calls


def test_malformed_unsupported_empty_and_deleted_files(tmp_path: Path) -> None:
    root = repo(tmp_path)
    store = SQLiteVectorStore(tmp_path / "index.sqlite3")
    service = IndexingService(FakeEmbeddingProvider(), store)
    first = asyncio.run(service.index(root))
    (root / "a.py").write_text("def invalid(:", encoding="utf-8")
    (root / "b.py").write_text("", encoding="utf-8")
    (root / "readme.md").write_text("text", encoding="utf-8")
    (root / "binary").write_bytes(b"\0")
    (root / "encoding").write_bytes(b"\xff")
    changed = asyncio.run(service.index(root))
    assert changed.chunk_count == 0
    assert {f.file_path for f in changed.files} == {"b.py"}
    assert {s.file_path for s in service.skipped} == {"a.py", "readme.md", "binary", "encoding"}
    assert store.load(first.repository.repository_id) is not None
    for file in root.iterdir():
        file.unlink()
    empty = asyncio.run(service.index(root))
    assert empty.chunk_count == 0 and empty.files == ()


def test_unreadable_ignore_file_preserves_existing_index(tmp_path: Path) -> None:
    root = repo(tmp_path)
    store = SQLiteVectorStore(tmp_path / "index.sqlite3")
    service = IndexingService(FakeEmbeddingProvider(), store)
    first = asyncio.run(service.index(root))
    before = store.load(first.repository.repository_id)
    (root / ".gitignore").symlink_to(tmp_path / "missing")
    with pytest.raises(StorageError, match="incomplete"):
        asyncio.run(service.index(root))
    assert store.load(first.repository.repository_id) == before


def test_store_conflict_and_transaction_rollback(tmp_path: Path) -> None:
    root = repo(tmp_path)
    path = tmp_path / "index.sqlite3"
    store = SQLiteVectorStore(path)
    manifest = asyncio.run(IndexingService(FakeEmbeddingProvider(), store).index(root))
    before = store.load(manifest.repository.repository_id)
    assert before is not None
    with pytest.raises(StorageError, match="changed"):
        store.save(before, None)
    assert store.load(manifest.repository.repository_id) == before
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TRIGGER reject_chunks BEFORE INSERT ON chunks "
            "BEGIN SELECT RAISE(ABORT, 'failure'); END"
        )
    with pytest.raises(CorruptedIndexError):
        store.save(before, manifest.generation)
    assert store.load(manifest.repository.repository_id) == before


def test_factory_builds_offline_service(tmp_path: Path) -> None:
    settings = CodeAtlasSettings(index=IndexSettings(storage_path=tmp_path / ".codeatlas"))
    manifest = asyncio.run(create_indexing_service(settings).index(repo(tmp_path)))
    assert manifest.embedding.provider == "fake"
    assert manifest.chunk_count == 2
    with pytest.raises(ValueError, match="batch_size"):
        IndexingService(
            FakeEmbeddingProvider(), SQLiteVectorStore(tmp_path / "other.db"), batch_size=0
        )


def test_cancelled_indexing_leaves_previous_generation(tmp_path: Path) -> None:
    root = repo(tmp_path)
    store = SQLiteVectorStore(tmp_path / "index.sqlite3")
    first = asyncio.run(IndexingService(FakeEmbeddingProvider(), store).index(root))
    before = store.load(first.repository.repository_id)
    (root / "a.py").write_text("X = 3\n", encoding="utf-8")

    class CancelledProvider(FakeEmbeddingProvider):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(IndexingService(CancelledProvider(), store).index(root))
    assert store.load(first.repository.repository_id) == before


def test_exclusion_change_removes_previously_indexed_files(tmp_path: Path) -> None:
    root = repo(tmp_path)
    store = SQLiteVectorStore(tmp_path / "index.db")
    asyncio.run(IndexingService(FakeEmbeddingProvider(), store).index(root))
    scanner = RepositoryScanner(ScanOptions(exclude=("b.py",)))
    manifest = asyncio.run(
        IndexingService(FakeEmbeddingProvider(), store, scanner=scanner).index(root)
    )
    assert {file.file_path for file in manifest.files} == {"a.py"}
    assert manifest.statistics.deleted_files == 1


def test_changed_file_reuses_unchanged_vectors(tmp_path: Path) -> None:
    root = repo(tmp_path)
    store = SQLiteVectorStore(tmp_path / "index.db")
    provider = FakeEmbeddingProvider()
    service = IndexingService(provider, store)
    first = asyncio.run(service.index(root))
    before = store.load(first.repository.repository_id)
    assert before is not None
    retained = [r for r in before.records if r.chunk.reference.file_path == "b.py"]
    (root / "a.py").write_text("X = 3\n", encoding="utf-8")
    provider.calls.clear()
    final = asyncio.run(service.index(root))
    assert provider.calls == [["X = 3\n"]]
    assert final.statistics.unchanged_files == 1
    snapshot = store.load(first.repository.repository_id)
    assert snapshot is not None
    assert [r for r in snapshot.records if r.chunk.reference.file_path == "b.py"] == retained
