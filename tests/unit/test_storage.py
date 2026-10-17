import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

from codeatlas.core.exceptions import CorruptedIndexError, StorageError
from codeatlas.indexing.service import IndexingService
from codeatlas.providers.fake import FakeEmbeddingProvider
from codeatlas.storage.models import IndexSnapshot
from codeatlas.storage.sqlite import SQLiteVectorStore


@pytest.fixture
def indexed(tmp_path: Path) -> tuple[SQLiteVectorStore, IndexSnapshot]:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def first():\n    return 1\n", encoding="utf-8")
    (root / "b.py").write_text("def second():\n    return 2\n", encoding="utf-8")
    store = SQLiteVectorStore(tmp_path / "index.db")
    manifest = asyncio.run(IndexingService(FakeEmbeddingProvider(), store).index(root))
    snapshot = store.load(manifest.repository.repository_id)
    assert snapshot is not None
    return store, snapshot


def test_cosine_search_and_repository_isolation(
    indexed: tuple[SQLiteVectorStore, IndexSnapshot],
) -> None:
    store, snapshot = indexed
    record = snapshot.records[0]
    generation = snapshot.manifest.generation
    results = store.search(
        snapshot.manifest.repository.repository_id,
        list(record.vector),
        top_k=1,
        expected_generation=generation,
    )
    assert len(results) == 1
    assert results[0].chunk == record.chunk
    assert results[0].score == pytest.approx(1.0)
    assert (
        store.search("other-repository", list(record.vector), expected_generation=generation) == ()
    )
    assert store.load("other-repository") is None
    with pytest.raises(ValueError, match="top_k"):
        store.search("other-repository", [], top_k=0, expected_generation=generation)
    for vector in ([0.0] * 8, [float("inf")] * 8, [1.0]):
        with pytest.raises(ValueError, match="Query vector"):
            store.search(
                snapshot.manifest.repository.repository_id,
                vector,
                expected_generation=generation,
            )


@pytest.mark.parametrize(
    "target", ["manifest", "chunk", "orphan", "count", "version", "repository"]
)
def test_corrupted_rows_are_rejected(
    indexed: tuple[SQLiteVectorStore, IndexSnapshot],
    target: str,
) -> None:
    store, snapshot = indexed
    with closing(sqlite3.connect(store.path)) as connection, connection:
        if target == "manifest":
            connection.execute("UPDATE manifests SET payload='not json'")
        elif target == "chunk":
            connection.execute("UPDATE chunks SET payload='{}'")
        elif target == "orphan":
            connection.execute("DELETE FROM manifests")
        else:
            data = snapshot.manifest.model_dump(mode="json")
            if target == "count":
                data["chunk_count"] = 100
            elif target == "version":
                data["schema_version"] = 99
            else:
                data["repository"]["repository_id"] = "different"
                data["files"] = []
                data["chunk_count"] = 0
                connection.execute("DELETE FROM chunks")
            import json

            connection.execute("UPDATE manifests SET payload=?", (json.dumps(data),))
    with pytest.raises(CorruptedIndexError):
        store.load(snapshot.manifest.repository.repository_id)
    if target == "manifest":
        with pytest.raises(CorruptedIndexError):
            store.save(snapshot, snapshot.manifest.generation)


def test_database_corruption_schema_and_access_errors(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.db"
    invalid.write_bytes(b"not sqlite")
    with pytest.raises(CorruptedIndexError):
        SQLiteVectorStore(invalid)
    schema = tmp_path / "schema.db"
    with closing(sqlite3.connect(schema)) as connection, connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(CorruptedIndexError, match="schema"):
        SQLiteVectorStore(schema)
    with pytest.raises(StorageError, match="access"):
        SQLiteVectorStore(tmp_path)
    with pytest.raises(StorageError, match="directory"):
        SQLiteVectorStore(invalid / "index.db")


def test_snapshot_rejects_inconsistent_data(
    indexed: tuple[SQLiteVectorStore, IndexSnapshot],
) -> None:
    _, snapshot = indexed
    for mutation in (
        "duplicate_file",
        "duplicate_chunk",
        "file_repo",
        "missing_parent",
        "dimension",
        "unknown_file",
    ):
        data = snapshot.model_dump(mode="json")
        if mutation == "duplicate_file":
            data["manifest"]["files"].append(data["manifest"]["files"][0])
        elif mutation == "duplicate_chunk":
            data["records"].append(data["records"][0])
        elif mutation == "file_repo":
            data["manifest"]["files"][0]["repository_id"] = "other"
        elif mutation == "missing_parent":
            data["records"][0]["chunk"]["parent_chunk_id"] = "missing"
        elif mutation == "dimension":
            data["records"][0]["vector"] = [1.0]
        else:
            data["records"][0]["chunk"]["reference"]["file_path"] = "missing.py"
        with pytest.raises(ValidationError):
            IndexSnapshot.model_validate(data)


def test_two_repositories_coexist(
    indexed: tuple[SQLiteVectorStore, IndexSnapshot], tmp_path: Path
) -> None:
    store, original = indexed
    second_root = tmp_path / "second"
    second_root.mkdir()
    (second_root / "main.py").write_text("X = 1\n", encoding="utf-8")
    second = asyncio.run(IndexingService(FakeEmbeddingProvider(), store).index(second_root))
    assert store.load(original.manifest.repository.repository_id) == original
    assert second.repository.repository_id != original.manifest.repository.repository_id
    assert (
        len(
            store.search(
                second.repository.repository_id,
                list(original.records[0].vector),
                expected_generation=second.generation,
            )
        )
        == 1
    )


def test_empty_index_search(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    store = SQLiteVectorStore(tmp_path / "index.db")
    manifest = asyncio.run(IndexingService(FakeEmbeddingProvider(), store).index(root))
    assert manifest.vector_dimensions is None
    assert (
        store.search(
            manifest.repository.repository_id,
            [1.0],
            expected_generation=manifest.generation,
        )
        == ()
    )
