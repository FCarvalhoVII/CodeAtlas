"""SQLite persistence with transactional snapshots and exact cosine search."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from codeatlas.core.exceptions import CorruptedIndexError, StorageError
from codeatlas.core.models import SearchResult
from codeatlas.storage.models import EmbeddedChunk, IndexManifest, IndexSnapshot


class SQLiteVectorStore:
    """A dependency-free local store intended for small repository indexes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._connection() as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                tables = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if version == 0 and not tables:
                    connection.execute(
                        "CREATE TABLE manifests "
                        "(repository_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                    )
                    connection.execute(
                        "CREATE TABLE chunks (repository_id TEXT NOT NULL, "
                        "chunk_id TEXT NOT NULL, payload TEXT NOT NULL, "
                        "PRIMARY KEY(repository_id, chunk_id))"
                    )
                    connection.execute("PRAGMA user_version=1")
                elif version != 1 or {row[0] for row in tables} != {"manifests", "chunks"}:
                    raise CorruptedIndexError("Unsupported index schema; use a new index path")
        except OSError:
            raise StorageError("Cannot create index directory") from None

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path, timeout=5)
            with connection:
                yield connection
        except sqlite3.DatabaseError as exc:
            if isinstance(exc, sqlite3.OperationalError):
                raise StorageError("Cannot access SQLite index") from None
            raise CorruptedIndexError("SQLite index is corrupted") from None
        finally:
            if connection is not None:
                connection.close()

    def load(self, repository_id: str) -> IndexSnapshot | None:
        with self._connection() as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT payload FROM manifests WHERE repository_id=?",
                (repository_id,),
            ).fetchone()
            records = connection.execute(
                "SELECT payload FROM chunks WHERE repository_id=? ORDER BY chunk_id",
                (repository_id,),
            ).fetchall()
        if row is None:
            if records:
                raise CorruptedIndexError("Index chunks have no manifest")
            return None
        try:
            snapshot = IndexSnapshot(
                manifest=IndexManifest.model_validate_json(row[0]),
                records=tuple(EmbeddedChunk.model_validate_json(item[0]) for item in records),
            )
            if snapshot.manifest.repository.repository_id != repository_id:
                raise ValueError("Repository mismatch")
            return snapshot
        except (ValidationError, ValueError):
            raise CorruptedIndexError("Invalid index manifest or chunk data") from None

    def save(self, snapshot: IndexSnapshot, expected_generation: str | None) -> None:
        repository_id = snapshot.manifest.repository.repository_id
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM manifests WHERE repository_id=?",
                (repository_id,),
            ).fetchone()
            try:
                actual = IndexManifest.model_validate_json(row[0]).generation if row else None
            except ValidationError:
                raise CorruptedIndexError("Invalid index manifest") from None
            if actual != expected_generation:
                raise StorageError("Index changed during indexing; retry the operation")
            connection.execute("DELETE FROM chunks WHERE repository_id=?", (repository_id,))
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?)",
                [
                    (repository_id, record.chunk.chunk_id, record.model_dump_json())
                    for record in snapshot.records
                ],
            )
            connection.execute(
                "INSERT OR REPLACE INTO manifests VALUES (?, ?)",
                (repository_id, snapshot.manifest.model_dump_json()),
            )

    def search(
        self,
        repository_id: str,
        vector: list[float],
        top_k: int = 8,
    ) -> tuple[SearchResult, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        snapshot = self.load(repository_id)
        if snapshot is None:
            return ()
        if not snapshot.records:
            return ()
        if (
            len(vector) != snapshot.manifest.vector_dimensions
            or not all(math.isfinite(value) for value in vector)
            or not any(vector)
        ):
            raise ValueError("Query vector must be finite, nonzero, and match index dimensions")
        query_norm = math.hypot(*vector)
        results: list[SearchResult] = []
        for record in snapshot.records:
            norm = math.hypot(*record.vector)
            score = sum(
                (a / query_norm) * (b / norm) for a, b in zip(vector, record.vector, strict=True)
            )
            results.append(
                SearchResult(
                    chunk=record.chunk,
                    score=max(-1.0, min(1.0, score)),
                    snippet=record.chunk.content,
                )
            )
        return tuple(sorted(results, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k])
