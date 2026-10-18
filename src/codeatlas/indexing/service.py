"""Incremental scan/parse/embed orchestration with atomic publication."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from codeatlas.core.exceptions import EmbeddingProviderError, PartialIndexingError, StorageError
from codeatlas.core.logging import log_event
from codeatlas.core.models import CodeChunk
from codeatlas.indexing.scanner import RepositoryScanner, ScanResult, SkippedFile
from codeatlas.parsing.service import RepositoryPreparer
from codeatlas.providers.base import EmbeddingProvider
from codeatlas.storage.base import VectorStore
from codeatlas.storage.models import (
    EmbeddedChunk,
    IndexManifest,
    IndexSnapshot,
    IndexStatistics,
)

_LOGGER = logging.getLogger(__name__)


class IndexingService:
    """Build all new embeddings before changing a persistent index."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        store: VectorStore,
        *,
        batch_size: int = 32,
        max_concurrency: int = 4,
        scanner: RepositoryScanner | None = None,
        preparer: RepositoryPreparer | None = None,
        pipeline_version: str = "python-ast-v1",
    ) -> None:
        if batch_size < 1 or max_concurrency < 1:
            raise ValueError("batch_size and max_concurrency must be positive")
        self.provider = provider
        self.store = store
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.scanner = scanner or RepositoryScanner()
        self.preparer = preparer or RepositoryPreparer()
        # Custom parsers must bump pipeline_version when their behavior changes.
        self.pipeline_id = hashlib.sha256(
            json.dumps(
                [
                    pipeline_version,
                    self.scanner.options.model_dump(),
                    self.preparer.chunker.max_lines,
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.skipped: tuple[SkippedFile, ...] = ()

    async def _embed_batches(
        self,
        repository_id: str,
        chunks: tuple[CodeChunk, ...],
    ) -> list[tuple[tuple[CodeChunk, ...], list[list[float]]]]:
        """Embed with bounded concurrency and clean up every task on failure."""

        batches = [
            chunks[offset : offset + self.batch_size]
            for offset in range(0, len(chunks), self.batch_size)
        ]
        if not batches:
            return []
        semaphore = asyncio.Semaphore(self.max_concurrency)
        aborted = asyncio.Event()
        completed = 0

        async def run(
            batch_number: int, batch: tuple[CodeChunk, ...]
        ) -> tuple[tuple[CodeChunk, ...], list[list[float]]]:
            nonlocal completed
            async with semaphore:
                if aborted.is_set():
                    raise asyncio.CancelledError
                log_event(
                    _LOGGER,
                    "embeddings_requested",
                    repository_id=repository_id,
                    batch=batch_number,
                    total_batches=len(batches),
                    count=len(batch),
                )
                try:
                    vectors = await self.provider.embed([chunk.content for chunk in batch])
                except asyncio.CancelledError:
                    raise
                except Exception:
                    aborted.set()
                    raise PartialIndexingError(batch_number, len(batches), completed) from None
                completed += 1
                return batch, vectors

        tasks = [
            asyncio.create_task(run(number, batch), name=f"codeatlas-embedding-{number}")
            for number, batch in enumerate(batches, start=1)
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            log_event(
                _LOGGER,
                "index_staging_aborted",
                repository_id=repository_id,
                completed_batches=completed,
                total_batches=len(batches),
            )
            raise

    async def index(self, root: Path) -> IndexManifest:
        scan = self.scanner.scan(root)
        self.skipped = scan.skipped
        if any(
            item.reason in {"file unreadable", "directory unreadable", "gitignore unreadable"}
            for item in scan.skipped
        ):
            raise StorageError("Cannot index an incomplete scan; resolve unreadable files first")
        repository_id = scan.repository.repository_id
        previous = self.store.load(repository_id)
        old_files = {file.file_path: file for file in previous.manifest.files} if previous else {}
        compatible = (
            previous is not None
            and previous.manifest.embedding == self.provider.profile
            and previous.manifest.pipeline_id == self.pipeline_id
        )
        unchanged = {
            file.metadata.file_path
            for file in scan.files
            if compatible
            and file.metadata.file_path in old_files
            and file.metadata.content_hash == old_files[file.metadata.file_path].content_hash
        }
        changed = tuple(file for file in scan.files if file.metadata.file_path not in unchanged)
        prepared = self.preparer.prepare(
            ScanResult(repository=scan.repository, files=changed, skipped=())
        )
        self.skipped += prepared.skipped
        failed = {item.file_path for item in prepared.skipped}
        files = tuple(file.metadata for file in scan.files if file.metadata.file_path not in failed)
        retained = (
            [record for record in previous.records if record.chunk.reference.file_path in unchanged]
            if previous
            else []
        )
        dimensions = previous.manifest.vector_dimensions if compatible and previous else None
        records = list(retained)
        for batch, vectors in await self._embed_batches(repository_id, prepared.chunks):
            if len(vectors) != len(batch):
                raise EmbeddingProviderError(
                    "Embedding provider returned an incorrect vector count"
                )
            try:
                for chunk, vector in zip(batch, vectors, strict=True):
                    record = EmbeddedChunk(chunk=chunk, vector=tuple(vector))
                    if dimensions is None:
                        dimensions = len(record.vector)
                    if len(record.vector) != dimensions:
                        raise ValueError("dimension mismatch")
                    records.append(record)
            except (ValidationError, ValueError, TypeError):
                raise EmbeddingProviderError(
                    "Embedding provider returned invalid vectors"
                ) from None
        current_paths = {file.file_path for file in files}
        manifest = IndexManifest(
            generation=uuid4().hex,
            repository=scan.repository,
            embedding=self.provider.profile,
            pipeline_id=self.pipeline_id,
            vector_dimensions=dimensions,
            files=files,
            chunk_count=len(records),
            updated_at=datetime.now(UTC),
            statistics=IndexStatistics(
                scanned_files=len(scan.files),
                unchanged_files=len(unchanged),
                updated_files=len(current_paths - unchanged),
                deleted_files=len(old_files.keys() - current_paths),
                embedded_chunks=len(prepared.chunks),
                skipped_files=len(self.skipped),
            ),
        )
        try:
            snapshot = IndexSnapshot(manifest=manifest, records=tuple(records))
        except ValidationError:
            raise EmbeddingProviderError(
                "Embedding profile does not match returned vectors"
            ) from None
        self.store.save(snapshot, previous.manifest.generation if previous else None)
        log_event(
            _LOGGER,
            "index_updated",
            repository_id=repository_id,
            chunks=manifest.chunk_count,
            embedded_chunks=manifest.statistics.embedded_chunks,
        )
        return manifest
