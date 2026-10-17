"""Incremental scan/parse/embed orchestration with atomic publication."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from codeatlas.core.exceptions import EmbeddingProviderError, StorageError
from codeatlas.core.logging import log_event
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
        scanner: RepositoryScanner | None = None,
        preparer: RepositoryPreparer | None = None,
        pipeline_version: str = "python-ast-v1",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.provider = provider
        self.store = store
        self.batch_size = batch_size
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
        for offset in range(0, len(prepared.chunks), self.batch_size):
            batch = prepared.chunks[offset : offset + self.batch_size]
            log_event(
                _LOGGER, "embeddings_requested", repository_id=repository_id, count=len(batch)
            )
            vectors = await self.provider.embed([chunk.content for chunk in batch])
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
