"""Validated persistence snapshots and embedding compatibility metadata."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import Field, StrictFloat, model_validator

from codeatlas.core.models import CodeChunk, DomainModel, IndexedFile, RepositoryIdentity


class EmbeddingProfile(DomainModel):
    """Configuration that identifies an embedding space."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimensions: int | None = Field(default=None, ge=1)


class EmbeddedChunk(DomainModel):
    """A source chunk and its finite, nonzero embedding."""

    chunk: CodeChunk
    vector: tuple[StrictFloat, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_vector(self) -> EmbeddedChunk:
        if not all(math.isfinite(value) for value in self.vector) or not any(self.vector):
            raise ValueError("Vectors must be finite and nonzero")
        return self


class IndexStatistics(DomainModel):
    """Counters for the last completed indexing run."""

    scanned_files: int = Field(ge=0)
    unchanged_files: int = Field(ge=0)
    updated_files: int = Field(ge=0)
    deleted_files: int = Field(ge=0)
    embedded_chunks: int = Field(ge=0)
    skipped_files: int = Field(ge=0)


class IndexManifest(DomainModel):
    """Versioned repository identity, configuration, file hashes, and counts."""

    schema_version: Literal[1] = 1
    generation: str = Field(min_length=1)
    repository: RepositoryIdentity
    embedding: EmbeddingProfile
    pipeline_id: str = Field(min_length=1)
    vector_dimensions: int | None = Field(default=None, ge=1)
    files: tuple[IndexedFile, ...]
    chunk_count: int = Field(ge=0)
    updated_at: datetime
    statistics: IndexStatistics


class IndexSnapshot(DomainModel):
    """An internally consistent snapshot, committed atomically by a store."""

    manifest: IndexManifest
    records: tuple[EmbeddedChunk, ...]

    @model_validator(mode="after")
    def validate_snapshot(self) -> IndexSnapshot:
        manifest = self.manifest
        repository_id = manifest.repository.repository_id
        files = {file.file_path: file for file in manifest.files}
        ids = {record.chunk.chunk_id for record in self.records}
        if len(files) != len(manifest.files) or len(ids) != len(self.records):
            raise ValueError("Duplicate files or chunk identifiers")
        if len(self.records) != manifest.chunk_count:
            raise ValueError("Chunk count does not match manifest")
        if any(file.repository_id != repository_id for file in manifest.files):
            raise ValueError("File belongs to a different repository")
        if (
            manifest.embedding.dimensions is not None
            and manifest.vector_dimensions is not None
            and manifest.embedding.dimensions != manifest.vector_dimensions
        ):
            raise ValueError("Embedding profile dimension mismatch")
        for record in self.records:
            chunk = record.chunk
            if chunk.repository_id != repository_id or chunk.reference.file_path not in files:
                raise ValueError("Chunk belongs to an unknown repository or file")
            if len(record.vector) != manifest.vector_dimensions:
                raise ValueError("Vector dimension mismatch")
            if chunk.parent_chunk_id is not None and chunk.parent_chunk_id not in ids:
                raise ValueError("Missing parent chunk")
        return self
