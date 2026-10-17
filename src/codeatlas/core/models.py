"""Validated models passed between CodeAtlas application layers."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DomainModel(BaseModel):
    """Base model for immutable, strictly validated domain data."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def _validate_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(value)
    if (
        not normalized
        or not path.parts
        or path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in path.parts
    ):
        raise ValueError("file_path must be a non-empty repository-relative path")
    return path.as_posix()


class SymbolType(StrEnum):
    """Language-neutral kinds of source symbols."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    INTERFACE = "interface"
    SECTION = "section"


class RepositoryIdentity(DomainModel):
    """Stable identity and resolved location of a repository."""

    repository_id: str = Field(min_length=1)
    root_path: Path
    name: str = Field(min_length=1)

    @field_validator("root_path")
    @classmethod
    def root_path_must_be_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("root_path must be absolute")
        return value


class IndexedFile(DomainModel):
    """Metadata used to detect and process repository file changes."""

    repository_id: str = Field(min_length=1)
    file_path: str
    language: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    modified_at: datetime
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    _relative_file_path = field_validator("file_path")(_validate_relative_path)


class SourceReference(DomainModel):
    """A verified location inside a repository source file."""

    file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    symbol_name: str | None = None

    _relative_file_path = field_validator("file_path")(_validate_relative_path)

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> SourceReference:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ParsedSymbol(DomainModel):
    """A structural source-code unit produced by a language parser."""

    repository_id: str = Field(min_length=1)
    language: str = Field(min_length=1)
    symbol_name: str = Field(min_length=1)
    symbol_type: SymbolType
    source_code: str = Field(min_length=1)
    reference: SourceReference
    parent_symbol: str | None = None
    imports: tuple[str, ...] = ()


class CodeChunk(DomainModel):
    """A semantically meaningful unit suitable for embedding and retrieval."""

    chunk_id: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    language: str = Field(min_length=1)
    content: str = Field(min_length=1)
    reference: SourceReference
    symbol_type: SymbolType
    parent_chunk_id: str | None = None


class SearchResult(DomainModel):
    """A ranked code chunk returned by retrieval."""

    chunk: CodeChunk
    score: float
    snippet: str = Field(min_length=1)
