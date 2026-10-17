"""Validated requests and evidence returned by repository retrieval."""

from pydantic import Field

from codeatlas.core.models import DomainModel, SourceReference, SourceText, SymbolType


class RetrievalQuery(DomainModel):
    """A repository-bounded natural-language search request."""

    repository_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=100)
    min_score: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)


class RetrievedEvidence(DomainModel):
    """A ranked stored chunk with both semantic and final scores."""

    chunk_id: str = Field(min_length=1)
    repository_id: str = Field(min_length=1)
    language: str = Field(min_length=1)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    semantic_score: float = Field(ge=-1, le=1, allow_inf_nan=False)
    snippet: SourceText
    reference: SourceReference
    symbol_type: SymbolType
    parent_chunk_id: str | None = None


class RetrievalResponse(DomainModel):
    """Evidence tied to the exact persisted generation searched."""

    repository_id: str = Field(min_length=1)
    index_generation: str = Field(min_length=1)
    results: tuple[RetrievedEvidence, ...]
