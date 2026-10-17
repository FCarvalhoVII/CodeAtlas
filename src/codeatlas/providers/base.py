"""Vendor-neutral provider contracts."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field

from codeatlas.core.models import DomainModel, SourceReference


class LLMRequest(DomainModel):
    """A provider-neutral text generation request."""

    question: str = Field(min_length=1)
    context: str
    allowed_references: tuple[SourceReference, ...] = ()


class LLMResponse(DomainModel):
    """Raw generated content returned by an LLM provider."""

    content: str = Field(min_length=1)


class EmbeddingProvider(Protocol):
    """Create vector representations for a batch of texts."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts while preserving input order."""
        ...


class LLMProvider(Protocol):
    """Generate an answer from a structured request."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for the supplied context."""
        ...
