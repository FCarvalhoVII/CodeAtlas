"""Validated grounded-answer requests, citations, and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from codeatlas.core.models import DomainModel, SourceReference

Confidence = Literal["high", "medium", "low", "insufficient"]


class AnswerQuery(DomainModel):
    """A repository-bounded question and optional retrieval overrides."""

    repository_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    min_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class AnswerCitation(DomainModel):
    """A model-selected evidence ID resolved to a stored source location."""

    evidence_id: str = Field(min_length=1)
    reference: SourceReference


class GroundedAnswer(DomainModel):
    """A structured answer whose citations were validated by the application."""

    repository_id: str = Field(min_length=1)
    index_generation: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    citations: tuple[AnswerCitation, ...]
    confidence: Confidence
    limitations: tuple[str, ...]
    inferences: tuple[str, ...]
    retrieved_evidence_ids: tuple[str, ...]
    insufficient_context: bool

    @model_validator(mode="after")
    def validate_grounding_state(self) -> GroundedAnswer:
        citation_ids = tuple(citation.evidence_id for citation in self.citations)
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("Citations must be unique")
        if not set(citation_ids) <= set(self.retrieved_evidence_ids):
            raise ValueError("Citations must reference retrieved evidence")
        if self.insufficient_context:
            if self.citations or self.confidence != "insufficient":
                raise ValueError("Insufficient answers cannot contain citations")
        elif not self.citations or self.confidence == "insufficient":
            raise ValueError("Grounded answers require citations and a confidence level")
        return self


class ModelAnswer(DomainModel):
    """Strict provider output before evidence IDs are resolved to references."""

    answer: str = Field(min_length=1)
    citation_ids: tuple[str, ...]
    confidence: Confidence
    limitations: tuple[str, ...]
    inferences: tuple[str, ...]
    insufficient_context: bool
