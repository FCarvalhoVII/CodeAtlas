"""Provider-independent retrieval and reranking contracts."""

from typing import Protocol

from codeatlas.core.models import SearchResult
from codeatlas.retrieval.models import RetrievalQuery, RetrievedEvidence


class Retriever(Protocol):
    """Retrieve evidence from one immutable index generation."""

    def retrieve(
        self,
        query: RetrievalQuery,
        vector: list[float],
        *,
        expected_generation: str,
    ) -> tuple[SearchResult, ...]: ...


class Reranker(Protocol):
    """Convert semantic candidates into normalized application evidence."""

    def rerank(
        self, query: str, candidates: tuple[SearchResult, ...]
    ) -> tuple[RetrievedEvidence, ...]: ...
