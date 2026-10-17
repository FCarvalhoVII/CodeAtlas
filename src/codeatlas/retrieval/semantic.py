"""Exact vector retrieval with bounded candidate expansion."""

from codeatlas.core.models import SearchResult
from codeatlas.retrieval.models import RetrievalQuery
from codeatlas.storage.base import VectorStore


class SemanticRetriever:
    """Fetch semantic candidates within a repository and index generation."""

    def __init__(self, store: VectorStore, candidate_multiplier: int = 4) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        self.store = store
        self.candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: RetrievalQuery,
        vector: list[float],
        *,
        expected_generation: str,
    ) -> tuple[SearchResult, ...]:
        return self.store.search(
            query.repository_id,
            vector,
            min(1000, query.top_k * self.candidate_multiplier),
            expected_generation=expected_generation,
        )
