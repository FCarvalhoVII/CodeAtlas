"""Application service for LLM-free repository search."""

from __future__ import annotations

import logging

from codeatlas.core.exceptions import (
    EmbeddingProviderError,
    IncompatibleIndexError,
    IndexNotFoundError,
)
from codeatlas.core.logging import log_event
from codeatlas.providers.base import EmbeddingProvider
from codeatlas.providers.validation import validate_vectors
from codeatlas.retrieval.base import Reranker, Retriever
from codeatlas.retrieval.models import RetrievalQuery, RetrievalResponse
from codeatlas.retrieval.reranking import MetadataReranker, deduplicate_overlaps
from codeatlas.retrieval.semantic import SemanticRetriever
from codeatlas.storage.base import VectorStore

_LOGGER = logging.getLogger(__name__)


class SearchService:
    """Embed one query and return evidence from an exact index generation."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        store: VectorStore,
        *,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        candidate_multiplier: int = 4,
        default_top_k: int = 8,
        default_min_score: float = 0,
    ) -> None:
        # Reuse request validation for configuration supplied by composition roots.
        defaults = RetrievalQuery(
            repository_id="configuration",
            text="configuration",
            top_k=default_top_k,
            min_score=default_min_score,
        )
        self.provider = provider
        self.store = store
        self.retriever = retriever or SemanticRetriever(store, candidate_multiplier)
        self.reranker = reranker or MetadataReranker()
        self.default_top_k = defaults.top_k
        self.default_min_score = defaults.min_score

    async def search_text(
        self,
        repository_id: str,
        text: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> RetrievalResponse:
        """Search using service defaults while allowing per-request overrides."""

        return await self.search(
            RetrievalQuery(
                repository_id=repository_id,
                text=text,
                top_k=self.default_top_k if top_k is None else top_k,
                min_score=self.default_min_score if min_score is None else min_score,
            )
        )

    async def search(self, query: RetrievalQuery) -> RetrievalResponse:
        snapshot = self.store.load(query.repository_id)
        if snapshot is None:
            raise IndexNotFoundError(f"No index found for repository: {query.repository_id}")
        manifest = snapshot.manifest
        if manifest.embedding != self.provider.profile:
            raise IncompatibleIndexError(
                "Query embedding provider does not match the indexed embedding profile"
            )
        if not snapshot.records:
            return RetrievalResponse(
                repository_id=query.repository_id,
                index_generation=manifest.generation,
                results=(),
            )
        vectors = await self.provider.embed([query.text])
        try:
            vectors = validate_vectors(vectors, manifest.vector_dimensions)
        except (TypeError, ValueError):
            raise EmbeddingProviderError(
                "Embedding provider returned an invalid query vector"
            ) from None
        if len(vectors) != 1:
            raise EmbeddingProviderError(
                "Embedding provider returned an incorrect query vector count"
            )
        candidates = self.retriever.retrieve(
            query,
            vectors[0],
            expected_generation=manifest.generation,
        )
        ranked = self.reranker.rerank(query.text, candidates)
        results = tuple(
            result for result in deduplicate_overlaps(ranked) if result.score >= query.min_score
        )[: query.top_k]
        if any(result.repository_id != query.repository_id for result in results):
            raise IncompatibleIndexError("Retriever returned evidence from another repository")
        log_event(
            _LOGGER,
            "retrieval_performed",
            repository_id=query.repository_id,
            index_generation=manifest.generation,
            candidate_count=len(candidates),
            result_count=len(results),
        )
        return RetrievalResponse(
            repository_id=query.repository_id,
            index_generation=manifest.generation,
            results=results,
        )
