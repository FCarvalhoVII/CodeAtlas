from __future__ import annotations

import pytest

from codeatlas.core.models import CodeChunk, SearchResult, SourceReference, SymbolType
from codeatlas.retrieval.models import RetrievalQuery
from codeatlas.retrieval.reranking import MetadataReranker, deduplicate_overlaps


def candidate(
    chunk_id: str,
    score: float,
    *,
    path: str = "src/auth/service.py",
    symbol: str = "src.auth.service.authenticateUser",
    start: int = 10,
    end: int = 20,
) -> SearchResult:
    content = "\n".join(f"line {line}" for line in range(start, end + 1))
    chunk = CodeChunk(
        chunk_id=chunk_id,
        repository_id="repo",
        language="python",
        content=content,
        reference=SourceReference(
            file_path=path,
            start_line=start,
            end_line=end,
            symbol_name=symbol,
        ),
        symbol_type=SymbolType.FUNCTION,
    )
    return SearchResult(chunk=chunk, score=score, snippet=content)


def test_reranking_normalizes_similarity_and_boosts_exact_metadata() -> None:
    reranker = MetadataReranker()
    semantic_winner = candidate("semantic", 1.0, symbol="other")
    metadata_winner = candidate("metadata", 0.9)

    results = reranker.rerank(
        "Where is authenticate user in service.py?",
        (semantic_winner, metadata_winner),
    )

    assert [result.chunk_id for result in results] == ["metadata", "semantic"]
    assert results[0].score == pytest.approx(0.96)
    assert results[0].semantic_score == 0.9
    assert results[0].reference == metadata_winner.chunk.reference
    assert results[0].snippet == metadata_winner.snippet


def test_reranking_is_deterministic_and_scores_are_bounded() -> None:
    reranker = MetadataReranker(semantic_weight=1, symbol_weight=0, path_weight=0)
    candidates = (candidate("b", -1), candidate("a", 1, start=30, end=40))

    first = reranker.rerank("unmatched", candidates)

    assert first == reranker.rerank("unmatched", candidates)
    assert [(result.chunk_id, result.score) for result in first] == [("a", 1), ("b", 0)]


def test_overlap_deduplication_keeps_best_but_preserves_distinct_ranges() -> None:
    reranker = MetadataReranker()
    ranked = reranker.rerank(
        "authenticate user",
        (
            candidate("best", 1, start=10, end=20),
            candidate("covered", 0.9, start=11, end=19),
            candidate("partial", 0.8, start=18, end=28),
            candidate("other-file", 0.7, path="src/api.py", start=11, end=19),
        ),
    )

    deduplicated = deduplicate_overlaps(ranked)

    assert [result.chunk_id for result in deduplicated] == [
        "best",
        "partial",
        "other-file",
    ]


def test_reranker_configuration_and_query_validation() -> None:
    for weights in ((-1, 1, 1), (0, 0, 0)):
        try:
            MetadataReranker(*weights)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid weights should be rejected")

    query = RetrievalQuery(repository_id=" repo ", text=" auth ", top_k=2)
    assert query.repository_id == "repo"
    assert query.text == "auth"
