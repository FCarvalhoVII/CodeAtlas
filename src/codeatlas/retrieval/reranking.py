"""Deterministic score normalization, metadata boosts, and overlap removal."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from codeatlas.core.models import SearchResult
from codeatlas.retrieval.models import RetrievedEvidence

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(value: str) -> set[str]:
    expanded = _CAMEL_BOUNDARY.sub(" ", value)
    return {token.casefold() for token in _TOKEN.findall(expanded)}


def _overlap(left: RetrievedEvidence, right: RetrievedEvidence) -> bool:
    first, second = left.reference, right.reference
    if first.file_path != second.file_path:
        return False
    intersection = (
        min(first.end_line, second.end_line) - max(first.start_line, second.start_line) + 1
    )
    if intersection <= 0:
        return False
    shorter = min(
        first.end_line - first.start_line + 1,
        second.end_line - second.start_line + 1,
    )
    return intersection / shorter >= 0.8


class MetadataReranker:
    """Favor exact symbol/path terms while keeping semantic score dominant."""

    def __init__(
        self,
        semantic_weight: float = 0.8,
        symbol_weight: float = 0.15,
        path_weight: float = 0.05,
    ) -> None:
        if any(weight < 0 for weight in (semantic_weight, symbol_weight, path_weight)):
            raise ValueError("Reranking weights must be nonnegative")
        total = semantic_weight + symbol_weight + path_weight
        if total <= 0:
            raise ValueError("At least one reranking weight must be positive")
        self.semantic_weight = semantic_weight / total
        self.symbol_weight = symbol_weight / total
        self.path_weight = path_weight / total

    def rerank(
        self, query: str, candidates: tuple[SearchResult, ...]
    ) -> tuple[RetrievedEvidence, ...]:
        query_tokens = _tokens(query)
        evidence: list[RetrievedEvidence] = []
        for candidate in candidates:
            chunk = candidate.chunk
            reference = chunk.reference
            symbol = reference.symbol_name or ""
            symbol_leaf = symbol.rsplit(".", 1)[-1]
            path = PurePosixPath(reference.file_path)
            symbol_match = bool(symbol and _tokens(symbol_leaf) <= query_tokens)
            path_match = any(
                terms and terms <= query_tokens
                for terms in (_tokens(path.name), _tokens(path.stem))
            )
            normalized_semantic = (candidate.score + 1) / 2
            score = (
                self.semantic_weight * normalized_semantic
                + self.symbol_weight * float(symbol_match)
                + self.path_weight * float(path_match)
            )
            evidence.append(
                RetrievedEvidence(
                    chunk_id=chunk.chunk_id,
                    repository_id=chunk.repository_id,
                    language=chunk.language,
                    score=max(0.0, min(1.0, score)),
                    semantic_score=candidate.score,
                    snippet=candidate.snippet,
                    reference=reference,
                    symbol_type=chunk.symbol_type,
                    parent_chunk_id=chunk.parent_chunk_id,
                )
            )
        return tuple(
            sorted(
                evidence,
                key=lambda item: (-item.score, -item.semantic_score, item.chunk_id),
            )
        )


def deduplicate_overlaps(
    results: tuple[RetrievedEvidence, ...],
) -> tuple[RetrievedEvidence, ...]:
    """Keep the highest-ranked result when one range mostly covers another."""

    selected: list[RetrievedEvidence] = []
    for result in results:
        if not any(_overlap(result, existing) for existing in selected):
            selected.append(result)
    return tuple(selected)
