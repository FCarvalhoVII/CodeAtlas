from __future__ import annotations

import json

import pytest

from codeatlas.answering.context import ContextAssembler, estimate_tokens
from codeatlas.core.models import SourceReference, SymbolType
from codeatlas.retrieval.models import RetrievedEvidence


def evidence(chunk_id: str, snippet: str, score: float = 1) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        repository_id="repo",
        language="python",
        score=score,
        semantic_score=score,
        snippet=snippet,
        reference=SourceReference(
            file_path=f"src/{chunk_id}.py",
            start_line=1,
            end_line=max(1, snippet.count("\n") + 1),
            symbol_name=chunk_id,
        ),
        symbol_type=SymbolType.FUNCTION,
    )


def test_context_budget_preserves_order_and_truncates_source() -> None:
    first = evidence("first", "x" * 100)
    second = evidence("second", "y" * 100, 0.9)
    assembler = ContextAssembler(max_chars=260, max_tokens=1000, max_evidence_chars=20)

    context = assembler.assemble((first, second))

    payload = json.loads(context.content)
    assert len(context.content) <= 260
    assert context.estimated_tokens <= 1000
    assert context.evidence[0] == first
    assert payload["evidence"][0]["evidence_id"] == "first"
    assert len(payload["evidence"][0]["snippet"]) <= 20
    assert "first" in context.truncated_evidence_ids
    assert tuple(item["evidence_id"] for item in payload["evidence"]) == tuple(
        item.chunk_id for item in context.evidence
    )


def test_context_obeys_estimated_token_budget() -> None:
    item = evidence("auth", "credentials = 'private'\n" * 20)
    assembler = ContextAssembler(max_chars=2000, max_tokens=55, max_evidence_chars=1000)

    context = assembler.assemble((item,))

    assert context.estimated_tokens <= 55
    assert len(context.evidence) == 1
    assert context.truncated_evidence_ids == ("auth",)


def test_context_can_admit_no_evidence_and_rejects_impossible_budgets() -> None:
    empty_size = len('{"evidence":[]}')
    assembler = ContextAssembler(
        max_chars=empty_size,
        max_tokens=estimate_tokens('{"evidence":[]}'),
        max_evidence_chars=10,
    )
    context = assembler.assemble((evidence("auth", "source"),))
    assert context.evidence == ()
    assert json.loads(context.content) == {"evidence": []}

    with pytest.raises(ValueError, match="empty evidence"):
        ContextAssembler(max_chars=2, max_tokens=1, max_evidence_chars=1)
    with pytest.raises(ValueError, match="positive"):
        ContextAssembler(max_chars=100, max_tokens=0, max_evidence_chars=1)


def test_token_estimate_uses_utf8_bytes() -> None:
    assert estimate_tokens("a") == 1
    assert estimate_tokens("éé") == 1
    assert estimate_tokens("abcdefgh") == 2
