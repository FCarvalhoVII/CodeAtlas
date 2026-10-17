"""Deterministic JSON context assembly with explicit size budgets."""

from __future__ import annotations

import json

from pydantic import Field

from codeatlas.core.models import DomainModel
from codeatlas.retrieval.models import RetrievedEvidence


class AssembledContext(DomainModel):
    """Serialized evidence plus the exact evidence admitted by the budget."""

    content: str = Field(min_length=1)
    evidence: tuple[RetrievedEvidence, ...]
    estimated_tokens: int = Field(ge=1)
    truncated_evidence_ids: tuple[str, ...] = ()


def estimate_tokens(value: str) -> int:
    """Estimate tokens as one per four UTF-8 bytes for deterministic budgeting."""

    return max(1, (len(value.encode("utf-8")) + 3) // 4)


class ContextAssembler:
    """Admit ranked evidence in order without exceeding configured budgets."""

    def __init__(
        self,
        *,
        max_chars: int = 24_000,
        max_tokens: int = 6_000,
        max_evidence_chars: int = 8_000,
    ) -> None:
        if max_chars < 2 or max_tokens < 1 or max_evidence_chars < 1:
            raise ValueError("Context budgets must be positive")
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.max_evidence_chars = max_evidence_chars
        if not self._fits(self._serialize([])):
            raise ValueError("Context budget cannot hold the empty evidence envelope")

    @staticmethod
    def _serialize(items: list[dict[str, object]]) -> str:
        return json.dumps({"evidence": items}, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _item(evidence: RetrievedEvidence, snippet: str) -> dict[str, object]:
        reference = evidence.reference
        return {
            "evidence_id": evidence.chunk_id,
            "file_path": reference.file_path,
            "start_line": reference.start_line,
            "end_line": reference.end_line,
            "symbol_name": reference.symbol_name,
            "language": evidence.language,
            "snippet": snippet,
        }

    def _fits(self, content: str) -> bool:
        return len(content) <= self.max_chars and estimate_tokens(content) <= self.max_tokens

    def assemble(self, evidence: tuple[RetrievedEvidence, ...]) -> AssembledContext:
        items: list[dict[str, object]] = []
        included: list[RetrievedEvidence] = []
        truncated: list[str] = []
        for result in evidence:
            snippet = result.snippet[: self.max_evidence_chars]
            was_truncated = len(snippet) < len(result.snippet)
            candidate = [*items, self._item(result, snippet)]
            content = self._serialize(candidate)
            if not self._fits(content):
                low, high = 0, len(snippet)
                while low < high:
                    middle = (low + high + 1) // 2
                    trial = self._serialize([*items, self._item(result, snippet[:middle])])
                    if self._fits(trial):
                        low = middle
                    else:
                        high = middle - 1
                if low == 0:
                    break
                snippet = snippet[:low]
                candidate = [*items, self._item(result, snippet)]
                was_truncated = True
            items = candidate
            included.append(result)
            if was_truncated:
                truncated.append(result.chunk_id)
        content = self._serialize(items)
        return AssembledContext(
            content=content,
            evidence=tuple(included),
            estimated_tokens=estimate_tokens(content),
            truncated_evidence_ids=tuple(truncated),
        )
