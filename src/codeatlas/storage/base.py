"""Storage contract with atomic replacement and optimistic concurrency."""

from typing import Protocol

from codeatlas.core.models import SearchResult
from codeatlas.storage.models import IndexSnapshot


class VectorStore(Protocol):
    """Persist repository snapshots and search within one repository."""

    def load(self, repository_id: str) -> IndexSnapshot | None: ...

    def save(self, snapshot: IndexSnapshot, expected_generation: str | None) -> None: ...

    def search(
        self,
        repository_id: str,
        vector: list[float],
        top_k: int = 8,
        *,
        expected_generation: str,
    ) -> tuple[SearchResult, ...]: ...
