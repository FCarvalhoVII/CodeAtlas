"""Deterministic providers for tests and offline development."""

from __future__ import annotations

import hashlib

from codeatlas.providers.base import LLMRequest, LLMResponse
from codeatlas.storage.models import EmbeddingProfile


class FakeEmbeddingProvider:
    """Map text to deterministic vectors without network access."""

    def __init__(self, dimensions: int = 8) -> None:
        if dimensions < 1 or dimensions > 32:
            raise ValueError("dimensions must be between 1 and 32")
        self.dimensions = dimensions
        self.profile = EmbeddingProfile(
            provider="fake", model="deterministic", dimensions=dimensions
        )
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([byte / 255 for byte in digest[: self.dimensions]])
        return vectors


class FakeLLMProvider:
    """Return a configured response and record requests."""

    def __init__(self, response: str = "Insufficient repository context.") -> None:
        self.response = response
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(content=self.response)
