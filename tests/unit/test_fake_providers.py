import asyncio

import pytest

from codeatlas.providers import (
    FakeEmbeddingProvider,
    FakeLLMProvider,
    LLMRequest,
)


def test_fake_embeddings_are_deterministic_and_record_calls() -> None:
    provider = FakeEmbeddingProvider(dimensions=4)

    first = asyncio.run(provider.embed(["alpha", "beta"]))
    second = asyncio.run(provider.embed(["alpha"]))

    assert first[0] == second[0]
    assert first[0] != first[1]
    assert all(len(vector) == 4 for vector in first)
    assert provider.calls == [["alpha", "beta"], ["alpha"]]


@pytest.mark.parametrize("dimensions", [0, 33])
def test_fake_embedding_dimensions_are_bounded(dimensions: int) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        FakeEmbeddingProvider(dimensions)


def test_fake_llm_returns_configured_response_and_records_request() -> None:
    provider = FakeLLMProvider("Grounded response")
    request = LLMRequest(
        question="Where?",
        instructions="Use only evidence.",
        context="Relevant source",
        allowed_evidence_ids=("evidence-1",),
    )

    response = asyncio.run(provider.generate(request))

    assert response.content == "Grounded response"
    assert provider.requests == [request]
