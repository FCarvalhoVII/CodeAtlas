import asyncio
import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from codeatlas.core.config import CodeAtlasSettings, EmbeddingSettings
from codeatlas.core.exceptions import ConfigurationError, EmbeddingProviderError
from codeatlas.providers.factory import create_embedding_provider
from codeatlas.providers.fake import FakeEmbeddingProvider
from codeatlas.providers.openai import OpenAIEmbeddingProvider, _NoRedirect


@pytest.mark.parametrize("kind", ["fake", "openai"])
def test_provider_contract_determinism_and_order(kind: str) -> None:
    async def transport(body: dict[str, object]) -> object:
        assert body["input"] == ["first", "second"]
        assert body["encoding_format"] == "float"
        assert body["dimensions"] == 2
        return {
            "data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
        }

    provider = (
        FakeEmbeddingProvider(2)
        if kind == "fake"
        else OpenAIEmbeddingProvider(
            SecretStr("private-test-key"), dimensions=2, transport=transport
        )
    )
    first = asyncio.run(provider.embed(["first", "second"]))
    assert first == asyncio.run(provider.embed(["first", "second"]))
    assert len(first) == 2 and all(len(vector) == 2 for vector in first)
    assert first[0] != first[1]
    assert asyncio.run(provider.embed([])) == []
    if kind == "openai":
        assert first == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"data": []},
        {"data": [{"index": 1, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": [float("nan")]}]},
        {"data": [{"index": 0, "embedding": [0.0]}]},
        {"data": [{"index": 0, "embedding": ["secret-response-value"]}]},
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]},
    ],
)
def test_malformed_responses_are_safe_errors(response: object) -> None:
    async def transport(body: dict[str, object]) -> object:
        return response

    provider = OpenAIEmbeddingProvider(SecretStr("private-test-key"), transport=transport)
    with pytest.raises(EmbeddingProviderError) as error:
        asyncio.run(provider.embed(["private-source"]))
    assert "private" not in str(error.value)
    assert "secret-response-value" not in str(error.value)


@pytest.mark.parametrize("texts", [[""], [" "], ["x" * 8193], ["x"] * 2049, ["x" * 8000] * 38])
def test_input_limits_reject_before_transport(texts: list[str]) -> None:
    provider = OpenAIEmbeddingProvider(SecretStr("private-test-key"))
    with patch("urllib.request.build_opener") as opener, pytest.raises(EmbeddingProviderError):
        asyncio.run(provider.embed(texts))
    opener.assert_not_called()


def test_http_request_and_transport_failure_mapping() -> None:
    provider = OpenAIEmbeddingProvider(SecretStr("private-test-key"), timeout=2)
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(
        {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}
    ).encode()
    with patch("urllib.request.build_opener") as factory:
        factory.return_value.open.return_value = response
        assert asyncio.run(provider.embed(["source"])) == [[1.0, 2.0]]
        request = factory.return_value.open.call_args.args[0]
        assert request.full_url == "https://api.openai.com/v1/embeddings"
        assert request.get_header("Authorization") == "Bearer private-test-key"
        assert factory.return_value.open.call_args.kwargs["timeout"] == 2
    failures = [
        urllib.error.HTTPError(
            "https://api.openai.com", 429, "private", Message(), io.BytesIO(b"private")
        ),
        TimeoutError("private-test-key"),
        ValueError("private-source"),
    ]
    for failure in failures:
        with patch("urllib.request.build_opener") as factory:
            factory.return_value.open.side_effect = failure
            with pytest.raises(EmbeddingProviderError) as error:
                asyncio.run(provider.embed(["source"]))
            assert "private" not in str(error.value)
            if isinstance(failure, urllib.error.HTTPError):
                assert failure.closed
    _NoRedirect().redirect_request(MagicMock(), None, 302, "", None, "https://other")


def test_transport_failure_is_safe() -> None:
    async def transport(body: dict[str, object]) -> object:
        raise OSError("private-source")

    provider = OpenAIEmbeddingProvider(
        SecretStr("private-test-key"), transport=transport, retry_base_delay=0
    )
    with pytest.raises(EmbeddingProviderError, match="after 3 attempts"):
        asyncio.run(provider.embed(["text"]))


def test_factory_credentials_and_configuration(tmp_path: Path) -> None:
    assert isinstance(create_embedding_provider(CodeAtlasSettings()), FakeEmbeddingProvider)
    settings = CodeAtlasSettings(
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        openai_api_key=SecretStr("private-test-key"),
    )
    assert isinstance(create_embedding_provider(settings), OpenAIEmbeddingProvider)
    invalid = [
        CodeAtlasSettings(embedding=EmbeddingSettings(provider="unknown")),
        CodeAtlasSettings(embedding=EmbeddingSettings(model="wrong")),
        CodeAtlasSettings(embedding=EmbeddingSettings(dimensions=33)),
        CodeAtlasSettings(
            embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small")
        ),
        CodeAtlasSettings(
            embedding=EmbeddingSettings(provider="openai"), openai_api_key=SecretStr("test")
        ),
    ]
    for config in invalid:
        with pytest.raises(ConfigurationError):
            create_embedding_provider(config)
    with pytest.raises(ConfigurationError):
        OpenAIEmbeddingProvider(SecretStr(""))
    with pytest.raises(ValueError, match="timeout"):
        OpenAIEmbeddingProvider(SecretStr("key"), timeout=0)
    with pytest.raises(ValueError, match="retry"):
        OpenAIEmbeddingProvider(SecretStr("key"), max_attempts=0)


def test_dimension_mismatch_rejected_by_adapter() -> None:
    async def transport(body: dict[str, object]) -> object:
        return {"data": [{"index": 0, "embedding": [1.0]}]}

    provider = OpenAIEmbeddingProvider(SecretStr("test"), dimensions=2, transport=transport)
    with pytest.raises(EmbeddingProviderError):
        asyncio.run(provider.embed(["text"]))


def test_transient_transport_failure_retries_then_succeeds() -> None:
    attempts = 0

    async def transport(body: dict[str, object]) -> object:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("private transient detail")
        return {"data": [{"index": 0, "embedding": [1.0]}]}

    provider = OpenAIEmbeddingProvider(
        SecretStr("private-test-key"),
        transport=transport,
        max_attempts=3,
        retry_base_delay=0,
    )

    assert asyncio.run(provider.embed(["text"])) == [[1.0]]
    assert attempts == 3


def test_cancellation_is_never_retried() -> None:
    attempts = 0

    async def transport(body: dict[str, object]) -> object:
        nonlocal attempts
        attempts += 1
        raise asyncio.CancelledError

    provider = OpenAIEmbeddingProvider(
        SecretStr("private-test-key"), transport=transport, retry_base_delay=0
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(provider.embed(["text"]))
    assert attempts == 1


@pytest.mark.parametrize("status, expected_attempts", [(400, 1), (503, 3)])
def test_http_retries_only_transient_statuses(status: int, expected_attempts: int) -> None:
    provider = OpenAIEmbeddingProvider(
        SecretStr("private-test-key"), max_attempts=3, retry_base_delay=0
    )

    def fail(*args: object, **kwargs: object) -> object:
        raise urllib.error.HTTPError(
            "https://api.openai.com", status, "private", Message(), io.BytesIO(b"private")
        )

    with patch("urllib.request.build_opener") as factory:
        factory.return_value.open.side_effect = fail
        with pytest.raises(EmbeddingProviderError) as error:
            asyncio.run(provider.embed(["text"]))

    assert factory.return_value.open.call_count == expected_attempts
    assert "private" not in str(error.value)
