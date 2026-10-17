from __future__ import annotations

import asyncio
import io
import json
import urllib.error
from email.message import Message
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from codeatlas.core.config import CodeAtlasSettings, LLMSettings
from codeatlas.core.exceptions import ConfigurationError, LLMProviderError
from codeatlas.providers.base import LLMRequest
from codeatlas.providers.factory import create_llm_provider
from codeatlas.providers.fake import FakeLLMProvider
from codeatlas.providers.openai_llm import OpenAILLMProvider, _NoRedirect


def request() -> LLMRequest:
    return LLMRequest(
        question="Where is auth?",
        instructions="Use evidence only.",
        context='{"evidence":[]}',
        allowed_evidence_ids=("chunk-1",),
    )


def structured_content() -> str:
    return json.dumps(
        {
            "answer": "Authentication is in the service.",
            "citation_ids": ["chunk-1"],
            "confidence": "high",
            "limitations": [],
            "inferences": [],
            "insufficient_context": False,
        }
    )


def api_response(content: str | None = None) -> dict[str, object]:
    return {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": content or structured_content()}],
            }
        ],
    }


def test_responses_payload_uses_strict_schema_and_preserves_output() -> None:
    bodies: list[dict[str, object]] = []

    async def transport(body: dict[str, object]) -> object:
        bodies.append(body)
        return api_response()

    provider = OpenAILLMProvider(
        SecretStr("private-test-key"),
        "test-model",
        max_output_tokens=321,
        transport=transport,
    )
    response = asyncio.run(provider.generate(request()))

    assert response.content == structured_content()
    body = bodies[0]
    assert body["model"] == "test-model"
    assert body["store"] is False
    assert body["max_output_tokens"] == 321
    assert body["text"]["format"]["type"] == "json_schema"  # type: ignore[index]
    assert body["text"]["format"]["strict"] is True  # type: ignore[index]
    assert body["input"][0]["role"] == "developer"  # type: ignore[index]
    assert "chunk-1" in body["input"][1]["content"]  # type: ignore[index]


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"status": "incomplete", "output": []},
        {"status": "completed", "output": []},
        {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}],
        },
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "one"},
                        {"type": "output_text", "text": "two"},
                    ],
                }
            ],
        },
    ],
)
def test_malformed_incomplete_and_refused_responses_are_safe(response: object) -> None:
    async def transport(body: dict[str, object]) -> object:
        return response

    provider = OpenAILLMProvider(SecretStr("private-key"), "model", transport=transport)
    with pytest.raises(LLMProviderError) as error:
        asyncio.run(provider.generate(request()))
    assert "private" not in str(error.value)


def test_transport_failure_is_mapped_without_sensitive_data() -> None:
    async def transport(body: dict[str, object]) -> object:
        raise TimeoutError("private source")

    provider = OpenAILLMProvider(SecretStr("private-key"), "model", transport=transport)
    with pytest.raises(LLMProviderError, match="transport") as error:
        asyncio.run(provider.generate(request()))
    assert "private" not in str(error.value)


def test_http_request_status_timeout_and_redirect_handling() -> None:
    provider = OpenAILLMProvider(SecretStr("private-key"), "model", timeout=2)
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(api_response()).encode()
    with patch("urllib.request.build_opener") as factory:
        factory.return_value.open.return_value = response
        assert provider._post(provider._body(request())) == api_response()
        sent = factory.return_value.open.call_args.args[0]
        assert sent.full_url == "https://api.openai.com/v1/responses"
        assert sent.get_header("Authorization") == "Bearer private-key"
        assert factory.return_value.open.call_args.kwargs["timeout"] == 2

    failure = urllib.error.HTTPError(
        "https://api.openai.com", 429, "private", Message(), io.BytesIO(b"private")
    )
    with patch("urllib.request.build_opener") as factory:
        factory.return_value.open.side_effect = failure
        with pytest.raises(LLMProviderError, match="429") as error:
            provider._post(provider._body(request()))
        assert "private" not in str(error.value)
        assert failure.closed
    _NoRedirect().redirect_request(MagicMock(), None, 302, "", None, "https://other")


def test_configuration_validation() -> None:
    for key, model in (("", "model"), ("key", ""), ("key", "deterministic")):
        with pytest.raises(ConfigurationError):
            OpenAILLMProvider(SecretStr(key), model)
    with pytest.raises(ValueError, match="positive"):
        OpenAILLMProvider(SecretStr("key"), "model", timeout=0)
    with pytest.raises(ValueError, match="positive"):
        OpenAILLMProvider(SecretStr("key"), "model", max_output_tokens=0)


def test_llm_factory_requires_explicit_safe_configuration() -> None:
    assert isinstance(create_llm_provider(CodeAtlasSettings()), FakeLLMProvider)
    settings = CodeAtlasSettings(
        llm=LLMSettings(provider="openai", model="structured-model"),
        openai_api_key=SecretStr("private-key"),
    )
    assert isinstance(create_llm_provider(settings), OpenAILLMProvider)
    invalid = (
        CodeAtlasSettings(llm=LLMSettings(provider="unknown")),
        CodeAtlasSettings(llm=LLMSettings(model="wrong")),
        CodeAtlasSettings(llm=LLMSettings(provider="openai", model="structured-model")),
        CodeAtlasSettings(
            llm=LLMSettings(provider="openai"),
            openai_api_key=SecretStr("private-key"),
        ),
    )
    for configuration in invalid:
        with pytest.raises(ConfigurationError):
            create_llm_provider(configuration)
