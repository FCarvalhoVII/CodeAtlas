"""OpenAI Responses API adapter with strict structured output."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from codeatlas.core.exceptions import ConfigurationError, LLMProviderError
from codeatlas.providers.base import LLMRequest, LLMResponse

Transport = Callable[[dict[str, object]], Awaitable[object]]

_ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citation_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", "insufficient"],
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
        "inferences": {"type": "array", "items": {"type": "string"}},
        "insufficient_context": {"type": "boolean"},
    },
    "required": [
        "answer",
        "citation_ids",
        "confidence",
        "limitations",
        "inferences",
        "insufficient_context",
    ],
    "additionalProperties": False,
}


class _Content(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["output_text", "refusal"]
    text: str | None = None
    refusal: str | None = None


class _Output(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    content: list[_Content] = Field(default_factory=list)


class _Response(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["completed", "failed", "in_progress", "cancelled", "queued", "incomplete"]
    output: list[_Output]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class OpenAILLMProvider:
    """Generate schema-constrained answers through the OpenAI Responses API."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        *,
        timeout: float = 60,
        max_output_tokens: int = 1200,
        transport: Transport | None = None,
    ) -> None:
        if not api_key.get_secret_value().strip():
            raise ConfigurationError("Set CODEATLAS_OPENAI_API_KEY for OpenAI answers")
        if not model.strip() or model == "deterministic":
            raise ConfigurationError("Select an OpenAI LLM model explicitly")
        if timeout <= 0 or max_output_tokens < 1:
            raise ValueError("Timeout and max_output_tokens must be positive")
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self._api_key = api_key
        self._transport = transport

    def _body(self, request: LLMRequest) -> dict[str, object]:
        user_content = (
            f"Question:\n{request.question}\n\n"
            f"Allowed evidence IDs:\n{json.dumps(request.allowed_evidence_ids)}\n\n"
            f"Repository evidence JSON:\n{request.context}"
        )
        return {
            "model": self.model,
            "store": False,
            "max_output_tokens": self.max_output_tokens,
            "input": [
                {"role": "developer", "content": request.instructions},
                {"role": "user", "content": user_content},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "grounded_code_answer",
                    "description": "A repository answer citing only supplied evidence IDs.",
                    "strict": True,
                    "schema": _ANSWER_SCHEMA,
                }
            },
        }

    def _post(self, body: dict[str, object]) -> object:
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.build_opener(_NoRedirect()).open(
                request, timeout=self.timeout
            ) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise LLMProviderError(f"OpenAI response request failed (HTTP {status})") from None
        except (OSError, ValueError):
            raise LLMProviderError("OpenAI response request failed or timed out") from None

    @staticmethod
    def _content(raw: object) -> str:
        try:
            response = _Response.model_validate(raw)
        except ValidationError:
            raise LLMProviderError("OpenAI returned a malformed response") from None
        if response.status != "completed":
            raise LLMProviderError(f"OpenAI response did not complete: {response.status}")
        texts: list[str] = []
        for output in response.output:
            for content in output.content:
                if content.type == "refusal":
                    raise LLMProviderError("OpenAI refused to generate the repository answer")
                if content.type == "output_text" and content.text is not None:
                    texts.append(content.text)
        if len(texts) != 1 or not texts[0].strip():
            raise LLMProviderError("OpenAI returned no single structured answer")
        return texts[0]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        body = self._body(request)
        try:
            raw = (
                await self._transport(body)
                if self._transport is not None
                else await asyncio.to_thread(self._post, body)
            )
        except LLMProviderError:
            raise
        except (OSError, TimeoutError, ValueError, TypeError):
            raise LLMProviderError("OpenAI response transport failed") from None
        return LLMResponse(content=self._content(raw))
