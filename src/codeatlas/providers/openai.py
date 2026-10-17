"""OpenAI embedding adapter using the documented JSON HTTP API."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, SecretStr, StrictFloat, StrictInt, ValidationError

from codeatlas.core.exceptions import ConfigurationError, EmbeddingProviderError
from codeatlas.providers.validation import validate_vectors
from codeatlas.storage.models import EmbeddingProfile

Transport = Callable[[dict[str, object]], Awaitable[object]]


class _Embedding(BaseModel):
    model_config = ConfigDict(extra="ignore")
    index: StrictInt
    embedding: list[StrictFloat]


class _Response(BaseModel):
    model_config = ConfigDict(extra="ignore")
    data: list[_Embedding]


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


class OpenAIEmbeddingProvider:
    """Explicit opt-in adapter: submitted chunk contents leave the local machine."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str = "text-embedding-3-small",
        *,
        dimensions: int | None = None,
        timeout: float = 30,
        transport: Transport | None = None,
    ) -> None:
        if not api_key.get_secret_value().strip():
            raise ConfigurationError("Set CODEATLAS_OPENAI_API_KEY for OpenAI embeddings")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.profile = EmbeddingProfile(provider="openai", model=model, dimensions=dimensions)
        self._api_key = api_key
        self.timeout = timeout
        self._transport = transport

    def _post(self, body: dict[str, object]) -> object:
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
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
            raise EmbeddingProviderError(
                f"OpenAI embeddings request failed (HTTP {status})"
            ) from None
        except (OSError, ValueError):
            raise EmbeddingProviderError("OpenAI embeddings request failed or timed out") from None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingProviderError("Embedding inputs must not be empty")
        # UTF-8 bytes conservatively bound tokens without adding a tokenizer dependency.
        if (
            any(len(text.encode()) > 8192 for text in texts)
            or sum(len(text.encode()) for text in texts) > 300_000
            or len(texts) > 2048
        ):
            raise EmbeddingProviderError(
                "Embedding input exceeds conservative request limits; reduce chunk or batch size"
            )
        body: dict[str, object] = {
            "model": self.profile.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.profile.dimensions is not None:
            body["dimensions"] = self.profile.dimensions
        try:
            raw = (
                await self._transport(body)
                if self._transport
                else await asyncio.to_thread(self._post, body)
            )
            response = _Response.model_validate(raw)
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(texts))):
                raise ValueError("invalid response indexes")
            return validate_vectors([item.embedding for item in ordered], self.profile.dimensions)
        except EmbeddingProviderError:
            raise
        except (ValidationError, ValueError, TypeError, OSError):
            raise EmbeddingProviderError(
                "OpenAI embeddings returned an invalid response or transport failed"
            ) from None
