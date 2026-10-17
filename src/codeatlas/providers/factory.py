"""Construct embedding providers from explicit application configuration."""

from codeatlas.core.config import CodeAtlasSettings
from codeatlas.core.exceptions import ConfigurationError
from codeatlas.providers.base import EmbeddingProvider
from codeatlas.providers.fake import FakeEmbeddingProvider
from codeatlas.providers.openai import OpenAIEmbeddingProvider


def create_embedding_provider(settings: CodeAtlasSettings) -> EmbeddingProvider:
    embedding = settings.embedding
    if embedding.provider == "fake":
        if embedding.model != "deterministic":
            raise ConfigurationError("The fake provider requires model='deterministic'")
        dimensions = embedding.dimensions if embedding.dimensions is not None else 8
        if dimensions > 32:
            raise ConfigurationError("Fake embedding dimensions must be between 1 and 32")
        return FakeEmbeddingProvider(dimensions)
    if embedding.provider == "openai":
        if settings.openai_api_key is None:
            raise ConfigurationError("Set CODEATLAS_OPENAI_API_KEY for OpenAI embeddings")
        if embedding.model == "deterministic":
            raise ConfigurationError("Select an OpenAI embedding model explicitly")
        return OpenAIEmbeddingProvider(
            settings.openai_api_key,
            embedding.model,
            dimensions=embedding.dimensions,
            timeout=embedding.timeout_seconds,
        )
    raise ConfigurationError(f"Unsupported embedding provider: {embedding.provider}")
