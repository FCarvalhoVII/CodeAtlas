"""Configuration loading from TOML and environment variables."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from codeatlas.core.exceptions import ConfigurationError

_FORBIDDEN_FILE_KEY_SUFFIXES = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


class ProviderSettings(BaseModel):
    """A provider selection with no embedded credentials."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider: str = Field(default="fake", min_length=1)
    model: str = Field(default="deterministic", min_length=1)


class IndexSettings(BaseModel):
    """Index and retrieval defaults."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=8, ge=1, le=100)
    storage_path: Path = Path(".codeatlas")
    batch_size: int = Field(default=32, ge=1, le=2048)
    max_concurrency: int = Field(default=4, ge=1, le=32)
    max_chunk_lines: int = Field(default=100, ge=1)
    min_score: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    candidate_multiplier: int = Field(default=4, ge=1, le=20)


class EmbeddingSettings(ProviderSettings):
    """Embedding-specific output and request settings."""

    dimensions: int | None = Field(default=None, ge=1)
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    max_attempts: int = Field(default=3, ge=1, le=5)
    retry_base_delay_seconds: float = Field(default=0.25, ge=0, le=10, allow_inf_nan=False)


class LLMSettings(ProviderSettings):
    """Text-generation provider request settings."""

    timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    max_output_tokens: int = Field(default=1200, ge=1)


class AnswerSettings(BaseModel):
    """Grounded-answer context and application timeout limits."""

    model_config = ConfigDict(extra="forbid")

    max_context_chars: int = Field(default=24_000, ge=256)
    max_context_tokens: int = Field(default=6_000, ge=64)
    max_evidence_chars: int = Field(default=8_000, ge=64)
    timeout_seconds: float = Field(default=75, gt=0, allow_inf_nan=False)


class LoggingSettings(BaseModel):
    """Application logging controls."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_output: bool = Field(default=True, alias="json")


class CodeAtlasSettings(BaseSettings):
    """Validated application settings; environment values override TOML values."""

    model_config = SettingsConfigDict(
        env_prefix="CODEATLAS_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    embedding: EmbeddingSettings = EmbeddingSettings()
    llm: LLMSettings = LLMSettings()
    answer: AnswerSettings = AnswerSettings()
    index: IndexSettings = IndexSettings()
    logging: LoggingSettings = LoggingSettings()
    openai_api_key: SecretStr | None = Field(default=None, repr=False, exclude=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls
        return env_settings, init_settings, dotenv_settings, file_secret_settings


def _find_forbidden_key(value: object, path: tuple[str, ...] = ()) -> str | None:
    if not isinstance(value, dict):
        return None
    for raw_key, nested in value.items():
        key = str(raw_key)
        current_path = (*path, key)
        normalized_key = key.lower()
        if any(
            normalized_key == suffix or normalized_key.endswith(f"_{suffix}")
            for suffix in _FORBIDDEN_FILE_KEY_SUFFIXES
        ):
            return ".".join(current_path)
        found = _find_forbidden_key(nested, current_path)
        if found:
            return found
    return None


def load_settings(config_path: Path | None = None) -> CodeAtlasSettings:
    """Load settings, allowing environment variables to override TOML values."""

    path = config_path or Path.cwd() / "codeatlas.toml"
    if config_path is not None and not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    data: dict[str, Any] = {}
    if path.is_file():
        try:
            with path.open("rb") as config_file:
                data = tomllib.load(config_file)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigurationError(f"Invalid TOML configuration in {path}: {exc}") from exc

        forbidden_key = _find_forbidden_key(data)
        if forbidden_key:
            raise ConfigurationError(
                f"Secret-like setting '{forbidden_key}' is not allowed in {path}; "
                "use an environment variable instead"
            )

    try:
        return CodeAtlasSettings(**data)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid CodeAtlas configuration: {exc}") from exc
