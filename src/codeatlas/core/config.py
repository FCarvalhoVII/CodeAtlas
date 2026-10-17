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

_FORBIDDEN_FILE_KEY_PARTS = ("api_key", "password", "secret", "token")


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

    embedding: ProviderSettings = ProviderSettings()
    llm: ProviderSettings = ProviderSettings()
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
        if any(part in key.lower() for part in _FORBIDDEN_FILE_KEY_PARTS):
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
