from pathlib import Path

import pytest

from codeatlas.core.config import load_settings
from codeatlas.core.exceptions import ConfigurationError


def test_load_settings_uses_defaults_when_default_file_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    assert settings.index.top_k == 8
    assert settings.embedding.provider == "fake"


def test_environment_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "codeatlas.toml"
    config_path.write_text("[index]\ntop_k = 3\n", encoding="utf-8")
    monkeypatch.setenv("CODEATLAS_INDEX__TOP_K", "12")

    settings = load_settings(config_path)

    assert settings.index.top_k == 12


def test_explicit_missing_config_is_an_actionable_error(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.toml"

    with pytest.raises(ConfigurationError, match="does not exist"):
        load_settings(config_path)


@pytest.mark.parametrize("environment_value", [None, "true"])
def test_logging_json_setting_from_toml_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment_value: str | None
) -> None:
    config_path = tmp_path / "codeatlas.toml"
    config_path.write_text("[logging]\njson = false\n", encoding="utf-8")
    monkeypatch.delenv("CODEATLAS_LOGGING__JSON", raising=False)
    if environment_value is not None:
        monkeypatch.setenv("CODEATLAS_LOGGING__JSON", environment_value)

    settings = load_settings(config_path)

    assert settings.logging.json_output is (environment_value is not None)
    assert settings.logging.model_dump(by_alias=True)["json"] is (environment_value is not None)


def test_invalid_toml_is_wrapped_as_configuration_error(tmp_path: Path) -> None:
    config_path = tmp_path / "codeatlas.toml"
    config_path.write_text("[index\ntop_k = 3", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid TOML"):
        load_settings(config_path)


def test_invalid_setting_is_wrapped_as_configuration_error(tmp_path: Path) -> None:
    config_path = tmp_path / "codeatlas.toml"
    config_path.write_text("[index]\ntop_k = 0\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid CodeAtlas configuration"):
        load_settings(config_path)


@pytest.mark.parametrize(
    "secret_name", ["api_key", "openai_api_key", "access_token", "password", "client_secret"]
)
def test_secrets_are_rejected_in_toml(tmp_path: Path, secret_name: str) -> None:
    config_path = tmp_path / "codeatlas.toml"
    config_path.write_text(f'[llm]\n{secret_name} = "do-not-store-this"\n', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="environment variable") as error:
        load_settings(config_path)

    assert "do-not-store-this" not in str(error.value)


def test_secret_from_environment_is_excluded_from_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEATLAS_OPENAI_API_KEY", "sk-private-value")

    settings = load_settings()

    assert settings.openai_api_key is not None
    assert "sk-private-value" not in repr(settings)
    assert "openai_api_key" not in settings.model_dump()


def test_indexing_settings_from_toml_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "codeatlas.toml"
    path.write_text(
        '[embedding]\nprovider = "fake"\ndimensions = 4\ntimeout_seconds = 10\n'
        "[index]\nbatch_size = 2\nmax_chunk_lines = 40\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEATLAS_INDEX__BATCH_SIZE", "3")
    settings = load_settings(path)
    assert settings.embedding.dimensions == 4
    assert settings.embedding.timeout_seconds == 10
    assert settings.index.batch_size == 3
    assert settings.index.max_chunk_lines == 40
