"""Structured logging with conservative secret redaction."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

_SENSITIVE_KEY_PARTS = ("api_key", "authorization", "credential", "password", "secret", "token")
_TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
)
_STANDARD_RECORD_KEYS = frozenset(logging.makeLogRecord({}).__dict__)


def _redact_string(value: str, sensitive_values: tuple[str, ...]) -> str:
    redacted = value
    for secret in sensitive_values:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact(value: object, sensitive_values: tuple[str, ...] = ()) -> object:
    """Recursively redact recognized credentials from log data."""

    if isinstance(value, str):
        return _redact_string(value, sensitive_values)
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS)
                else redact(nested, sensitive_values)
            )
            for key, nested in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact(item, sensitive_values) for item in value)
    if isinstance(value, list):
        return [redact(item, sensitive_values) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object per line."""

    def __init__(self, sensitive_values: Iterable[str] = ()) -> None:
        super().__init__()
        self._sensitive_values = tuple(sensitive_values)

    def format(self, record: logging.LogRecord) -> str:
        fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_KEYS and not key.startswith("_")
        }
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **fields,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload, self._sensitive_values), default=str, sort_keys=True)


class RedactingTextFormatter(logging.Formatter):
    """Format human-readable logs while removing registered credentials."""

    def __init__(self, sensitive_values: Iterable[str] = ()) -> None:
        super().__init__("%(levelname)s %(name)s %(message)s")
        self._sensitive_values = tuple(sensitive_values)

    def format(self, record: logging.LogRecord) -> str:
        return _redact_string(super().format(record), self._sensitive_values)


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = True,
    sensitive_values: Iterable[str] = (),
) -> None:
    """Configure the root logger without exposing registered credentials."""

    handler = logging.StreamHandler()
    formatter: logging.Formatter
    if json_output:
        formatter = JsonFormatter(sensitive_values)
    else:
        formatter = RedactingTextFormatter(sensitive_values)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    """Emit a named structured event."""

    logger.info(event, extra={"event": event, **fields})
