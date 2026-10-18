import json
import logging

import pytest

from codeatlas.core.logging import (
    JsonFormatter,
    RedactingTextFormatter,
    configure_logging,
    log_event,
    redact,
)


def test_redact_handles_sensitive_keys_and_nested_values() -> None:
    value = {
        "api_key": "plain-secret",
        "nested": {"authorization": "Bearer abcdefghijklmnop"},
        "request": {"prompt": "private prompt", "source_code": "private source"},
        "safe": "visible",
        "estimated_context_tokens": 42,
    }

    result = redact(value)

    assert result == {
        "api_key": "[REDACTED]",
        "nested": {"authorization": "[REDACTED]"},
        "request": {"prompt": "[REDACTED]", "source_code": "[REDACTED]"},
        "safe": "visible",
        "estimated_context_tokens": 42,
    }


def test_json_formatter_redacts_registered_secret_and_token_pattern() -> None:
    formatter = JsonFormatter(["literal-private-value"])
    record = logging.LogRecord(
        name="codeatlas.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="provider failed for literal-private-value with sk-abcdefgh12345678",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "provider failed for [REDACTED] with [REDACTED]"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "codeatlas.test"


def test_json_formatter_keeps_structured_event_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="codeatlas.indexing",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="repository scan started",
        args=(),
        exc_info=None,
    )
    record.event = "repository_scan_started"
    record.file_count = 4

    payload = json.loads(formatter.format(record))

    assert payload["event"] == "repository_scan_started"
    assert payload["file_count"] == 4


def test_text_formatter_redacts_secrets() -> None:
    formatter = RedactingTextFormatter(["private-value"])
    record = logging.LogRecord(
        name="codeatlas.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="failed with private-value",
        args=(),
        exc_info=None,
    )

    assert formatter.format(record) == "ERROR codeatlas.test failed with [REDACTED]"


def test_configure_logging_and_log_event_emit_json(capsys: pytest.CaptureFixture[str]) -> None:
    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    try:
        configure_logging(level="INFO", sensitive_values=["private-value"])
        logger = logging.getLogger("codeatlas.test")

        log_event(logger, "scan_started", repository="example", detail="private-value")

        captured = capsys.readouterr()
        payload = json.loads(captured.err)
        assert payload["event"] == "scan_started"
        assert payload["repository"] == "example"
        assert payload["detail"] == "[REDACTED]"
    finally:
        root_logger.handlers = previous_handlers
        root_logger.setLevel(previous_level)
