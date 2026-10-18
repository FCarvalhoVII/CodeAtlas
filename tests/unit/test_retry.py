import asyncio

import pytest

from codeatlas.providers.retry import retry_transient


def test_retry_rejects_invalid_configuration() -> None:
    async def operation() -> str:
        return "ok"

    with pytest.raises(ValueError, match="attempts"):
        asyncio.run(
            retry_transient(
                operation,
                retry_on=(OSError,),
                max_attempts=0,
                base_delay=0,
            )
        )


def test_retry_reports_attempt_and_delay() -> None:
    attempts = 0
    retries: list[tuple[int, float]] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError
        return "ok"

    result = asyncio.run(
        retry_transient(
            operation,
            retry_on=(OSError,),
            max_attempts=2,
            base_delay=0,
            on_retry=lambda attempt, delay: retries.append((attempt, delay)),
        )
    )

    assert result == "ok"
    assert retries == [(1, 0)]
