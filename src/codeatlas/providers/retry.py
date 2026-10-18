"""Small async retry primitive for explicitly classified transient operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

ResultT = TypeVar("ResultT")


async def retry_transient(
    operation: Callable[[], Awaitable[ResultT]],
    *,
    retry_on: tuple[type[BaseException], ...],
    max_attempts: int,
    base_delay: float,
    on_retry: Callable[[int, float], None] | None = None,
) -> ResultT:
    """Retry classified failures with bounded deterministic exponential backoff."""

    if max_attempts < 1 or base_delay < 0:
        raise ValueError("Retry attempts must be positive and delay must be non-negative")
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except retry_on:
            if attempt == max_attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            if on_retry is not None:
                on_retry(attempt, delay)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable retry state")
