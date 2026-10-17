"""Shared embedding response validation."""

import math


def validate_vectors(
    vectors: list[list[float]], dimensions: int | None = None
) -> list[list[float]]:
    for vector in vectors:
        if dimensions is None:
            dimensions = len(vector)
        if (
            not vector
            or len(vector) != dimensions
            or not any(vector)
            or not all(math.isfinite(value) for value in vector)
        ):
            raise ValueError("Invalid embedding vector")
    return vectors
