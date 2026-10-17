"""Physical source lines, preserving whitespace and original newline sequences."""

from io import StringIO


def source_lines(source: str) -> list[str]:
    """Split only on CR/LF, matching Python's source line numbering."""

    return StringIO(source, newline="").readlines()
