from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from codeatlas.core.models import IndexedFile, RepositoryIdentity, SourceReference


def test_source_reference_normalizes_windows_separator() -> None:
    reference = SourceReference(
        file_path="src\\codeatlas\\core.py",
        start_line=3,
        end_line=8,
        symbol_name="run",
    )

    assert reference.file_path == "src/codeatlas/core.py"


@pytest.mark.parametrize(
    "file_path",
    [
        "",
        ".",
        "../secret.py",
        "/tmp/source.py",
        "src/../../x.py",
        "C:\\tmp\\source.py",
        "unsafe\npath.py",
        "unsafe\x00path.py",
    ],
)
def test_source_reference_rejects_path_outside_repository(file_path: str) -> None:
    with pytest.raises(ValidationError):
        SourceReference(file_path=file_path, start_line=1, end_line=1)


def test_source_reference_rejects_reversed_lines() -> None:
    with pytest.raises(ValidationError, match="end_line"):
        SourceReference(file_path="main.py", start_line=10, end_line=2)


def test_repository_root_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match="root_path"):
        RepositoryIdentity(repository_id="repo-1", root_path=Path("relative"), name="example")


def test_indexed_file_requires_sha256_hash() -> None:
    with pytest.raises(ValidationError, match="content_hash"):
        IndexedFile(
            repository_id="repo-1",
            file_path="main.py",
            language="python",
            size_bytes=10,
            modified_at=datetime.now(UTC),
            content_hash="not-a-sha256",
        )
