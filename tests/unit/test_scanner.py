import hashlib
import os
from pathlib import Path

import pytest

from codeatlas.core.exceptions import InvalidRepositoryError
from codeatlas.indexing.scanner import RepositoryScanner, ScanOptions, resolve_repository


def write(root: Path, name: str, content: bytes = b"x = 1\n") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_scan_metadata_and_preserves_source(tmp_path: Path) -> None:
    source = b"  # comment\n\n"
    path = write(tmp_path, "src/main.py", source)
    result = RepositoryScanner().scan(tmp_path)
    assert result.repository.root_path == tmp_path.resolve()
    assert result == RepositoryScanner().scan(tmp_path)
    (file,) = result.files
    assert file.source == source.decode()
    assert file.metadata.file_path == "src/main.py"
    assert file.metadata.language == "python"
    assert file.metadata.size_bytes == len(source)
    assert file.metadata.content_hash == hashlib.sha256(source).hexdigest()
    assert file.metadata.modified_at.timestamp() == pytest.approx(path.stat().st_mtime)


def test_filters_binary_generated_and_symlinks(tmp_path: Path) -> None:
    write(tmp_path, "src/keep.py")
    write(tmp_path, "src/exclude.py")
    write(tmp_path, "node_modules/dep.py")
    write(tmp_path, "binary.py", b"abc\0def")
    write(tmp_path, "README.md")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("raise RuntimeError('never read')", encoding="utf-8")
    (tmp_path / "link.py").symlink_to(outside)
    (tmp_path / "linked_dir").symlink_to(tmp_path.parent, target_is_directory=True)
    (tmp_path / "broken.py").symlink_to(tmp_path / "missing")
    options = ScanOptions(include=("*.py",), exclude=("exclude.py",))
    result = RepositoryScanner(options).scan(tmp_path)
    assert [f.metadata.file_path for f in result.files] == ["src/keep.py"]
    assert {s.file_path for s in result.skipped} >= {
        "node_modules",
        "binary.py",
        "link.py",
        "linked_dir",
        "broken.py",
        "README.md",
    }


def test_nested_gitignore_negation_and_anchoring(tmp_path: Path) -> None:
    write(tmp_path, ".gitignore", b"*.py\n!keep.py\n/root.txt\nignored/\n")
    write(tmp_path, "keep.py")
    write(tmp_path, "drop.py")
    write(tmp_path, "root.txt")
    write(tmp_path, "ignored/keep.py")
    write(tmp_path, "nested/root.txt")
    write(tmp_path, "nested/.gitignore", b"!restore.py\nkeep.py\n")
    write(tmp_path, "nested/restore.py")
    write(tmp_path, "nested/keep.py")
    paths = {f.metadata.file_path for f in RepositoryScanner().scan(tmp_path).files}
    assert paths == {
        ".gitignore",
        "keep.py",
        "nested/.gitignore",
        "nested/root.txt",
        "nested/restore.py",
    }
    unfiltered = RepositoryScanner(ScanOptions(respect_gitignore=False)).scan(tmp_path)
    assert "drop.py" in {f.metadata.file_path for f in unfiltered.files}


def test_empty_invalid_oversized_and_non_utf8(tmp_path: Path) -> None:
    assert RepositoryScanner().scan(tmp_path).files == ()
    with pytest.raises(InvalidRepositoryError):
        RepositoryScanner().scan(tmp_path / "missing")
    path = write(tmp_path, "large.py", b"x" * 101)
    with pytest.raises(InvalidRepositoryError):
        RepositoryScanner().scan(path)
    write(tmp_path, "invalid.txt", b"\xff")
    write(tmp_path, "empty.py", b"")
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "pipe")
    result = RepositoryScanner(ScanOptions(max_file_bytes=100)).scan(tmp_path)
    assert [f.metadata.file_path for f in result.files] == ["empty.py"]
    assert {s.file_path for s in result.skipped} >= {"large.py", "invalid.txt"}


def test_python_encoding_cookie_and_other_languages(tmp_path: Path) -> None:
    write(tmp_path, "latin.py", b"# coding: latin-1\nname = 'caf\xe9'\n")
    write(tmp_path, "main.rs", b"fn main() {}\n")
    files = RepositoryScanner().scan(tmp_path).files
    assert files[0].source.endswith("name = 'café'\n")
    assert files[1].metadata.language == "rust"


def test_symlinked_gitignore_is_never_read(tmp_path: Path) -> None:
    write(tmp_path, "keep.py")
    outside = tmp_path.parent / f"{tmp_path.name}-ignore"
    outside.write_text("*.py\n", encoding="utf-8")
    (tmp_path / ".gitignore").symlink_to(outside)
    result = RepositoryScanner().scan(tmp_path)
    assert [f.metadata.file_path for f in result.files] == ["keep.py"]
    assert any(s.reason == "gitignore unreadable" for s in result.skipped)


def test_content_change_updates_hash_and_repository_identity_is_root_specific(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    path = write(first_root, "main.py")
    write(second_root, "main.py")
    scanner = RepositoryScanner()
    first = scanner.scan(first_root)
    path.write_text("x = 2\n", encoding="utf-8")
    changed = scanner.scan(first_root)
    assert first.repository == changed.repository
    assert first.files[0].metadata.content_hash != changed.files[0].metadata.content_hash
    assert scanner.scan(second_root).repository.repository_id != first.repository.repository_id


def test_repository_identity_can_be_resolved_without_scanning(tmp_path: Path) -> None:
    identity = resolve_repository(tmp_path)

    assert identity == RepositoryScanner().scan(tmp_path).repository
    with pytest.raises(InvalidRepositoryError):
        resolve_repository(tmp_path / "missing")


def test_control_characters_in_paths_are_skipped_safely(tmp_path: Path) -> None:
    write(tmp_path, "safe.py")
    write(tmp_path, "unsafe\nname.py")
    write(tmp_path, "unsafe\tdirectory/main.py")

    result = RepositoryScanner().scan(tmp_path)

    assert [file.metadata.file_path for file in result.files] == ["safe.py"]
    assert {(item.file_path, item.reason) for item in result.skipped} >= {
        ("unsafe\\nname.py", "unsafe path"),
        ("unsafe\\tdirectory", "unsafe path"),
    }
