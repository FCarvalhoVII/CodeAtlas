"""Bounded repository scanning without following symbolic links."""

from __future__ import annotations

import errno
import hashlib
import io
import os
import stat
import tokenize
from datetime import UTC, datetime
from pathlib import Path

from pathspec import GitIgnoreSpec
from pydantic import ConfigDict, Field

from codeatlas.core.exceptions import InvalidRepositoryError
from codeatlas.core.models import DomainModel, IndexedFile, RepositoryIdentity

_DEFAULT_EXCLUDES = (
    ".git/",
    "node_modules/",
    ".venv/",
    "venv/",
    "dist/",
    "build/",
    "coverage/",
    "__pycache__/",
    ".codeatlas/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
)
_LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".md": "markdown",
    ".toml": "toml",
}


class ScanOptions(DomainModel):
    """Gitignore-style filters; excludes take precedence over includes."""

    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    respect_gitignore: bool = True
    max_file_bytes: int = Field(default=1_000_000, ge=1)


class SourceFile(DomainModel):
    """Metadata and decoded source from a single bounded read."""

    metadata: IndexedFile
    # Empty files are valid scanner input; source whitespace must survive validation.
    source: str = Field(default="", strict=True)

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)


class SkippedFile(DomainModel):
    """A relative path and a safe diagnostic, without source contents."""

    file_path: str
    reason: str


class ScanResult(DomainModel):
    """An immutable repository snapshot with explicit skipped paths."""

    repository: RepositoryIdentity
    files: tuple[SourceFile, ...]
    skipped: tuple[SkippedFile, ...]


def _read_at(directory_fd: int, name: str, limit: int) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("not a regular file")
        if metadata.st_size > limit:
            raise ValueError("file exceeds size limit")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("file exceeds size limit")
        return data, metadata


class RepositoryScanner:
    """Scan text files on POSIX using directory-relative, no-follow reads."""

    def __init__(self, options: ScanOptions | None = None) -> None:
        self.options = options or ScanOptions()

    def scan(self, root: Path) -> ScanResult:
        if os.name != "posix":
            raise InvalidRepositoryError("Safe repository scanning currently requires POSIX")
        try:
            resolved = root.resolve(strict=True)
            if not resolved.is_dir():
                raise InvalidRepositoryError(f"Repository is not a directory: {root}")
        except (OSError, RuntimeError) as exc:
            raise InvalidRepositoryError(f"Cannot resolve repository: {root}") from exc
        repository = RepositoryIdentity(
            repository_id=hashlib.sha256(os.fsencode(resolved)).hexdigest(),
            root_path=resolved,
            name=resolved.name or str(resolved),
        )
        include = GitIgnoreSpec.from_lines(self.options.include)
        exclude = GitIgnoreSpec.from_lines((*_DEFAULT_EXCLUDES, *self.options.exclude))
        rules: dict[Path, tuple[tuple[Path, GitIgnoreSpec], ...]] = {}
        files: list[SourceFile] = []
        skipped: list[SkippedFile] = []

        def skip(path: str, reason: str) -> None:
            skipped.append(SkippedFile(file_path=path, reason=reason))

        def walk_error(error: OSError) -> None:
            skip(os.path.relpath(error.filename or resolved, resolved), "directory unreadable")

        for directory, directories, names, directory_fd in os.fwalk(
            resolved,
            follow_symlinks=False,
            onerror=walk_error,
        ):
            base = Path(directory)
            inherited = rules.get(base.parent, ()) if base != resolved else ()
            if self.options.respect_gitignore and ".gitignore" in names:
                try:
                    data, _ = _read_at(directory_fd, ".gitignore", self.options.max_file_bytes)
                    inherited += (
                        (base, GitIgnoreSpec.from_lines(data.decode("utf-8").splitlines())),
                    )
                except (OSError, ValueError, UnicodeError):
                    skip(
                        (base / ".gitignore").relative_to(resolved).as_posix(),
                        "gitignore unreadable",
                    )
            rules[base] = inherited

            def ignored(
                path: Path,
                is_directory: bool,
                active_rules: tuple[tuple[Path, GitIgnoreSpec], ...] = inherited,
            ) -> bool:
                relative = path.relative_to(resolved).as_posix() + ("/" if is_directory else "")
                if exclude.match_file(relative):
                    return True
                matched = False
                for rule_base, spec in active_rules:
                    decision = spec.check_file(
                        path.relative_to(rule_base).as_posix() + ("/" if is_directory else "")
                    ).include
                    if decision is not None:
                        matched = decision
                return matched

            for name in sorted(directories):
                path = base / name
                relative = path.relative_to(resolved).as_posix()
                if path.is_symlink() or ignored(path, True):
                    directories.remove(name)
                    skip(relative, "symlink" if path.is_symlink() else "ignored")
            directories.sort()
            for name in sorted(names):
                path = base / name
                relative = path.relative_to(resolved).as_posix()
                if ignored(path, False) or not include.match_file(relative):
                    skip(relative, "ignored")
                    continue
                try:
                    data, metadata = _read_at(directory_fd, name, self.options.max_file_bytes)
                    if b"\0" in data:
                        skip(relative, "binary file")
                        continue
                    language = _LANGUAGES.get(path.suffix.lower(), "text")
                    encoding = (
                        tokenize.detect_encoding(io.BytesIO(data).readline)[0]
                        if language == "python"
                        else "utf-8"
                    )
                    source = data.decode(encoding)
                except OSError as exc:
                    skip(relative, "symlink" if exc.errno == errno.ELOOP else "file unreadable")
                    continue
                except (UnicodeError, SyntaxError, LookupError):
                    skip(relative, "unsupported encoding")
                    continue
                except ValueError:
                    skip(relative, "oversized or nonregular file")
                    continue
                files.append(
                    SourceFile(
                        metadata=IndexedFile(
                            repository_id=repository.repository_id,
                            file_path=relative,
                            language=language,
                            size_bytes=len(data),
                            modified_at=datetime.fromtimestamp(metadata.st_mtime, UTC),
                            content_hash=hashlib.sha256(data).hexdigest(),
                        ),
                        source=source,
                    )
                )
        return ScanResult(repository=repository, files=tuple(files), skipped=tuple(skipped))
