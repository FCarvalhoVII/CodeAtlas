# Repository scanning and structural preparation

Milestone 2 provides an in-process API for discovering source and producing traceable chunks. Embedding, persistence, and the public CLI follow in later milestones.

```python
from pathlib import Path

from codeatlas.indexing.scanner import RepositoryScanner, ScanOptions
from codeatlas.parsing.service import RepositoryPreparer

scanner = RepositoryScanner(
    ScanOptions(include=("*.py", "*.pyi"), exclude=("tests/",), max_file_bytes=1_000_000)
)
scan = scanner.scan(Path("./my-project"))
prepared = RepositoryPreparer().prepare(scan)

for chunk in prepared.chunks:
    ref = chunk.reference
    print(f"{ref.file_path}:{ref.start_line}-{ref.end_line} {ref.symbol_name}")

for skipped in prepared.skipped:
    print(f"Skipped {skipped.file_path}: {skipped.reason}")
```

## Discovery

The root is resolved and must be an existing directory. Repository IDs are SHA-256 hashes of the resolved root path, so moving a repository changes its identity. File paths are relative POSIX paths. Traversal order is deterministic.

Scanning currently requires POSIX (Linux/macOS). Directory-relative file descriptors, `O_NOFOLLOW`, and `os.fwalk(follow_symlinks=False)` prevent following file or directory symlinks, including linked `.gitignore` files. The explicitly supplied root may resolve through a symlink. Nonregular files are skipped without blocking on FIFOs. Unreadable directories and files produce diagnostics rather than stopping discovery.

Default exclusions cover `.git`, `node_modules`, `.venv`, `venv`, `dist`, `build`, `coverage`, `__pycache__`, `.codeatlas`, and tool caches. Include and exclude patterns use Gitignore syntax through `pathspec`; excludes and Gitignore rules take precedence over includes. An empty include tuple selects no files. Files below excluded directories are not visited or individually reported.

Root and nested `.gitignore` files are supported, including negation, directory patterns, and patterns anchored to the containing directory. Nested rules override matching ancestor rules. As with Git, negation cannot restore a file inside a directory that has already been pruned. Global Git excludes, `.git/info/exclude`, and Git's tracked-file state are not consulted. Set `respect_gitignore=False` to disable repository ignore files.

Reads are bounded by `max_file_bytes` (default 1 MB). Null-containing files and undecodable text are skipped. Python encoding cookies are honored; other text uses UTF-8. Metadata contains language, byte count, UTC modification time, and SHA-256 of the original bytes. Unknown extensions remain `text` so scanning is not limited to Python.

The scan retains decoded source alongside metadata so parsing uses the same snapshot as hashing. This bounds each file read, but total snapshot memory grows with the repository. Concurrently modified repositories are not an atomic filesystem snapshot; rescan after changes. These APIs neither import repository modules nor send source to an external service.

## Parsing

`Parser` is a language-neutral protocol, and `ParserRegistry` dispatches by the scanned language identifier. `RepositoryPreparer` registers `PythonParser` by default; callers can supply a custom registry and chunker. Unsupported languages and malformed Python files are reported in `PreparationResult.skipped`; other files continue processing.

The Python parser uses only `ast.parse`. It extracts modules, classes, functions, async functions, methods, nested definitions, docstrings, and import statements. Names include the module path and lexical parents. Definitions inside control flow retain their enclosing lexical symbol. Imports are file-wide context on each symbol, not a resolved dependency graph.

References use inclusive physical line ranges. Decorators are included; indentation, newline sequences, and final newlines are preserved. Empty or whitespace-only files produce no symbols. Comments outside definitions remain in module sections; comments inside a definition remain in that symbol's source. A trailing comment after a definition belongs to a module section according to its physical location.

## Semantic chunks

Ordinary definitions remain whole. Module-level text outside outermost definitions becomes section chunks, preserving imports, constants, comments, and executable statements. Classes include their full bodies and also expose separate method chunks; nested definitions likewise overlap their enclosing symbol intentionally. Retrieval deduplication belongs to Milestone 4.

Symbols and sections larger than `SemanticChunker(max_lines=200)` are split into contiguous physical line ranges. This fallback can split a statement; it preserves exact source text and qualified symbol identity rather than claiming each part is independently parseable. The file byte limit also bounds unusually long individual lines. Blank-only parts are omitted.

Every chunk carries repository identity, language, kind, and a source reference. IDs are deterministic hashes of repository identity, reference, kind, and content. Child chunks point to the first emitted chunk of their enclosing symbol when available. A module containing only definitions has no module section chunk, so its top-level definitions have no parent chunk ID. Source references still retain qualified symbol names.
