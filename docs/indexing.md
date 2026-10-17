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

## Incremental indexing and persistence

`IndexingService` orchestrates discovery, preparation, batched asynchronous embeddings, and storage. `create_indexing_service(load_settings())` assembles the configured embedding provider, chunker, and SQLite store. The factory uses `index.max_chunk_lines=100`; constructing `SemanticChunker` directly retains its 200-line default.

```python
import asyncio
from pathlib import Path

from codeatlas.indexing.service import IndexingService
from codeatlas.providers.fake import FakeEmbeddingProvider
from codeatlas.storage.sqlite import SQLiteVectorStore

store = SQLiteVectorStore(Path(".codeatlas/index.sqlite3"))
provider = FakeEmbeddingProvider()
service = IndexingService(provider, store, batch_size=32)
manifest = asyncio.run(service.index(Path("./my-project")))

snapshot = store.load(manifest.repository.repository_id)
assert snapshot is not None
print(manifest.statistics)
```

The manifest persists schema version, repository identity, provider/model/dimension profile, pipeline fingerprint, actual vector dimensions, successful file metadata and hashes, generation ID, UTC update time, and last-run statistics. Chunk records retain full source references, source text, parent IDs, and vectors. `service.skipped` exposes current-run diagnostics; these diagnostic messages are not persisted.

For a compatible index, unchanged hashes reuse existing vectors without parsing or embedding again. Changed/new files are prepared and embedded; deleted, ignored, binary, oversized, or no-longer-decodable files lose their previous records. Files with unsupported languages or Python syntax errors are reported and have no persisted successful hash, so later runs retry them. Empty Python files are successfully tracked with no chunks. An unreadable file/directory or unreadable `.gitignore` aborts publication, preventing an incomplete scan from deleting valid index entries.

Changing provider, model, configured dimensions, scanner options, chunk size, or pipeline version rebuilds the repository's embeddings. Custom parser registries must supply a new `pipeline_version` when behavior changes. When remote dimensions are unspecified, the first vector establishes the dimension and all subsequent vectors must match. Nonfinite, zero, inconsistent, or missing vectors fail the run.

All embedding batches finish before a write transaction starts. SQLite atomically replaces one repository's vectors and manifest. A failed batch, cancellation, validation error, or failed database transaction leaves the preceding generation intact. Optimistic generation checks reject a writer whose base generation was replaced by another indexing run. Cancellation during the HTTP adapter's blocking worker does not stop an already-started remote request, but no snapshot is published from a cancelled run.

`VectorStore` is an infrastructure-neutral protocol. The initial `SQLiteVectorStore` stores JSON payloads in SQLite tables and computes exact cosine similarity in Python, restricted to an explicit repository ID. Multiple repositories can share a database. The adapter validates persisted manifests, repository ownership, dimensions, chunk counts, and parent existence on read. Unsupported schema versions fail explicitly; automatic migrations are not implemented.

SQLite is deliberately simple and requires no external vector database or Docker. Search loads one repository snapshot and scans its vectors, so this adapter targets small local repositories, not high-volume search. Snapshot publication rewrites all records for that repository, while only changed files incur parsing/embedding work. Scan, preparation, and staging retain repository data in memory. SQLite work is synchronous and serialized on each call. A later adapter can implement the same contract with a specialized vector engine.

The low-level `store.search(repository_id, vector, top_k=8)` returns stored chunks, similarity scores, and exact snippets. Query embedding, profile compatibility checks for user queries, reranking, and retrieval deduplication belong to Milestone 4.

Keep index files private: they contain repository source in plain text. The default `.codeatlas/` directory is excluded by the scanner and Git ignore rules. If choosing a custom storage location inside a repository, add that directory to the repository's `.gitignore` or pass explicit scanner exclusions. Reopening the database in a new process requires only its path and repository ID; no source imports or provider credentials are needed to inspect stored data.
