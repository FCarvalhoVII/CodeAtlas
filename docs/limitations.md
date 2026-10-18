# Known limitations

- Safe scanning currently requires POSIX filesystem primitives. The supported CI operating systems are Linux and macOS.
- Only Python is parsed structurally. Other recognized text languages are scanned and then reported as unsupported by the default parser registry.
- The scanner bounds each file to 1 MB by default, but the full accepted repository snapshot, staged chunks, vectors, and SQLite search candidates remain in memory.
- Filesystem scanning is not an atomic snapshot. A file can change between directory enumeration and opening; the bytes actually read remain internally consistent for hashing and parsing.
- Control characters in repository-relative paths are rejected to keep persistence, logs, and terminal output safe.
- SQLite performs exact cosine search in Python and rewrites a repository snapshot on publication. It targets small local repositories rather than high-volume or multi-user workloads.
- The default fake embeddings are deterministic workflow fixtures and do not provide meaningful semantic ranking.
- Context token counts use a conservative byte estimate rather than a model tokenizer. Evidence may be truncated under the configured budgets.
- Citation validation proves that a returned evidence ID and source range exist in the supplied context. It does not prove claim-level entailment.
- OpenAI embedding retries can repeat billable requests after ambiguous transient failures. LLM generation is not retried automatically.
- Cancellation cannot stop a blocking standard-library HTTP worker already running in a thread. Its result is discarded and no cancelled index generation is published.
- Index schema version 1 has no automatic migration. An incompatible database must be cleared or moved before reindexing.
- Local indexes contain source snippets in plain text. Encryption at rest and access control are delegated to the host filesystem.
