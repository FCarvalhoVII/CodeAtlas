# Command-line interface

The CodeAtlas CLI covers the full local workflow: build an index, inspect it, retrieve evidence, ask grounded questions, and remove the index. Install the project from the repository root:

```bash
python -m venv .venv
python -m pip install -e "."
codeatlas --help
```

For development tools, install `.[dev]` instead.

## Repository and configuration selection

`codeatlas index` takes the repository as a positional path. Every other command accepts `--repository` or `-r` and defaults to the current directory:

```bash
codeatlas index ./my-project
codeatlas search "request validation" --repository ./my-project
codeatlas ask "Where is the request validated?" --repository ./my-project
codeatlas status --repository ./my-project
```

Repository identity is derived from the resolved absolute path. Commands therefore select the same persisted index even when the supplied path contains `.` or `..` components. CodeAtlas does not maintain an implicit active-repository file.

All commands load `./codeatlas.toml` when it exists. `--config PATH` selects an explicit TOML file. Environment variables override TOML settings. The configured `index.storage_path` is relative to the command's working directory, so automation should either use a stable working directory or configure an absolute storage path.

## Index

```bash
codeatlas index <repository-path> [--config PATH]
```

The command prints the resolved repository before scanning so long operations have immediate progress feedback. It then reports file and chunk counts, incremental changes, skipped files, and the published generation. Repeated indexing reuses unchanged chunks.

## Search

```bash
codeatlas search <query> [-r PATH] [--top-k N] [--config PATH]
```

Search embeds the query and prints ranked evidence. Each result includes its score, repository-relative file, exact line range, symbol when available, and stored source snippet. It does not call the configured LLM.

## Ask

```bash
codeatlas ask <question> [-r PATH] [--top-k N] [--config PATH]
```

Ask retrieves source evidence and passes the bounded context to the configured LLM. CodeAtlas validates every evidence identifier returned by the provider, then prints only source references resolved from the persisted index. Output includes confidence, citations, inferences, and limitations when present.

The default fake provider keeps the workflow offline and returns an explicit insufficient-context answer. A substantive answer requires a configured LLM provider. Remote providers receive the question and selected source snippets and may incur charges; review the [provider and privacy guide](providers.md) first.

## Status and removal

```bash
codeatlas status [-r PATH] [--config PATH]
codeatlas clear [-r PATH] [--config PATH]
```

Status prints the stored repository path, generation, update time, file and chunk counts, and embedding profile. Clear removes only the selected repository's snapshot. It prompts before deletion; automation can pass `--yes` or `-y`:

```bash
codeatlas clear --repository ./my-project --yes
```

Declining the prompt leaves the index in place and exits successfully.

## Exit codes

Expected failures use stable exit codes so scripts can distinguish their cause:

| Code | Meaning |
| ---: | --- |
| `0` | Command completed, including an empty search or a declined removal. |
| `1` | Other expected CodeAtlas failure. |
| `2` | Invalid command usage or validated input. |
| `3` | Missing, unsafe, or invalid configuration. |
| `4` | Invalid or inaccessible repository path. |
| `5` | Missing, incompatible, corrupted, or inaccessible index. |
| `6` | Embedding or LLM provider failure. |

Messages for expected failures are written to standard error and do not include credentials or source snippets. Typer also uses exit code `2` when command-line arguments or option values are invalid.

## Common failures

- **No index found:** run `codeatlas index` with the same repository, configuration, storage path, and working directory used by the query command.
- **Provider does not match the index:** reindex after changing the embedding provider, model, or dimensions.
- **Missing OpenAI credential:** export `CODEATLAS_OPENAI_API_KEY`; secrets are intentionally rejected in TOML.
- **Insufficient repository context:** broaden the question, increase retrieval limits within the configured bounds, or index the repository again after source changes.
