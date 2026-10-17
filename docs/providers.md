# Embedding providers

`EmbeddingProvider` exposes `profile` (provider/model/configured dimensions) and `async embed(texts)`. Responses preserve input order. Indexing verifies vector count, finite values, nonzero vectors, and a consistent dimension. The profile is part of index compatibility; use a distinct profile when changing an embedding space.

## Offline development

The default provider is `FakeEmbeddingProvider`, with model `deterministic` and eight dimensions. It hashes source text into reproducible vectors and records calls for tests. It supports 1–32 dimensions. These vectors exercise orchestration and storage; their similarity is not semantic code relevance.

Ordinary tests use fake embeddings or simulated OpenAI transport responses. No API key or external AI request is required.

## OpenAI adapter

Enable remote embeddings explicitly in `codeatlas.toml`:

```toml
[embedding]
provider = "openai"
model = "text-embedding-3-small"
timeout_seconds = 30
# Optional dimensions must be supported by the selected model.
# dimensions = 512

[index]
storage_path = ".codeatlas"
batch_size = 32
max_chunk_lines = 100
```

Supply credentials only through the environment:

```bash
export CODEATLAS_OPENAI_API_KEY=your-key
```

`create_embedding_provider(settings)` constructs the configured adapter and rejects unknown providers, missing credentials, or the fake model name used with OpenAI. Provider objects may also be constructed directly with `SecretStr` credentials and injected into `IndexingService`.

The adapter follows the [official OpenAI Create embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create): it sends model, batched text input, `encoding_format="float"`, and optional dimensions to `https://api.openai.com/v1/embeddings`. It restores response order using returned indexes and rejects duplicate/missing indexes and malformed vectors. It uses the Python standard library in an asynchronous worker thread, with a 30-second socket timeout by default; redirects are disabled. There are no automatic retries.

To stay conservatively within request limits without adding a tokenizer, the adapter accepts at most 8,192 UTF-8 bytes per input, 300,000 summed UTF-8 bytes, and 2,048 inputs. These byte limits are stricter than the API token limits and may reject text the API could accept. Reduce `index.max_chunk_lines` or `index.batch_size` when necessary; a single oversized source line may still need manual handling. Failed requests preserve the previous index. Error messages include HTTP status when available, but omit response bodies, source text, and credentials.

Enabling OpenAI sends the full text of each new or changed chunk to an external service and can incur charges. Configuration changes can resend all indexed chunks. Source includes comments, literals, docstrings, and any embedded secrets in selected files: CodeAtlas does not sanitize source before embedding it. Choose scanner exclusions appropriate for your repository. The adapter is tested against simulated transport/HTTP responses; live account/model availability is not verified by the automated suite.

## Extension points

A new provider implements the protocol and supplies an accurate embedding profile. Map expected network/provider failures to `EmbeddingProviderError`, avoid logging submitted source or credentials, and preserve batch ordering. Contract tests should cover counts, dimensions, invalid vectors, and errors without relying on live services. The LLM protocol and fake remain available; production LLM integration belongs to Milestone 5.
