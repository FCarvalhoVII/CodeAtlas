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
max_attempts = 3
retry_base_delay_seconds = 0.25
# Optional dimensions must be supported by the selected model.
# dimensions = 512

[index]
storage_path = ".codeatlas"
batch_size = 32
max_concurrency = 4
max_chunk_lines = 100
```

Supply credentials only through the environment:

```bash
export CODEATLAS_OPENAI_API_KEY=your-key
```

`create_embedding_provider(settings)` constructs the configured adapter and rejects unknown providers, missing credentials, or the fake model name used with OpenAI. Provider objects may also be constructed directly with `SecretStr` credentials and injected into `IndexingService`.

The adapter follows the [official OpenAI Create embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create): it sends model, batched text input, `encoding_format="float"`, and optional dimensions to `https://api.openai.com/v1/embeddings`. It restores response order using returned indexes and rejects duplicate/missing indexes and malformed vectors. It uses the Python standard library in an asynchronous worker thread, with a 30-second socket timeout by default; redirects are disabled.

Embedding requests retry only explicitly transient failures: connection/timeouts and HTTP 408, 409, 429, and 5xx. The default is three total attempts with deterministic exponential delays of 0.25 and 0.5 seconds. `embedding.max_attempts` and `embedding.retry_base_delay_seconds` control these bounds. Validation failures and other 4xx responses fail immediately. Cancellation is never retried. An ambiguous transport failure may mean a repeated request is billable.

To stay conservatively within request limits without adding a tokenizer, the adapter accepts at most 8,192 UTF-8 bytes per input, 300,000 summed UTF-8 bytes, and 2,048 inputs. These byte limits are stricter than the API token limits and may reject text the API could accept. Reduce `index.max_chunk_lines` or `index.batch_size` when necessary; a single oversized source line may still need manual handling. Failed requests preserve the previous index. Error messages include HTTP status when available, but omit response bodies, source text, and credentials.

Enabling OpenAI sends the full text of each new or changed chunk to an external service and can incur charges. Configuration changes can resend all indexed chunks. Source includes comments, literals, docstrings, and any embedded secrets in selected files: CodeAtlas does not sanitize source before embedding it. Choose scanner exclusions appropriate for your repository. The adapter is tested against simulated transport/HTTP responses; live account/model availability is not verified by the automated suite.

## OpenAI answer adapter

The LLM remains offline by default: `provider = "fake"` and `model = "deterministic"` return a stable insufficient-context response. Enable OpenAI explicitly:

```toml
[llm]
provider = "openai"
model = "your-structured-output-capable-model"
timeout_seconds = 60
max_output_tokens = 1200

[answer]
max_context_chars = 24000
max_context_tokens = 6000
max_evidence_chars = 8000
timeout_seconds = 75
```

The adapter uses the Responses API and `text.format` with a strict JSON Schema, following the [official OpenAI Structured Outputs guidance](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses). It sends a developer instruction plus a user message containing the question, allowed evidence IDs, and bounded repository evidence JSON. Requests set `store` to false. The response must complete with exactly one `output_text`; refusals, incomplete responses, malformed envelopes, and multiple text outputs map to safe `LLMProviderError` messages.

There are two timeout layers. `llm.timeout_seconds` bounds the HTTP request and `answer.timeout_seconds` bounds the full provider coroutine. Cancellation cannot stop a blocking HTTP worker that has already started, but its result is discarded. Redirects and automatic LLM retries are disabled because a repeated generation can duplicate cost and provider-side effects. HTTP response bodies, prompts, source snippets, and credentials are never included in mapped errors or normal logs.

Remote answering sends the developer's question and selected code chunks, including comments, docstrings, literals, and potential embedded secrets, to the configured provider. Source locations and symbols are included as context metadata. Index and answer only repositories whose data policy permits this transfer. The automated suite uses fake or simulated transports and requires no live API credentials.

## Provider implementation guide

A new embedding provider implements `EmbeddingProvider`, exposes an accurate immutable `EmbeddingProfile`, and returns one finite nonzero vector per input in the original order. Provider/model/dimensions define index compatibility. Reject input before network access when practical and map sanitized failures to `EmbeddingProviderError`.

A new answer provider implements `LLMProvider` and returns `LLMResponse` containing the strict JSON consumed by `ModelAnswer`. It must treat developer instructions and repository context as separate trust domains. The application remains responsible for evidence-ID validation; adapters must not create source paths or line ranges.

For either protocol:

- accept credentials through secret-aware settings or constructor injection, never project TOML;
- bound payloads, timeouts, redirects, and response parsing;
- retry only operations and failure classes whose duplicate execution is understood;
- allow `CancelledError` to propagate and release owned resources in `finally` blocks;
- never log request bodies, source, prompts, model output, credentials, or raw provider error bodies;
- test ordering, malformed output, timeout, cancellation, and safe error messages with an injected transport;
- document exactly which repository data crosses the process boundary.
