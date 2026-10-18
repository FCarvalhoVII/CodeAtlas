"""Domain-specific exceptions exposed by CodeAtlas."""


class CodeAtlasError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(CodeAtlasError):
    """Raised when configuration is missing, malformed, or unsafe."""


class InvalidRepositoryError(CodeAtlasError):
    """Raised when a repository path cannot be indexed safely."""


class ParsingError(CodeAtlasError):
    """Raised when source code cannot be parsed."""


class ProviderError(CodeAtlasError):
    """Base class for failures reported by an external provider."""


class EmbeddingProviderError(ProviderError):
    """Raised when an embedding provider request fails."""


class PartialIndexingError(EmbeddingProviderError):
    """Report a failed embedding batch without publishing partial results."""

    def __init__(self, failed_batch: int, total_batches: int, completed_batches: int) -> None:
        self.failed_batch = failed_batch
        self.total_batches = total_batches
        self.completed_batches = completed_batches
        super().__init__(
            f"Embedding batch {failed_batch}/{total_batches} failed after "
            f"{completed_batches} completed batches; the previous index was preserved"
        )


class LLMProviderError(ProviderError):
    """Raised when an LLM provider request fails."""


class StorageError(CodeAtlasError):
    """Raised when the local or remote index cannot be accessed."""


class CorruptedIndexError(StorageError):
    """Raised when persisted index data is invalid or incompatible."""


class IndexNotFoundError(StorageError):
    """Raised when a repository has no persisted index."""


class IncompatibleIndexError(StorageError):
    """Raised when a query provider cannot search an existing embedding space."""
