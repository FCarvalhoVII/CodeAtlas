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


class LLMProviderError(ProviderError):
    """Raised when an LLM provider request fails."""


class StorageError(CodeAtlasError):
    """Raised when the local or remote index cannot be accessed."""


class CorruptedIndexError(StorageError):
    """Raised when persisted index data is invalid or incompatible."""
