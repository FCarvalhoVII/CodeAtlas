"""Assemble a local indexing service from validated settings."""

from codeatlas.core.config import CodeAtlasSettings
from codeatlas.indexing.service import IndexingService
from codeatlas.parsing.chunking import SemanticChunker
from codeatlas.parsing.service import RepositoryPreparer
from codeatlas.providers.factory import create_embedding_provider
from codeatlas.storage.sqlite import SQLiteVectorStore


def create_indexing_service(settings: CodeAtlasSettings) -> IndexingService:
    provider = create_embedding_provider(settings)
    return IndexingService(
        provider,
        SQLiteVectorStore(settings.index.storage_path / "index.sqlite3"),
        batch_size=settings.index.batch_size,
        max_concurrency=settings.index.max_concurrency,
        preparer=RepositoryPreparer(chunker=SemanticChunker(settings.index.max_chunk_lines)),
    )
