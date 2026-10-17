"""Assemble repository search from validated application settings."""

from codeatlas.core.config import CodeAtlasSettings
from codeatlas.providers.factory import create_embedding_provider
from codeatlas.retrieval.service import SearchService
from codeatlas.storage.sqlite import SQLiteVectorStore


def create_search_service(settings: CodeAtlasSettings) -> SearchService:
    """Create search over the same configured SQLite embedding space as indexing."""

    return SearchService(
        create_embedding_provider(settings),
        SQLiteVectorStore(settings.index.storage_path / "index.sqlite3"),
        candidate_multiplier=settings.index.candidate_multiplier,
        default_top_k=settings.index.top_k,
        default_min_score=settings.index.min_score,
    )
