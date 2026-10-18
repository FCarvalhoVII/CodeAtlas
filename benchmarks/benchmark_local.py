"""Reproducible offline indexing and retrieval benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

from codeatlas.indexing.service import IndexingService
from codeatlas.providers.fake import FakeEmbeddingProvider
from codeatlas.retrieval.service import SearchService
from codeatlas.storage.sqlite import SQLiteVectorStore


def _fixture(root: Path, file_count: int) -> None:
    root.mkdir()
    for number in range(file_count):
        (root / f"module_{number:04d}.py").write_text(
            f"class Service{number}:\n"
            f'    """Process record {number}."""\n\n'
            "    def process(self, value: int) -> int:\n"
            f"        return value + {number}\n",
            encoding="utf-8",
        )


async def _measure(file_count: int, query_count: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="codeatlas-benchmark-") as directory:
        base = Path(directory)
        repository = base / "repository"
        _fixture(repository, file_count)
        provider = FakeEmbeddingProvider()
        store = SQLiteVectorStore(base / "index.sqlite3")
        indexing = IndexingService(provider, store, batch_size=32, max_concurrency=4)

        started = time.perf_counter()
        manifest = await indexing.index(repository)
        cold_index_seconds = time.perf_counter() - started

        provider.calls.clear()
        started = time.perf_counter()
        unchanged = await indexing.index(repository)
        unchanged_index_seconds = time.perf_counter() - started

        search = SearchService(provider, store, default_top_k=8)
        durations: list[float] = []
        for number in range(query_count):
            started = time.perf_counter()
            await search.search_text(
                manifest.repository.repository_id,
                f"process record {number % file_count}",
            )
            durations.append(time.perf_counter() - started)

        ordered = sorted(durations)
        percentile_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return {
            "fixture": {
                "python_files": file_count,
                "chunks": manifest.chunk_count,
                "queries": query_count,
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor() or "unknown",
            },
            "measurements_seconds": {
                "cold_index": round(cold_index_seconds, 6),
                "unchanged_index": round(unchanged_index_seconds, 6),
                "search_mean": round(statistics.fmean(durations), 6),
                "search_p95": round(ordered[percentile_index], 6),
            },
            "verification": {
                "unchanged_files": unchanged.statistics.unchanged_files,
                "unchanged_embedding_batches": len(provider.calls) - query_count,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=250)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.files < 1 or arguments.queries < 1:
        parser.error("--files and --queries must be positive")

    result = asyncio.run(_measure(arguments.files, arguments.queries))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
