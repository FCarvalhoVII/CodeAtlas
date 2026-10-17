"""Convert a scan snapshot into chunks with per-file parsing diagnostics."""

from codeatlas.core.exceptions import ParsingError
from codeatlas.core.models import CodeChunk, DomainModel
from codeatlas.indexing.scanner import ScanResult, SkippedFile
from codeatlas.parsing.base import ParserRegistry
from codeatlas.parsing.chunking import SemanticChunker
from codeatlas.parsing.python import PythonParser


class PreparationResult(DomainModel):
    """Structural repository output, ready for future embedding."""

    chunks: tuple[CodeChunk, ...]
    skipped: tuple[SkippedFile, ...]


class RepositoryPreparer:
    """Parse supported files and continue when individual files cannot be parsed."""

    def __init__(
        self,
        registry: ParserRegistry | None = None,
        chunker: SemanticChunker | None = None,
    ) -> None:
        if registry is None:
            registry = ParserRegistry()
            registry.register("python", PythonParser())
        self.registry = registry
        self.chunker = chunker or SemanticChunker()

    def prepare(self, scan: ScanResult) -> PreparationResult:
        chunks: list[CodeChunk] = []
        skipped = list(scan.skipped)
        for file in scan.files:
            try:
                chunks.extend(self.chunker.chunk(self.registry.parse(file)))
            except ParsingError as exc:
                skipped.append(SkippedFile(file_path=file.metadata.file_path, reason=str(exc)))
        return PreparationResult(chunks=tuple(chunks), skipped=tuple(skipped))
