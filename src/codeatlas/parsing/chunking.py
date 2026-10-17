"""Symbol-based chunks with bounded, traceable line splitting."""

from __future__ import annotations

import hashlib
import json

from codeatlas.core.models import CodeChunk, ParsedSymbol, SourceReference, SymbolType
from codeatlas.parsing.source import source_lines


class SemanticChunker:
    """Keep symbols together and retain module text outside top-level definitions."""

    def __init__(self, max_lines: int = 200) -> None:
        if max_lines < 1:
            raise ValueError("max_lines must be positive")
        self.max_lines = max_lines

    def chunk(self, symbols: tuple[ParsedSymbol, ...]) -> tuple[CodeChunk, ...]:
        chunks: list[CodeChunk] = []
        first_chunks: dict[str, str] = {}

        def emit(symbol: ParsedSymbol, start: int, lines: list[str], kind: SymbolType) -> None:
            parent = first_chunks.get(symbol.parent_symbol or "")
            for offset in range(0, len(lines), self.max_lines):
                part = lines[offset : offset + self.max_lines]
                content = "".join(part)
                if not content.strip():
                    continue
                reference = SourceReference(
                    file_path=symbol.reference.file_path,
                    start_line=start + offset,
                    end_line=start + offset + len(part) - 1,
                    symbol_name=symbol.symbol_name,
                )
                identity = json.dumps(
                    [
                        symbol.repository_id,
                        reference.model_dump(),
                        kind.value,
                        content,
                    ],
                    sort_keys=True,
                )
                chunk_id = hashlib.sha256(identity.encode()).hexdigest()
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        repository_id=symbol.repository_id,
                        language=symbol.language,
                        content=content,
                        reference=reference,
                        symbol_type=kind,
                        parent_chunk_id=parent,
                    )
                )
                first_chunks.setdefault(symbol.symbol_name, chunk_id)

        for symbol in symbols:
            lines = source_lines(symbol.source_code)
            if symbol.symbol_type != SymbolType.MODULE:
                emit(symbol, symbol.reference.start_line, lines, symbol.symbol_type)
                continue
            occupied: set[int] = set()
            for child in symbols:
                if child.parent_symbol == symbol.symbol_name:
                    occupied.update(range(child.reference.start_line, child.reference.end_line + 1))
            section: list[str] = []
            section_start = symbol.reference.start_line
            for number, line in enumerate(lines, symbol.reference.start_line):
                if number in occupied:
                    if section:
                        emit(symbol, section_start, section, SymbolType.SECTION)
                        section = []
                else:
                    if not section:
                        section_start = number
                    section.append(line)
            if section:
                emit(symbol, section_start, section, SymbolType.SECTION)
        return tuple(chunks)
