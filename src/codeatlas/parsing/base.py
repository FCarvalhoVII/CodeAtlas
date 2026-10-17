"""Parser contracts and explicit language dispatch."""

from typing import Protocol

from codeatlas.core.exceptions import ParsingError
from codeatlas.core.models import ParsedSymbol
from codeatlas.indexing.scanner import SourceFile


class Parser(Protocol):
    """Extract structural symbols without importing repository modules."""

    def parse(self, file: SourceFile) -> tuple[ParsedSymbol, ...]: ...


class ParserRegistry:
    """Map language identifiers to replaceable parser implementations."""

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}

    def register(self, language: str, parser: Parser) -> None:
        self._parsers[language] = parser

    def parse(self, file: SourceFile) -> tuple[ParsedSymbol, ...]:
        parser = self._parsers.get(file.metadata.language)
        if parser is None:
            raise ParsingError(f"Unsupported language: {file.metadata.language}")
        return parser.parse(file)
