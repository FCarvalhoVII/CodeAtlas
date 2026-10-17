from datetime import UTC, datetime

import pytest

from codeatlas.core.exceptions import ParsingError
from codeatlas.core.models import IndexedFile, SymbolType
from codeatlas.indexing.scanner import SourceFile
from codeatlas.parsing.base import ParserRegistry
from codeatlas.parsing.chunking import SemanticChunker
from codeatlas.parsing.python import PythonParser


def source_file(source: str, language: str = "python") -> SourceFile:
    return SourceFile(
        metadata=IndexedFile(
            repository_id="repo",
            file_path="src/example.py",
            language=language,
            size_bytes=len(source.encode()),
            modified_at=datetime.now(UTC),
            content_hash="a" * 64,
        ),
        source=source,
    )


SOURCE = '''"""Module docs."""
import os
from . import helper

VALUE = 2

@decorate(
    VALUE,
)
class Example:
    """Class docs."""

    @staticmethod
    async def run(
        value: int,
    ) -> int:
        """Method docs."""
        def nested():
            return value
        return nested()

# trailing section
RESULT = VALUE
'''


def test_ast_structure_decorators_async_and_docstrings() -> None:
    symbols = PythonParser().parse(source_file(SOURCE))
    module, cls, method, nested = symbols
    assert module.symbol_type == SymbolType.MODULE
    assert module.docstring == "Module docs."
    assert module.imports == ("import os", "from . import helper")
    assert cls.reference.start_line == 7
    assert cls.docstring == "Class docs."
    assert method.symbol_type == SymbolType.METHOD
    assert method.is_async
    assert method.docstring == "Method docs."
    assert method.reference.start_line == 13
    assert nested.parent_symbol == method.symbol_name
    assert nested.symbol_type == SymbolType.FUNCTION
    for symbol in symbols:
        ref = symbol.reference
        assert symbol.source_code == "".join(
            SOURCE.splitlines(keepends=True)[ref.start_line - 1 : ref.end_line]
        )


@pytest.mark.parametrize("source", ["", " \n\n"])
def test_empty_source(source: str) -> None:
    assert PythonParser().parse(source_file(source)) == ()


@pytest.mark.parametrize("source", ["def broken(:\n", "\0"])
def test_malformed_source_is_domain_error(source: str) -> None:
    with pytest.raises(ParsingError, match=r"src/example\.py"):
        PythonParser().parse(source_file(source))


def test_registry_dispatch_and_unsupported_language() -> None:
    registry = ParserRegistry()
    file = source_file("x = 1\n")
    with pytest.raises(ParsingError, match="Unsupported"):
        registry.parse(file)
    registry.register("python", PythonParser())
    assert registry.parse(file)[0].symbol_type == SymbolType.MODULE
    with pytest.raises(ParsingError, match="Unsupported"):
        PythonParser().parse(source_file("text", "text"))


@pytest.mark.parametrize("max_lines", [3, 200])
def test_chunks_preserve_exact_source_lines_and_parent_relationships(max_lines: int) -> None:
    symbols = PythonParser().parse(source_file(SOURCE))
    chunker = SemanticChunker(max_lines=max_lines)
    chunks = chunker.chunk(symbols)
    assert chunks == chunker.chunk(symbols)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    by_id = {c.chunk_id: c for c in chunks}
    covered: set[int] = set()
    for chunk in chunks:
        ref = chunk.reference
        assert 1 <= ref.start_line <= ref.end_line <= len(SOURCE.splitlines())
        assert ref.end_line - ref.start_line + 1 <= max_lines
        assert chunk.content == "".join(
            SOURCE.splitlines(keepends=True)[ref.start_line - 1 : ref.end_line]
        )
        covered.update(range(ref.start_line, ref.end_line + 1))
        if chunk.parent_chunk_id:
            assert chunk.parent_chunk_id in by_id
        if ref.symbol_name == "src.example.Example.run":
            assert by_id[chunk.parent_chunk_id or ""].reference.symbol_name == "src.example.Example"
    assert all(
        number in covered for number, line in enumerate(SOURCE.splitlines(), 1) if line.strip()
    )
    assert any(c.symbol_type == SymbolType.SECTION and "RESULT" in c.content for c in chunks)


def test_large_symbol_keeps_identity_and_indentation() -> None:
    source = "def run():\n" + "    value = 1\n" * 10
    chunks = SemanticChunker(3).chunk(PythonParser().parse(source_file(source)))
    assert len(chunks) == 4
    assert {c.reference.symbol_name for c in chunks} == {"src.example.run"}
    assert chunks[1].content.startswith("    ")
    with pytest.raises(ValueError):
        SemanticChunker(0)


def test_definitions_inside_control_flow_and_function_classes() -> None:
    source = (
        "if True:\n    def outer():\n        class Local:\n"
        "            def method(self):\n                pass\n"
    )
    symbols = PythonParser().parse(source_file(source))
    assert [s.symbol_type for s in symbols] == [
        SymbolType.MODULE,
        SymbolType.FUNCTION,
        SymbolType.CLASS,
        SymbolType.METHOD,
    ]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_physical_lines_preserve_unicode_separators_and_newlines(newline: str) -> None:
    source = newline.join(['TEXT = "first\u2028second"', "def run():", "    return TEXT", ""])
    symbols = PythonParser().parse(source_file(source))
    assert symbols[1].reference.start_line == 2
    assert symbols[1].reference.end_line == 3
    assert symbols[1].source_code == f"def run():{newline}    return TEXT{newline}"
    chunks = SemanticChunker().chunk(symbols)
    assert chunks[0].content == f'TEXT = "first\u2028second"{newline}'
    assert chunks[1].content == symbols[1].source_code


def test_module_comments_without_definitions_are_preserved() -> None:
    source = "# module comment\n\n"
    chunks = SemanticChunker(1).chunk(PythonParser().parse(source_file(source)))
    assert len(chunks) == 1
    assert chunks[0].content == "# module comment\n"
