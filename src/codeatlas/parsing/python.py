"""Python AST extraction with decorator-aware source references."""

from __future__ import annotations

import ast
from pathlib import PurePosixPath

from codeatlas.core.exceptions import ParsingError
from codeatlas.core.models import ParsedSymbol, SourceReference, SymbolType
from codeatlas.indexing.scanner import SourceFile
from codeatlas.parsing.source import source_lines


class PythonParser:
    """Extract modules and nested definitions; never evaluate source code."""

    def parse(self, file: SourceFile) -> tuple[ParsedSymbol, ...]:
        if file.metadata.language != "python":
            raise ParsingError(f"Unsupported language: {file.metadata.language}")
        try:
            tree = ast.parse(file.source, filename=file.metadata.file_path)
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise ParsingError(f"Cannot parse Python file: {file.metadata.file_path}") from exc
        lines = source_lines(file.source)
        if not lines or not file.source.strip():
            return ()
        imports = tuple(
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
        )
        module_name = (
            PurePosixPath(file.metadata.file_path).with_suffix("").as_posix().replace("/", ".")
        )
        symbols: list[ParsedSymbol] = []

        def add(
            name: str,
            kind: SymbolType,
            start: int,
            end: int,
            parent: str | None,
            docstring: str | None,
            is_async: bool = False,
        ) -> None:
            symbols.append(
                ParsedSymbol(
                    repository_id=file.metadata.repository_id,
                    language="python",
                    symbol_name=name,
                    symbol_type=kind,
                    source_code="".join(lines[start - 1 : end]),
                    reference=SourceReference(
                        file_path=file.metadata.file_path,
                        start_line=start,
                        end_line=end,
                        symbol_name=name,
                    ),
                    parent_symbol=parent,
                    imports=imports,
                    docstring=docstring,
                    is_async=is_async,
                )
            )

        add(module_name, SymbolType.MODULE, 1, len(lines), None, ast.get_docstring(tree))

        def visit(node: ast.AST, parent: str, parent_kind: SymbolType) -> None:
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{parent}.{node.name}"
                kind = (
                    SymbolType.CLASS
                    if isinstance(node, ast.ClassDef)
                    else (
                        SymbolType.METHOD
                        if parent_kind == SymbolType.CLASS
                        else SymbolType.FUNCTION
                    )
                )
                start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
                add(
                    name,
                    kind,
                    start,
                    node.end_lineno or node.lineno,
                    parent,
                    ast.get_docstring(node),
                    isinstance(node, ast.AsyncFunctionDef),
                )
                for child in ast.iter_child_nodes(node):
                    visit(child, name, kind)
            else:
                for child in ast.iter_child_nodes(node):
                    visit(child, parent, parent_kind)

        for node in tree.body:
            visit(node, module_name, SymbolType.MODULE)
        return tuple(symbols)
