"""Qualified Python function identities and body measurements."""

import ast
import io
import tokenize
from dataclasses import dataclass
from typing import override


@dataclass(frozen=True)
class Function:
    name: str
    line: int
    code_lines: int
    longest_run: int
    body_nodes: int
    occurrence: int
    docstring_statements: int


class PythonSource(ast.NodeVisitor):
    def __init__(self, text: str) -> None:
        self.lines = text.splitlines()
        self.tree = ast.parse(text)
        self.comment_lines = {
            token.start[0]
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type == tokenize.COMMENT
            and self.lines[token.start[0] - 1].lstrip().startswith("#")
        }
        self.names: list[str] = []
        self.functions: dict[int, Function] = {}
        self.overload_lines: set[int] = set()
        self.overloads = self.overload_names()
        self.doc_lines = self.docstring_lines()
        self.visit(self.tree)

    def overload_names(self) -> set[str]:
        names = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module in ("typing", "typing_extensions"):
                names.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "overload"
                )
            elif isinstance(node, ast.Import):
                names.update(
                    f"{alias.asname or alias.name}.overload"
                    for alias in node.names
                    if alias.name in ("typing", "typing_extensions")
                )
        return names

    @staticmethod
    def docstrings(node: ast.AST) -> list[ast.Expr]:
        return [
            child.body[0]
            for child in ast.walk(node)
            if isinstance(child, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and ast.get_docstring(child, clean=False) is not None
            and isinstance(child.body[0], ast.Expr)
        ]

    def docstring_lines(self) -> set[int]:
        return {
            row
            for docstring in self.docstrings(self.tree)
            for row in range(docstring.lineno, (docstring.end_lineno or docstring.lineno) + 1)
        }

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.append(node.name)
        self.generic_visit(node)
        self.names.pop()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if any(ast.unparse(decorator) in self.overloads for decorator in node.decorator_list):
            self.overload_lines.add(node.lineno)
            return
        self.names.append(node.name)
        code, run = self.body_counts(node)
        name = ".".join(self.names)
        occurrence = sum(item.name == name for item in self.functions.values())
        nodes = self.body_nodes(node)
        self.functions[node.lineno] = Function(
            name, node.lineno, code, run, nodes, occurrence, self.statement_docstrings(node)
        )
        self.generic_visit(node)
        self.names.pop()

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def body_counts(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
        excluded = self.doc_lines.copy()
        for child in ast.walk(node):
            if child is node or not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            start = min([child.lineno, *(item.lineno for item in child.decorator_list)])
            excluded.update(range(start, (child.end_lineno or child.lineno) + 1))
        code = run = longest = 0
        for row in range(node.body[0].lineno, (node.end_lineno or node.lineno) + 1):
            if row in excluded:
                run = 0
                continue
            line = self.lines[row - 1].strip()
            if not line or row in self.comment_lines:
                run = 0
            else:
                code += 1
                run += 1
                longest = max(longest, run)
        return code, longest

    @staticmethod
    def body_nodes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        excluded = {
            child for docstring in PythonSource.docstrings(node) for child in ast.walk(docstring)
        }
        return sum(
            child not in excluded for statement in node.body for child in ast.walk(statement)
        )

    @staticmethod
    def statement_docstrings(node: ast.AST) -> int:
        """Count documentation included in Ruff's statement total.

        Returns:
            The number of function docstrings reached without entering class bodies.
        """
        if isinstance(node, ast.ClassDef):
            return 0
        own = int(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and ast.get_docstring(node, clean=False) is not None
        )
        return own + sum(
            PythonSource.statement_docstrings(child) for child in ast.iter_child_nodes(node)
        )
