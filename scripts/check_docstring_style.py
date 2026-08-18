#!/usr/bin/env python3
"""Reject Python docstrings that do not use Sphinx field-list syntax."""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DOCSTRING_OWNERS = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)
LEGACY_SECTIONS = {
    "Args",
    "Arguments",
    "Parameters",
    "Keyword Args",
    "Keyword Arguments",
    "Other Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Attributes",
    "Public Properties",
    "Methods",
    "Notes",
    "Examples",
    "Warnings",
    "See Also",
    "References",
    "Example",
    "Note",
    "Warning",
}
NONCANONICAL_FIELDS = re.compile(
    r"^\s*:(?:returns|yields|Example|example|reference|Yields|raises):",
    re.MULTILINE,
)
PARAM_FIELD = re.compile(r"^\s*:(?:param|type)(?:\s+([^:]*))?:", re.MULTILINE)
PARAM_NAME = re.compile(r"\*{0,2}[A-Za-z_]\w*")
DOCUMENTED_PARAM = re.compile(r"^\s*:param\s+([^:]+):", re.MULTILINE)
SPHINX_FIELD = re.compile(
    r"^\s+:(?:param|type|return|rtype|raises|yield|ytype|ivar|vartype|keyword)\b"
)


def repository_python_files() -> list[Path]:
    """Return tracked and non-ignored untracked Python files.

    ``git ls-files --cached`` still lists files whose deletion is staged but not
    committed, so paths that no longer exist on disk are filtered out; a deleted
    file has no docstrings to check.

    :return: Deterministically ordered Python paths in the repository.
    :rtype: list[Path]
    """
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "*.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    candidates = (ROOT / name for name in result.stdout.splitlines() if name)
    return [path for path in candidates if path.is_file()]


def python_files(paths: Iterable[Path]) -> list[Path]:
    """Resolve explicit files and directories to Python paths.

    :param paths: Files or directories supplied on the command line.
    :type paths: Iterable[Path]
    :return: Deterministically ordered Python paths.
    :rtype: list[Path]
    """
    resolved: set[Path] = set()
    for path in paths:
        candidate = path if path.is_absolute() else ROOT / path
        if candidate.is_file() and candidate.suffix == ".py":
            resolved.add(candidate)
        elif candidate.is_dir():
            resolved.update(item for item in candidate.rglob("*.py") if item.is_file())
    return sorted(resolved)


def docstrings(tree: ast.AST):
    """Yield source nodes and raw docstrings from an abstract syntax tree.

    :param tree: Parsed Python syntax tree.
    :type tree: ast.AST
    :yield: Pairs containing a docstring owner and its text.
    :ytype: tuple[ast.AST, str]
    """
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_OWNERS) or not node.body:
            continue
        expression = node.body[0]
        if not (
            isinstance(expression, ast.Expr)
            and isinstance(expression.value, ast.Constant)
            and isinstance(expression.value.value, str)
        ):
            continue
        yield node, expression.value.value


def legacy_section_lines(docstring: str) -> list[int]:
    """Return relative line numbers containing legacy section headings.

    :param docstring: Docstring text to inspect.
    :type docstring: str
    :return: One-based relative line numbers.
    :rtype: list[int]
    """
    lines = docstring.splitlines()
    violations: set[int] = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1] in LEGACY_SECTIONS:
            violations.add(index + 1)
        if (
            stripped in LEGACY_SECTIONS
            and index + 1 < len(lines)
            and re.fullmatch(r"\s*-{3,}\s*", lines[index + 1])
        ):
            violations.add(index + 1)
    return sorted(violations)


def inspect_docstring(docstring: str) -> list[tuple[int, str]]:
    """Return relative-line conformance violations for one docstring.

    :param docstring: Docstring text to inspect.
    :type docstring: str
    :return: Relative line numbers paired with diagnostic messages.
    :rtype: list[tuple[int, str]]
    """
    docstring = inspect.cleandoc(docstring)
    violations = [
        (line, "legacy Google/NumPy section; use Sphinx fields")
        for line in legacy_section_lines(docstring)
    ]
    for match in NONCANONICAL_FIELDS.finditer(docstring):
        line = docstring.count("\n", 0, match.start()) + 1
        violations.append((line, "noncanonical Sphinx field"))
    for match in PARAM_FIELD.finditer(docstring):
        name = (match.group(1) or "").strip()
        if PARAM_NAME.fullmatch(name):
            continue
        line = docstring.count("\n", 0, match.start()) + 1
        violations.append(
            (line, "use ':param name:' and a separate ':type name:' field")
        )
    for line_number, line in enumerate(docstring.splitlines(), start=1):
        if SPHINX_FIELD.match(line):
            violations.append((line_number, "Sphinx fields must not be indented"))
    return sorted(set(violations))


def check_files(paths: Iterable[Path]) -> list[str]:
    """Return all syntax and docstring-style violations.

    :param paths: Python files to inspect.
    :type paths: Iterable[Path]
    :return: Repository-relative diagnostics.
    :rtype: list[str]
    """
    violations: list[str] = []
    for path in paths:
        name = path.resolve().relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            violations.append(f"{name}: cannot parse source: {exc}")
            continue
        for node, docstring in docstrings(tree):
            for relative_line, message in inspect_docstring(docstring):
                line = node.body[0].lineno + relative_line - 1
                violations.append(f"{name}:{line}: {message}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args
                accepted = {
                    argument.arg
                    for argument in (
                        *arguments.posonlyargs,
                        *arguments.args,
                        *arguments.kwonlyargs,
                    )
                }
                if arguments.vararg is not None:
                    accepted.add(arguments.vararg.arg)
                if arguments.kwarg is not None:
                    accepted.add(arguments.kwarg.arg)
                clean = inspect.cleandoc(docstring)
                for match in DOCUMENTED_PARAM.finditer(clean):
                    documented = match.group(1).strip().lstrip("*")
                    if documented in accepted:
                        continue
                    relative_line = clean.count("\n", 0, match.start()) + 1
                    line = node.body[0].lineno + relative_line - 1
                    violations.append(
                        f"{name}:{line}: documented parameter {documented!r} "
                        f"is absent from {node.name}()"
                    )
    return violations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Optional explicit argument vector.
    :type argv: list[str] | None
    :return: Parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the repository docstring-style check.

    :param argv: Optional explicit argument vector.
    :type argv: list[str] | None
    :return: Process exit status.
    :rtype: int
    """
    args = parse_args(argv)
    paths = python_files(args.paths) if args.paths else repository_python_files()
    violations = check_files(paths)
    if violations:
        for violation in violations:
            print(f"ERROR {violation}", file=sys.stderr)
        return 1
    print(f"Docstring style check passed: {len(paths)} Python files use Sphinx fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
