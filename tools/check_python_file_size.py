#!/usr/bin/env python3
"""Enforce a non-docstring source-line limit for Python files."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Iterable

DEFAULT_PATHS = ("synkit", "Test", "tools")
DEFAULT_MAX_LINES = 1000
DEFAULT_BASELINE = Path("tools/python_file_size_baseline.json")
DOCSTRING_OWNERS = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


def docstring_lines(tree: ast.AST) -> set[int]:
    """Return physical line numbers occupied by recognized docstrings."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_OWNERS) or not node.body:
            continue
        first = node.body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def count_non_docstring_lines(path: Path) -> int:
    """Count physical source lines except recognized docstring lines."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    excluded = docstring_lines(tree)
    return sum(
        line_number not in excluded
        for line_number, _ in enumerate(source.splitlines(), start=1)
    )


def python_files(paths: Iterable[Path]) -> list[Path]:
    """Resolve files and directories to a deterministic Python-file list."""
    resolved: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            resolved.add(path)
        elif path.is_dir():
            resolved.update(item for item in path.rglob("*.py") if item.is_file())
    return sorted(resolved)


def load_baseline(path: Path) -> dict[str, int]:
    """Load grandfathered limits for existing oversized files."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    if not isinstance(files, dict) or not all(
        isinstance(name, str) and isinstance(limit, int)
        for name, limit in files.items()
    ):
        raise ValueError(f"Invalid file-size baseline: {path}")
    return files


def repository_name(path: Path, root: Path) -> str:
    """Return a stable POSIX path relative to the repository root."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def check_files(
    paths: Iterable[Path],
    *,
    root: Path,
    max_lines: int,
    baseline: dict[str, int],
) -> tuple[list[str], list[str]]:
    """Return policy violations and active baseline notices."""
    violations: list[str] = []
    notices: list[str] = []
    observed: dict[str, int] = {}

    for path in python_files(paths):
        name = repository_name(path, root)
        try:
            count = count_non_docstring_lines(path)
        except (OSError, SyntaxError) as exc:
            violations.append(f"{name}: cannot count source lines: {exc}")
            continue
        observed[name] = count
        if count <= max_lines:
            if name in baseline:
                violations.append(
                    f"{name}: now has {count} lines; remove its stale baseline entry"
                )
            continue

        legacy_limit = baseline.get(name)
        if legacy_limit is None:
            violations.append(
                f"{name}: {count} non-docstring lines exceeds limit {max_lines}"
            )
        elif count > legacy_limit:
            violations.append(
                f"{name}: {count} lines exceeds legacy baseline {legacy_limit} "
                f"(target {max_lines})"
            )
        else:
            notices.append(
                f"{name}: {count}/{legacy_limit} legacy lines (target {max_lines})"
            )

    for name in sorted(set(baseline) - set(observed)):
        violations.append(f"{name}: baseline entry is stale or outside checked paths")
    return violations, notices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path(path) for path in DEFAULT_PATHS],
    )
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_lines < 1:
        raise ValueError("--max-lines must be positive")
    root = Path(__file__).resolve().parents[1]
    paths = [path if path.is_absolute() else root / path for path in args.paths]
    baseline_path = (
        args.baseline if args.baseline.is_absolute() else root / args.baseline
    )
    violations, notices = check_files(
        paths,
        root=root,
        max_lines=args.max_lines,
        baseline=load_baseline(baseline_path),
    )
    for notice in notices:
        print(f"BASELINE {notice}")
    if violations:
        for violation in violations:
            print(f"ERROR {violation}", file=sys.stderr)
        return 1
    print(
        f"Python file-size check passed: maximum {args.max_lines} "
        "non-docstring lines."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
