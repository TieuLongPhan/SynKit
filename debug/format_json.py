"""
Merge and/or reformat generated mechanistic JSON files.

The formatter renders each transition arrow (a 3-element list
[kind, src, dst]) on a single compact line, while the rest of the structure
keeps standard 2-space indentation.

Before:
  "expected": [
    [
      "LP-/B+",
      [1],
      [1, 2]
    ],
    ...
  ]

After:
  "expected": [
    ["LP-/B+", [1], [1, 2]],
    ["B-/LP+", [3, 4], [3]]
  ]

Usage:
  python debug/format_json.py
  python debug/format_json.py debug/data/mech
  python debug/format_json.py debug/data/mech/polar.json
  python debug/format_json.py --merge
  python debug/format_json.py --merge --clean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _is_transition(obj) -> bool:
    return (
        isinstance(obj, list)
        and len(obj) == 3
        and isinstance(obj[0], str)
        and isinstance(obj[1], list)
        and isinstance(obj[2], list)
    )


def _is_transition_list(obj) -> bool:
    return (
        isinstance(obj, list) and len(obj) > 0 and all(_is_transition(e) for e in obj)
    )


def fmt(obj, indent: int = 2, level: int = 0) -> str:
    pad = " " * (indent * level)
    pad1 = " " * (indent * (level + 1))

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        parts = []
        for k, v in obj.items():
            parts.append(f"{pad1}{json.dumps(k)}: {fmt(v, indent, level + 1)}")
        return "{\n" + ",\n".join(parts) + "\n" + pad + "}"

    if _is_transition(obj):
        return json.dumps(obj, ensure_ascii=False)

    if _is_transition_list(obj):
        rows = [f"{pad1}{json.dumps(e, ensure_ascii=False)}" for e in obj]
        return "[\n" + ",\n".join(rows) + "\n" + pad + "]"

    if isinstance(obj, list):
        if not obj:
            return "[]"
        parts = [f"{pad1}{fmt(v, indent, level + 1)}" for v in obj]
        return "[\n" + ",\n".join(parts) + "\n" + pad + "]"

    return json.dumps(obj, ensure_ascii=False)


def reformat_file(path: Path) -> bool:
    """Return True if the file was changed."""
    original = path.read_text(encoding="utf-8")
    try:
        data = json.loads(original)
    except json.JSONDecodeError as e:
        print(f"  SKIP (invalid JSON): {path} — {e}")
        return False
    formatted = fmt(data) + "\n"
    if formatted == original:
        return False
    path.write_text(formatted, encoding="utf-8")
    return True


def _collect_files(target: str | None) -> list[Path]:
    if target is None:
        target = "debug/data/mech"

    p = Path(target)
    if p.is_file():
        return [p]
    if p.is_dir():
        return [f for f in p.rglob("*.json") if f.name != "index.json"]
    # treat as glob pattern rooted at cwd
    return sorted(Path(".").glob(target))


def _step_file_sort_key(path: Path) -> tuple[int, str]:
    parts = path.stem.split("_")
    try:
        step_count = int(parts[1])
    except (IndexError, ValueError):
        step_count = 10**9
    return step_count, path.name


def _collect_step_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("polar_*_step*.json"), key=_step_file_sort_key)


def merge_step_files(input_dir: Path, output_file: Path, clean: bool = False) -> int:
    """Merge polar_N_step(s).json files and add number_arrow to every row."""
    files = _collect_step_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No polar_*_step*.json files found in {input_dir}")

    merged = []
    for path in files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list in {path}")

        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Expected object rows in {path}")
            if "epd" not in row:
                raise ValueError(f"Missing 'epd' in row from {path}")

            row = dict(row)
            row["number_arrow"] = len(row["epd"])
            merged.append(row)

    merged.sort(key=lambda row: row["index"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(fmt(merged) + "\n", encoding="utf-8")

    if clean:
        for path in files:
            if path.resolve() != output_file.resolve():
                path.unlink()

    return len(merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="JSON file, directory, or glob to format. Defaults to debug/data/mech.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge polar_N_step(s).json files into one JSON array.",
    )
    parser.add_argument(
        "--input-dir",
        default="debug/data/mech",
        help="Directory containing polar_N_step(s).json files for --merge.",
    )
    parser.add_argument(
        "--output",
        default="debug/data/mech/polar.json",
        help="Output file for --merge.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="With --merge, remove the source polar_N_step(s).json files after writing output.",
    )
    args = parser.parse_args()

    if args.merge:
        count = merge_step_files(
            Path(args.input_dir), Path(args.output), clean=args.clean
        )
        print(f"merged {count} row(s) → {args.output}")
        return

    files = _collect_files(args.target)

    changed = 0
    for f in sorted(files):
        if reformat_file(f):
            print(f"  reformatted: {f}")
            changed += 1

    print(f"\n{changed} file(s) reformatted, {len(files) - changed} unchanged.")


if __name__ == "__main__":
    main()
