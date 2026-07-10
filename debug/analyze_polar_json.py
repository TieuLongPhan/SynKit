"""
Analyze generated polar mechanistic JSON.

Reports:
  - number of rows
  - distribution of number_arrow / len(epd)
  - action-label counts in epd and epd_lw
  - common action sequences
  - original and compact arrow-code distributions

Usage:
  python debug/analyze_polar_json.py
  python debug/analyze_polar_json.py debug/data/mech/polar.json --top 20
  python debug/analyze_polar_json.py --json-out debug/data/mech/polar_stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _pct(count: int, total: int) -> float:
    return round((count / total) * 100, 4) if total else 0.0


def _transition_labels(epd: list[list[Any]]) -> list[str]:
    return [step[0] for step in epd]


def _transition_sequence(epd: list[list[Any]]) -> str:
    return " ; ".join(_transition_labels(epd))


def _counter_table(
    counter: Counter, total: int, top: int | None = None
) -> list[dict[str, Any]]:
    items = counter.most_common(top)
    return [
        {
            "value": value,
            "count": count,
            "percent": _pct(count, total),
        }
        for value, count in items
    ]


def analyze(path: Path, top: int) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected top-level JSON list in {path}")

    number_arrow = Counter()
    epd_actions = Counter()
    epd_lw_actions = Counter()
    epd_sequences = Counter()
    epd_lw_sequences = Counter()
    arrow_codes = Counter()
    original_arrow_codes = Counter()
    orbital_classes = Counter()

    bad_number_arrow = []

    for row in rows:
        epd = row["epd"]
        epd_lw = row["epd_lw"]
        n_arrow = row.get("number_arrow")

        if n_arrow != len(epd):
            bad_number_arrow.append(row.get("index"))

        number_arrow[len(epd)] += 1
        epd_actions.update(_transition_labels(epd))
        epd_lw_actions.update(_transition_labels(epd_lw))
        epd_sequences[_transition_sequence(epd)] += 1
        epd_lw_sequences[_transition_sequence(epd_lw)] += 1
        arrow_codes[row["arrow_code"]] += 1
        original_arrow_codes[row["original_arrow_code"]] += 1
        orbital_classes[row.get("orbital_class")] += 1

    total_rows = len(rows)
    total_steps = sum(
        number_arrow_count * n for n, number_arrow_count in number_arrow.items()
    )

    return {
        "input": str(path),
        "total_rows": total_rows,
        "total_arrow_steps": total_steps,
        "unique_epd_actions": len(epd_actions),
        "unique_epd_lw_actions": len(epd_lw_actions),
        "unique_epd_sequences": len(epd_sequences),
        "unique_epd_lw_sequences": len(epd_lw_sequences),
        "unique_arrow_codes": len(arrow_codes),
        "unique_original_arrow_codes": len(original_arrow_codes),
        "bad_number_arrow_indices": bad_number_arrow[:50],
        "number_arrow_distribution": _counter_table(number_arrow, total_rows),
        "epd_action_counts": _counter_table(epd_actions, total_steps),
        "epd_lw_action_counts": _counter_table(epd_lw_actions, total_steps),
        "top_epd_sequences": _counter_table(epd_sequences, total_rows, top),
        "top_epd_lw_sequences": _counter_table(epd_lw_sequences, total_rows, top),
        "top_arrow_codes": _counter_table(arrow_codes, total_rows, top),
        "top_original_arrow_codes": _counter_table(
            original_arrow_codes, total_rows, top
        ),
        "orbital_class_counts": _counter_table(orbital_classes, total_rows),
    }


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows:
        print(f"{row['count']:>8}  {row['percent']:>8.4f}%  {row['value']}")


def print_summary(summary: dict[str, Any]) -> None:
    print(f"input: {summary['input']}")
    print(f"total_rows: {summary['total_rows']}")
    print(f"total_arrow_steps: {summary['total_arrow_steps']}")
    print(f"unique_epd_actions: {summary['unique_epd_actions']}")
    print(f"unique_epd_lw_actions: {summary['unique_epd_lw_actions']}")
    print(f"unique_epd_sequences: {summary['unique_epd_sequences']}")
    print(f"unique_epd_lw_sequences: {summary['unique_epd_lw_sequences']}")
    print(f"unique_arrow_codes: {summary['unique_arrow_codes']}")
    print(f"unique_original_arrow_codes: {summary['unique_original_arrow_codes']}")
    print(f"bad_number_arrow_indices: {summary['bad_number_arrow_indices']}")

    _print_table("number_arrow distribution", summary["number_arrow_distribution"])
    _print_table("epd action counts", summary["epd_action_counts"])
    _print_table("epd_lw action counts", summary["epd_lw_action_counts"])
    _print_table("top epd sequences", summary["top_epd_sequences"])
    _print_table("top epd_lw sequences", summary["top_epd_lw_sequences"])
    _print_table("top original arrow codes", summary["top_original_arrow_codes"])
    _print_table("orbital class counts", summary["orbital_class_counts"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_file",
        nargs="?",
        default="debug/data/mech/polar.json",
        help="Generated polar JSON file.",
    )
    parser.add_argument(
        "--top", type=int, default=15, help="Number of top sequences/codes to show."
    )
    parser.add_argument(
        "--json-out", default=None, help="Optional path to write stats JSON."
    )
    args = parser.parse_args()

    summary = analyze(Path(args.json_file), top=args.top)
    print_summary(summary)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
