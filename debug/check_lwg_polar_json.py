"""
Run LWGEditor over generated polar.json.

This is a debugging script for the sigma/pi-only Lewis graph editor. It checks
whether each typed mechanism can be applied and whether the final graph matches
the product by structure, charge, and exported SMILES.

Usage:
  python debug/check_lwg_polar_json.py
  python debug/check_lwg_polar_json.py --limit 100
  python debug/check_lwg_polar_json.py --rows-out debug/data/mech/lwg_polar_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synkit.Graph.Mech.lwg_editor import (
    LWGEditor,
    _mapped_charge_signature,
    _mapped_edge_signature,
)

DEFAULT_JSON = Path("debug/data/mech/polar.json")
DEFAULT_OUT = Path("debug/data/mech/lwg_polar_check.json")


def _pct(count: int, total: int) -> float:
    return round(count / total * 100.0, 4) if total else 0.0


def _sequence(epd_lw: list[list[Any]]) -> str:
    return " ; ".join(str(step[0]) for step in epd_lw)


def _counter_table(
    counter: Counter, total: int, top: int | None = None
) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count, "percent": _pct(count, total)}
        for value, count in counter.most_common(top)
    ]


def _diff_dict(left: dict[Any, Any], right: dict[Any, Any]) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    missing_left = [key for key in keys if key not in left]
    missing_right = [key for key in keys if key not in right]
    changed = {
        str(key): {"final": left[key], "product": right[key]}
        for key in keys
        if key in left and key in right and left[key] != right[key]
    }
    return {
        "missing_final": missing_left,
        "missing_product": missing_right,
        "changed": changed,
    }


def _failure_kind(row_result: dict[str, Any]) -> str:
    if not row_result["ok"]:
        return "error"
    if not row_result["charge_match"]:
        return "charge"
    if not row_result["smiles_match"]:
        return "smiles"
    if not row_result["structural_match"]:
        return "structural_diagnostic"
    return "none"


def _add_example(
    examples: dict[str, list[dict[str, Any]]],
    kind: str,
    row: dict[str, Any],
    row_result: dict[str, Any],
    max_examples: int,
    extra: dict[str, Any] | None = None,
) -> None:
    if kind == "none" or len(examples[kind]) >= max_examples:
        return

    example = {
        "index": row.get("index"),
        "number_arrow": row.get("number_arrow"),
        "orbital_class": row.get("orbital_class"),
        "arrow_code": row.get("arrow_code"),
        "original_arrow_code": row.get("original_arrow_code"),
        "sequence": _sequence(row.get("epd_lw", [])),
        "ok": row_result["ok"],
        "error_type": row_result.get("error_type"),
        "error": row_result.get("error"),
        "structural_match": row_result.get("structural_match"),
        "charge_match": row_result.get("charge_match"),
        "smiles_match": row_result.get("smiles_match"),
        "matches_product": row_result.get("matches_product"),
        "final_smiles": row_result.get("final_smiles"),
        "product_smiles": row_result.get("product_smiles"),
    }
    if extra:
        example.update(extra)
    examples[kind].append(example)


def check_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    top: int,
    max_examples: int,
    progress_every: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    editor = LWGEditor()
    selected = rows[:limit] if limit is not None else rows

    totals = Counter()
    by_number_arrow: dict[int, Counter] = defaultdict(Counter)
    by_orbital_class: dict[str, Counter] = defaultdict(Counter)
    by_sequence: dict[str, Counter] = defaultdict(Counter)
    errors = Counter()
    failure_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_results: list[dict[str, Any]] = []

    started = time.time()
    for position, row in enumerate(selected, start=1):
        sequence = _sequence(row["epd_lw"])
        number_arrow = int(row.get("number_arrow", len(row["epd_lw"])))
        orbital_class = str(row.get("orbital_class"))

        row_result: dict[str, Any] = {
            "position": position - 1,
            "index": row.get("index"),
            "number_arrow": number_arrow,
            "orbital_class": orbital_class,
            "sequence": sequence,
            "ok": False,
            "error_type": None,
            "error": None,
            "structural_match": False,
            "charge_match": False,
            "smiles_match": False,
            "matches_product": False,
            "final_smiles": None,
            "product_smiles": None,
        }

        extra: dict[str, Any] = {}
        try:
            result = editor.apply(row["rsmi"], row["epd_lw"])
            row_result.update(
                {
                    "ok": True,
                    "structural_match": result.structural_match,
                    "charge_match": result.charge_match,
                    "smiles_match": result.smiles_match,
                    "matches_product": result.matches_product,
                    "final_smiles": result.final_smiles,
                    "product_smiles": result.product_smiles,
                }
            )

            if not result.structural_match:
                extra["edge_diff"] = _diff_dict(
                    _mapped_edge_signature(result.final_graph),
                    _mapped_edge_signature(result.product_graph),
                )
            if not result.charge_match:
                extra["charge_diff"] = _diff_dict(
                    _mapped_charge_signature(result.final_graph),
                    _mapped_charge_signature(result.product_graph),
                )
        except Exception as exc:
            row_result["error_type"] = type(exc).__name__
            row_result["error"] = str(exc)
            errors[type(exc).__name__] += 1

        totals["rows"] += 1
        totals["ok"] += int(row_result["ok"])
        totals["error"] += int(not row_result["ok"])
        totals["structural_match"] += int(row_result["structural_match"])
        totals["charge_match"] += int(row_result["charge_match"])
        totals["smiles_match"] += int(row_result["smiles_match"])
        totals["matches_product"] += int(row_result["matches_product"])
        totals["final_smiles_none"] += int(row_result["final_smiles"] is None)
        totals["product_smiles_none"] += int(row_result["product_smiles"] is None)

        kind = _failure_kind(row_result)
        by_number_arrow[number_arrow][kind] += 1
        by_orbital_class[orbital_class][kind] += 1
        by_sequence[sequence][kind] += 1
        _add_example(failure_examples, kind, row, row_result, max_examples, extra)

        row_results.append(row_result)

        if progress_every and position % progress_every == 0:
            elapsed = time.time() - started
            rate = position / elapsed if elapsed else 0.0
            print(
                f"checked {position:,}/{len(selected):,} rows "
                f"({rate:.1f} rows/s, matches={totals['matches_product']:,})",
                flush=True,
            )

    total_rows = totals["rows"]
    summary = {
        "total_rows": total_rows,
        "elapsed_seconds": round(time.time() - started, 3),
        "limit": limit,
        "counts": dict(totals),
        "rates": {
            "ok": _pct(totals["ok"], total_rows),
            "error": _pct(totals["error"], total_rows),
            "structural_match": _pct(totals["structural_match"], total_rows),
            "charge_match": _pct(totals["charge_match"], total_rows),
            "smiles_match": _pct(totals["smiles_match"], total_rows),
            "matches_product": _pct(totals["matches_product"], total_rows),
            "final_smiles_none": _pct(totals["final_smiles_none"], total_rows),
            "product_smiles_none": _pct(totals["product_smiles_none"], total_rows),
        },
        "errors": _counter_table(errors, totals["error"]),
        "by_number_arrow": {
            str(key): dict(value) for key, value in sorted(by_number_arrow.items())
        },
        "by_orbital_class": {
            key: dict(value) for key, value in sorted(by_orbital_class.items())
        },
        "top_sequences": [
            {"sequence": sequence, **dict(counter), "total": sum(counter.values())}
            for sequence, counter in sorted(
                by_sequence.items(),
                key=lambda item: sum(item[1].values()),
                reverse=True,
            )[:top]
        ],
        "failure_examples": dict(failure_examples),
    }
    return summary, row_results


def _write_row_csv(path: Path, row_results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "position",
        "index",
        "number_arrow",
        "orbital_class",
        "sequence",
        "ok",
        "error_type",
        "error",
        "structural_match",
        "charge_match",
        "smiles_match",
        "matches_product",
        "final_smiles",
        "product_smiles",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in fields} for row in row_results
        )


def print_summary(summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    rates = summary["rates"]
    print(f"total_rows: {summary['total_rows']:,}")
    print(f"elapsed_seconds: {summary['elapsed_seconds']}")
    print(f"ok: {counts['ok']:,} ({rates['ok']}%)")
    print(f"errors: {counts['error']:,} ({rates['error']}%)")
    print(
        "matches: "
        f"structural={counts['structural_match']:,} ({rates['structural_match']}%), "
        f"charge={counts['charge_match']:,} ({rates['charge_match']}%), "
        f"smiles={counts['smiles_match']:,} ({rates['smiles_match']}%), "
        f"all={counts['matches_product']:,} ({rates['matches_product']}%)"
    )
    print(
        "smiles_none: "
        f"final={counts['final_smiles_none']:,} ({rates['final_smiles_none']}%), "
        f"product={counts['product_smiles_none']:,} ({rates['product_smiles_none']}%)"
    )

    if summary["errors"]:
        print("\nerrors:")
        for row in summary["errors"]:
            print(f"  {row['count']:>8}  {row['percent']:>8.4f}%  {row['value']}")

    print("\nby_number_arrow:")
    for number_arrow, counter in summary["by_number_arrow"].items():
        print(f"  {number_arrow}: {counter}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", nargs="?", default=str(DEFAULT_JSON))
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), help="Summary JSON output path."
    )
    parser.add_argument(
        "--rows-out", default=None, help="Optional row-level CSV output path."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional row limit for smoke checks."
    )
    parser.add_argument(
        "--top", type=int, default=25, help="Number of top sequences to keep."
    )
    parser.add_argument(
        "--examples", type=int, default=20, help="Failure examples per kind."
    )
    parser.add_argument(
        "--progress-every", type=int, default=5000, help="Progress print interval."
    )
    args = parser.parse_args()

    json_path = Path(args.json_file)
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected top-level JSON list in {json_path}")

    summary, row_results = check_rows(
        rows,
        limit=args.limit,
        top=args.top,
        max_examples=args.examples,
        progress_every=args.progress_every,
    )
    summary["input"] = str(json_path)
    print_summary(summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote summary: {out}")

    if args.rows_out:
        rows_out = Path(args.rows_out)
        _write_row_csv(rows_out, row_results)
        print(f"wrote rows: {rows_out}")


if __name__ == "__main__":
    main()
