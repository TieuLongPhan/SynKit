#!/usr/bin/env python3
"""Build a compact, manuscript-ready summary of the full benchmark matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import iter_jsonl, read_json, write_json  # noqa: E402


def _method_counts(report: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        str(method["method"]): dict(method["counts"]) for method in report["methods"]
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _direction_summary(
    rows: list[dict[str, Any]],
    direction: str,
) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    totals = Counter()
    for row in rows:
        record = row["directions"][direction]
        status = str(record.get("status", "MISSING"))
        statuses[status] += 1
        if status != "PASS":
            continue
        for key in (
            "mapping_count",
            "rewrite_count",
            "serialized_count",
            "unique_reaction_count",
        ):
            totals[key] += int(record[key])
    return {
        "statuses": dict(sorted(statuses.items())),
        "pass_solution_totals": dict(totals),
    }


def _matched_comparison(
    tuple_rows: list[dict[str, Any]],
    legacy_rows: list[dict[str, Any]],
    direction: str,
) -> dict[str, Any]:
    tuple_by_id = {int(row["record_id"]): row for row in tuple_rows}
    legacy_by_id = {int(row["record_id"]): row for row in legacy_rows}
    totals = {
        "tuple": Counter(),
        "typesGH": Counter(),
    }
    matched = 0
    for record_id in sorted(set(tuple_by_id) & set(legacy_by_id)):
        tuple_record = tuple_by_id[record_id]["directions"][direction]
        legacy_record = legacy_by_id[record_id]["directions"][direction]
        if (
            tuple_record.get("status") != "PASS"
            or legacy_record.get("status") != "PASS"
        ):
            continue
        matched += 1
        for key in (
            "mapping_count",
            "rewrite_count",
            "serialized_count",
            "unique_reaction_count",
        ):
            totals["tuple"][key] += int(tuple_record[key])
            totals["typesGH"][key] += int(legacy_record[key])

    deltas = {}
    for key, legacy_value in totals["typesGH"].items():
        tuple_value = totals["tuple"][key]
        deltas[key] = {
            "tuple_minus_typesGH": tuple_value - legacy_value,
            "tuple_reduction_percent": (
                100.0 * (legacy_value - tuple_value) / legacy_value
                if legacy_value
                else None
            ),
        }
    return {
        "matched_pass_cases": matched,
        "solution_totals": {name: dict(values) for name, values in totals.items()},
        "deltas": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = args.results.resolve()

    expansion_general = read_json(results / "expansion-general-all.json")
    expansion_radical = read_json(results / "expansion-radical-all.json")
    tuple_report = read_json(results / "replay-general-tuple-all.json")
    legacy_report = read_json(results / "replay-general-typesGH-all.json")
    radical_report = read_json(results / "replay-radical-tuple-all.json")
    tuple_rows = _load_cases(results / "replay-general-tuple-all-cases.jsonl.gz")
    legacy_rows = _load_cases(results / "replay-general-typesGH-all-cases.jsonl.gz")
    radical_rows = _load_cases(results / "replay-radical-tuple-all-cases.jsonl.gz")

    report = {
        "schema": "synkit.full-partial-aam-and-replay-summary/1",
        "representation_names": {
            "tuple": "electron-aware Lewis-labelled graph (tuple)",
            "typesGH": "legacy atom-bond graph (typesGH)",
            "paper_short_labels": {
                "tuple": "electron-aware LLG",
                "typesGH": "legacy atom-bond graph",
            },
        },
        "expansion": {
            "general": {
                "rows": expansion_general["selection"]["rows"],
                "independent_reference": True,
                "method_counts": _method_counts(expansion_general),
            },
            "radical": {
                "rows": expansion_radical["selection"]["rows"],
                "independent_reference": False,
                "method_counts": _method_counts(expansion_radical),
                "interpretation": (
                    "Acceptance requires source-map anchors because radical "
                    "fishhooks address those labels; it is completion validity, "
                    "not independent AAM accuracy."
                ),
            },
        },
        "replay": {
            "general": {
                "rows": len(tuple_rows),
                "tuple": {
                    "wall_seconds": tuple_report["timing_seconds"]["wall"],
                    "failure_count": tuple_report["failure_count"],
                    "forward": _direction_summary(tuple_rows, "forward"),
                    "inverse": _direction_summary(tuple_rows, "inverse"),
                },
                "typesGH": {
                    "wall_seconds": legacy_report["timing_seconds"]["wall"],
                    "failure_count": legacy_report["failure_count"],
                    "forward": _direction_summary(legacy_rows, "forward"),
                    "inverse": _direction_summary(legacy_rows, "inverse"),
                },
                "matched_pass_comparison": {
                    direction: _matched_comparison(
                        tuple_rows,
                        legacy_rows,
                        direction,
                    )
                    for direction in ("forward", "inverse")
                },
            },
            "radical": {
                "rows": len(radical_rows),
                "representation": "tuple",
                "wall_seconds": radical_report["timing_seconds"]["wall"],
                "failure_count": radical_report["failure_count"],
                "forward": _direction_summary(radical_rows, "forward"),
                "inverse": _direction_summary(radical_rows, "inverse"),
            },
        },
    }
    output = args.output or results / "full-benchmark-summary.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
