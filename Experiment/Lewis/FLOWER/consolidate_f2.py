#!/usr/bin/env python3
"""Consolidate the durable F2 prefix and shards into exact residual sets."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
DEFAULT_SELECTION = HERE / "f2-failure-rows"
DEFAULT_RUNS = HERE / "f2-runs"
DEFAULT_OUTPUT = HERE / "f2-consolidated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=DEFAULT_SELECTION,
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_selection(path: Path) -> list[int]:
    rows = [int(line) for line in path.read_text().splitlines() if line]
    if rows != sorted(set(rows)):
        raise ValueError(f"Rows must be sorted and unique: {path}")
    return rows


def _iter_cases(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
    except EOFError:
        return


def _write_rows(path: Path, rows: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{row}\n" for row in rows),
        encoding="utf-8",
    )


def consolidate_f2(
    selection_dir: Path,
    runs_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify exact coverage and write FAIL and timeout-only row sets."""
    selection_dir = selection_dir.resolve()
    runs_dir = runs_dir.resolve()
    output_dir = output_dir.resolve()
    direction_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    reports = []

    for batch in range(1, 11):
        selection = _read_selection(selection_dir / f"batch-{batch:02d}.rows")
        case_paths = [
            runs_dir
            / f"failure-batch-{batch:02d}-definitive-prefix"
            / "cases.jsonl.gz",
            *[
                runs_dir
                / f"failure-batch-{batch:02d}-shard-{shard:02d}"
                / "cases.jsonl.gz"
                for shard in (1, 2)
            ],
        ]
        cases = [case for path in case_paths for case in _iter_cases(path)]
        observed = [int(case["batch_row"]) for case in cases]
        if sorted(observed) != selection or len(observed) != len(set(observed)):
            raise ValueError(f"Coverage mismatch in batch {batch:02d}")

        fail_rows = []
        timeout_rows = []
        local_rows: Counter[str] = Counter()
        for case in cases:
            statuses = {
                direction: result["status"]
                for direction, result in case["directions"].items()
            }
            for direction, status in statuses.items():
                direction_counts[f"{direction}:{status.lower()}"] += 1
            if "FAIL" in statuses.values():
                category = "fail"
                fail_rows.append(int(case["batch_row"]))
            elif "ERROR" in statuses.values():
                category = "timeout_only"
                timeout_rows.append(int(case["batch_row"]))
            else:
                category = "pass"
            row_counts[category] += 1
            local_rows[category] += 1

        fail_path = output_dir / "residual-fail" / (f"batch-{batch:02d}.rows")
        timeout_path = output_dir / "timeout-only" / (f"batch-{batch:02d}.rows")
        _write_rows(fail_path, sorted(fail_rows))
        _write_rows(timeout_path, sorted(timeout_rows))
        reports.append(
            {
                "batch": batch,
                "selected_rows": len(selection),
                "row_counts": dict(sorted(local_rows.items())),
                "residual_fail_rows": str(fail_path),
                "timeout_only_rows": str(timeout_path),
            }
        )

    manifest = {
        "schema": "synkit.flower-f2-consolidated/1",
        "classification": {
            "fail": "at least one direction has status FAIL",
            "timeout_only": (
                "no direction has status FAIL and at least one direction "
                "has status ERROR"
            ),
            "pass": "all attempted directions have status PASS",
        },
        "selected_rows": sum(row_counts.values()),
        "row_counts": dict(sorted(row_counts.items())),
        "direction_counts": dict(sorted(direction_counts.items())),
        "batches": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = consolidate_f2(
        args.selection_dir,
        args.runs_dir,
        args.output_dir,
    )
    counts = manifest["row_counts"]
    print(
        f"Consolidated {manifest['selected_rows']} rows: "
        f"{counts.get('pass', 0)} pass, "
        f"{counts.get('timeout_only', 0)} timeout-only, "
        f"{counts.get('fail', 0)} residual fail."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
