#!/usr/bin/env python3
"""Freeze original FLOWER row selections containing historical replay FAILs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = HERE / "replay-runs"
DEFAULT_BATCHES = HERE / "full-reaction-batches"
DEFAULT_OUTPUT = HERE / "f2-failure-rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _has_failure(case: dict[str, Any]) -> bool:
    directions = case.get("directions", {})
    return any(
        isinstance(result, dict) and result.get("status") == "FAIL"
        for result in directions.values()
    )


def prepare_failure_rows(
    runs_dir: Path,
    batches_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write deterministic sparse-row files and their provenance manifest."""
    runs_dir = runs_dir.resolve()
    batches_dir = batches_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for index in range(1, 11):
        name = f"full-batch-{index:02d}"
        bugs = runs_dir / name / "bugs.jsonl"
        batch = batches_dir / f"batch-{index:02d}-of-10.txt.gz"
        if not bugs.is_file() or not batch.is_file():
            raise FileNotFoundError(
                f"Missing historical evidence for batch {index:02d}"
            )
        selected = set()
        with bugs.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at {bugs}:{line_number}") from exc
                if _has_failure(case):
                    selected.add(int(case["batch_row"]))
        if not selected:
            raise ValueError(f"No FAIL rows selected for batch {index:02d}")

        rows_path = output_dir / f"batch-{index:02d}.rows"
        rows_path.write_text(
            "".join(f"{row}\n" for row in sorted(selected)),
            encoding="utf-8",
        )
        reports.append(
            {
                "batch": index,
                "dataset": str(batch),
                "bugs": str(bugs),
                "rows_file": str(rows_path),
                "selected_rows": len(selected),
                "first_row": min(selected),
                "last_row": max(selected),
            }
        )

    manifest = {
        "schema": "synkit.flower-f2-failure-rows/1",
        "selection": ("historical row has at least one direction with status FAIL"),
        "total_rows": sum(report["selected_rows"] for report in reports),
        "batches": reports,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = prepare_failure_rows(
        args.runs_dir,
        args.batches_dir,
        args.output_dir,
    )
    print(
        f"Wrote {manifest['total_rows']} historical FAIL row selections "
        f"across {len(manifest['batches'])} batches."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
