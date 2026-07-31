#!/usr/bin/env python3
"""Derive exact FLOWER replay suffixes from durable historical case rows."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = HERE / "replay-runs"
DEFAULT_BATCHES = HERE / "full-reaction-batches"
DEFAULT_OUTPUT = HERE / "f2-resume"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _count_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _durable_prefix(path: Path) -> tuple[int, bool]:
    """Return the maximal verified contiguous prefix and gzip completeness."""
    expected = 1
    complete = True
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
                row = int(case["batch_row"])
                if row != expected:
                    raise ValueError(
                        f"Non-contiguous row at {path}:{line_number}: "
                        f"expected {expected}, found {row}"
                    )
                expected += 1
    except EOFError:
        complete = False
    return expected - 1, complete


def prepare_resume(
    runs_dir: Path,
    batches_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write an exact, reproducible suffix-resume manifest."""
    runs_dir = runs_dir.resolve()
    batches_dir = batches_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for index in range(1, 11):
        batch = batches_dir / f"batch-{index:02d}-of-10.txt.gz"
        cases = runs_dir / f"full-batch-{index:02d}" / "cases.jsonl.gz"
        if not batch.is_file() or not cases.is_file():
            raise FileNotFoundError(f"Missing resume evidence for batch {index:02d}")
        total = _count_rows(batch)
        completed, gzip_complete = _durable_prefix(cases)
        if completed > total:
            raise ValueError(f"Historical batch {index:02d} exceeds its source length")
        remaining = total - completed
        reports.append(
            {
                "batch": index,
                "dataset": str(batch),
                "historical_cases": str(cases),
                "source_rows": total,
                "durable_prefix_rows": completed,
                "historical_gzip_complete": gzip_complete,
                "resume_offset": completed,
                "first_resume_row": completed + 1 if remaining else None,
                "last_resume_row": total if remaining else None,
                "remaining_rows": remaining,
            }
        )

    total_rows = sum(report["source_rows"] for report in reports)
    durable_rows = sum(report["durable_prefix_rows"] for report in reports)
    manifest = {
        "schema": "synkit.flower-f2-resume/1",
        "completion_definition": (
            "durable_prefix_rows / source_rows; a durable prefix is the "
            "maximal contiguous sequence of parsed case records beginning "
            "at batch_row 1"
        ),
        "source_rows": total_rows,
        "durable_prefix_rows": durable_rows,
        "remaining_rows": total_rows - durable_rows,
        "completion_fraction": durable_rows / total_rows,
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
    manifest = prepare_resume(
        args.runs_dir,
        args.batches_dir,
        args.output_dir,
    )
    print(
        f"Verified {manifest['durable_prefix_rows']} / "
        f"{manifest['source_rows']} durable rows "
        f"({manifest['completion_fraction']:.6%}); "
        f"{manifest['remaining_rows']} rows remain."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
