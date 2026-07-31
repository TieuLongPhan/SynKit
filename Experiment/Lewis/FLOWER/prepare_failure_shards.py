#!/usr/bin/env python3
"""Shard unprocessed F2 failure rows after a durable sparse-run prefix."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_ROWS = HERE / "f2-failure-rows"
DEFAULT_RUNS = HERE / "f2-runs"
DEFAULT_OUTPUT = HERE / "f2-failure-shards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-dir", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shards", type=int, default=2)
    return parser.parse_args()


def _read_rows(path: Path) -> list[int]:
    rows = [int(line) for line in path.read_text().splitlines() if line]
    if rows != sorted(set(rows)):
        raise ValueError(f"Rows must be sorted and unique: {path}")
    return rows


def _read_durable_rows(path: Path) -> tuple[list[int], bool]:
    rows = []
    complete = True
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    rows.append(int(json.loads(line)["batch_row"]))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"Malformed case at {path}:{line_number}") from exc
    except EOFError:
        complete = False
    return rows, complete


def _partition(rows: list[int], count: int) -> list[list[int]]:
    base, extra = divmod(len(rows), count)
    shards = []
    start = 0
    for index in range(count):
        stop = start + base + (index < extra)
        shards.append(rows[start:stop])
        start = stop
    return shards


def prepare_failure_shards(
    rows_dir: Path,
    runs_dir: Path,
    output_dir: Path,
    shard_count: int,
) -> dict[str, Any]:
    """Write disjoint remaining-row shards with prefix provenance."""
    if shard_count < 1:
        raise ValueError("Shard count must be positive")
    rows_dir = rows_dir.resolve()
    runs_dir = runs_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for batch in range(1, 11):
        selected_path = rows_dir / f"batch-{batch:02d}.rows"
        cases_path = runs_dir / f"failure-batch-{batch:02d}" / "cases.jsonl.gz"
        selected = _read_rows(selected_path)
        durable, gzip_complete = _read_durable_rows(cases_path)
        if durable != selected[: len(durable)]:
            raise ValueError(
                f"Durable rows are not a selected prefix in batch {batch:02d}"
            )
        remaining = selected[len(durable) :]
        shard_reports = []
        for shard, rows in enumerate(
            _partition(remaining, shard_count),
            start=1,
        ):
            path = output_dir / f"batch-{batch:02d}-shard-{shard:02d}.rows"
            path.write_text(
                "".join(f"{row}\n" for row in rows),
                encoding="utf-8",
            )
            shard_reports.append(
                {
                    "shard": shard,
                    "rows_file": str(path),
                    "rows": len(rows),
                    "first_row": rows[0] if rows else None,
                    "last_row": rows[-1] if rows else None,
                }
            )
        reports.append(
            {
                "batch": batch,
                "selected_rows": len(selected),
                "durable_prefix_rows": len(durable),
                "prefix_gzip_complete": gzip_complete,
                "remaining_rows": len(remaining),
                "shards": shard_reports,
            }
        )

    manifest = {
        "schema": "synkit.flower-f2-failure-shards/1",
        "partition": (
            "selected rows = durable ordered prefix disjoint-union "
            "contiguous remaining shards"
        ),
        "selected_rows": sum(item["selected_rows"] for item in reports),
        "durable_prefix_rows": sum(item["durable_prefix_rows"] for item in reports),
        "remaining_rows": sum(item["remaining_rows"] for item in reports),
        "shard_count_per_batch": shard_count,
        "batches": reports,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = prepare_failure_shards(
        args.rows_dir,
        args.runs_dir,
        args.output_dir,
        args.shards,
    )
    print(
        f"Preserved {manifest['durable_prefix_rows']} durable rows; "
        f"wrote {manifest['remaining_rows']} rows into "
        f"{10 * manifest['shard_count_per_batch']} shards."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
