#!/usr/bin/env python3
"""Partition per-batch sparse row sets into deterministic disjoint shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--shards", type=int, default=2)
    return parser.parse_args()


def _read_rows(path: Path) -> list[int]:
    rows = [int(line) for line in path.read_text().splitlines() if line]
    if rows != sorted(set(rows)):
        raise ValueError(f"Rows must be sorted and unique: {path}")
    return rows


def _partition(rows: list[int], count: int) -> list[list[int]]:
    base, extra = divmod(len(rows), count)
    shards = []
    start = 0
    for index in range(count):
        stop = start + base + (index < extra)
        shards.append(rows[start:stop])
        start = stop
    return shards


def shard_row_sets(
    rows_dir: Path,
    output_dir: Path,
    shard_count: int,
) -> dict[str, Any]:
    """Write shards and prove their union equals each source row set."""
    if shard_count < 1:
        raise ValueError("Shard count must be positive")
    rows_dir = rows_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for batch in range(1, 11):
        source = rows_dir / f"batch-{batch:02d}.rows"
        rows = _read_rows(source)
        shard_reports = []
        observed = []
        for shard, shard_rows in enumerate(
            _partition(rows, shard_count),
            start=1,
        ):
            path = output_dir / f"batch-{batch:02d}-shard-{shard:02d}.rows"
            path.write_text(
                "".join(f"{row}\n" for row in shard_rows),
                encoding="utf-8",
            )
            observed.extend(shard_rows)
            shard_reports.append(
                {
                    "shard": shard,
                    "path": str(path),
                    "rows": len(shard_rows),
                    "first_row": (shard_rows[0] if shard_rows else None),
                    "last_row": (shard_rows[-1] if shard_rows else None),
                }
            )
        if observed != rows:
            raise AssertionError(f"Shard union mismatch in batch {batch:02d}")
        reports.append(
            {
                "batch": batch,
                "source": str(source),
                "rows": len(rows),
                "shards": shard_reports,
            }
        )

    manifest = {
        "schema": "synkit.flower-sparse-row-shards/1",
        "partition": "source rows = ordered disjoint-union of shards",
        "source_rows": sum(report["rows"] for report in reports),
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
    manifest = shard_row_sets(
        args.rows_dir,
        args.output_dir,
        args.shards,
    )
    print(
        f"Partitioned {manifest['source_rows']} rows into "
        f"{10 * manifest['shard_count_per_batch']} shards."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
