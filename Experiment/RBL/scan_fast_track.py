#!/usr/bin/env python3
"""Run fusion-free RBL fast track over every paired USPTO-50K record."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import io
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.RBL.benchmark import (  # noqa: E402
    DEFAULT_DATA,
    RULE_EXTRACTION_DESCRIPTION,
    _evaluate_unbounded,
    load_records,
)

DEFAULT_OUTPUT = HERE / "fast_track_all.json.gz"
DEFAULT_SUMMARY = HERE / "fast_track_all_summary.json"


def _quiet_worker() -> None:
    logging.disable(logging.CRITICAL)


def _scan_one(record: dict[str, str]) -> dict[str, Any]:
    result = _evaluate_unbounded(record, "fast_track")
    return {
        "R_id": record["R_id"],
        "status": result["status"],
        "seconds": result["seconds"],
        "n_candidates": result.get("n_candidates", 0),
        "candidates": result.get("candidates", []),
        "stop_mode": result.get("stop_mode"),
        "stop_reason": result.get("stop_reason"),
        "error": result.get("error"),
        "normalization_violations": result.get("rule_audit", {}).get(
            "normalization_violations"
        ),
    }


def _sample_evenly(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return list(records)
    if limit == 1:
        return [records[0]]
    indices = [
        round(index * (len(records) - 1) / (limit - 1)) for index in range(limit)
    ]
    return [records[index] for index in indices]


def _write_gzip_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8") as text_handle:
                json.dump(
                    payload,
                    text_handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                text_handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional smoke-test prefix; omit to scan every record.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.data)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive when provided.")
        records = records[: args.limit]
    by_id = {record["R_id"]: record for record in records}
    started = perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_quiet_worker,
    ) as executor:
        iterator = executor.map(_scan_one, records, chunksize=args.chunksize)
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index % args.progress_every == 0 or index == len(records):
                counts = Counter(item["status"] for item in results)
                print(
                    f"[{index}/{len(records)}] {dict(sorted(counts.items()))}",
                    flush=True,
                )

    counts = Counter(result["status"] for result in results)
    unsolved = []
    for result in results:
        if result["status"] == "solved":
            continue
        source = by_id[result["R_id"]]
        unsolved.append({**source, **result})
    payload = {
        "dataset": str(args.data.resolve()),
        "method": "fast_track",
        "fusion_used": False,
        "rule_extraction": RULE_EXTRACTION_DESCRIPTION,
        "success_definition": (
            "candidate equals complete after SynKit Standardize with "
            "remove_aam=True and ignore_stereo=True"
        ),
        "records_total": len(records),
        "statuses": dict(sorted(counts.items())),
        "wall_seconds": perf_counter() - started,
        "worker_seconds": sum(result["seconds"] for result in results),
        "normalization_violations": sum(
            result["normalization_violations"] or 0
            for result in results
            if result["normalization_violations"] is not None
        ),
        "unsolved_records": unsolved,
    }
    _write_gzip_json(payload, args.output)
    summary = {
        key: payload[key]
        for key in (
            "dataset",
            "method",
            "fusion_used",
            "rule_extraction",
            "success_definition",
            "records_total",
            "statuses",
            "wall_seconds",
            "worker_seconds",
            "normalization_violations",
        )
    }
    summary["unsolved_count"] = len(unsolved)
    summary["unsolved_sample"] = _sample_evenly(unsolved, args.sample_size)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
