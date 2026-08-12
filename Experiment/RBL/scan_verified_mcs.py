#!/usr/bin/env python3
"""Run verified exact/global maximum-MCS fusion on raw fast-track failures."""

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

from Experiment.RBL.benchmark import evaluate  # noqa: E402

DEFAULT_INPUT = HERE / "fast_track_all.json.gz"
DEFAULT_OUTPUT = HERE / "verified_mcs_all.json.gz"
DEFAULT_SUMMARY = HERE / "verified_mcs_all_summary.json"
DEFAULT_CHECKPOINT = HERE / "verified_mcs_checkpoint.jsonl"


def _quiet_worker() -> None:
    logging.disable(logging.CRITICAL)


def _run_one(arguments: tuple[dict[str, Any], float]) -> dict[str, Any]:
    record, timeout = arguments
    result = evaluate(record, "verified", timeout)
    return {
        "R_id": record["R_id"],
        "raw_fast_track_status": record["status"],
        **result,
    }


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
                    payload, text_handle, ensure_ascii=False, separators=(",", ":")
                )
                text_handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid resource and selection bounds before opening files."""
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be at least 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive when supplied")


def _load_checkpoint(
    path: Path,
    selected_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Load and validate resumable per-record results."""
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        result = json.loads(line)
        record_id = result["R_id"]
        if record_id in completed:
            raise ValueError(f"Duplicate {record_id} in checkpoint line {line_number}")
        completed[record_id] = result
    foreign_ids = set(completed) - selected_ids
    if foreign_ids:
        raise ValueError(
            "Checkpoint contains IDs outside the selected input: "
            f"{sorted(foreign_ids)[:5]}"
        )
    return completed


def _run_pending(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> None:
    """Evaluate pending records and durably append each completed result."""
    pending = [record for record in records if record["R_id"] not in completed]
    arguments = [(record, args.timeout) for record in pending]
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("a", encoding="utf-8") as checkpoint_handle:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_quiet_worker,
        ) as executor:
            iterator = executor.map(_run_one, arguments, chunksize=args.chunksize)
            for result in iterator:
                completed[result["R_id"]] = result
                checkpoint_handle.write(
                    json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
                )
                checkpoint_handle.flush()
                index = len(completed)
                if index % args.progress_every == 0 or index == len(records):
                    counts = Counter(item["status"] for item in completed.values())
                    print(
                        f"[{index}/{len(records)}] {dict(sorted(counts.items()))}",
                        flush=True,
                    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        source = json.load(handle)
    records = source["unsolved_records"]
    if args.limit is not None:
        records = records[: args.limit]

    started = perf_counter()
    selected_ids = {record["R_id"] for record in records}
    completed = _load_checkpoint(args.checkpoint, selected_ids)
    _run_pending(args, records, completed)

    results = [completed[record["R_id"]] for record in records]

    by_id = {result["R_id"]: result for result in results}
    combined = [{**record, "verified_mcs": by_id[record["R_id"]]} for record in records]
    statuses = Counter(result["status"] for result in results)
    by_raw_status: dict[str, Counter[str]] = {}
    for raw_status in sorted({record["status"] for record in records}):
        by_raw_status[raw_status] = Counter(
            record["verified_mcs"]["status"]
            for record in combined
            if record["status"] == raw_status
        )
    payload = {
        "source": str(args.input.resolve()),
        "records_total": len(records),
        "input_field": "raw",
        "method": "verified",
        "timeout_seconds_per_record": args.timeout,
        "scope": {
            "matcher": "exact global MCS",
            "overlap_scope": "maximum common subgraphs only",
            "component_matching": False,
            "automorphism_pruning": False,
            "max_mappings_per_pair": 0,
            "fusion_backend": "categorical_pushout",
            "termination": "mapping_scope_exhausted",
            "globally_complete_over_all_overlaps": False,
        },
        "success_definition": (
            "candidate equals balanced canonical complete after AAM/stereo removal"
        ),
        "statuses": dict(sorted(statuses.items())),
        "by_raw_fast_track_status": {
            raw_status: dict(sorted(counts.items()))
            for raw_status, counts in by_raw_status.items()
        },
        "wall_seconds": perf_counter() - started,
        "worker_seconds": sum(result["seconds"] for result in results),
        "records": combined,
    }
    _write_gzip_json(payload, args.output)
    summary = {key: value for key, value in payload.items() if key != "records"}
    summary["solved_R_ids"] = [
        result["R_id"] for result in results if result["status"] == "solved"
    ]
    summary["timeout_R_ids"] = [
        result["R_id"] for result in results if result["status"] == "timeout"
    ]
    summary["error_R_ids"] = [
        result["R_id"] for result in results if result["status"] == "error"
    ]
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
