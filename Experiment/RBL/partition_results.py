#!/usr/bin/env python3
"""Partition safe RBL records by raw- and complete-input fast-track success."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "uspto_50k_rbl.json.gz"
DEFAULT_RAW_SCAN = HERE / "fast_track_all.json.gz"
DEFAULT_COMPLETE_RETRY = HERE / "complete_retry_hcomplete.json.gz"
DEFAULT_OUTPUT = HERE / "solution_partition.json"


def _load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--raw-scan", type=Path, default=DEFAULT_RAW_SCAN)
    parser.add_argument(
        "--complete-retry",
        type=Path,
        default=DEFAULT_COMPLETE_RETRY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = _load_gzip_json(args.data)
    raw_scan = _load_gzip_json(args.raw_scan)
    complete_retry = _load_gzip_json(args.complete_retry)

    all_ids = [record["R_id"] for record in dataset]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Safe dataset contains duplicate R_id values")
    raw_failures = {record["R_id"]: record for record in raw_scan["unsolved_records"]}
    complete_records = {record["R_id"]: record for record in complete_retry["records"]}
    if set(raw_failures) != set(complete_records):
        raise ValueError("Complete retry does not cover exactly the raw failures")

    raw_exact_ids = [
        record_id for record_id in all_ids if record_id not in raw_failures
    ]
    complete_only_ids = [
        record_id
        for record_id in all_ids
        if record_id in complete_records
        and complete_records[record_id]["complete_retry"]["retry_status"] == "solved"
    ]
    failed_both_ids = [
        record_id
        for record_id in all_ids
        if record_id in complete_records
        and complete_records[record_id]["complete_retry"]["retry_status"] != "solved"
    ]

    partition = set(raw_exact_ids) | set(complete_only_ids) | set(failed_both_ids)
    if partition != set(all_ids):
        raise ValueError("Result groups do not cover the safe dataset")
    if sum(map(len, (raw_exact_ids, complete_only_ids, failed_both_ids))) != len(
        partition
    ):
        raise ValueError("Result groups overlap")

    complete_only_by_raw_status: dict[str, list[str]] = {}
    failed_both_by_raw_status: dict[str, list[str]] = {}
    for target, identifiers in (
        (complete_only_by_raw_status, complete_only_ids),
        (failed_both_by_raw_status, failed_both_ids),
    ):
        grouped: dict[str, list[str]] = {}
        for record_id in identifiers:
            grouped.setdefault(raw_failures[record_id]["status"], []).append(record_id)
        target.update(dict(sorted(grouped.items())))

    total = len(all_ids)
    payload = {
        "records_total": total,
        "method": "fusion-free fast_track for both passes",
        "passes": {
            "raw": "input_field=raw; exact target=complete",
            "complete": (
                "input_field=complete; exact target=complete; run only on raw failures"
            ),
        },
        "groups": {
            "solved_on_raw_fast_track": {
                "count": len(raw_exact_ids),
                "percent_of_all": 100 * len(raw_exact_ids) / total,
                "R_ids": raw_exact_ids,
            },
            "solved_only_on_complete_input": {
                "count": len(complete_only_ids),
                "percent_of_all": 100 * len(complete_only_ids) / total,
                "percent_of_raw_failures": (
                    100 * len(complete_only_ids) / len(raw_failures)
                ),
                "by_raw_status_counts": {
                    status: len(identifiers)
                    for status, identifiers in complete_only_by_raw_status.items()
                },
                "by_raw_status_R_ids": complete_only_by_raw_status,
                "R_ids": complete_only_ids,
            },
            "failed_on_both_inputs": {
                "count": len(failed_both_ids),
                "percent_of_all": 100 * len(failed_both_ids) / total,
                "by_raw_status_counts": {
                    status: len(identifiers)
                    for status, identifiers in failed_both_by_raw_status.items()
                },
                "by_raw_status_R_ids": failed_both_by_raw_status,
                "R_ids": failed_both_ids,
            },
        },
        "checks": {
            "disjoint": True,
            "covers_all_records": True,
            "group_count_sum": sum(
                map(len, (raw_exact_ids, complete_only_ids, failed_both_ids))
            ),
            "raw_failure_statuses": dict(
                sorted(
                    Counter(
                        record["status"] for record in raw_failures.values()
                    ).items()
                )
            ),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({name: group["count"] for name, group in payload["groups"].items()})
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
