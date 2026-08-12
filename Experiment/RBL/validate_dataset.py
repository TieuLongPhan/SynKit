#!/usr/bin/env python3
"""Independently validate the rebuilt safe RBL benchmark artifact."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synkit.Chem.Reaction.standardize import Standardize  # noqa: E402

from Experiment.RBL.benchmark import extract_normalized_rule  # noqa: E402
from Experiment.RBL.build_dataset import (  # noqa: E402
    _mapped_reaction_issue,
    _reaction_balance,
)

DEFAULT_DATA = HERE / "uspto_50k_rbl.json.gz"
DEFAULT_REPORT = HERE / "uspto_50k_rbl_validation.json"
EXPECTED_FIELDS = {"R_id", "raw", "complete", "aam"}


def _validate_one(record: dict[str, str]) -> dict[str, Any]:
    issues = []
    record_id = record.get("R_id", "<missing>")
    if set(record) != EXPECTED_FIELDS:
        issues.append(
            {
                "code": "schema",
                "fields": sorted(record),
            }
        )
    standardize = Standardize()
    raw = standardize.fit(
        record.get("raw", ""),
        remove_aam=True,
        ignore_stereo=True,
        remove_invalid=False,
    )
    complete = standardize.fit(
        record.get("complete", ""),
        remove_aam=True,
        ignore_stereo=True,
        remove_invalid=False,
    )
    if raw != record.get("raw"):
        issues.append({"code": "raw_not_canonical", "observed": raw})
    if complete != record.get("complete"):
        issues.append({"code": "complete_not_canonical", "observed": complete})

    aam = record.get("aam", "")
    mapped_issue = _mapped_reaction_issue(aam)
    if mapped_issue is not None:
        issues.append({"code": "invalid_aam", "detail": mapped_issue})
    aam_complete = standardize.fit(
        aam,
        remove_aam=True,
        ignore_stereo=True,
        remove_invalid=False,
    )
    if aam_complete != record.get("complete"):
        issues.append(
            {
                "code": "aam_complete_mismatch",
                "observed": aam_complete,
            }
        )

    element_balanced, charge_balanced = _reaction_balance(record.get("complete", ""))
    if not element_balanced:
        issues.append({"code": "element_unbalanced_ground_truth"})
    if not charge_balanced:
        issues.append({"code": "charge_unbalanced_ground_truth"})

    hydrogen_candidates = None
    hydrogen_exhaustive = None
    try:
        _rule, audit = extract_normalized_rule(record)
        completion = audit["hydrogen_completion"]
        hydrogen_candidates = completion["candidates"]
        hydrogen_exhaustive = completion["exhaustive"]
        if completion["status"] != "unambiguous" or not hydrogen_exhaustive:
            issues.append(
                {
                    "code": "hydrogen_not_unambiguous",
                    "completion": completion,
                }
            )
    except Exception as exc:
        issues.append(
            {
                "code": "rule_extraction_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return {
        "R_id": record_id,
        "valid": not issues,
        "issues": issues,
        "hydrogen_candidates": hydrogen_candidates,
        "hydrogen_exhaustive": hydrogen_exhaustive,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )
    parser.add_argument("--chunksize", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be at least 1")
    with gzip.open(args.data, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError("Dataset must be a JSON list")

    ids = [record.get("R_id") for record in records]
    duplicate_ids = sorted(
        str(record_id) for record_id, count in Counter(ids).items() if count > 1
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(_validate_one, records, chunksize=args.chunksize))

    failures = [result for result in results if not result["valid"]]
    issue_counts = Counter(
        issue["code"] for result in failures for issue in result["issues"]
    )
    candidate_counts = Counter(
        result["hydrogen_candidates"]
        for result in results
        if result["hydrogen_candidates"] is not None
    )
    report = {
        "dataset": str(args.data.resolve()),
        "records": len(records),
        "valid_records": len(records) - len(failures),
        "invalid_records": len(failures),
        "duplicate_R_ids": duplicate_ids,
        "issue_counts": dict(sorted(issue_counts.items())),
        "hydrogen_candidate_counts": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
        "failures": failures,
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({key: value for key, value in report.items() if key != "failures"})
    )
    print(f"Wrote {args.report}")
    if failures or duplicate_ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
