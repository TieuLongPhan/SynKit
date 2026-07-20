#!/usr/bin/env python3
"""Evaluate generated expansion candidates with the current AAMValidator."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # noqa: E402
    POLAR_DATASET,
    iter_jsonl,
    read_json,
    sha256,
    timing_summary,
    write_json,
    write_jsonl,
)
from synkit.Chem.Reaction import AAMValidator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=POLAR_DATASET)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    dataset = args.dataset.resolve()
    candidates_path = args.candidates.resolve()
    references = {int(row["R-id"]): str(row["smart"]) for row in read_json(dataset)}
    validator = AAMValidator(strip_unbalanced_maps=True)
    counts: Counter[str] = Counter()
    durations: list[float] = []
    failures: list[dict[str, object]] = []
    started = time.perf_counter()

    for index, row in enumerate(iter_jsonl(candidates_path), start=1):
        status = str(row.get("status"))
        if status != "OUTPUT":
            counts["generation_error"] += 1
            failures.append(row)
            continue
        record_id = int(row["record_id"])
        case_started = time.perf_counter()
        try:
            verdict = validator.smiles_checks(
                str(row["candidate"]),
                references[record_id],
                constitutional_only=True,
            )
            its = bool(verdict["ITS"])
            rc = bool(verdict["RC"])
            counts[f"its_equivalent:{str(its).lower()}"] += 1
            counts[f"rc_equivalent:{str(rc).lower()}"] += 1
            counts[f"both_equivalent:{str(its and rc).lower()}"] += 1
            if not (its and rc):
                failures.append({**row, "its_equivalent": its, "rc_equivalent": rc})
        except Exception as exc:
            counts["validation_error"] += 1
            failures.append(
                {
                    **row,
                    "validation_error_type": type(exc).__name__,
                    "validation_message": str(exc),
                }
            )
        durations.append(time.perf_counter() - case_started)
        if args.progress_every and index % args.progress_every == 0:
            print(f"validation: {index}", file=sys.stderr, flush=True)

    wall = time.perf_counter() - started
    report = {
        "schema": "synkit.partial-expand-evaluation/1",
        "ground_truth": "smart",
        "validation": {
            "validator": "synkit.Chem.Reaction.AAMValidator",
            "checks": ["ITS", "RC"],
            "constitutional_only": True,
            "strip_unbalanced_maps": True,
        },
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "candidates": {"path": str(candidates_path), "sha256": sha256(candidates_path)},
        "counts": dict(sorted(counts.items())),
        "failure_count": len(failures),
        "timing_seconds": {**timing_summary(durations), "wall": wall},
    }
    write_json(args.output, report)
    if args.failures:
        write_jsonl(args.failures, failures)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
