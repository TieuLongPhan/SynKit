#!/usr/bin/env python3
"""Audit guarded SynKit 2.0 AAM completion on the full radical corpus."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (
    RADICAL_DATASET,
    sha256,
    timing_summary,
    write_json,
    write_jsonl,
)  # noqa: E402
from synkit.Mechanism.radical_data import complete_radical_aam  # noqa: E402

GATES = (
    "all_atoms_mapped",
    "unique_atom_maps",
    "balanced_atom_maps",
    "source_anchors_preserved",
    "constitution_preserved",
    "radical_state_preserved",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=RADICAL_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    with dataset.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if args.limit is not None:
        rows = rows[: args.limit]

    counts: Counter[str] = Counter()
    durations: list[float] = []
    failures: list[dict[str, object]] = []
    started = time.perf_counter()
    for row_number, row in enumerate(rows, start=1):
        source = row[0].strip().split(None, 1)[0]
        case_started = time.perf_counter()
        result = complete_radical_aam(source)
        durations.append(time.perf_counter() - case_started)
        counts[f"status:{result.status.lower()}"] += 1
        counts[f"method:{str(result.method).lower()}"] += 1
        counts[f"usable:{str(result.usable).lower()}"] += 1
        for gate in GATES:
            counts[f"{gate}:{str(bool(getattr(result, gate))).lower()}"] += 1
        counts[f"fallback_used:{str(result.fallback_used).lower()}"] += 1
        if not result.usable or not all(bool(getattr(result, gate)) for gate in GATES):
            failures.append({"row_number": row_number, **result.to_dict()})
        if args.progress_every and row_number % args.progress_every == 0:
            print(
                f"radical expansion: {row_number}/{len(rows)}",
                file=sys.stderr,
                flush=True,
            )

    wall = time.perf_counter() - started
    report = {
        "schema": "synkit.radical-expansion-audit/1",
        "implementation": "SynKit 2.0 development checkout",
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "failure_count": len(failures),
        "timing_seconds": {**timing_summary(durations), "wall": wall},
        "reference_policy": {
            "independent_full_aam": False,
            "aamvalidator_its_equivalence": "not_applicable",
            "reason": (
                "The source reactions are partially mapped and therefore do not "
                "define an independent full ITS/AAM reference. Guarded completion "
                "gates are reported instead of a tautological self-comparison."
            ),
        },
    }
    write_json(args.output, report)
    if args.failures:
        write_jsonl(args.failures, failures)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
