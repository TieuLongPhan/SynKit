#!/usr/bin/env python3
"""Generate full-AAM candidates with a selected SynKit checkout.

Run this script once with the current checkout and once with an isolated
another source tree.  Validation is deliberately a separate current-checkout
step so both generations are judged by the same AAMValidator implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import sys
import time

from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import synkit  # noqa: F401
except ModuleNotFoundError:
    # A source checkout is not necessarily installed in the benchmark env.
    # Do this only as a fallback so an explicit PYTHONPATH can select another tree.
    sys.path.insert(0, str(HERE.parents[2]))

from common import (
    POLAR_DATASET,
    read_json,
    sha256,
    timing_summary,
    write_json,
)  # noqa: E402
from common import open_text  # noqa: E402

from synkit.Graph.ITS.its_expand import ITSExpand  # noqa: E402


def has_radical(reaction: str) -> bool:
    for side in reaction.split(">>"):
        molecule = Chem.MolFromSmiles(side)
        if molecule is None:
            raise ValueError("RDKit rejected reaction endpoint")
        if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
            return True
    return False


def expand(reaction: str, mode: str) -> tuple[str, dict[str, object]]:
    report_method = getattr(ITSExpand, "expand_aam_with_its_report", None)
    if report_method is None:
        if mode != "core":
            raise ValueError("SynKit v1 supports only the core expansion mode")
        return ITSExpand.expand_aam_with_its(reaction, use_G=True), {}

    if mode == "core":
        report = report_method(
            reaction,
            preserve_older_map=True,
            constitutional_only=True,
        )
    else:
        report = report_method(
            reaction,
            preserve_older_map=True,
            fallback_to_other_side=True,
            require_constitution_preservation=True,
            fold_unmapped_explicit_hydrogens=True,
            ignore_stereochemistry=True,
            explicit_hydrogen=True,
            preserve_radical_state=has_radical(reaction),
            constitutional_only=True,
        )
    return report.rsmi, report.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=POLAR_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--mode", choices=("core", "guarded"), default="core")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    dataset = args.dataset.resolve()
    rows = read_json(dataset)
    if not isinstance(rows, list):
        raise ValueError("Polar benchmark root must be a JSON list")
    selected = rows[
        args.offset : None if args.limit is None else args.offset + args.limit
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    durations: list[float] = []
    counts: Counter[str] = Counter()
    started = time.perf_counter()
    with open_text(args.output, "wt") as handle:
        for index, row in enumerate(selected, start=1):
            case_started = time.perf_counter()
            output: dict[str, object] = {
                "record_id": int(row["R-id"]),
                "implementation": args.implementation,
                "mode": args.mode,
            }
            try:
                candidate, evidence = expand(str(row["partial"]), args.mode)
                output.update(candidate=candidate, evidence=evidence, status="OUTPUT")
                counts["output"] += 1
            except Exception as exc:
                output.update(
                    status="ERROR",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                counts["error"] += 1
            elapsed = time.perf_counter() - case_started
            durations.append(elapsed)
            output["generation_seconds"] = elapsed
            handle.write(json.dumps(output, sort_keys=True) + "\n")
            if args.progress_every and index % args.progress_every == 0:
                print(
                    f"expansion {args.implementation}: {index}/{len(selected)}",
                    file=sys.stderr,
                    flush=True,
                )

    wall = time.perf_counter() - started
    summary = {
        "schema": "synkit.partial-expand-generation/1",
        "implementation": args.implementation,
        "mode": args.mode,
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "selection": {
            "offset": args.offset,
            "limit": args.limit,
            "rows": len(selected),
        },
        "counts": dict(sorted(counts.items())),
        "timing_seconds": {**timing_summary(durations), "wall": wall},
        "environment": {"python": platform.python_version()},
        "candidate_file": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
