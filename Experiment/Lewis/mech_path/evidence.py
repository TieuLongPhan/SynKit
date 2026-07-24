#!/usr/bin/env python
"""Collect reproducible Sprint 9 evidence from executable MechanismBench data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synkit.Mechanism.evidence import collect_evidence, write_evidence_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Data" / "MechanismBench",
        help="Directory containing polar.json, radical.json, and stereo.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the generated evidence JSON.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Strict stepwise replay timing repetitions per record (default: 3).",
    )
    args = parser.parse_args()
    report = collect_evidence(args.benchmark_dir, repetitions=args.repetitions)
    write_evidence_report(report, args.output)


if __name__ == "__main__":
    main()
