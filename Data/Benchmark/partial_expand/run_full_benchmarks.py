#!/usr/bin/env python3
"""Run the complete partial-AAM expansion and bidirectional replay matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"


def _run(command: list[str], output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"skip completed report: {output}", flush=True)
        return
    print("run: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--limit", type=int, help="Use a common pilot limit")
    parser.add_argument(
        "--only",
        choices=("all", "expansion", "replay"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    common = []
    if args.limit is not None:
        common = ["--limit", str(args.limit)]

    if args.only in {"all", "expansion"}:
        for suite in ("general", "radical"):
            output = results / f"expansion-{suite}-all.json"
            command = [
                sys.executable,
                str(HERE / "benchmark_expansion_comparison.py"),
                "--suite",
                suite,
                "--output",
                str(output),
                "--cases",
                str(results / f"expansion-{suite}-all-cases.jsonl.gz"),
                "--progress-every",
                str(args.progress_every),
                *common,
            ]
            _run(command, output, args.force)

    if args.only in {"all", "replay"}:
        configurations = (
            ("polar", "tuple", "general-tuple"),
            ("polar", "typesGH", "general-typesGH"),
            ("radical", "tuple", "radical-tuple"),
        )
        for suite, representation, label in configurations:
            output = results / f"replay-{label}-all.json"
            command = [
                sys.executable,
                str(HERE / "benchmark_rule_replay.py"),
                "--suite",
                suite,
                "--implementation",
                "current",
                "--representation",
                representation,
                "--output",
                str(output),
                "--cases",
                str(results / f"replay-{label}-all-cases.jsonl.gz"),
                "--failures",
                str(results / f"replay-{label}-all-failures.jsonl.gz"),
                "--progress-every",
                str(args.progress_every),
                *common,
            ]
            _run(command, output, args.force)

    required = (
        "expansion-general-all.json",
        "expansion-radical-all.json",
        "replay-general-tuple-all.json",
        "replay-general-typesGH-all.json",
        "replay-radical-tuple-all.json",
    )
    if all((results / name).exists() for name in required):
        summary = results / "full-benchmark-summary.json"
        subprocess.run(
            [
                sys.executable,
                str(HERE / "summarize_full_benchmarks.py"),
                "--results",
                str(results),
                "--output",
                str(summary),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
