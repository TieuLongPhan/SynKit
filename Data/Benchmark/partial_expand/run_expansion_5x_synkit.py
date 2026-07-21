#!/usr/bin/env python3
"""Repeat SynKit minimal partial-AAM expansion on one corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "benchmark_expansion_comparison.py"
ROOT = HERE.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("general", "radical"), default="general")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Pilot only; omit for full corpus")
    parser.add_argument("--case-timeout", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: results/synkit-<suite>-expansion-5x",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values) if values else None,
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def aggregate_reports(report_paths: list[Path], output: Path) -> None:
    grouped: dict[str, list[dict[str, float]]] = {}
    inputs = []
    for path in report_paths:
        report = json.loads(path.read_text())
        rows = int(report["selection"]["rows"])
        inputs.append(
            {
                "path": str(path.resolve()),
                "dataset": report["dataset"],
                "rows": rows,
            }
        )
        independent_reference = bool(report["reference_policy"]["independent_full_aam"])
        for method in report["methods"]:
            counts = method["counts"]
            generated = int(counts.get("output", 0))
            generation = method["timing_seconds"]["stages"]["generation"]
            metrics = {
                "mean_generation_seconds_per_attempt": generation["total"] / rows,
                "generation_success_rate": generated / rows if rows else 0.0,
                "accepted_coverage": counts.get("accepted:true", 0) / rows,
            }
            if independent_reference:
                metrics.update(
                    {
                        "its_accuracy_generated": (
                            counts.get("aam_validator_its:true", 0) / generated
                            if generated
                            else 0.0
                        ),
                        "rc_accuracy_generated": (
                            counts.get("aam_validator_rc:true", 0) / generated
                            if generated
                            else 0.0
                        ),
                    }
                )
            else:
                metrics["radical_state_preservation_generated"] = (
                    counts.get("radical_state_preserved:true", 0) / generated
                    if generated
                    else 0.0
                )
            grouped.setdefault(method["method"], []).append(metrics)
    aggregates = []
    for method, repetitions in sorted(grouped.items()):
        names = sorted(repetitions[0])
        aggregates.append(
            {
                "method": method,
                "repetitions": len(repetitions),
                "metrics": {
                    name: _summary([item[name] for item in repetitions])
                    for name in names
                },
            }
        )
    payload = {
        "schema": "synkit.partial-aam-repeated-comparison/1",
        "reference_policy": report["reference_policy"],
        "runtime_policy": "generation time per attempted input",
        "inputs": inputs,
        "aggregates": aggregates,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_repetitions(
    *,
    repetitions: int,
    suite: str,
    output_dir: Path,
    limit: int | None,
    case_timeout: float,
    progress_every: int,
    force: bool,
) -> Path:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[Path] = []
    for repetition in range(1, repetitions + 1):
        stem = f"synkit-run-{repetition:02d}"
        report = output_dir / f"{stem}.json"
        cases = output_dir / f"{stem}-cases.jsonl.gz"
        if not force and (report.exists() or cases.exists()):
            raise FileExistsError(f"Refusing to overwrite {stem}; pass --force")
        command = [
            sys.executable,
            str(BENCHMARK),
            "--methods",
            "synkit",
            "--suite",
            suite,
            "--case-timeout",
            str(case_timeout),
            "--progress-every",
            str(progress_every),
            "--output",
            str(report),
            "--cases",
            str(cases),
        ]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        print(f"Running SynKit repetition {repetition}/{repetitions}", flush=True)
        subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
        reports.append(report)
    aggregate = output_dir / "aggregate.json"
    if aggregate.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {aggregate}; pass --force")
    aggregate_reports(reports, aggregate)
    print(f"Wrote aggregate report: {aggregate}")
    return aggregate


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (
        HERE / "results" / f"synkit-{args.suite}-expansion-5x"
    )
    run_repetitions(
        repetitions=args.repetitions,
        suite=args.suite,
        output_dir=output_dir.resolve(),
        limit=args.limit,
        case_timeout=args.case_timeout,
        progress_every=args.progress_every,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
