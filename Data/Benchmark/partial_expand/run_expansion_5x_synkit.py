#!/usr/bin/env python3
"""Run SynKit partial-AAM completion five times on both full corpora."""

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
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("synkit", "synkit_rsmi", "synkit_noanchors"),
        default=["synkit"],
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=("general", "radical"),
        default=["general", "radical"],
    )
    parser.add_argument("--limit", type=int, help="Pilot only; omit for full datasets")
    parser.add_argument("--case-timeout", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "synkit-expansion-5x",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _ratio(counts: dict[str, int], numerator: str, denominator: int) -> float:
    return counts.get(numerator, 0) / denominator if denominator else 0.0


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values) if values else None,
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def aggregate_reports(report_paths: list[Path], output: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    inputs = []
    for path in report_paths:
        report = json.loads(path.read_text())
        suite = report["suite"]
        rows = int(report["selection"]["rows"])
        inputs.append(
            {
                "path": str(path.resolve()),
                "suite": suite,
                "dataset": report["dataset"],
                "rows": rows,
            }
        )
        for method in report["methods"]:
            counts = method["counts"]
            generated = int(counts.get("output", 0))
            generation = method["timing_seconds"]["stages"]["generation"]
            metrics = {
                "mean_generation_seconds_per_attempt": generation["total"] / rows,
                "generation_success_rate": generated / rows if rows else 0.0,
                "accepted_coverage": _ratio(counts, "accepted:true", rows),
                "anchor_preservation_rate_generated": _ratio(
                    counts, "source_anchors_preserved:true", generated
                ),
            }
            if suite == "general":
                metrics["its_accuracy_generated"] = _ratio(
                    counts, "aam_validator_its:true", generated
                )
                metrics["rc_accuracy_generated"] = _ratio(
                    counts, "aam_validator_rc:true", generated
                )
            else:
                metrics["structural_accuracy_generated"] = _ratio(
                    counts, "extension_valid_without_anchors:true", generated
                )
            grouped.setdefault((suite, method["method"]), []).append(metrics)

    aggregates = []
    for (suite, method), repetitions in sorted(grouped.items()):
        names = sorted({name for item in repetitions for name in item})
        aggregates.append(
            {
                "suite": suite,
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
        "accuracy_policy": {
            "general": "ITS/RC accuracy among successfully generated candidates",
            "radical": "structural validity excluding source-anchor preservation",
            "anchors": "reported separately, never used as accuracy acceptance",
            "runtime": "generation time per attempted input, including failed attempts",
        },
        "inputs": inputs,
        "aggregates": aggregates,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_repetitions(
    *,
    methods: list[str],
    repetitions: int,
    output_dir: Path,
    limit: int | None,
    case_timeout: float,
    progress_every: int,
    force: bool,
    command_prefix: list[str],
    extra_args: list[str] | None = None,
    child_env: dict[str, str] | None = None,
    suites: list[str] | tuple[str, ...] = ("general", "radical"),
) -> Path:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths: list[Path] = []
    for suite in suites:
        for repetition in range(1, repetitions + 1):
            stem = f"{suite}-run-{repetition:02d}"
            report = output_dir / f"{stem}.json"
            cases = output_dir / f"{stem}-cases.jsonl.gz"
            if not force and (report.exists() or cases.exists()):
                raise FileExistsError(
                    f"Refusing to overwrite {stem}; pass --force to replace it"
                )
            command = command_prefix + [
                str(BENCHMARK),
                "--suite",
                suite,
                "--methods",
                *methods,
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
            if extra_args:
                command.extend(extra_args)
            print(f"Running {suite} repetition {repetition}/{repetitions}", flush=True)
            subprocess.run(command, cwd=ROOT, env=child_env, check=True)
            report_paths.append(report)
    aggregate = output_dir / "aggregate.json"
    if aggregate.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {aggregate}; pass --force to replace it"
        )
    aggregate_reports(report_paths, aggregate)
    print(f"Wrote aggregate report: {aggregate}")
    return aggregate


def main() -> int:
    args = parse_args()
    run_repetitions(
        methods=args.methods,
        repetitions=args.repetitions,
        output_dir=args.output_dir.resolve(),
        limit=args.limit,
        case_timeout=args.case_timeout,
        progress_every=args.progress_every,
        force=args.force,
        command_prefix=[sys.executable],
        child_env=os.environ.copy(),
        suites=args.suites,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
