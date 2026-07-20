#!/usr/bin/env python3
"""Run historical GM, RB1, and RB2 five times in the ``aam`` conda env.

Generation runs in the reconstructed 5 May 2025 dependency stack.  The parent
process evaluates all generated candidates with the current common evaluator,
keeping source-anchor preservation separate from AAM accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, TextIO

from run_expansion_5x_synkit import aggregate_reports

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GENERAL_DATASET = ROOT / "Data" / "Benchmark" / "benchmark.json.gz"
RADICAL_DATASET = ROOT / "Data" / "Benchmark" / "radical" / "all.csv"


class CaseTimeout(TimeoutError):
    """Raised when one historical generation exceeds its wall-time ceiling."""


def raise_timeout(_signum, _frame) -> None:
    raise CaseTimeout("External partial-AAM generation exceeded the case timeout")


def open_text(path: Path, mode: str) -> TextIO:
    compressed = (
        path.suffix == ".gz" if "w" in mode else path.read_bytes()[:2] == b"\x1f\x8b"
    )
    opener = gzip.open if compressed else open
    return opener(path, mode, encoding="utf-8")


def load_sources(path: Path, suite: str) -> list[dict[str, Any]]:
    if suite == "general":
        with open_text(path, "rt") as handle:
            return [
                {
                    "record_id": int(row["R-id"]),
                    "source": str(row["partial"]),
                    "reference": str(row["smart"]),
                }
                for row in json.load(handle)
            ]
    records = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if row:
                records.append(
                    {
                        "record_id": row_number,
                        "source": row[0].strip().split(None, 1)[0],
                        "reference": None,
                    }
                )
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Historical external generation worker"
    )
    parser.add_argument("--suite", choices=("general", "radical"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-timeout", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(arguments)

    # Imported only in the conda worker. PYTHONPATH selects the April checkout.
    from partialaams.aam_expand import partial_aam_extension_from_smiles
    import partialaams
    import synkit

    synkit_version = importlib.metadata.version("synkit")
    if synkit_version != "0.0.6":
        raise RuntimeError(f"Expected SynKit 0.0.6 in aam, found {synkit_version}")

    rows = load_sources(args.dataset.resolve(), args.suite)
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    method_reports = []
    with open_text(args.output, "wt") as handle:
        for method in args.methods:
            durations: list[float] = []
            counts: Counter[str] = Counter()
            wall_started = time.perf_counter()
            for index, row in enumerate(rows, start=1):
                started = time.perf_counter()
                previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
                signal.setitimer(signal.ITIMER_REAL, args.case_timeout)
                output: dict[str, Any] = {
                    "method": method,
                    "record_id": row["record_id"],
                }
                try:
                    candidate = partial_aam_extension_from_smiles(
                        row["source"],
                        method={"rb1": "extend", "rb2": "extend_g"}.get(method, method),
                    )
                    output.update(status="OUTPUT", candidate=candidate)
                    counts["output"] += 1
                except Exception as exc:
                    output.update(
                        status="ERROR",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                    counts["error"] += 1
                    if isinstance(exc, CaseTimeout):
                        counts["timeout"] += 1
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    signal.signal(signal.SIGALRM, previous_handler)
                elapsed = time.perf_counter() - started
                durations.append(elapsed)
                output["generation_seconds"] = elapsed
                handle.write(json.dumps(output, sort_keys=True) + "\n")
                if args.progress_every and index % args.progress_every == 0:
                    print(
                        f"external {args.suite}/{method}: {index}/{len(rows)}",
                        file=sys.stderr,
                        flush=True,
                    )
            method_reports.append(
                {
                    "method": method,
                    "counts": dict(counts),
                    "generation_seconds": {
                        "count": len(durations),
                        "total": sum(durations),
                        "mean": statistics.mean(durations) if durations else 0.0,
                        "median": statistics.median(durations) if durations else 0.0,
                    },
                    "wall_seconds": time.perf_counter() - wall_started,
                }
            )
    summary = {
        "schema": "synkit.external-partial-aam-generation/1",
        "suite": args.suite,
        "rows": len(rows),
        "environment": {
            "python": platform.python_version(),
            "synkit": synkit_version,
            "synkit_source": synkit.__file__,
            "gmapache": importlib.metadata.version("gmapache"),
            "rdkit": importlib.metadata.version("rdkit"),
            "networkx": importlib.metadata.version("networkx"),
            "partialaams_source": partialaams.__file__,
        },
        "methods": method_reports,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def timing_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "total": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[round((len(ordered) - 1) * 0.95)],
        "maximum": ordered[-1],
        "total": sum(ordered),
    }


def evaluate_generation(
    *,
    suite: str,
    dataset: Path,
    generated: Path,
    generation_summary: Path,
    report_path: Path,
    cases_path: Path,
) -> None:
    from benchmark_expansion_comparison import completion_gates
    from synkit.Chem.Reaction import AAMValidator

    sources = {row["record_id"]: row for row in load_sources(dataset, suite)}
    generation = json.loads(generation_summary.read_text())
    rows = int(generation["rows"])
    validator = AAMValidator(strip_unbalanced_maps=True) if suite == "general" else None
    counts_by_method: dict[str, Counter[str]] = defaultdict(Counter)
    generation_times: dict[str, list[float]] = defaultdict(list)
    validation_times: dict[str, list[float]] = defaultdict(list)

    with (
        open_text(generated, "rt") as source_handle,
        open_text(cases_path, "wt") as case_handle,
    ):
        for line in source_handle:
            raw = json.loads(line)
            method = str(raw["method"])
            counts = counts_by_method[method]
            generation_times[method].append(float(raw["generation_seconds"]))
            case = {
                "method": method,
                "record_id": raw["record_id"],
                "generation_seconds": raw["generation_seconds"],
            }
            if raw["status"] != "OUTPUT":
                counts["error"] += 1
                case.update(
                    status="ERROR",
                    error_type=raw.get("error_type"),
                    message=raw.get("message"),
                )
                case_handle.write(json.dumps(case, sort_keys=True) + "\n")
                continue

            counts["output"] += 1
            record = sources[int(raw["record_id"])]
            candidate = str(raw["candidate"])
            validation_started = time.perf_counter()
            try:
                gates = completion_gates(record["source"], candidate)
                for gate, passed in gates.items():
                    counts[f"{gate}:{str(passed).lower()}"] += 1
                extension_valid = all(
                    passed
                    for gate, passed in gates.items()
                    if gate != "source_anchors_preserved"
                )
                anchored_valid = all(gates.values())
                counts[
                    "extension_valid_without_anchors:" f"{str(extension_valid).lower()}"
                ] += 1
                counts[
                    "anchor_preserving_extension_valid:"
                    f"{str(anchored_valid).lower()}"
                ] += 1
                verdicts = None
                if validator is not None:
                    verdicts = {
                        key: bool(value)
                        for key, value in validator.smiles_checks(
                            candidate,
                            record["reference"],
                            constitutional_only=True,
                        ).items()
                        if key in {"ITS", "RC"}
                    }
                    for check, passed in verdicts.items():
                        counts[
                            f"aam_validator_{check.lower()}:{str(passed).lower()}"
                        ] += 1
                accepted = (
                    extension_valid if verdicts is None else all(verdicts.values())
                )
                counts[f"accepted:{str(accepted).lower()}"] += 1
                case.update(
                    status="PASS" if accepted else "FAIL",
                    gates=gates,
                    reference_checks=verdicts,
                    extension_valid_without_anchors=extension_valid,
                    anchor_preserving_extension_valid=anchored_valid,
                )
                if not accepted:
                    case["candidate"] = candidate
            except Exception as exc:
                counts["validation_error"] += 1
                case.update(
                    status="ERROR",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    candidate=candidate,
                )
            validation_times[method].append(time.perf_counter() - validation_started)
            case_handle.write(json.dumps(case, sort_keys=True) + "\n")

    methods = []
    for generated_method in generation["methods"]:
        method = generated_method["method"]
        methods.append(
            {
                "method": method,
                "label": {
                    "gm": "GM",
                    "rb1": "RB1 (PartialAAMs extend)",
                    "rb2": "RB2 (PartialAAMs extend_g)",
                }[method],
                "counts": dict(sorted(counts_by_method[method].items())),
                "timing_seconds": {
                    "wall": generated_method["wall_seconds"],
                    "rows_per_second": rows / generated_method["wall_seconds"],
                    "stages": {
                        "generation": timing_summary(generation_times[method]),
                        "validation": timing_summary(validation_times[method]),
                    },
                },
            }
        )
    report = {
        "schema": "synkit.partial-aam-method-comparison/1",
        "suite": suite,
        "dataset": {"path": str(dataset.resolve()), "sha256": sha256(dataset)},
        "selection": {
            "rows": rows,
            "all_dataset_rows": len(sources),
            "limit": rows if rows != len(sources) else None,
        },
        "reference_policy": {
            "accuracy_excludes_anchor_preservation": True,
            "source_anchors_preserved_reported_separately": True,
        },
        "external_generation_environment": generation["environment"],
        "methods": methods,
        "case_file": str(cases_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda-env", default="aam")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Pilot only; omit for full datasets")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--case-timeout", type=float, default=10.0)
    parser.add_argument(
        "--partialaams",
        type=Path,
        default=Path("/tmp/PartialAAMs-008"),
        help="PartialAAMs checkout at 008173e (last revision before 5 May 2025)",
    )
    parser.add_argument(
        "--gmapache",
        type=Path,
        default=Path("/tmp/GranMapache-4c8"),
        help="GranMapache checkout at 4c8f292 (last revision before 5 May 2025)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "external-aam-expansion-5x",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    partialaams = args.partialaams.resolve()
    gmapache = args.gmapache.resolve()
    for path in (partialaams, gmapache):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(partialaams)
    reports = []

    for suite, dataset in (
        ("general", GENERAL_DATASET),
        ("radical", RADICAL_DATASET),
    ):
        for repetition in range(1, args.repetitions + 1):
            for method in ("gm", "rb1", "rb2"):
                stem = f"{suite}-{method}-run-{repetition:02d}"
                generated = output_dir / f"{stem}-generated.jsonl.gz"
                generation_summary = output_dir / f"{stem}-generation.json"
                report = output_dir / f"{stem}.json"
                cases = output_dir / f"{stem}-cases.jsonl.gz"
                targets = (generated, generation_summary, report, cases)
                if not args.force and any(path.exists() for path in targets):
                    raise FileExistsError(f"Refusing to overwrite {stem}; pass --force")
                command = [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    args.conda_env,
                    "python",
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--suite",
                    suite,
                    "--dataset",
                    str(dataset),
                    "--methods",
                    method,
                    "--case-timeout",
                    str(args.case_timeout),
                    "--progress-every",
                    str(args.progress_every),
                    "--output",
                    str(generated),
                    "--summary",
                    str(generation_summary),
                ]
                if args.limit is not None:
                    command.extend(["--limit", str(args.limit)])
                print(
                    f"Running external {suite}/{method} repetition "
                    f"{repetition}/{args.repetitions}",
                    flush=True,
                )
                subprocess.run(command, cwd=ROOT, env=child_env, check=True)
                evaluate_generation(
                    suite=suite,
                    dataset=dataset,
                    generated=generated,
                    generation_summary=generation_summary,
                    report_path=report,
                    cases_path=cases,
                )
                reports.append(report)

    aggregate = output_dir / "aggregate.json"
    aggregate_reports(reports, aggregate)
    payload = json.loads(aggregate.read_text())
    payload["external_environment"] = {
        "conda_env": args.conda_env,
        "partialaams_path": str(partialaams),
        "partialaams_commit": git_commit(partialaams),
        "gmapache_path": str(gmapache),
        "gmapache_commit": git_commit(gmapache),
        "historical_csv_container_commit": "edfbcbec2f2ea635f2148236599a14a59b1f1710",
    }
    aggregate.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote aggregate report: {aggregate}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(worker_main(sys.argv[2:]))
    raise SystemExit(main())
