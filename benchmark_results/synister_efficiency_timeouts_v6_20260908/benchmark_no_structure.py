#!/usr/bin/env python3
"""Compare isolated implementations on an explicit, frozen timeout cohort."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"


def implementation_hash(root):
    digest = hashlib.sha256()
    for path in sorted((root / "synkit").rglob("*.py")):
        name = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(name).to_bytes(4, "little"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def run_task(task, row, source_root, seconds):
    sys.path.insert(0, source_root)
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    from synkit.Chem.Mapper import (
        GlobalShellConfig,
        analyze_reference_blinded_global_shell,
        blinded_mapped_reaction_problem,
    )

    started = time.perf_counter()
    record = dict(task)
    try:
        reaction = row["mapped_reaction"]
        if "|" in reaction:
            reaction, identifier = reaction.rsplit("|", 1)
            if identifier != row["reaction_id"]:
                raise ValueError("reaction provenance mismatch")
        problem = blinded_mapped_reaction_problem(
            reaction, heavy_only=True, blind_seed="synister-global-v1"
        )
        result = analyze_reference_blinded_global_shell(
            problem.lgp,
            problem.reference_mapping,
            target_mode=task["mode"],
            config=GlobalShellConfig(time_limit_seconds=seconds, structure_analysis=False),
        )
        record["result"] = result.as_dict()
    except Exception as error:  # noqa: BLE001 -- preserve explicit per-task failures
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["wall_seconds"] = time.perf_counter() - started
    record["worker_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.workers < 1:
        parser.error("seconds and workers must be positive")
    source = args.source_root.resolve()
    if not (source / "synkit" / "__init__.py").is_file():
        parser.error("source root must contain the synkit package")
    selection = json.loads(args.selection.read_text())
    tasks = selection["tasks"]
    if not tasks or any(task["historical_status"] != "timeout" for task in tasks):
        parser.error("every selected shell must be a historical timeout")
    keys = [(task["source_line"], task["mode"]) for task in tasks]
    if len(set(keys)) != len(keys):
        parser.error("duplicate shell tasks")
    dataset = Path(selection["dataset"])
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != selection["dataset_sha256"]:
        parser.error("frozen dataset checksum mismatch")
    with gzip.open(dataset, "rt") as stream:
        rows = {int(row["source_line"]): row for row in csv.DictReader(stream)}
    for task in tasks:
        if rows[task["source_line"]]["reaction_id"] != task["reaction_id"]:
            parser.error("selected reaction identity mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[: args.workers]
    os.sched_setaffinity(0, cpus)
    manifest = {
        "source_root": str(source),
        "implementation_sha256": implementation_hash(source),
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "python": sys.executable,
        "seconds_per_shell": args.seconds,
        "workers": args.workers,
        "cpus": cpus,
        "memory_limit_per_worker_bytes": 4 * 1024**3,
        "tasks": len(tasks),
        "configuration": "Diagnostic only: structure_analysis=False; defaults except time_limit_seconds",
        "rss_scope": "Worker lifetime high-water mark, not an isolated per-task peak",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(
                run_task, task, rows[task["source_line"]], str(source), args.seconds
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            name = f"{record['source_line']}_{record['mode']}.json"
            (args.output / name).write_text(json.dumps(record, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "done": len(records),
                        "total": len(tasks),
                        "source_line": record["source_line"],
                        "mode": record["mode"],
                        "status": record.get("result", {}).get("status", "error"),
                        "wall_seconds": record["wall_seconds"],
                    }
                ),
                flush=True,
            )
    unchanged = implementation_hash(source) == manifest["implementation_sha256"]
    summary = {
        "tasks": len(records),
        "complete": sum(
            record.get("result", {}).get("complete", False) for record in records
        ),
        "errors": sum("error" in record for record in records),
        "wall_seconds": time.perf_counter() - started,
        "implementation_unchanged": unchanged,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if unchanged and not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
