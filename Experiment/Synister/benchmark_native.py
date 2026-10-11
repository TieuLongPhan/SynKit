#!/usr/bin/env python3
"""Run explicit mixed-mode selections with the optional native orbit backend."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"


def source_hash(root):
    digest = hashlib.sha256()
    paths = sorted(
        path for path in (root / "synkit").rglob("*") if path.suffix in {".py", ".cpp"}
    )
    for path in paths:
        name = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(name).to_bytes(4, "little"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mapping-cap", type=int, default=100000)
    parser.add_argument(
        "--scheduler", choices=("static", "frontier"), default="frontier"
    )
    parser.add_argument("--slice-nodes", type=int, default=8192)
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Write the identical public result without indentation or separator whitespace",
    )
    parser.add_argument(
        "--wall-budget",
        action="store_true",
        help="Start the search deadline before case preparation; full output time is audited separately",
    )
    parser.add_argument("--immutable-json", action="store_true",
        help="Serialize immutable class-count tuples directly; identical public JSON")
    args = parser.parse_args()
    source = args.source_root.resolve()
    if not (source / "synkit/__init__.py").exists():
        parser.error("source root must contain synkit")
    selection = json.loads(args.selection.read_text())
    if any(task["mode"] not in {"minimal", "reference_cd"} for task in selection["tasks"]):
        parser.error("unsupported target mode")
    dataset = Path(selection["dataset"])
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != selection["dataset_sha256"]:
        parser.error("dataset hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*.json")):
        parser.error("output directory already contains records")
    sys.path.insert(0, str(source))
    from synkit.Chem.Mapper import GlobalShellConfig, blinded_mapped_reaction_problem
    from synkit.Chem.Mapper.native_analysis import (
        analyze_reference_blinded_native_shell,
    )

    with gzip.open(dataset, "rt") as stream:
        rows = {int(row["source_line"]): row for row in csv.DictReader(stream)}
    initial_hash = source_hash(source)
    manifest = {
        "source_root": str(source),
        "benchmark_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "native_build_manifest": str(args.library.with_suffix(".json").resolve()),
        "native_build_manifest_sha256": (
            hashlib.sha256(args.library.with_suffix(".json").read_bytes()).hexdigest()
            if args.library.with_suffix(".json").exists()
            else None
        ),
        "diagnostic_environment": {
            name: os.environ.get(name)
            for name in (
                "SYNKIT_NATIVE_PROFILE",
                "SYNKIT_NATIVE_PATTERN_CACHE",
                "GLIBC_TUNABLES",
                "LD_PRELOAD",
            )
        },
        "source_sha256_python_and_cpp": initial_hash,
        "library": str(args.library.resolve()),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "dataset_sha256": selection["dataset_sha256"],
        "workers_per_case": args.workers,
        "coordinator_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "python_version": sys.version,
        "analysis_wall_scope": "before case construction through complete public result",
        "end_to_end_wall_scope": "before case construction through JSON serialization and file close",
        "search_deadline_scope": "case preparation onward"
        if args.wall_budget
        else "worker readiness onward",
        "seconds": args.seconds,
        "mapping_cap": args.mapping_cap,
        "scheduler": args.scheduler,
        "slice_nodes": args.slice_nodes,
        "mapping_cap_scope": "retained_worker_double_orbit_representatives",
        "absolute_case_deadline": args.wall_budget,
        "output_json_format": "compact" if args.compact_json else "indent=2",
        "immutable_json_payload": args.immutable_json,
        "time_scope": (
            "deadline starts before case preparation; output completion separately measured"
            if args.wall_budget
            else "search, classification, transfer, exact merge; excludes setup and public result formatting"
        ),
        "worker_address_space_bytes": 4 * 1024**3,
        "parent_address_space_limit": None,
        "tasks_run_sequentially": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    case_timings = []
    for task in selection["tasks"]:
        started = time.perf_counter()
        cpu_started = time.process_time()
        child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        record = dict(task)
        try:
            row = rows[task["source_line"]]
            if row["reaction_id"] != task["reaction_id"]:
                raise ValueError("reaction identifier mismatch")
            reaction = row["mapped_reaction"]
            if "|" in reaction:
                reaction, identifier = reaction.rsplit("|", 1)
                if identifier != row["reaction_id"]:
                    raise ValueError("reaction suffix mismatch")
            problem = blinded_mapped_reaction_problem(
                reaction,
                heavy_only=True,
                blind_seed="synister-global-v1",
            )
            result = analyze_reference_blinded_native_shell(
                problem.lgp,
                problem.reference_mapping,
                library_path=args.library,
                target_mode=task["mode"],
                workers=args.workers,
                scheduler=args.scheduler,
                slice_nodes=args.slice_nodes,
                **({"deadline": started + args.seconds} if args.wall_budget else {}),
                config=GlobalShellConfig(
                    time_limit_seconds=args.seconds, max_mappings=args.mapping_cap
                ),
            )
            record["result"] = result.as_dict(copy_sequences=not args.immutable_json)
        except Exception as error:  # noqa: BLE001 -- retain each explicit benchmark failure
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            if hasattr(error, "diagnostics"):
                record["error"]["diagnostics"] = error.diagnostics
        child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        record["aggregate_cpu_seconds"] = (
            time.process_time() - cpu_started
            + child_after.ru_utime + child_after.ru_stime
            - child_before.ru_utime - child_before.ru_stime
        )
        record["wall_seconds"] = time.perf_counter() - started
        record["parent_peak_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        record["child_peak_rss_kib"] = resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss
        record["source_sha256_python_and_cpp"] = initial_hash
        name = f"{task['source_line']}_{task['mode']}.json"
        serialization_started = time.perf_counter()
        payload = (
            json.dumps(record, separators=(",", ":"))
            if args.compact_json
            else json.dumps(record, indent=2)
        ) + "\n"
        serialized = time.perf_counter()
        (args.output / name).write_text(payload)
        closed = time.perf_counter()
        case_timings.append(
            {
                "source_line": task["source_line"],
                "mode": task["mode"],
                "analysis_wall_seconds": record["wall_seconds"],
                "serialization_seconds": serialized - serialization_started,
                "write_close_seconds": closed - serialized,
                "end_to_end_wall_seconds": closed - started,
                "output_bytes": len(payload.encode()),
                "complete_below_60_seconds": bool(
                    record.get("result", {}).get("complete")
                )
                and closed - started < 60,
            }
        )
        (args.output / "case_timings.json").write_text(
            json.dumps(case_timings, indent=2) + "\n"
        )
        result = record.get("result", {})
        print(
            json.dumps(
                {
                    "source_line": task["source_line"],
                "mode": task["mode"],
                    "complete": result.get("complete"),
                    "representative_count": result.get("representative_solution_count"),
                    "wall_seconds": record["wall_seconds"],
                    "end_to_end_wall_seconds": case_timings[-1][
                        "end_to_end_wall_seconds"
                    ],
                    "error": record.get("error"),
                }
            ),
            flush=True,
        )
    manifest["source_unchanged"] = source_hash(source) == initial_hash
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["source_unchanged"]:
        raise RuntimeError("implementation changed during benchmark")


if __name__ == "__main__":
    main()
