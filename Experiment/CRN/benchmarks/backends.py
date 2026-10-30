#!/usr/bin/env python
"""Compare complete Python/C++ CRN enumeration on identical biochemical inputs."""

from pathlib import Path
import argparse
import hashlib
import json
import multiprocessing as mp
from statistics import median
from time import perf_counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Experiment.CRN.tools.common import environment, source_hashes, write_report
from synkit.CRN import (
    build_native,
    enumerate_structural_sets,
    enumerate_semiflows,
    read_sbml,
)
from synkit.CRN.Petrinet.minimal_semiflows import minimal_semiflows
from synkit.CRN.Props.stoich import stoichiometric_matrix
from Experiment.CRN.benchmarks import CASE_STUDY_MODULES, build_kegg_crn
from Experiment.CRN.benchmarks.scaling import generate_network

TASKS = ("siphons", "traps", "p_semiflows", "t_semiflows")


def _worker(connection, network, backend, task, library, budget, max_nodes):
    connection.send("ready")
    started = perf_counter()
    try:
        if task in {"siphons", "traps"}:
            if backend == "cpp":
                outcome = enumerate_structural_sets(
                    network,
                    kind="siphon" if task == "siphons" else "trap",
                    backend=backend,
                    library_path=library,
                    time_limit=budget,
                    max_nodes=max_nodes,
                )
                items = outcome.value["sets"]
            else:
                # Compare with the production set-based implementation, not the
                # slower experimental NumPy bitset reference adapter.
                from synkit.CRN.Petrinet.structure import find_siphons, find_traps
                from synkit.CRN.Petrinet.siphon_search import SiphonSearchLimit
                from synkit.CRN.analysis import AnalysisResult

                try:
                    groups = (find_siphons if task == "siphons" else find_traps)(
                        network, names="id", strict_limit=True, max_nodes=max_nodes
                    )
                    items = sorted(sorted(group) for group in groups)
                    outcome = AnalysisResult(
                        {"sets": items}, method="production Python set closure"
                    )
                except SiphonSearchLimit:
                    items = []
                    outcome = AnalysisResult(
                        {"sets": []},
                        "inconclusive",
                        False,
                        termination_reason="node_limit",
                    )
        else:
            matrix = stoichiometric_matrix(network)
            kind = "p" if task == "p_semiflows" else "t"
            if backend == "cpp":
                outcome = enumerate_semiflows(
                    matrix, kind=kind, library_path=library, time_limit=budget
                )
                items = outcome.value["flows"]
            else:
                from synkit.CRN.analysis import AnalysisResult

                items = minimal_semiflows(matrix, kind=kind, backend="python")
                outcome = AnalysisResult({"flows": items})
        digest = hashlib.sha256(
            json.dumps(
                sorted(tuple(item) for item in items), separators=(",", ":")
            ).encode()
        ).hexdigest()
        details = {k: v for k, v in outcome.value.items() if k not in {"flows", "sets"}}
        connection.send(
            {
                "seconds": perf_counter() - started,
                "complete": outcome.complete,
                "termination_reason": outcome.termination_reason,
                "error": None,
                "result_count": len(items),
                "result_sha256": digest,
                "statistics": details,
            }
        )
    except Exception as error:
        connection.send(
            {
                "seconds": None,
                "complete": False,
                "termination_reason": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    finally:
        connection.close()


def _sample(network, backend, task, library, budget, max_nodes):
    # A parent process enforces the Python budget too; results sent over the
    # pipe contain only summaries, so large cones cannot deadlock pipe writes.
    context = mp.get_context(
        "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    )
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(sender, network, backend, task, library, budget, max_nodes),
    )
    started = perf_counter()
    process.start()
    sender.close()
    try:
        if not receiver.poll(30):
            raise TimeoutError("worker startup")
        if receiver.recv() != "ready":
            raise RuntimeError("worker handshake")
        startup = perf_counter() - started
        # Native search has its own deadline; allow bounded serialization and
        # independent witness verification after that deadline.
        if receiver.poll(budget + (1.0 if backend == "cpp" else 0.05)):
            result = receiver.recv()
        else:
            result = {
                "seconds": None,
                "complete": False,
                "termination_reason": "wall_time_limit",
                "error": None,
            }
        result["worker_startup_seconds"] = startup
        return result
    except (EOFError, TimeoutError, RuntimeError) as exc:
        return {
            "seconds": None,
            "complete": False,
            "termination_reason": "worker_failure",
            "error": str(exc),
        }
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        receiver.close()


def benchmark_report(
    *,
    library=None,
    biomodels_dir=None,
    repeats=3,
    budget=5.0,
    max_nodes=200000,
    tasks=("p_semiflows", "siphons"),
    synthetic_sizes=(),
    modules=CASE_STUDY_MODULES,
):
    if (
        repeats < 1
        or budget <= 0
        or max_nodes < 1
        or not tasks
        or any(task not in TASKS for task in tasks)
        or any(size < 1 for size in synthetic_sizes)
        or any(mid not in CASE_STUDY_MODULES for mid in modules)
    ):
        raise ValueError("Invalid repeats, budget, node limit, tasks, sizes or modules")
    networks = [
        (mid, lambda mid=mid: build_kegg_crn(mid, drop_currency=False))
        for mid in modules
    ]
    if biomodels_dir is not None:
        networks += [
            (p.stem, lambda p=p: read_sbml(p))
            for p in sorted(Path(biomodels_dir).glob("*.xml"))
        ]
    networks += [
        (f"reversible_chain_{n}", lambda n=n: generate_network("reversible_chain", n))
        for n in synthetic_sizes
    ]
    rows, comparisons = [], []
    checks = {"completed_backends_agree": True, "no_backend_errors": True}
    backends = ["python", "cpp"] if library is not None else ["python"]
    for name, build in networks:
        started = perf_counter()
        network = build()
        construction_seconds = perf_counter() - started
        if not network.n_reactions:
            continue
        for task in tasks:
            samples = {backend: [] for backend in backends}
            for repeat in range(repeats):
                for backend in (
                    backends if repeat % 2 == 0 else list(reversed(backends))
                ):
                    samples[backend].append(
                        _sample(network, backend, task, library, budget, max_nodes)
                    )
            complete_digests = set()
            medians = {}
            for backend in backends:
                values = samples[backend]
                complete_digests.update(
                    row["result_sha256"] for row in values if row["complete"]
                )
                checks["no_backend_errors"] &= all(
                    row.get("error") is None for row in values
                )
                finished = all(row["complete"] for row in values)
                elapsed = median(row["seconds"] for row in values) if finished else None
                medians[backend] = elapsed
                rows.append(
                    {
                        "network": name,
                        "task": task,
                        "n_species": network.n_species,
                        "n_reactions": network.n_reactions,
                        "backend": backend,
                        "construction_seconds": construction_seconds,
                        "samples": values,
                        "median_seconds": elapsed,
                        "all_samples_complete": finished,
                    }
                )
            checks["completed_backends_agree"] &= len(complete_digests) <= 1
            python_seconds, cpp_seconds = medians.get("python"), medians.get("cpp")
            comparisons.append(
                {
                    "network": name,
                    "task": task,
                    "speedup": (
                        python_seconds / cpp_seconds
                        if python_seconds and cpp_seconds and len(complete_digests) <= 1
                        else None
                    ),
                    "reason": (
                        "output_mismatch"
                        if len(complete_digests) > 1
                        else (
                            "complete_comparison"
                            if python_seconds and cpp_seconds
                            else "incomplete_comparison"
                        )
                    ),
                }
            )
    complete = bool(rows) and all(row["all_samples_complete"] for row in rows)
    checks["all_measurements_complete"] = complete
    return {
        "schema": "synkit.crn-backend-benchmark/2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "complete": complete,
        "environment": environment(),
        "source_sha256": source_hashes(),
        "native_manifest": (
            json.loads(Path(library).with_suffix(".json").read_text())
            if library is not None
            else None
        ),
        "checks": checks,
        "measurements": rows,
        "comparisons": comparisons,
        "parameters": {
            "repeats": repeats,
            "budget": budget,
            "max_nodes": max_nodes,
            "tasks": list(tasks),
            "synthetic_sizes": list(synthetic_sizes),
            "modules": list(modules),
            "order": "alternating; isolated workers; startup excluded",
        },
        "claim_boundary": "Preparation, API overhead and exact native witness checks included. Ratios require every repeat of both backends to complete and outputs to agree. Limits are not speedups.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--build-native", type=Path)
    parser.add_argument("--boost-include-dir", type=Path)
    parser.add_argument(
        "--biomodels-dir", type=Path, help="local XML cache; never downloads"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=float, default=5)
    parser.add_argument("--max-nodes", type=int, default=200000)
    parser.add_argument(
        "--tasks", nargs="+", choices=TASKS, default=["p_semiflows", "siphons"]
    )
    parser.add_argument("--synthetic-sizes", nargs="*", type=int, default=[])
    parser.add_argument(
        "--modules",
        nargs="*",
        choices=CASE_STUDY_MODULES,
        default=list(CASE_STUDY_MODULES),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    library = (
        build_native(
            args.build_native,
            boost_include_dir=args.boost_include_dir,
            with_semiflows=any("semiflows" in t for t in args.tasks),
        )
        if args.build_native
        else args.library
    )
    raise SystemExit(
        write_report(
            benchmark_report(
                library=library,
                biomodels_dir=args.biomodels_dir,
                repeats=args.repeats,
                budget=args.budget,
                max_nodes=args.max_nodes,
                tasks=args.tasks,
                synthetic_sizes=args.synthetic_sizes,
                modules=args.modules,
            ),
            args.output,
        )
    )
