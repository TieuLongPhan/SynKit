"""Serial repeated timings of selected development inputs using frozen solvers."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from Experiment.Synister.propagation_phases import publish


def measure(task):
    """Counterbalance repeated calls, checking each complete indexed output."""
    os.sched_setaffinity(0, {task["cpu"]})
    from synkit.Chem.Mapper.identifiability import parse_reaction
    from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
    from synkit.Chem.Mapper.exact.propagation import enumerate_synister_cp_mappings
    from Experiment.Synister.mapping_check import check_mappings

    r, p = parse_reaction(task["reaction"])
    functions = {
        "legacy": enumerate_distance_mappings,
        "synister_cp": enumerate_synister_cp_mappings,
    }
    trials = []
    for repetition in range(task["repeats"]):
        order = (
            ["legacy", "synister_cp"]
            if (task["index"] + repetition) % 2 == 0
            else ["synister_cp", "legacy"]
        )
        trial, sets = {}, {}
        for name in order:
            maps = []
            before = time.perf_counter()
            result = functions[name](
                [r.graph(), p.graph()],
                CD="minimal",
                binary=False,
                initial_mapping=task["seed"],
                max_bijections=None,
                max_mappings=100000,
                time_limit_seconds=60,
                tolerance=0,
                symmetry_pruning=True,
                expand_symmetry=True,
                symmetry_node_properties=("charges", "hcounts"),
                collect_mappings=False,
                mapping_callback=lambda mapping, cost: maps.append(tuple(mapping)),
            )
            elapsed = time.perf_counter() - before
            minimum = (
                None if result.minimum_cost is None else int(2 * result.minimum_cost)
            )
            if not result.complete:
                raise ValueError(
                    "Serial timing selection must complete for both engines"
                )
            if len(set(maps)) != len(maps):
                raise ValueError("Duplicate mapping")
            check_mappings(r, p, maps, minimum)
            sets[name] = set(maps)
            trial[name] = {
                "seconds": elapsed,
                "minimum_doubled_cd": minimum,
                "mapping_count": len(maps),
                "complete": True,
            }
        if (
            sets["legacy"] != sets["synister_cp"]
            or trial["legacy"]["minimum_doubled_cd"]
            != trial["synister_cp"]["minimum_doubled_cd"]
        ):
            raise ValueError("Repeated complete outputs disagree")
        trials.append(trial)
    medians = {
        name: statistics.median(t[name]["seconds"] for t in trials)
        for name in functions
    }
    return {
        "benchmark_id": task["benchmark_id"],
        "trials": trials,
        "median_seconds": medians,
        "median_legacy_over_new": medians["legacy"] / medians["synister_cp"],
    }


def run(comparison, output, cases, repeats):
    """Bind serial measurements to an already audited immutable solver tree."""
    comparison, output = comparison.resolve(), output.resolve()
    protocol = json.loads((comparison / "protocol.json").read_text())
    audit = json.loads((comparison / "audit.json").read_text())
    if not audit["all_recorded_outputs_consistent"]:
        raise ValueError("Comparison has not passed its independent audit")
    source = comparison / "frozen_source"
    for name, expected in protocol["source_sha256"].items():
        if sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source changed: " + name)
    rows = {
        row["benchmark_id"]: row
        for row in json.loads((comparison / "inputs.json").read_text())
    }
    seeds = json.loads((comparison / "prepared_seeds.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    script = output / "worker.py"
    shutil.copyfile(__file__, script)
    cpu = min(os.sched_getaffinity(0))
    publish(
        output / "protocol.json",
        {
            "comparison": str(comparison),
            "cases": cases,
            "repeats": repeats,
            "cpu": cpu,
            "single_thread": True,
            "order": "counterbalanced by input/repetition parity",
            "selection": "retrospective development timing diagnostic, not an independent cohort",
            "comparison_protocol_sha256": sha256(
                (comparison / "protocol.json").read_bytes()
            ).hexdigest(),
            "comparison_audit_sha256": sha256(
                (comparison / "audit.json").read_bytes()
            ).hexdigest(),
            "script_sha256": sha256(script.read_bytes()).hexdigest(),
        },
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(source),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    results = []
    for index, key in enumerate(cases):
        task = dict(
            rows[key], seed=seeds[key]["mapping"], cpu=cpu, repeats=repeats, index=index
        )
        process = subprocess.run(
            [sys.executable, str(script), "--worker"],
            input=json.dumps(task),
            text=True,
            capture_output=True,
            cwd=source,
            env=env,
            timeout=repeats * 150,
        )
        if process.returncode:
            raise RuntimeError(process.stderr[-4000:])
        result = json.loads(process.stdout)
        publish(output / (key + ".json"), result)
        results.append(result)
        print(
            json.dumps(
                {
                    "benchmark_id": key,
                    "median_legacy_over_new": result["median_legacy_over_new"],
                }
            ),
            flush=True,
        )
    summary = {
        "inputs": len(results),
        "repeats": repeats,
        "median_paired_ratio": statistics.median(
            r["median_legacy_over_new"] for r in results
        ),
        "median_seconds_sum": {
            name: sum(r["median_seconds"][name] for r in results)
            for name in ("legacy", "synister_cp")
        },
    }
    publish(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(measure(json.load(sys.stdin))))
    else:
        run(args.comparison, args.directory, args.cases, args.repeats)
