"""Controlled output-size experiments, with three repeats and analytic counts.

These disconnected atoms, repeated edges and paths are mathematical graph
tests, not chemically representative reactions. Both variants export all
indexed maps; the symmetry-enabled variant expands its verified subgroup.
"""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from math import factorial
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from Experiment.Synister.ablation_benchmark import compare_records
from Experiment.Synister.enumeration_benchmark import encode


def families():
    rows = []
    def append(name, family, parameter, numbers, bonds, expected):
        n = len(numbers)
        endpoint = {"atomic_numbers": numbers, "charges": [0]*n, "hcounts": [0]*n, "bonds": bonds}
        rows.append({"benchmark_id": name, "family": family, "parameter": parameter,
                     "reactant": endpoint, "product": endpoint, "query": "zero_distance",
                     "target_doubled_cd": 0, "expected_indexed_maps": expected})
    distinct = [z for z in range(2, 119) if z != 6]
    for k in range(2, 10):
        append(f"fixed12_k{k}", "fixed_atom_count_variable_output", k,
               [6]*k+distinct[:12-k], [], factorial(k))
    for n in (4, 8, 12, 24, 48, 72, 96):
        append(f"fixed2_n{n}", "fixed_output_variable_atom_count", n,
               [6, 6]+distinct[:n-2], [], 2)
    for k in range(1, 8):
        append(f"edges_k{k}", "repeated_two_atom_components", k,
               [6]*(2*k), [[2*i, 2*i+1, 2] for i in range(k)], 2**k*factorial(k))
    for n in (6, 12, 24, 48):
        append(f"path_n{n}", "connected_path_control", n, [6]*n,
               [[i, i+1, 2] for i in range(n-1)], 2)
    return rows


def make_tasks(rows, *, repeats=3, seconds=15, memory_gib=6, max_maps=100000):
    result = []
    for repeat in range(repeats):
        for index, row in enumerate(rows):
            order = ("full", "no_symmetry") if (repeat+index) % 2 == 0 else ("no_symmetry", "full")
            for variant in order:
                result.append({"task_id": f'{row["benchmark_id"]}.zero_distance.r{repeat}.{variant}',
                               "benchmark_id": row["benchmark_id"],
                               "reactant": row["reactant"], "product": row["product"],
                               "query": "zero_distance", "repeat": repeat, "variant": variant,
                               "target_doubled_cd": 0, "seconds": seconds,
                               "memory_gib": memory_gib, "max_maps": max_maps})
    return result


def execute(task, directory):
    path = directory/"cases"/f'{task["task_id"]}.json'
    map_path = directory/"maps"/path.name
    if path.exists() or map_path.exists():
        raise FileExistsError(path)
    task = {**task, "map_path": str(map_path.resolve())}
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONHASHSEED="0")
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, "-m", "Experiment.Synister.scaling_worker"],
                                 input=encode(task), capture_output=True, text=True, env=env,
                                 timeout=task["seconds"]+15)
        if process.returncode:
            record = {"complete": False, "minimum_proved": False, "termination": "process_failure",
                      "returncode": process.returncode, "stderr": process.stderr[-4000:]}
        else:
            record = json.loads(process.stdout)
            record["stderr"] = process.stderr[-4000:]
    except subprocess.TimeoutExpired:
        record = {"complete": False, "minimum_proved": False, "termination": "external_time_limit"}
    record.update(task=task, parent_seconds=time.perf_counter()-started)
    with path.open("x") as stream:
        stream.write(encode(record))
    print(json.dumps({"task": task["task_id"], "complete": record["complete"],
                      "termination": record["termination"], "seconds": record["parent_seconds"]}), flush=True)
    return record


def analytic_checks(rows, records):
    lookup = {row["benchmark_id"]: row for row in rows}
    checks = []
    for record in records:
        expected = lookup[record["task"]["benchmark_id"]]["expected_indexed_maps"]
        count = record.get("mapping_count", 0)
        valid = count <= expected and (not record["complete"] or count == expected)
        checks.append({"task_id": record["task"]["task_id"], "expected": expected,
                       "returned": count, "complete": record["complete"], "consistent": valid})
    return checks


def run(args):
    if args.after is not None:
        previous = json.loads((args.after/"audit.json").read_text())
        if not previous["all_output_comparisons_consistent"]:
            raise ValueError("Preceding study has inconsistent outputs")
    rows = families()
    tasks = make_tasks(rows, repeats=args.repeats, seconds=args.seconds,
                       memory_gib=args.memory_gib, max_maps=args.max_maps)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"cases").mkdir()
    (args.output/"maps").mkdir()
    paths = sorted(Path("synkit/Chem/Mapper").rglob("*.py"))
    paths += [Path(__file__), Path("Experiment/Synister/ablation_worker.py"),
              Path("Experiment/Synister/scaling_worker.py"),
              Path("Experiment/Synister/ablation_benchmark.py"), Path("Experiment/Synister/global_milp.py"),
              Path("Experiment/Synister/audit_enumeration_benchmark.py"), Path("Experiment/Synister/enumeration_benchmark.py")]
    saved = {"inputs.json": rows, "tasks.json": tasks,
             "sources.json": {str(p): p.read_text() for p in paths}}
    for name, value in saved.items():
        (args.output/name).write_text(encode(value))
    manifest = {"schema": "synister.output-scaling.v1", "selected": len(rows), "attempts": len(tasks),
                "repeats": args.repeats, "seconds": args.seconds, "external_seconds": args.seconds+15,
                "memory_gib": args.memory_gib, "max_maps": args.max_maps, "workers": 1,
                "threads_per_worker": 1, "variants": ["full", "no_symmetry"],
                "python": sys.version, "platform": platform.platform(),
                "dependencies": {name: version(name) for name in ("numpy", "scipy", "networkx", "rdkit")},
                "file_sha256": {name: sha256((args.output/name).read_bytes()).hexdigest() for name in saved},
                "after": str(args.after) if args.after is not None else None,
                "after_audit_sha256": sha256((args.after/"audit.json").read_bytes()).hexdigest() if args.after else None,
                "output_unit": "all_indexed_atom_maps", "atom_order": "Same saved endpoint order in every repeat",
                "input_scope": "Synthetic Endpoint graphs; no chemical SMILES parser or valence interpretation",
                "variant_order": "Alternates by repeat and case index; every attempt retained",
                "seed": "Both variants recompute the same feasible heuristic; seed time included",
                "count_proofs": {
                    "fixed_atom_count_variable_output": "k! permutations of the k identical isolated carbons; other elements unique",
                    "fixed_output_variable_atom_count": "2! permutations of the two isolated carbons; other elements unique",
                    "repeated_two_atom_components": "k! component permutations and 2^k independent edge reversals",
                    "connected_path_control": "Identity and reversal are the only automorphisms of a path with at least two vertices"},
                "scope": "Mathematical stress tests, not representative chemical reactions or asymptotic runtime proofs"}
    (args.output/"manifest.json").write_text(encode(manifest))
    # Serial attempts support interpretable repeated timings without competing
    # experiment workers; unrelated operating-system activity is not excluded.
    records = [execute(task, args.output) for task in tasks]
    comparisons = compare_records(records, args.output)
    checks = analytic_checks(rows, records)
    summary = {"selected": len(rows), "attempts": len(records), "repeats": args.repeats,
               "by_variant": {v: dict(Counter("complete" if r["complete"] else r["termination"]
                                               for r in records if r["task"]["variant"] == v))
                              for v in ("full", "no_symmetry")},
               "analytic_count_checks": checks, "comparisons": comparisons,
               "all_consistent": all(c["consistent"] for c in checks+comparisons),
               "all_record_hashes": {p.name: sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((args.output/"cases").glob("*.json"))}}
    (args.output/"summary.json").write_text(encode(summary))
    print(encode({k: v for k, v in summary.items() if k not in ("analytic_count_checks", "comparisons", "all_record_hashes")}))
    return summary


def audit(directory):
    from synkit.Chem.Mapper.identifiability import extract_label
    from Experiment.Synister.scaling_worker import endpoint
    manifest = json.loads((directory/"manifest.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    for name, expected in manifest["file_sha256"].items():
        if sha256((directory/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Changed input, task plan or source snapshot")
    rows = json.loads((directory/"inputs.json").read_text())
    if rows != families():
        raise ValueError("Synthetic families differ from the declared construction")
    tasks = json.loads((directory/"tasks.json").read_text())
    if tasks != make_tasks(rows, repeats=manifest["repeats"], seconds=manifest["seconds"],
                           memory_gib=manifest["memory_gib"], max_maps=manifest["max_maps"]):
        raise ValueError("Task accounting differs from protocol")
    expected = {t["task_id"]: t for t in tasks}
    paths = sorted((directory/"cases").glob("*.json"))
    if {p.stem for p in paths} != set(expected):
        raise ValueError("Missing or unexpected attempted tasks")
    records, checked = [], 0
    for path in paths:
        if sha256(path.read_bytes()).hexdigest() != summary["all_record_hashes"][path.name]:
            raise ValueError("Changed attempt record")
        record = json.loads(path.read_text())
        task = record["task"]
        if {k: task[k] for k in expected[path.stem]} != expected[path.stem]:
            raise ValueError("Executed task differs from plan")
        if "mapping_sha256" in record:
            map_path = directory/"maps"/path.name
            output = json.loads(map_path.read_text())
            if sha256(map_path.read_bytes()).hexdigest() != record["mapping_sha256"] or map_path.stat().st_size != record["output_bytes"]:
                raise ValueError("Changed mapping output")
            if len(output) != record["mapping_count"] or len(set(map(tuple, output))) != len(output):
                raise ValueError("Duplicate map or wrong count")
            r, p = endpoint(task["reactant"]), endpoint(task["product"])
            for mapping in output:
                if extract_label(r, p, mapping).weighted_distance != 0:
                    raise ValueError("Exported map is not at CD zero")
            checked += len(output)
        elif record["complete"]:
            raise ValueError("Complete result lacks a mapping file")
        records.append(record)
    checks = analytic_checks(rows, records)
    comparisons = compare_records(records, directory)
    if len(records) != summary["attempts"] or len(rows) != summary["selected"]:
        raise ValueError("Summary denominator mismatch")
    by_variant = {v: dict(Counter("complete" if r["complete"] else r["termination"]
                                  for r in records if r["task"]["variant"] == v))
                  for v in ("full", "no_symmetry")}
    if by_variant != summary["by_variant"]:
        raise ValueError("Completion accounting mismatch")
    return {"all_verified": all(c["consistent"] for c in checks+comparisons),
            "selected": len(rows), "attempts": len(records), "validated_saved_maps": checked,
            "by_variant": by_variant, "summary_sha256": sha256((directory/"summary.json").read_bytes()).hexdigest(),
            "auditor_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--memory-gib", type=int, default=6)
    parser.add_argument("--max-maps", type=int, default=100000)
    args = parser.parse_args()
    if args.audit:
        report = audit(args.audit)
        with args.output.open("x") as stream:
            stream.write(encode(report))
        print(encode(report))
        raise SystemExit(0 if report["all_verified"] else 1)
    if min(args.repeats, args.seconds, args.memory_gib, args.max_maps) <= 0:
        parser.error("Counts and limits must be positive")
    raise SystemExit(0 if run(args)["all_consistent"] else 1)
