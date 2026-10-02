"""Input-selected five-configuration experiment; no historical result is overwritten."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from Experiment.Synister.ablation_worker import CONFIGURATIONS
from Experiment.Synister.audit_enumeration_benchmark import compare_outputs
from Experiment.Synister.enumeration_benchmark import encode


def select_inputs(directory, per_bin):
    """Subset the matched experiment by input hash, never by completion."""
    buckets = defaultdict(list)
    for row in json.loads((directory/"inputs.json").read_text()):
        buckets[row["source"], row["size_bin"]].append(row)
    return [row for key in sorted(buckets)
            for row in sorted(buckets[key], key=lambda x: x["selection_sha256"])[:per_bin]]


def derive_tasks(rows, directory, *, seconds, memory_gib, max_maps, repeats=1,
                 variants=tuple(CONFIGURATIONS)):
    """Three planned queries; unavailable relative targets remain explicit."""
    tasks, unavailable = [], []
    for index, row in enumerate(rows):
        proofs = []
        for method in ("synister", "milp"):
            path = directory/"cases"/f'{row["benchmark_id"]}.minimum.{method}.json'
            record = json.loads(path.read_text())
            if record.get("minimum_proved"):
                proofs.append({"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
                               "minimum_doubled_cd": record["minimum_doubled_cd"]})
        values = {p["minimum_doubled_cd"] for p in proofs}
        if len(values) > 1:
            raise ValueError("Conflicting minimum proofs")
        queries = [("minimum", "minimal")]
        if values:
            best = next(iter(values))
            queries.extend((("at_minimum", best), ("plus_2", best+4)))
        else:
            unavailable.append({"benchmark_id": row["benchmark_id"],
                                "queries": ["at_minimum", "plus_2"],
                                "reason": "No minimum proved in the matched experiment"})
        for repeat in range(repeats):
            order = list(variants)
            shift = (index+repeat) % len(order)
            order = order[shift:]+order[:shift]
            if repeat % 2:
                order.reverse()
            for query, target in queries:
                for variant in order:
                    tasks.append({"task_id": f'{row["benchmark_id"]}.{query}.r{repeat}.{variant}',
                                  "benchmark_id": row["benchmark_id"], "reaction": row["reaction"],
                                  "query": query, "repeat": repeat, "variant": variant,
                                  "target_doubled_cd": target, "minimum_proofs": proofs if target != "minimal" else [],
                                  "seconds": seconds, "memory_gib": memory_gib, "max_maps": max_maps})
    return tasks, unavailable


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
        process = subprocess.run([sys.executable, "-m", "Experiment.Synister.ablation_worker"],
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


def compare_records(records, directory):
    groups = defaultdict(list)
    for record in records:
        task = record["task"]
        # Repeats share the mathematical query, so they must agree as well.
        groups[task["benchmark_id"], task["query"]].append(record)
    comparisons = []
    for (case, query), group in sorted(groups.items()):
        outputs = {r["task"]["task_id"]: json.loads((directory/"maps"/f'{r["task"]["task_id"]}.json').read_text())
                   if "mapping_sha256" in r else [] for r in group}
        minima = {r["minimum_doubled_cd"] for r in group if r.get("minimum_proved")}
        if len(minima) > 1:
            raise ValueError("Variants disagree on proved minima")
        for a, b in combinations(group, 2):
            aid, bid = a["task"]["task_id"], b["task"]["task_id"]
            check = compare_outputs(a, b, outputs[aid], outputs[bid])
            comparisons.append({"benchmark_id": case, "query": query, "first": aid, "second": bid, **check})
    return comparisons


def run(args):
    if not (args.matched/"summary.json").exists():
        raise ValueError("The matched experiment must finish before timed ablations begin")
    validation = getattr(args, "validation", None)
    validation_record = None
    if validation is not None:
        control = json.loads((validation/"summary.json").read_text())
        protocol = json.loads((validation/"manifest.json").read_text())
        sources = json.loads((validation/"sources.json").read_text())
        if (not control["all_passed"] or not control["queries"]
                or sha256((validation/"cases.jsonl").read_bytes()).hexdigest() != control["records_sha256"]
                or sha256((validation/"manifest.json").read_bytes()).hexdigest() != control["manifest_sha256"]
                or sha256((validation/"sources.json").read_bytes()).hexdigest() != protocol["sources_sha256"]):
            raise ValueError("Ablation validation is unsuccessful or changed")
        for path in [*Path("synkit/Chem/Mapper").rglob("*.py"),
                     Path("Experiment/Synister/ablation_worker.py"), Path("Experiment/Synister/global_milp.py")]:
            if sources.get(str(path)) != path.read_text():
                raise ValueError(f"Runtime source differs from validated variant: {path}")
        validation_record = {"directory": str(validation), "queries": control["queries"],
                             "summary_sha256": sha256((validation/"summary.json").read_bytes()).hexdigest()}
    rows = select_inputs(args.matched, args.per_bin)
    variants = tuple(args.variants)
    if not variants or len(set(variants)) != len(variants) or set(variants)-CONFIGURATIONS.keys():
        raise ValueError("Invalid configuration list")
    tasks, unavailable = derive_tasks(rows, args.matched, seconds=args.seconds,
                                      memory_gib=args.memory_gib, max_maps=args.max_maps,
                                      repeats=args.repeats, variants=variants)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("cases", "maps"):
        (args.output/name).mkdir()
    sources = sorted(Path("synkit/Chem/Mapper").rglob("*.py"))
    sources += [Path(__file__), Path("Experiment/Synister/ablation_worker.py"),
                Path("Experiment/Synister/enumeration_benchmark.py"),
                Path("Experiment/Synister/global_milp.py"),
                Path("Experiment/Synister/audit_enumeration_benchmark.py")]
    saved = {"inputs.json": rows, "tasks.json": tasks, "unavailable_queries.json": unavailable,
             "sources.json": {str(p): p.read_text() for p in sources}}
    for name, value in saved.items():
        (args.output/name).write_text(encode(value))
    manifest = {"schema": "synister.enumeration-ablations.v1", "selected": len(rows),
                "attempts": len(tasks), "planned_queries_per_reaction": 3, "variants": variants,
                "repeats": args.repeats, "seconds": args.seconds, "external_seconds": args.seconds+15,
                "memory_gib": args.memory_gib, "max_maps": args.max_maps, "workers": args.workers,
                "threads_per_worker": 1, "python": sys.version, "platform": platform.platform(),
                "dependencies": {name: version(name) for name in ("numpy", "scipy", "networkx", "rdkit")},
                "matched_directory": str(args.matched),
                "matched_summary_sha256": sha256((args.matched/"summary.json").read_bytes()).hexdigest(),
                "literal_validation": validation_record,
                "file_sha256": {name: sha256((args.output/name).read_bytes()).hexdigest() for name in saved},
                "selection": "First per-bin inputs by saved reaction hash in each source; no outcome filtering",
                "seed": "Element-wise incident-histogram LAP and two improving swap sweeps; charged to each seeded attempt",
                "basic_bounds_scope": "Both assignment bounds and all profile bounds removed; committed-cost pruning and symmetry retained",
                "prechecks": "No separate bond-mass/congruence precheck; common production validation applies",
                "output_unit": "all_indexed_atom_maps"}
    (args.output/"manifest.json").write_text(encode(manifest))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda task: execute(task, args.output), tasks))
    comparisons = compare_records(records, args.output)
    summary = {"selected": len(rows), "attempts": len(records),
               "unavailable_numeric_queries": 2*len(unavailable),
               "by_variant": {v: dict(Counter("complete" if r["complete"] else r["termination"]
                                               for r in records if r["task"]["variant"] == v)) for v in variants},
               "comparisons": comparisons,
               "all_output_comparisons_consistent": all(c["consistent"] for c in comparisons),
               "all_record_hashes": {p.name: sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((args.output/"cases").glob("*.json"))}}
    (args.output/"summary.json").write_text(encode(summary))
    print(encode({k: v for k, v in summary.items() if k not in ("comparisons", "all_record_hashes")}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-bin", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-gib", type=int, default=6)
    parser.add_argument("--max-maps", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--variants", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument("--validation", type=Path,
                        default=Path("paper/synister/evidence/ablation_validation_v1"))
    args = parser.parse_args()
    if min(args.per_bin, args.seconds, args.workers, args.memory_gib, args.max_maps, args.repeats) <= 0:
        parser.error("Counts and limits must be positive")
    raise SystemExit(0 if run(args)["all_output_comparisons_consistent"] else 1)
