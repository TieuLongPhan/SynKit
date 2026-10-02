"""Matched binary-CD backend comparison on twenty tractable real reactions.

Both algorithms receive a supplied numeric target and return all indexed maps.
Literal permutations establish targets and reference sets outside timed calls.
The selected inputs come from the input-only all-distance validation sample.
"""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def encode(value):
    return json.dumps(value, indent=2, allow_nan=False)+"\n"


def binary_endpoints(reaction):
    from synkit.Chem.Mapper.identifiability import Endpoint, parse_reaction
    return tuple(Endpoint(e.atomic_numbers, e.charges, e.hcounts,
                          tuple((i, j, 2) for i, j, _ in e.bonds)) for e in parse_reaction(reaction))


def solve(r, p, target, backend, *, seconds=None, max_maps=None):
    started = time.perf_counter()
    from synkit.Chem.Mapper.exact.hybrid import enumerate_hybrid_distance_mappings
    from synkit.Chem.Mapper.identifiability import extract_label
    if backend not in ("assignment", "edit_support"):
        raise ValueError("Unknown binary backend")
    maximum = 2*(len(r.bonds)+len(p.bonds))
    maps, first = [], None
    def emit(mapping, cost):
        nonlocal first
        if first is None:
            first = time.perf_counter()-started
        maps.append(tuple(mapping))
    # Identical cheap necessary conditions for both methods. Retain these
    # outcomes separately from empty sets established by either search.
    precheck = "bond_mass_upper_bound" if target > maximum else "binary_congruence" if (maximum-target) % 4 else None
    if precheck:
        result = {"complete": True, "termination": "proved_empty_precheck", "precheck": precheck,
                  "backend": backend, "visited_nodes": 0, "backend_statistics": {}}
    else:
        options = dict(symmetry_pruning=True, expand_symmetry=True,
                       symmetry_node_properties=("charges", "hcounts")) if backend == "assignment" else {}
        raw = enumerate_hybrid_distance_mappings(
            [r.graph(), p.graph()], CD=target/2, binary=True, backend=backend,
            max_bijections=None, max_edit_support_pairs=None, tolerance=0,
            time_limit_seconds=None if seconds is None else max(0, seconds-(time.perf_counter()-started)),
            max_mappings=max_maps,
            collect_mappings=False, mapping_callback=emit, compute_minimum_cost=False, **options)
        if raw.selected_mapping_count != len(maps):
            raise ValueError("Callback count differs from backend count")
        result = {"complete": raw.complete, "termination": raw.truncation_reason or raw.status,
                  "precheck": None, "backend": raw.backend, "visited_nodes": raw.visited_nodes,
                  "backend_statistics": raw.backend_statistics}
    finished = time.perf_counter()
    if len(set(maps)) != len(maps):
        raise ValueError("Duplicate indexed map")
    if any(2*extract_label(r, p, m).weighted_distance != target for m in maps):
        raise ValueError("Incorrect binary chemical distance")
    result.update(mappings=sorted(maps), mapping_count=len(maps), first_map_seconds=first,
                  search_seconds=finished-started, checking_seconds=time.perf_counter()-finished,
                  output_unit="all_indexed_atom_maps", minimum_proved=False)
    return result


def worker(task):
    limit = int(task["memory_gib"]*1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    started = time.perf_counter()
    try:
        r, p = binary_endpoints(task["reaction"])
        parsed = time.perf_counter()
        result = solve(r, p, task["target_doubled_cd"], task["method"],
                       seconds=max(0, task["seconds"]-(parsed-started)), max_maps=task["max_maps"])
        exporting = time.perf_counter()
        data = json.dumps(result.pop("mappings"), separators=(",", ":")).encode()
        with Path(task["map_path"]).open("xb") as stream:
            stream.write(data)
        result.update(mapping_sha256=sha256(data).hexdigest(), output_bytes=len(data),
                      parse_seconds=parsed-started, export_seconds=time.perf_counter()-exporting,
                      end_to_end_seconds=time.perf_counter()-started)
    except Exception as exc:
        result = {"complete": False, "termination": "worker_error", "error_type": type(exc).__name__, "error": str(exc)}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(worker_seconds=time.perf_counter()-started,
                  cpu_seconds=usage.ru_utime+usage.ru_stime, peak_rss_kib=usage.ru_maxrss)
    return result


def execute(task, output):
    record_path = output/"cases"/f'{task["task_id"]}.json'
    map_path = output/"maps"/record_path.name
    if record_path.exists() or map_path.exists():
        raise FileExistsError(record_path)
    task = {**task, "map_path": str(map_path.resolve())}
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONHASHSEED="0")
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, "-m", "Experiment.Synister.binary_backend_benchmark", "--worker"],
                                 input=encode(task), capture_output=True, text=True, env=env,
                                 timeout=task["seconds"]+15)
        record = json.loads(process.stdout) if process.returncode == 0 else {
            "complete": False, "termination": "process_failure", "returncode": process.returncode}
        record["stderr"] = process.stderr[-4000:]
    except subprocess.TimeoutExpired:
        record = {"complete": False, "termination": "external_time_limit"}
    record.update(task=task, parent_seconds=time.perf_counter()-started)
    record_path.write_text(encode(record))
    print(json.dumps({"task": task["task_id"], "complete": record["complete"],
                      "termination": record["termination"], "seconds": record["parent_seconds"]}), flush=True)
    return record


def prepare(selection, *, seconds, memory_gib, max_maps):
    from Experiment.Synister.all_distance_oracle import literal_sets
    rows, targets, tasks = [], [], []
    for index, original in enumerate(json.loads(selection.read_text())["selected"]):
        row = {**original, "benchmark_id": f"binary_{index:03d}"}
        r, p = binary_endpoints(row["reaction"])
        oracle = literal_sets(r, p)
        if sum(map(len, oracle.values())) != row["compatible_maps"]:
            raise ValueError("Binary conversion changed the compatible mapping inventory")
        best = min(oracle)
        targets.append({"benchmark_id": row["benchmark_id"], "minimum_doubled_cd": best,
                        "queries": {label: {"target_doubled_cd": best+offset,
                                            "mappings": sorted(oracle.get(best+offset, set()))}
                                    for label, offset in (("at_minimum", 0), ("plus_1", 2), ("plus_2", 4))}})
        for query, values in targets[-1]["queries"].items():
            order = ("assignment", "edit_support") if index % 2 == 0 else ("edit_support", "assignment")
            for method in order:
                tasks.append({"task_id": f'{row["benchmark_id"]}.{query}.{method}',
                              "benchmark_id": row["benchmark_id"], "reaction": row["reaction"],
                              "method": method, "query": query, "target_doubled_cd": values["target_doubled_cd"],
                              "seconds": seconds, "memory_gib": memory_gib, "max_maps": max_maps})
        rows.append(row)
    return rows, targets, tasks


def audit(output, replay_oracle=True):
    from Experiment.Synister.all_distance_oracle import literal_sets
    from synkit.Chem.Mapper.identifiability import extract_label
    manifest = json.loads((output/"manifest.json").read_text())
    summary = json.loads((output/"summary.json").read_text())
    for name, digest in manifest["file_sha256"].items():
        if sha256((output/name).read_bytes()).hexdigest() != digest:
            raise ValueError("Changed inputs, oracle, task plan or sources")
    rows = {r["benchmark_id"]: r for r in json.loads((output/"inputs.json").read_text())}
    references = {r["benchmark_id"]: r for r in json.loads((output/"oracle.json").read_text())}
    if replay_oracle:
        for key, row in rows.items():
            oracle = literal_sets(*binary_endpoints(row["reaction"]))
            if min(oracle) != references[key]["minimum_doubled_cd"]:
                raise ValueError("Independent binary minimum differs")
            for query in references[key]["queries"].values():
                if set(map(tuple, query["mappings"])) != oracle.get(query["target_doubled_cd"], set()):
                    raise ValueError("Independent binary mapping set differs")
    tasks = json.loads((output/"tasks.json").read_text())
    expected = {t["task_id"]: t for t in tasks}
    paths = sorted((output/"cases").glob("*.json"))
    if len(expected) != len(tasks) or {p.stem for p in paths} != set(expected):
        raise ValueError("Missing or unexpected attempt")
    checks, records, maps_checked = [], [], 0
    for path in paths:
        if sha256(path.read_bytes()).hexdigest() != summary["all_record_hashes"][path.name]:
            raise ValueError("Changed case record")
        record = json.loads(path.read_text())
        task = record["task"]
        if {k: task[k] for k in expected[path.stem]} != expected[path.stem]:
            raise ValueError("Executed task differs from plan")
        query = references[task["benchmark_id"]]["queries"][task["query"]]
        if task["target_doubled_cd"] != query["target_doubled_cd"] or task["reaction"] != rows[task["benchmark_id"]]["reaction"]:
            raise ValueError("Target or input differs from independent oracle")
        for key in ("seconds", "memory_gib", "max_maps"):
            if task[key] != manifest[key]:
                raise ValueError("Resource limit mismatch")
        actual = set()
        if "mapping_sha256" in record:
            map_path = output/"maps"/path.name
            maps = json.loads(map_path.read_text())
            actual = set(map(tuple, maps))
            if (sha256(map_path.read_bytes()).hexdigest() != record["mapping_sha256"]
                    or len(maps) != record["mapping_count"] or len(actual) != len(maps)
                    or map_path.stat().st_size != record["output_bytes"]):
                raise ValueError("Mapping file, count or uniqueness mismatch")
            r, p = binary_endpoints(task["reaction"])
            if any(2*extract_label(r, p, m).weighted_distance != task["target_doubled_cd"] for m in maps):
                raise ValueError("Incorrect binary CD")
            maps_checked += len(maps)
        elif record["complete"]:
            raise ValueError("Complete result has no mapping output")
        wanted = set(map(tuple, query["mappings"]))
        valid = actual <= wanted and (not record["complete"] or actual == wanted)
        checks.append({"task_id": path.stem, "complete": record["complete"],
                       "expected_count": len(wanted), "returned_count": len(actual), "consistent": valid})
        records.append(record)
    if len(rows) != summary["selected"] or len(records) != summary["attempts"] or len(records) != 6*len(rows):
        raise ValueError("Study denominator mismatch")
    return {"all_verified": all(c["consistent"] for c in checks), "selected": len(rows),
            "attempts": len(records), "validated_saved_maps": maps_checked, "oracle_replayed": replay_oracle,
            "by_method": {m: dict(Counter("complete" if r["complete"] else r["termination"]
                                          for r in records if r["task"]["method"] == m))
                          for m in ("assignment", "edit_support")}, "checks": checks,
            "summary_sha256": sha256((output/"summary.json").read_bytes()).hexdigest(),
            "auditor_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


def run(args):
    if args.after is not None:
        preceding = json.loads((args.after/"audit.json").read_text())
        if not preceding["all_repeat_comparisons_consistent"]:
            raise ValueError("Preceding repeated study did not pass its audit")
    rows, references, tasks = prepare(args.selection, seconds=args.seconds,
                                      memory_gib=args.memory_gib, max_maps=args.max_maps)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"cases").mkdir()
    (args.output/"maps").mkdir()
    paths = sorted(Path("synkit/Chem/Mapper").rglob("*.py"))
    paths += [Path(__file__), Path("Experiment/Synister/all_distance_oracle.py"), Path("Experiment/Synister/worked_oracle.py")]
    saved = {"inputs.json": rows, "oracle.json": references, "tasks.json": tasks,
             "sources.json": {str(p): p.read_text() for p in paths}}
    for name, value in saved.items():
        (args.output/name).write_text(encode(value))
    manifest = {"schema": "synister.binary-backends.v1", "seconds": args.seconds,
                "external_seconds": args.seconds+15, "memory_gib": args.memory_gib,
                "max_maps": args.max_maps, "workers": 1, "threads_per_worker": 1,
                "python": sys.version, "platform": platform.platform(),
                "dependencies": {name: version(name) for name in ("numpy", "scipy", "networkx", "rdkit")},
                "selection_file": str(args.selection), "selection_sha256": sha256(args.selection.read_bytes()).hexdigest(),
                "file_sha256": {name: sha256((args.output/name).read_bytes()).hexdigest() for name in saved},
                "query_scope": "Supplied numeric binary CD; literal minimum proof is outside both timed calls",
                "assignment_options": "Verified product symmetry with full indexed-map expansion; no seed",
                "edit_support_options": "No support-pair cap; same time/output/memory limits; no seed",
                "shared_prechecks": "Bond mass and binary congruence; reported separately",
                "output_unit": "all_indexed_atom_maps",
                "scope": "Twenty input-selected tractable real reactions; not the weighted objective or a representative performance sample"}
    (args.output/"manifest.json").write_text(encode(manifest))
    records = [execute(task, args.output) for task in tasks]
    summary = {"selected": len(rows), "attempts": len(records),
               "all_record_hashes": {p.name: sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((args.output/"cases").glob("*.json"))}}
    (args.output/"summary.json").write_text(encode(summary))
    checked = audit(args.output)
    (args.output/"audit.json").write_text(encode(checked))
    print(encode({k: v for k, v in checked.items() if k != "checks"}))
    return checked


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False))
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("paper/synister/evidence/all_distance_validation_v1/real_selection.json"))
    parser.add_argument("--after", type=Path)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--memory-gib", type=int, default=6)
    parser.add_argument("--max-maps", type=int, default=100000)
    args = parser.parse_args()
    if min(args.seconds, args.memory_gib, args.max_maps) <= 0:
        parser.error("Resource limits must be positive")
    raise SystemExit(0 if run(args)["all_verified"] else 1)
