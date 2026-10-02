"""Input-selected, matched global all-solutions benchmark; preserves every attempt."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def encode(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def size_bin(n):
    return min((n-1)//20, 4)


def select(paths, per_bin):
    from synkit.Chem.Mapper.identifiability import parse_reaction
    selected, accounting = [], []
    for source, path in zip(("FlowER", "Rhea"), paths):
        buckets = [[] for _ in range(5)]
        rejected, seen = [], set()
        for row in json.loads(path.read_text()):
            digest = sha256(row["reaction"].encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            try:
                r, _ = parse_reaction(row["reaction"])
                n = len(r.atomic_numbers)
                if n > 512:
                    raise ValueError("Over shared 512-atom domain")
            except ValueError as exc:
                rejected.append({"case_id": row["case_id"], "reason": str(exc)})
                continue
            record = {**row, "source": source, "input_file": str(path), "atoms": n,
                      "size_bin": size_bin(n), "selection_sha256": digest}
            buckets[size_bin(n)].append(record)
        for bucket in buckets:
            bucket.sort(key=lambda x: x["selection_sha256"])
        chosen = [row for bucket in buckets for row in bucket[:per_bin]]
        remainder = sorted([row for bucket in buckets for row in bucket[per_bin:]],
                           key=lambda x: x["selection_sha256"])
        chosen += remainder[:max(0, 5*per_bin-len(chosen))]
        selected.extend(chosen)
        accounting.append({"source": source, "available_by_bin": list(map(len, buckets)),
                           "requested_per_bin": per_bin, "selected": len(chosen),
                           "selected_by_bin": dict(Counter(x["size_bin"] for x in chosen)),
                           "unsupported": rejected, "input_sha256": sha256(path.read_bytes()).hexdigest()})
    for i, row in enumerate(selected):
        row["benchmark_id"] = f"reaction_{i:03d}"
    return selected, accounting


def execute(task, output):
    key = task["task_id"]
    record_path = output / "cases" / f"{key}.json"
    map_path = output / "maps" / f"{key}.json"
    if record_path.exists() or map_path.exists():
        raise FileExistsError(key)
    task = {**task, "map_path": str(map_path.resolve())}
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, "-m", "Experiment.Synister.enumeration_worker"],
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
    record_path.write_text(encode(record))
    print(json.dumps({"task": key, "complete": record["complete"],
                      "termination": record["termination"], "seconds": record["parent_seconds"]}), flush=True)
    return record


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    (output / "maps").mkdir()
    rows, accounting = select([args.flower, args.rhea], args.per_bin)
    (output / "inputs.json").write_text(encode(rows))
    (output / "selection.json").write_text(encode(accounting))
    paths = sorted(Path("synkit/Chem/Mapper").rglob("*.py"))
    paths += [Path(__file__), Path("Experiment/Synister/global_milp.py"),
              Path("Experiment/Synister/enumeration_worker.py")]
    sources = {str(p): p.read_text() for p in paths}
    (output / "sources.json").write_text(encode(sources))
    manifest = {"schema": "synister.matched-enumeration.v1", "platform": platform.platform(),
                "processor": platform.processor(), "python": sys.version,
                "seconds": args.seconds, "external_seconds": args.seconds+15,
                "memory_gib": args.memory_gib, "max_maps": args.max_maps,
                "workers": args.workers, "threads_per_worker": 1,
                "methods": ["synister", "milp"], "output_unit": "all_indexed_atom_maps",
                "seed": "No externally supplied seed for either method",
                "dependencies": {name: version(name) for name in ("numpy", "scipy", "networkx", "rdkit")},
                "sources_sha256": sha256((output/"sources.json").read_bytes()).hexdigest(),
                "inputs_sha256": sha256((output/"inputs.json").read_bytes()).hexdigest(),
                "reference_source": str(args.references) if args.references else None,
                "reference_sha256": sha256(args.references.read_bytes()).hexdigest() if args.references else None,
                "selection_rule": "Per source: requested count per atom bin, ascending input hash; shortfalls from remaining hashes"}
    (output / "manifest.json").write_text(encode(manifest))
    def task(row, label, target, method):
        return {"task_id": f'{row["benchmark_id"]}.{label}.{method}', "benchmark_id": row["benchmark_id"],
                "reaction": row["reaction"], "method": method, "query": label,
                "target_doubled_cd": target, "seconds": args.seconds,
                "memory_gib": args.memory_gib, "max_maps": args.max_maps}
    minimum_tasks = []
    for index, row in enumerate(rows):
        for method in (("synister", "milp") if index % 2 == 0 else ("milp", "synister")):
            minimum_tasks.append(task(row, "minimum", "minimal", method))
    (output / "minimum_tasks.json").write_text(encode(minimum_tasks))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        minima = list(pool.map(lambda t: execute(t, output), minimum_tasks))
    reference_maps = ({r["reaction_id"]: r["mapped_reaction"] for r in json.loads(args.references.read_text())}
                      if args.references else {})
    numeric_tasks, target_records = [], []
    from synkit.Chem.Mapper.identifiability import parse_reaction
    from synkit.Chem.Mapper.prediction_adapter import align_mapped_prediction
    from Experiment.Synister.global_milp import doubled_distance
    for row in rows:
        proofs = [(r["task"]["method"], r["minimum_doubled_cd"]) for r in minima
                  if r["task"]["benchmark_id"] == row["benchmark_id"] and r.get("minimum_proved")]
        values = {cost for _, cost in proofs}
        if len(values) > 1:
            raise ValueError(f'Minimum disagreement for {row["benchmark_id"]}: {proofs}')
        record = {"benchmark_id": row["benchmark_id"], "minimum_proofs": proofs,
                  "relative_targets_available": bool(proofs), "reference_status": "not_available"}
        targets = []
        if proofs:
            best = next(iter(values))
            targets.extend((label, best+offset) for label, offset in
                           (("at_minimum", 0), ("plus_1", 2), ("plus_2", 4), ("plus_4", 8)))
        if row["source"] == "FlowER" and row["reaction_id"] in reference_maps:
            try:
                r, p = parse_reaction(row["reaction"])
                aligned = align_mapped_prediction(row["reaction"], reference_maps[row["reaction_id"]])
                cost = doubled_distance(r, p, aligned.mapping)
                targets.append(("reference_cd", cost))
                record.update(reference_status="valid", reference_doubled_cd=cost)
            except ValueError as exc:
                record.update(reference_status="invalid", reference_error=str(exc))
        record["targets"] = targets
        target_records.append(record)
        for label, cost in targets:
            for method in ("synister", "milp"):
                numeric_tasks.append(task(row, label, cost, method))
    (output / "target_derivation.json").write_text(encode(target_records))
    (output / "numeric_tasks.json").write_text(encode(numeric_tasks))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        numeric = list(pool.map(lambda t: execute(t, output), numeric_tasks))
    records = minima + numeric
    comparisons = []
    for key in sorted({r["task"]["task_id"].rsplit(".", 1)[0] for r in records}):
        pair = [r for r in records if r["task"]["task_id"].rsplit(".", 1)[0] == key]
        both = len(pair) == 2 and all(r["complete"] for r in pair)
        equal = (pair[0]["mapping_count"] == pair[1]["mapping_count"] and
                 pair[0]["mapping_sha256"] == pair[1]["mapping_sha256"]) if both else None
        comparisons.append({"query": key, "both_complete": both, "mapping_sets_equal": equal})
    summary = {"selected": len(rows), "attempts": len(records),
               "by_method": {method: dict(Counter(r["termination"] for r in records
                                                    if r["task"]["method"] == method))
                             for method in ("synister", "milp")},
               "comparisons": comparisons,
               "unexplained_disagreements": sum(r["mapping_sets_equal"] is False for r in comparisons),
               "all_record_hashes": {p.name: sha256(p.read_bytes()).hexdigest()
                                      for p in sorted((output/"cases").glob("*.json"))}}
    (output / "summary.json").write_text(encode(summary))
    print(encode(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flower", type=Path, default=Path("paper/synister/evidence/identifiability_c1_primary_v1/inputs.json"))
    parser.add_argument("--rhea", type=Path, default=Path("paper/synister/evidence/identifiability_c2_primary_v1/inputs.json"))
    parser.add_argument("--references", type=Path, default=Path("paper/synister/evidence/identifiability_c1_selection_v1/references.json"))
    parser.add_argument("--per-bin", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-gib", type=int, default=6)
    parser.add_argument("--max-maps", type=int, default=100000)
    args = parser.parse_args()
    if min(args.per_bin, args.seconds, args.workers, args.memory_gib, args.max_maps) <= 0:
        parser.error("Counts and resource limits must be positive")
    raise SystemExit(1 if run(args)["unexplained_disagreements"] else 0)
