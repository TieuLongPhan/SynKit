"""Reconcile ablation attempts and independently rescore all saved outputs."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from Experiment.Synister.ablation_benchmark import compare_records
from synkit.Chem.Mapper.identifiability import extract_label, parse_reaction


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def audit(directory):
    manifest = json.loads((directory/"manifest.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    for name, expected in manifest["file_sha256"].items():
        if digest(directory/name) != expected:
            raise ValueError("Changed manifest input or source snapshot")
    tasks = json.loads((directory/"tasks.json").read_text())
    rows = {r["benchmark_id"]: r for r in json.loads((directory/"inputs.json").read_text())}
    expected = {t["task_id"]: t for t in tasks}
    paths = sorted((directory/"cases").glob("*.json"))
    if len(expected) != len(tasks) or {p.stem for p in paths} != set(expected):
        raise ValueError("Missing, extra or duplicate ablation attempts")
    records, checked_maps, seeds = [], 0, defaultdict(set)
    for path in paths:
        if digest(path) != summary["all_record_hashes"][path.name]:
            raise ValueError("Changed case record")
        record = json.loads(path.read_text())
        task = record["task"]
        if {key: task[key] for key in expected[path.stem]} != expected[path.stem]:
            raise ValueError("Attempt differs from planned task")
        if task["reaction"] != rows[task["benchmark_id"]]["reaction"]:
            raise ValueError("Changed reaction input")
        for key in ("seconds", "memory_gib", "max_maps"):
            if task[key] != manifest[key]:
                raise ValueError("Resource limit mismatch")
        target = task["target_doubled_cd"]
        if task["query"] == "minimum":
            if target != "minimal" or task["minimum_proofs"]:
                raise ValueError("Minimum query received a supplied optimum")
            target = record.get("minimum_doubled_cd")
        else:
            proofs = task["minimum_proofs"]
            if not proofs:
                raise ValueError("Numeric relative query lacks minimum proof")
            for proof in proofs:
                saved = Path(proof["path"])
                original = json.loads(saved.read_text())
                if (digest(saved) != proof["sha256"] or not original.get("minimum_proved")
                        or original["minimum_doubled_cd"] != proof["minimum_doubled_cd"]):
                    raise ValueError("Invalid saved minimum proof")
            values = {p["minimum_doubled_cd"] for p in proofs}
            offset = {"at_minimum": 0, "plus_2": 4}[task["query"]]
            if len(values) != 1 or target != next(iter(values))+offset:
                raise ValueError("Incorrect relative target")
        r, p = parse_reaction(task["reaction"])
        if "mapping_sha256" in record:
            map_path = directory/"maps"/path.name
            output = json.loads(map_path.read_text())
            if digest(map_path) != record["mapping_sha256"] or map_path.stat().st_size != record["output_bytes"]:
                raise ValueError("Changed map file")
            if len(output) != record["mapping_count"] or len(set(map(tuple, output))) != len(output):
                raise ValueError("Incorrect map count or duplicate")
            if task["query"] == "minimum" and output and not record.get("minimum_proved"):
                raise ValueError("Unproved minimum exported as final result")
            if any(2*extract_label(r, p, m).weighted_distance != target for m in output):
                raise ValueError("Incorrect mapping CD")
            checked_maps += len(output)
        elif record["complete"]:
            raise ValueError("Complete task has no saved maps")
        seed = record.get("seed_mapping")
        if seed is not None:
            if task["variant"] == "no_seed" or 2*extract_label(r, p, seed).weighted_distance != record["seed_doubled_cd"]:
                raise ValueError("Incorrect seed")
            seeds[task["benchmark_id"]].add(tuple(seed))
        elif "mapping_sha256" in record and task["variant"] != "no_seed":
            raise ValueError("Seeded configuration lacks a seed")
        records.append(record)
    if any(len(values) != 1 for values in seeds.values()):
        raise ValueError("Seed varies between configurations or repeats")
    unavailable = json.loads((directory/"unavailable_queries.json").read_text())
    if len(records)+2*len(unavailable)*len(manifest["variants"])*manifest["repeats"] != 3*len(rows)*len(manifest["variants"])*manifest["repeats"]:
        raise ValueError("Planned query denominator mismatch")
    comparisons = compare_records(records, directory)
    counts = {v: dict(Counter("complete" if r["complete"] else r["termination"]
                             for r in records if r["task"]["variant"] == v)) for v in manifest["variants"]}
    if counts != summary["by_variant"] or summary["attempts"] != len(records) or summary["selected"] != len(rows):
        raise ValueError("Summary accounting mismatch")
    return {"schema": "synister.ablation-audit.v1", "attempts": len(records), "selected": len(rows),
            "validated_saved_maps": checked_maps, "by_variant": counts,
            "all_output_comparisons_consistent": all(c["consistent"] for c in comparisons),
            "comparisons": comparisons, "manifest_sha256": digest(directory/"manifest.json"),
            "summary_sha256": digest(directory/"summary.json"), "auditor_sha256": digest(Path(__file__))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.directory)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))
    raise SystemExit(0 if report["all_output_comparisons_consistent"] else 1)
