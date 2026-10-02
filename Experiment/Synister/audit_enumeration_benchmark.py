"""Check saved maps, all attempts and partial-set containment independently."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from synkit.Chem.Mapper.identifiability import extract_label, parse_reaction


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def compare_outputs(first, second, first_maps, second_maps):
    """An incomplete result can still falsify another method's complete set."""
    a, b = set(map(tuple, first_maps)), set(map(tuple, second_maps))
    both = first["complete"] and second["complete"]
    missing_from_first = b-a if first["complete"] else set()
    missing_from_second = a-b if second["complete"] else set()
    return {"both_complete": both, "equal_when_complete": a == b if both else None,
            "missing_from_first_complete_set": len(missing_from_first),
            "missing_from_second_complete_set": len(missing_from_second),
            "consistent": not missing_from_first and not missing_from_second}


def audit(directory):
    manifest = json.loads((directory/"manifest.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    if file_hash(directory/"inputs.json") != manifest["inputs_sha256"]:
        raise ValueError("Input digest mismatch")
    if file_hash(directory/"sources.json") != manifest["sources_sha256"]:
        raise ValueError("Source snapshot digest mismatch")
    rows = {r["benchmark_id"]: r for r in json.loads((directory/"inputs.json").read_text())}
    tasks = json.loads((directory/"minimum_tasks.json").read_text()) + json.loads((directory/"numeric_tasks.json").read_text())
    expected = {t["task_id"]: t for t in tasks}
    if len(expected) != len(tasks):
        raise ValueError("Duplicate task identifier")
    paths = sorted((directory/"cases").glob("*.json"))
    if {p.stem for p in paths} != set(expected):
        raise ValueError("Missing or unexpected attempted task records")
    groups, records, maps = defaultdict(list), {}, {}
    for path in paths:
        if file_hash(path) != summary["all_record_hashes"][path.name]:
            raise ValueError("Case digest mismatch")
        record = json.loads(path.read_text())
        task = record["task"]
        if {k: task[k] for k in expected[path.stem]} != expected[path.stem]:
            raise ValueError("Executed task differs from saved plan")
        if task["reaction"] != rows[task["benchmark_id"]]["reaction"]:
            raise ValueError("Reaction changed between selection and execution")
        for key in ("seconds", "memory_gib", "max_maps"):
            if task[key] != manifest[key]:
                raise ValueError("Methods did not receive declared resource limits")
        if record["complete"] and record["termination"] in ("external_time_limit", "process_failure", "worker_error"):
            raise ValueError("Failed attempt marked complete")
        map_path = directory/"maps"/path.name
        output = []
        if "mapping_sha256" in record:
            if file_hash(map_path) != record["mapping_sha256"]:
                raise ValueError("Mapping file digest mismatch")
            output = json.loads(map_path.read_text())
            if len(output) != record["mapping_count"] or len(set(map(tuple, output))) != len(output):
                raise ValueError("Wrong mapping count or duplicate output")
            if map_path.stat().st_size != record["output_bytes"]:
                raise ValueError("Incorrect output byte count")
            r, p = parse_reaction(task["reaction"])
            target = task["target_doubled_cd"]
            if target == "minimal":
                target = record.get("minimum_doubled_cd")
                if output and not record.get("minimum_proved"):
                    raise ValueError("Provisional optimizer exported as final minimum map")
            for mapping in output:
                if 2*extract_label(r, p, mapping).weighted_distance != target:
                    raise ValueError("Independent label extraction found an incorrect CD")
        elif record["complete"]:
            raise ValueError("Complete result has no saved output")
        records[path.stem], maps[path.stem] = record, output
        groups[path.stem.rsplit(".", 1)[0]].append(path.stem)
    comparisons = []
    for query, ids in sorted(groups.items()):
        if len(ids) != 2 or {records[i]["task"]["method"] for i in ids} != {"synister", "milp"}:
            raise ValueError("Unmatched query methods")
        first, second = ids
        if records[first]["task"]["target_doubled_cd"] != records[second]["task"]["target_doubled_cd"]:
            raise ValueError("Different supplied targets")
        comparison = compare_outputs(records[first], records[second], maps[first], maps[second])
        comparison.update(query=query, first=first, second=second)
        comparisons.append(comparison)
    if len(records) != summary["attempts"] or len(rows) != summary["selected"]:
        raise ValueError("Summary denominator mismatch")
    for row in rows:
        proofs = {r["minimum_doubled_cd"] for r in records.values()
                  if r["task"]["benchmark_id"] == row and r["task"]["query"] == "minimum"
                  and r.get("minimum_proved")}
        if len(proofs) > 1:
            raise ValueError("Conflicting proved minima")
        for r in records.values():
            if r["task"]["benchmark_id"] != row:
                continue
            label = r["task"]["query"]
            offsets = {"at_minimum": 0, "plus_1": 2, "plus_2": 4, "plus_4": 8}
            if label in offsets and (not proofs or r["task"]["target_doubled_cd"] != next(iter(proofs))+offsets[label]):
                raise ValueError("Relative query has no supporting minimum proof")
    return {"schema": "synister.matched-enumeration-audit.v1", "selected": len(rows),
            "attempts": len(records), "matched_queries": len(comparisons),
            "both_complete": sum(c["both_complete"] for c in comparisons),
            "all_output_comparisons_consistent": all(c["consistent"] for c in comparisons),
            "validated_saved_maps": sum(len(m) for m in maps.values()),
            "completion_by_method": {method: dict(Counter(
                "complete" if r["complete"] else r["termination"] for r in records.values()
                if r["task"]["method"] == method)) for method in ("synister", "milp")},
            "comparisons": comparisons, "manifest_sha256": file_hash(directory/"manifest.json"),
            "summary_sha256": file_hash(directory/"summary.json"), "auditor_sha256": file_hash(Path(__file__))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.directory)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))
    raise SystemExit(0 if report["all_output_comparisons_consistent"] else 1)
