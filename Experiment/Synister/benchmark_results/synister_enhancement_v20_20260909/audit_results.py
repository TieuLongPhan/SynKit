"""Audit exact observables and full-output timings against the complete V14 cohort."""
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

R = Path(__file__).resolve().parent
BASE = R.parent / "synister_enhancement_v14_20260909/cohort_1200"
selection = json.loads((R / "selection_1200.json").read_text())
expected = {(t["source_line"], t["mode"]) for t in selection["tasks"]}
assert len(expected) == len(selection["tasks"]) == 1200

def differences(a, b):
    changes = []
    for field in ("target", "minimum_cost", "reference_cd", "reference_class_observed", "reference_is_global_minimum_proven", "labeled_solution_count"):
        if a[field] != b[field]:
            changes.append(field)
    for field in ("atom_change_counts", "bond_change_counts"):
        values = []
        for result in (a, b):
            rc = result["reaction_center"]
            values.append({tuple(row[:-1]): Fraction(row[-1], rc["frequency_denominator"]) for row in rc[field] if row[-1]})
        if values[0] != values[1]:
            changes.append(field)
    for field in ("its_class_counts", "template_class_counts"):
        values = [sorted((name, int(count) * int(result["symmetry_group_order"])) for name, count in result["structure"][field]) for result in (a, b)]
        if values[0] != values[1]:
            changes.append(field)
    return changes

def audit(directory, require_all=False):
    timings_path = directory / "case_timings.json"
    if not timings_path.exists():
        return None
    timings = json.loads(timings_path.read_text())
    seen = set()
    failures, incomplete, over_budget = [], [], []
    count = structures = strict = comparisons = 0
    aggregate_cpu = output_bytes = 0
    max_parent_rss = max_child_rss = 0
    for timing in timings:
        key = timing["source_line"], timing["mode"]
        assert key in expected and key not in seen, key
        seen.add(key)
        name = f"{key[0]}_{key[1]}.json"
        doc = json.loads((directory / name).read_text())
        assert (doc["source_line"], doc["mode"]) == key
        result = doc.get("result", {})
        aggregate_cpu += doc.get("aggregate_cpu_seconds", 0)
        output_bytes += timing["output_bytes"]
        max_parent_rss = max(max_parent_rss, doc.get("parent_peak_rss_kib", 0))
        max_child_rss = max(max_child_rss, doc.get("child_peak_rss_kib", 0))
        if not result.get("complete") or not result.get("structure", {}).get("complete"):
            incomplete.append({"key": key, "error": doc.get("error"), "reason": result.get("truncation_reason")})
        count += bool(result.get("complete"))
        structures += bool(result.get("complete") and result.get("structure", {}).get("complete"))
        passed = bool(result.get("complete") and result.get("structure", {}).get("complete") and timing["end_to_end_wall_seconds"] < 60)
        strict += passed
        assert bool(timing["complete_below_60_seconds"]) == bool(result.get("complete") and timing["end_to_end_wall_seconds"] < 60)
        if timing["end_to_end_wall_seconds"] >= 60:
            over_budget.append({"key": key, "seconds": timing["end_to_end_wall_seconds"], "complete": result.get("complete", False)})
        if not result.get("complete") or not result.get("structure", {}).get("complete"):
            continue
        assert result["symmetry_quotient_complete"] and result["labeled_solution_count"] is not None
        old = json.loads((BASE / name).read_text())["result"]
        assert old["complete"] and old["structure"]["complete"] and old["symmetry_quotient_complete"]
        changes = differences(old, result)
        for field in ("its_class_counts", "template_class_counts"):
            if sum(c for _, c in result["structure"][field]) != result["representative_solution_count"]:
                changes.append(field + ".sum")
        if key == (13067, "minimal"):
            independent = json.loads((R / "13067_independent_classification.json").read_text())["python"]["result"]
            changes += ["independent:" + c for c in differences(independent, result)]
        if changes:
            failures.append({"key": key, "fields": changes})
        comparisons += 1
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["source_unchanged"]
    assert manifest["library_sha256"] == hashlib.sha256(Path(manifest["library"]).read_bytes()).hexdigest()
    for timing in timings:
        name = "{}_{}.json".format(timing["source_line"], timing["mode"])
        doc = json.loads((directory / name).read_text())
        assert doc["source_sha256_python_and_cpp"] == manifest["source_sha256_python_and_cpp"]
    if require_all:
        assert seen == expected
        disk_keys = set()
        for path in directory.glob("*.json"):
            if path.name in {"manifest.json", "case_timings.json", "origins.json"}:
                continue
            doc = json.loads(path.read_text())
            assert (doc["source_line"], doc["mode"]) not in disk_keys
            disk_keys.add((doc["source_line"], doc["mode"]))
        assert disk_keys == expected
    return dict(processed=len(seen), search_complete=count, structure_complete=structures,
                strict_complete_below_60=strict, comparisons=comparisons,
                failures=failures, incomplete=incomplete, over_budget=over_budget,
                missing_count=len(expected - seen) if require_all else None,
                total_full_output_seconds=sum(t["end_to_end_wall_seconds"] for t in timings),
                max_full_output_seconds=max(t["end_to_end_wall_seconds"] for t in timings),
                aggregate_cpu_seconds=aggregate_cpu, output_bytes=output_bytes,
                parent_peak_rss_kib=max_parent_rss, child_peak_rss_kib=max_child_rss,
                source_sha256=manifest["source_sha256_python_and_cpp"],
                library_sha256=manifest["library_sha256"])

if __name__ == "__main__":
    reports = {}
    for directory in [R / "hard_pilot", R / "cohort_1200", *sorted(R.glob("hard_repeat_*"))]:
        if directory.is_dir() and (directory / "manifest.json").exists():
            try:
                finished = json.loads((directory / "manifest.json").read_text()).get("source_unchanged", False)
            except json.JSONDecodeError:
                finished = False
            if not finished:
                continue  # An active sentinel run is audited after it finishes.
            result = audit(directory, directory.name == "cohort_1200")
            if result is not None:
                reports[directory.name] = result
    (R / "audit_summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    assert all(not r["failures"] for r in reports.values())
