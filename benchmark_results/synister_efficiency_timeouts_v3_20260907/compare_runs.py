"""Compare frozen cohorts, excluding order-dependent representative hashes."""
import json
import statistics
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "synister_efficiency_timeouts_v2_20260907"


def normalized_counts(result, key):
    rc = result["reaction_center"]
    denominator = rc["frequency_denominator"]
    return {tuple(row[:-1]): Fraction(row[-1], denominator) for row in rc[key]}


def compare():
    report = {"cohorts": {}}
    speedups = []
    for cohort in ("development", "heldout"):
        before = PREVIOUS / ("profile_" + cohort)
        after = ROOT / ("conditioned_" + cohort)
        old_manifest = json.loads((before / "manifest.json").read_text())
        new_manifest = json.loads((after / "manifest.json").read_text())
        for key in ("selection_sha256", "seconds_per_shell", "workers", "cpus", "memory_limit_per_worker_bytes"):
            assert old_manifest[key] == new_manifest[key], key
        selection = json.loads((ROOT / (cohort + "_selection.json")).read_text())
        data = {"versions": {}, "new_completions": [], "completion_regressions": [], "common_complete_checks": [], "seed_improvements": []}
        records = {}
        for label, directory in (("before", before), ("after", after)):
            records[label] = [json.loads((directory / (str(task["source_line"]) + "_" + task["mode"] + ".json")).read_text()) for task in selection["tasks"]]
            summary = json.loads((directory / "summary.json").read_text())
            assert summary["errors"] == 0 and summary["implementation_unchanged"]
            summary["observed_task_wall_seconds"] = sum(r["wall_seconds"] for r in records[label])
            summary["max_worker_rss_mib"] = max(r["worker_peak_rss_kib"] for r in records[label]) / 1024
            summary["complete_by_mode"] = {mode: sum(r["mode"] == mode and r["result"]["complete"] for r in records[label]) for mode in ("minimal", "reference_cd")}
            data["versions"][label] = summary
        for old, new in zip(records["before"], records["after"]):
            a, b = old["result"], new["result"]
            row = {key: new[key] for key in ("source_line", "reaction_id", "mode")}
            row["wall_seconds"] = new["wall_seconds"]
            if a["complete"] and not b["complete"]:
                data["completion_regressions"].append(row)
            if not a["complete"] and b["complete"]:
                data["new_completions"].append(row)
            seed_a = a["backend_statistics"]["seed"]
            seed_b = b["backend_statistics"]["seed"]
            if seed_a.get("available") and seed_b.get("available"):
                assert seed_b["cost"] <= seed_a["cost"]
                if seed_b["cost"] < seed_a["cost"]:
                    data["seed_improvements"].append(dict(row, before=seed_a["cost"], after=seed_b["cost"], repair_seconds=seed_b["repair"]["elapsed_seconds"]))
            if not (a["complete"] and b["complete"]):
                continue
            for key in ("minimum_cost", "reference_cd", "reference_class_observed", "reference_is_global_minimum_proven"):
                assert a[key] == b[key], (row, key)
            if a["labeled_solution_count"] is not None and b["labeled_solution_count"] is not None:
                assert int(a["labeled_solution_count"]) == int(b["labeled_solution_count"]), row
            if a["symmetry_quotient_complete"] and b["symmetry_quotient_complete"]:
                for key in ("atom_change_counts", "bond_change_counts"):
                    assert normalized_counts(a, key) == normalized_counts(b, key), (row, key)
            sa, sb = a["structure"], b["structure"]
            if sa["complete"] and sb["complete"] and a["symmetry_quotient_complete"] and b["symmetry_quotient_complete"]:
                for key in ("its_class_counts", "template_class_counts"):
                    ca = {name: count * int(a["symmetry_group_order"]) for name, count in sa[key]}
                    cb = {name: count * int(b["symmetry_group_order"]) for name, count in sb[key]}
                    assert ca == cb, (row, key)
            speedup = old["wall_seconds"] / new["wall_seconds"]
            speedups.append(speedup)
            data["common_complete_checks"].append(dict(row, mathematical_outputs_match=True, wall_speedup=speedup))
        report["cohorts"][cohort] = data
    report["combined"] = {
        "tasks": sum(c["versions"]["after"]["tasks"] for c in report["cohorts"].values()),
        "before_complete": sum(c["versions"]["before"]["complete"] for c in report["cohorts"].values()),
        "after_complete": sum(c["versions"]["after"]["complete"] for c in report["cohorts"].values()),
        "common_complete_checked": len(speedups),
        "median_common_complete_speedup": statistics.median(speedups),
    }
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["combined"], indent=2))
    for name, data in report["cohorts"].items():
        print(name, "new:", data["new_completions"], "regressions:", data["completion_regressions"])


if __name__ == "__main__":
    compare()
