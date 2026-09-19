"""Audit completed regressions using mathematical invariants."""
import json
from fractions import Fraction
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def normalize(result, key):
    rc = result["reaction_center"]
    return {tuple(row[:-1]): Fraction(row[-1], rc["frequency_denominator"]) for row in rc[key]}

def audit(directory):
    references = json.loads((ROOT / "regression_references.json").read_text())
    failures = []
    checked = 0
    classification_regressions = []
    for key, path in references.items():
        candidate = directory / (key + ".json")
        if not candidate.exists():
            continue
        old = json.loads(Path(path).read_text())["result"]
        new = json.loads(candidate.read_text()).get("result", {})
        if not new.get("complete"):
            failures.append([key, "completion"])
            continue
        checked += 1
        if old["structure"]["complete"] and not new["structure"]["complete"]:
            classification_regressions.append([key, new["structure"]["incomplete_reason"]])
        if old["symmetry_quotient_complete"] and not new["symmetry_quotient_complete"]:
            failures.append([key, "symmetry_quotient_complete"])
        if old["labeled_solution_count"] is not None and new["labeled_solution_count"] is None:
            failures.append([key, "known_labeled_count"])
        for field in ("minimum_cost", "reference_cd", "reference_class_observed", "reference_is_global_minimum_proven"):
            if old[field] != new[field]:
                failures.append([key, field])
        if old["labeled_solution_count"] is not None and new["labeled_solution_count"] is not None:
            if int(old["labeled_solution_count"]) != int(new["labeled_solution_count"]):
                failures.append([key, "labeled_solution_count"])
        if old["symmetry_quotient_complete"] and new["symmetry_quotient_complete"]:
            for field in ("atom_change_counts", "bond_change_counts"):
                if normalize(old, field) != normalize(new, field):
                    failures.append([key, field])
            if old["structure"]["complete"] and new["structure"]["complete"]:
                for field in ("its_class_counts", "template_class_counts"):
                    a = {name: int(count) * int(old["symmetry_group_order"]) for name, count in old["structure"][field]}
                    b = {name: int(count) * int(new["symmetry_group_order"]) for name, count in new["structure"][field]}
                    if a != b:
                        failures.append([key, field])
    report = dict(checked=checked, expected=len(references), failures=failures, classification_regressions=classification_regressions)
    (directory / "regression_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return report

if __name__ == "__main__":
    import sys
    audit(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "component_regression"))
