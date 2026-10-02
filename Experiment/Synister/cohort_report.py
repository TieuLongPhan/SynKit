"""Compute rational bounds and margin diagnostics from an audited archive."""

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path

from Experiment.Synister.audit_development import sha
from synkit.Chem.Mapper.cohort_evaluation import paired_cohort_bounds


def report(directory, margin):
    audit = json.loads((directory / "audit.json").read_text())
    assert sha(directory / "summary.json") == audit["summary_sha256"]
    assert sha(directory / "manifest.json") == audit["manifest_sha256"]
    rows = audit["resolved_rows"]
    n = audit["summary"]["common_valid_predictions"]
    intervals = [(row["lower"], row["upper"]) for row in rows] + [None] * (n-len(rows))
    bounds = paired_cohort_bounds(intervals, margin=margin)
    scope = audit["summary"].get("scope")
    replication = scope == "prospective_C2_replication"
    confirmation = scope == "prospective_C1_confirmation" or replication
    if confirmation:
        lock = json.loads((directory / "execution_lock.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        if sha(directory / "execution_lock.json") != manifest["execution_lock_sha256"]:
            raise ValueError("Confirmation execution lock hash mismatch")
        if margin != Fraction(lock["margin"]):
            raise ValueError("Confirmation margin differs from the execution lock")
    return {
        "scope": ("prospective C2 equal-row common-valid replication cohort; independent per-row label choices"
                  if replication else "prospective C1 equal-group common-valid confirmation cohort; independent per-row label choices"
                  if confirmation else "descriptive equal-row common-valid development cohort; independent per-row label choices"),
        "audit_sha256": sha(directory / "audit.json"), "margin": str(margin),
        "margin_status": ("fixed pre-outcome C2 margin inherited from C1" if replication else "fixed pre-outcome C1 margin" if confirmation else
                          "proposed development interpretation, not a confirmation decision"),
        "common_valid": n, "resolved": len(rows),
        "multiple_bond_label_orbits": sum(row["bond_label_orbits"] > 1 for row in rows),
        "local_sign_reversals": sum(Fraction(row["lower"]) < 0 < Fraction(row["upper"]) for row in rows),
        "bounds": {key: str(value) if isinstance(value, Fraction) else value for key, value in asdict(bounds).items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--margin", type=Fraction, default=Fraction(1, 50))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.directory, args.margin)
    if args.output:
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2))
