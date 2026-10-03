"""Derive manuscript counts from four unchanged frozen chemical reports."""

import argparse
import json
from pathlib import Path

from Experiment.Stereo.chemical_result_contracts import require, summarize
from Experiment.Stereo.provenance import ROOT, capture_source, finish_source, sha256

DEFAULT_INPUTS = {
    "coordination_annotation": "Experiment/Stereo/Data/Revision/Chemical/independent_annotation.json",
    "coordination_production": "Experiment/Stereo/Data/Revision/Chemical/production_validation.json",
    "rota_annotation": "Experiment/Stereo/Data/Revision/Chemical/rota_structural_annotation.json",
    "rota_production": "Experiment/Stereo/Data/Revision/Chemical/rota_production_validation.json",
}


def resolve_inputs(inputs, root=ROOT):
    """Verify every original input checksum and annotation/production binding."""
    require(set(inputs) == set(DEFAULT_INPUTS), "Chemical numerical summary lacks a required source report.")
    reports = {}
    for role, specification in inputs.items():
        path = root / specification["path"]
        require(sha256(path) == specification["sha256"], "Chemical numerical source checksum differs.")
        reports[role] = json.loads(path.read_text())
    for annotation, production in (("coordination_annotation", "coordination_production"),
                                   ("rota_annotation", "rota_production")):
        require(reports[production]["annotation_sha256"] == inputs[annotation]["sha256"],
                "Chemical production is not bound to the supplied frozen annotation.")
    return reports


def make_report(paths=None):
    """Retain both immutable report lineage and task-specific numerical scopes."""
    paths = paths or {role: ROOT / name for role, name in DEFAULT_INPUTS.items()}
    source = capture_source(paths.values())
    inputs = {role: {"path": str(path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT)
                                 else path.resolve()), "sha256": sha256(path)}
              for role, path in paths.items()}
    reports = resolve_inputs(inputs)
    summary = summarize(*(reports[role] for role in DEFAULT_INPUTS))
    return {
        "schema": "synkit.chemical-manuscript-summary/1", "provenance": finish_source(source),
        "inputs": inputs, "summary": summary, "passed": True,
        "claim_boundaries": {
            "coordination": "Source-generated initial MACE geometries; retained OH and supplied tetrahedral frames; "
                            "two frozen geometry refusals; production covers certified source/mirror states, "
                            "not every unsourced reference pose.",
            "negatives": "Independently sourced constitutional biaryl symmetry controls; original/renumbered "
                         "perception; no barrier or persistence claim.",
            "rota": "Fixed status-withheld, source-category/support-assisted development-corpus review; "
                    "all sixty frozen loci; seven aromatic supports are conditional configuration tests, "
                    "not ordinary single-bond atropisomerism; class total is a sum per case.",
            "review": "Automated independent mathematical annotation and replay, not independent human review.",
        },
    }


def validate_summary(report, root=ROOT):
    """Refuse summary values that do not reproduce from exact linked records."""
    reports = resolve_inputs(report["inputs"], root)
    expected = summarize(*(reports[role] for role in DEFAULT_INPUTS))
    require(report["summary"] == expected and report["passed"],
            "Chemical manuscript summary does not reproduce from its frozen inputs.")
    require(report["provenance"]["source_unchanged_during_run"]
            and report["provenance"]["inputs_unchanged_during_run"],
            "Chemical summary source/input stability is not verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for role in DEFAULT_INPUTS:
        parser.add_argument("--" + role.replace("_", "-"), type=Path, default=ROOT / DEFAULT_INPUTS[role])
    args = parser.parse_args()
    report = make_report({role: getattr(args, role) for role in DEFAULT_INPUTS})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
