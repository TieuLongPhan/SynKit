"""Published representation counts must retain states, strings and numbering coverage."""

from copy import deepcopy
import json

import pytest

from Experiment.Stereo.manuscript_tables import ROOT, validate_report


@pytest.fixture
def report():
    path = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/rdkit_representation_report.json"
    return json.loads(path.read_text())


def specification(report):
    return {"schema": report["schema"], "scope": "finite represented-geometry probes",
            "expected": {"/provenance/rdkit": "2026.03.1"}, "required_correctness": True}


def test_archived_rdkit_representation_report_supports_its_declared_counts(report):
    validate_report(report, specification(report))


@pytest.mark.parametrize("defect", ["state_omitted", "numbering_omitted", "false_invariance", "round_trip_failed"])
def test_representation_contract_refuses_incomplete_or_contradictory_evidence(report, defect):
    changed = deepcopy(report)
    group = changed["groups"][0]
    if defect == "state_omitted":
        group["records"].pop()
    elif defect == "numbering_omitted":
        group["records"][0]["atom_permutations_checked"] -= 1
    elif defect == "false_invariance":
        group["records"][0]["distinct_outputs_after_renumbering"].append("different complete string")
    else:
        group["records"][0]["round_trip_preserved"] = False
    with pytest.raises(ValueError):
        validate_report(changed, specification(changed))
