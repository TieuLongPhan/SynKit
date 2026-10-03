"""Reconciled protocol counts must agree with task-specific retained evidence."""

from copy import deepcopy
import json

import pytest

from Experiment.Stereo.manuscript_tables import ROOT, validate_report

NAMES = ["relations", "perception_conformance", "rota_loci", "elements", "axial_negatives", "cip_contract"]


def load_report(name):
    path = ROOT / "Experiment/Stereo/Data/Revision/ProtocolReconciled" / (name + ".json")
    return json.loads(path.read_text())


def specification(report, name):
    return {"schema": report["schema"], "scope": "declared task-specific " + name,
            "expected": {"/schema": report["schema"]},
            "required_correctness": name in ("relations", "perception_conformance", "axial_negatives")}


@pytest.mark.parametrize("name", NAMES)
def test_archived_protocol_counts_reproduce_from_retained_records(name):
    report = load_report(name)
    validate_report(report, specification(report, name))


@pytest.mark.parametrize("name", NAMES)
def test_protocol_contract_refuses_contradictory_evidence_even_with_unchanged_success_flags(name):
    report = deepcopy(load_report(name))
    if name == "relations":
        report["records"][0]["observed"] = "different_relation"
    elif name == "perception_conformance":
        report["records"][0]["status"] = "error"
    elif name == "rota_loci":
        report["records"][0]["true_positive_pairs"] = [[5, 6]]
    elif name == "elements":
        report["records"][0]["canonical_frames"].append("unjustified additional frame")
    elif name == "axial_negatives":
        report["records"][0]["axes"][0]["carrier_status"] = "confirmed"
    else:
        report["maximum_exact_records_from_supplied_smiles"] = 300
    with pytest.raises(ValueError):
        validate_report(report, specification(report, name))
