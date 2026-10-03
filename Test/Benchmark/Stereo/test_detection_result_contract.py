"""Fresh carrier scores must reproduce from every retained reference/prediction set."""

from copy import deepcopy
import json

import pytest

from Experiment.Stereo.manuscript_tables import ROOT, validate_report


@pytest.fixture
def report():
    path = ROOT / "Experiment/Stereo/Data/Revision/ProtocolReconciled/retained_detection.json"
    return json.loads(path.read_text())


def specification(report):
    return {"schema": report["schema"], "scope": "all three task-specific carrier datasets",
            "expected": {"/coverage/dataset_rows": 1208}, "required_correctness": False}


def test_archived_detection_report_reproduces_from_locus_sets(report):
    validate_report(report, specification(report))


@pytest.mark.parametrize("defect", ["duplicated_row", "inflated_recovery", "hidden_unevaluated_row"])
def test_detection_contract_refuses_inconsistent_coverage_and_recovery(report, defect):
    changed = deepcopy(report)
    dataset = changed["datasets"]["chiralfinder_rota"]
    if defect == "duplicated_row":
        dataset["records"][-1] = deepcopy(dataset["records"][0])
    elif defect == "inflated_recovery":
        dataset["confirmed"]["true_positive_loci"] += 1
    else:
        dataset["records"][0]["detection_error"] = "case_timeout"
    with pytest.raises(ValueError):
        validate_report(changed, specification(changed))


@pytest.mark.parametrize("defect", ["retained_recovery", "retained_extra", "renumbering", "family", "miss_scope"])
def test_detection_contract_refuses_contradictory_assistance_and_explanation_counts(report, defect):
    changed = deepcopy(report)
    dataset = changed["datasets"]["cip_validation_suite"]
    if defect == "retained_recovery":
        dataset["two_setting_comparison"]["local_neighbor_orientation_retained"]["true_positive_loci"] -= 1
    elif defect == "retained_extra":
        dataset["two_setting_comparison"]["local_neighbor_orientation_retained"]["additional_detected_loci"] = 0
    elif defect == "renumbering":
        dataset["neighbor_configuration_assisted_renumbering"]["invariant_records"] -= 1
    elif defect == "family":
        dataset["by_reference_class"]["tetrahedral"]["broad"]["recovered_loci"] += 1
    else:
        dataset["configuration_erased_miss_structure"]["by_atom_element"] = {"N": 283}
    with pytest.raises(ValueError):
        validate_report(changed, specification(changed))
