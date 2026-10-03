"""Chemical numerical claims must join unchanged complete truth and production."""

from copy import deepcopy
import json

import pytest

from Experiment.Stereo.chemical_result_contracts import summarize
from Experiment.Stereo.chemical_results import DEFAULT_INPUTS, make_report, validate_summary
from Experiment.Stereo.manuscript_tables import ROOT


@pytest.fixture(scope="module")
def reports():
    return [json.loads((ROOT / name).read_text()) for name in DEFAULT_INPUTS.values()]


def test_full_chemical_summary_separates_refusals_reference_poses_and_observed_configurations(reports):
    result = summarize(*reports)
    assert result["coordination"]["selected"] == 40
    assert result["coordination"]["certified"] == 38
    assert result["coordination"]["unresolved"] == 2
    assert result["coordination"]["reference_assignments"] == 900
    assert result["coordination"]["reference_classes"] == 230
    assert result["coordination"]["code_computations"] == 152
    assert result["negatives"]["perception_computations"] == 96
    assert result["rota"]["formal_assignments"] == 128
    assert result["rota"]["sum_per_case_classes"] == 98
    assert result["rota"]["aromatic_axes"] == 7
    assert result["rota"]["code_computations"] == 384


def test_summary_accepts_byte_identical_reproduction_inputs_outside_the_checkout(tmp_path):
    paths = {}
    for role, source in DEFAULT_INPUTS.items():
        path = tmp_path / (role + ".json")
        path.write_bytes((ROOT / source).read_bytes())
        paths[role] = path
    summary = make_report(paths)
    assert summary["summary"]["coordination"]["certified"] == 38
    validate_summary(summary)


@pytest.mark.parametrize("defect", [
    "hidden_refusal", "geometry_failed", "reference_incomplete", "constant_codes", "renumbering_failed",
    "negative_confirmed", "negative_substituted", "reference_replay_omitted",
])
def test_coordination_counts_refuse_incomplete_changed_or_contradictory_evidence(reports, defect):
    annotation, production, rota_truth, rota_results = deepcopy(reports)
    if defect == "hidden_refusal":
        production["coordination"]["unresolved"].pop()
    elif defect == "geometry_failed":
        annotation["coordination"]["accepted"][0]["source_audit"]["geometry"]["normalized_rmsd"] = 1
    elif defect == "reference_incomplete":
        annotation["coordination"]["constitutional_groups"][0]["reference"]["partition"][0].pop()
    elif defect == "constant_codes":
        for record in production["coordination"]["records"]:
            record["code_interning_ids"] = {key: 1 for key in record["code_interning_ids"]}
    elif defect == "renumbering_failed":
        production["coordination"]["records"][0]["code_interning_ids"]["renumbered"] = -1
    elif defect == "negative_confirmed":
        production["negative_controls"]["records"][0]["production_original"][0]["carrier_status"] = "confirmed"
    elif defect == "negative_substituted":
        production["negative_controls"]["records"][0]["id"] = "different-source"
    else:
        production["independent_reference_replay"].pop()
    with pytest.raises(ValueError):
        summarize(annotation, production, rota_truth, rota_results)


@pytest.mark.parametrize("defect", [
    "rota_status_unblinded", "aromatic_as_ordinary", "rota_partition_changed",
    "rota_false_confirmation", "rota_truth_changed",
])
def test_rota_counts_refuse_changed_truth_partitions_and_confirmation(reports, defect):
    annotation, production, rota_truth, rota_results = deepcopy(reports)
    if defect == "rota_status_unblinded":
        rota_truth["records"][0]["algorithm_confirmed_rejected_status_read"] = True
    elif defect == "aromatic_as_ordinary":
        record = next(r for r in rota_truth["records"] if r["independent_family"] == "axis"
                      and r["input_facts"]["source_bond_type"] == "AROMATIC")
        record["input_facts"]["ordinary_single_bond_topology"] = True
    elif defect == "rota_partition_changed":
        rota_results["records"][0]["production_partition"] = [[[0]], [[1]]]
    elif defect == "rota_false_confirmation":
        rota_results["records"][0]["perception"]["original"][-1]["observed_confirmed"] = True
    elif defect == "rota_truth_changed":
        for mode in ("original", "renumbered"):
            checks = rota_results["records"][0]["perception"][mode]
            checks[0]["expected"] = "confirmed"
            checks[0]["observed"] = ["confirmed"]
            checks[-1]["expected_confirmed"] = checks[-1]["observed_confirmed"] = True
    with pytest.raises(ValueError):
        summarize(annotation, production, rota_truth, rota_results)


@pytest.mark.parametrize("defect", ["changed_count", "changed_input_hash", "changed_annotation_binding"])
def test_numerical_summary_is_bound_to_original_frozen_report_bytes(tmp_path, reports, defect):
    paths = {}
    for role, report in zip(DEFAULT_INPUTS, reports):
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps(report))
        paths[role] = path
    # Use the original paths: writing JSON afresh would break the annotations'
    # existing production checksum bindings even before the chosen mutation.
    summary = make_report()
    if defect == "changed_count":
        summary["summary"]["rota"]["formal_assignments"] += 1
    elif defect == "changed_input_hash":
        summary["inputs"]["coordination_annotation"]["sha256"] = "0" * 64
    else:
        original = json.loads(paths["coordination_production"].read_text())
        original["annotation_sha256"] = "0" * 64
        paths["coordination_production"].write_text(json.dumps(original))
        import hashlib
        summary["inputs"]["coordination_production"] = {
            "path": str(paths["coordination_production"]),
            "sha256": hashlib.sha256(paths["coordination_production"].read_bytes()).hexdigest(),
        }
    with pytest.raises(ValueError):
        validate_summary(summary)
