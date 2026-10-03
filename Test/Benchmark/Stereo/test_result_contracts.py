"""Adversarial tests for report provenance, input scope, and numerical claims."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
from rdkit import Chem

from Experiment.Stereo.Perception.cip_labels import _rdkit_coordinate_result, _score_records
from Experiment.Stereo.Perception.corpus_overlap import structure_keys
from Experiment.Stereo.manuscript_tables import MANIFEST_SCHEMA, build_ledger, validate_report
from Experiment.Stereo.provenance import ROOT


def _cip_report():
    records = [{"id": "one", "expected": ["1R"], "predicted": ["1S"], "exact": False}]
    return {
        "schema": "synkit.cip-native-validation/4", "protocol": "smiles_only",
        "records": records, "parse_failures": [], "overall": _score_records(records),
        "dataset": {"coordinate_sha256": None},
        "coverage": {"parsed_records": 1, "input_records": 1},
    }


def _manifest(tmp_path, report=None):
    report = report or _cip_report()
    path = tmp_path / "result.json"
    payload = (json.dumps(report) + "\n").encode()
    path.write_bytes(payload)
    return {"schema": MANIFEST_SCHEMA, "reports": [{
        "path": "result.json", "sha256": hashlib.sha256(payload).hexdigest(),
        "schema": report["schema"], "scope": "one named SMILES-only fixture",
        "expected": {"/coverage/input_records": 1, "/protocol": "smiles_only"},
        "claims": [{"id": "ExactRecords", "pointer": "/overall/exact_records",
                    "denominator": "/coverage/input_records"}],
    }]}


def test_label_ledger_distinguishes_wrong_and_missing_labels(tmp_path):
    report = _cip_report()
    assert report["overall"]["wrong_labels"] == report["overall"]["missing_labels"] == 1
    ledger = build_ledger(_manifest(tmp_path, report), tmp_path)
    assert ledger[0]["value"] == 0 and ledger[0]["denominator"] == 1


def test_changed_artifact_cannot_keep_its_old_provenance(tmp_path):
    manifest = _manifest(tmp_path)
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        build_ledger(manifest, tmp_path)


@pytest.mark.parametrize("change", ["coordinates", "denominator", "exact_flag", "wrong_count", "duplicate_id"])
def test_incompatible_or_fabricated_cip_claims_are_refused(tmp_path, change):
    report = _cip_report()
    if change == "coordinates":
        report["dataset"]["coordinate_sha256"] = "borrowed-coordinate-evidence"
    elif change == "denominator":
        report["coverage"]["input_records"] = 2
    elif change == "exact_flag":
        report["records"][0]["exact"] = True
    elif change == "wrong_count":
        report["overall"]["wrong_labels"] = 0
    else:
        report["parse_failures"] = ["one"]
    with pytest.raises(ValueError):
        build_ledger(_manifest(tmp_path, report), tmp_path)


def test_conflicting_duplicate_claims_are_refused(tmp_path):
    manifest = _manifest(tmp_path)
    second = deepcopy(manifest["reports"][0])
    second["claims"][0]["pointer"] = "/overall/wrong_labels"
    manifest["reports"].append(second)
    with pytest.raises(ValueError, match="Contradictory"):
        build_ledger(manifest, tmp_path)


def test_partial_or_failed_oracle_cannot_supply_required_correctness():
    report = {"schema": "synkit.independent-stereograph-audit/1", "passed": True,
              "records": [{"passed": False}], "sources_unchanged_during_run": True,
              "mutation_controls": {"passed": True}, "summary": {"cases": 1}}
    spec = {"schema": report["schema"], "scope": "full declared catalogue",
            "expected": {"/summary/cases": 63}, "required_correctness": True}
    with pytest.raises(ValueError, match="contract"):
        validate_report(report, spec)
    spec["expected"]["/summary/cases"] = 1
    with pytest.raises(ValueError, match="correctness"):
        validate_report(report, spec)


def test_constitution_deduplication_erases_stereo_but_preserves_isotopes_and_charge():
    right = structure_keys("C[C@H](F)Cl")
    left = structure_keys("C[C@@H](F)Cl")
    assert right[0] != left[0] and right[1] == left[1]
    assert structure_keys("[13CH4]")[1] != structure_keys("C")[1]
    assert structure_keys("[NH4+]")[1] != structure_keys("N")[1]
    assert structure_keys("[CH3:1][CH3:2]")[1] == structure_keys("CC")[1]


def test_coordinate_comparator_refuses_ambiguous_label_transport(monkeypatch):
    import Experiment.Stereo.Perception.cip_labels as cip

    molecule = Chem.MolFromSmiles("CC")
    monkeypatch.setattr(cip, "_constitutional_matches", lambda *_args: ((0, 1), (1, 0)))
    monkeypatch.setattr(cip, "_rdkit_labels", lambda _molecule: {"1R"})
    monkeypatch.setattr(Chem, "AssignStereochemistryFrom3D", lambda *_args, **_kwargs: None)
    report = _rdkit_coordinate_result({"ID": "ambiguous", "recommended_labels": ["1R"]}, molecule, molecule)
    assert report["status"] == "ambiguous_coordinate_transport"
    assert not report["exact"] and report["predicted"] == []


def test_full_reproduction_commands_satisfy_the_actual_canonicalization_cli(tmp_path):
    """Catch missing required artifacts before scheduling an expensive full study."""
    from Experiment.Stereo.reproduce import stages
    from Experiment.Stereo.Canonicalization.configuration_free_global import _parser as global_parser
    from Experiment.Stereo.Canonicalization.configuration_free_local import _parser as local_parser

    args = SimpleNamespace(output_dir=tmp_path, tier="full", jobs=2,
                           cip_path=tmp_path / "input.smi", coordinate_path=tmp_path / "input.sdf",
                           stereomolgraph=None, allow_expensive_raw=False)
    executed = 0
    for _name, argv in stages(args):
        if argv[2].endswith("configuration_free_local"):
            parsed = local_parser().parse_args(argv[3:])
            assert parsed.output.parent == parsed.table.parent == tmp_path
            executed += 1
        if argv[2].endswith("configuration_free_global"):
            parsed = global_parser().parse_args(argv[3:])
            assert parsed.enumeration_mode == "formal"
            assert parsed.output.parent == parsed.table.parent == tmp_path
            executed += 1
    assert executed == 7


@pytest.mark.parametrize("change", ["missing_trial", "duplicate_trial", "censored_code", "code_disagreement",
                                    "censor_as_completed_time", "incorrect_outcome_count"])
def test_controlled_cost_ledger_rejects_corrupted_coverage_and_censoring(change):
    path = ROOT / "Experiment/Stereo/Data/Revision/Performance/controlled_full.json"
    report = json.loads(path.read_text())
    spec = {"schema": report["schema"], "scope": "all 540 prospectively planned cost trials",
            "expected": {"/summary/generators/planned": 270}, "required_correctness": True}
    validate_report(report, spec)
    if change == "missing_trial":
        report["records"].pop()
    elif change == "duplicate_trial":
        report["records"][-1] = deepcopy(report["records"][-2])
    elif change == "censored_code":
        next(r for r in report["records"] if r["status"] == "timeout")["full_code_intern_id"] = 1
    elif change == "code_disagreement":
        report["records"][0]["full_code_intern_id"] = -1
    elif change == "censor_as_completed_time":
        report["summary"]["generators"]["completed_only"]["worker_wall_seconds"]["count"] += 1
    else:
        report["summary"]["generators"]["outcomes"]["complete"] += 1
    with pytest.raises(ValueError):
        validate_report(report, spec)


@pytest.mark.parametrize("change", ["missing_pipeline", "merge_classes", "false_group_count", "runtime",
                                    "profile_boundary"])
def test_registration_ledger_rejects_fabricated_groups_and_performance(change):
    report = json.loads((ROOT / "Experiment/Stereo/Data/Revision/Performance/registration_full.json").read_text())
    spec = {"schema": report["schema"], "scope": "all declared sizes, repetitions and pipelines",
            "expected": {"/protocol/repetitions": 3}, "required_correctness": True}
    validate_report(report, spec)
    if change == "missing_pipeline":
        report["records"].pop()
    elif change == "merge_classes":
        report["records"][0]["observed_partition"] = [list(range(report["records"][0]["size"]))]
    elif change == "false_group_count":
        report["records"][0]["group_count"] += 1
    elif change == "runtime":
        report["summary"][0]["pairwise"]["total_seconds_completed_only"]["median"] = 0
    else:
        report["boundary_examples"]["profile_pair"][0]["observed_equal"] = False
    with pytest.raises(ValueError):
        validate_report(report, spec)
