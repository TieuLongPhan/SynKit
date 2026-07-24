"""Integrity gates for designed stereo-perception data.

These tests validate the data contract only. They deliberately do not call the
perception engine or calculate benchmark accuracy.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from Experiment.Stereo.Perception.conformance import (
    benchmark_perception_conformance,
)
from Experiment.Stereo.Perception.family_executors import (
    neutralize_rdkit_configuration,
)
from synkit.Chem.Molecule.stereo_perception import (
    StereoConfigurationState,
    detect_potential_stereo_elements,
)
from synkit.Graph.Stereo import descriptors_from_rdkit, stereo_from_dict

ROOT = Path(__file__).resolve().parents[4]
DATASET = (
    ROOT
    / "Experiment"
    / "Stereo"
    / "Data"
    / "Perception"
    / "perception_conformance_cases.json"
)
FROZEN_REPORT = DATASET.with_name("perception_conformance_report.json")

FAMILIES = {
    "tetrahedral",
    "double_bond",
    "cumulene_axis",
    "extended_cis_trans",
    "atrop_axis",
    "square_planar",
    "trigonal_bipyramidal",
    "octahedral",
    "helical",
    "planar_chirality",
}
CASE_TYPES = {
    "configured_positive",
    "unconfigured_positive",
    "negative_near_miss",
    "unsupported_or_ambiguous",
}
DESCRIPTOR_CLASSES = {
    "tetrahedral": "tetrahedral",
    "double_bond": "planar_bond",
    "cumulene_axis": "cumulene_axis",
    "extended_cis_trans": "extended_cis_trans",
    "atrop_axis": "atrop_bond",
    "square_planar": "square_planar",
    "trigonal_bipyramidal": "trigonal_bipyramidal",
    "octahedral": "octahedral",
    "helical": "helical",
    "planar_chirality": "planar_chirality",
}


def _dataset() -> dict[str, Any]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def test_dataset_fingerprint_is_frozen_in_the_stereo_registry() -> None:
    manifest = json.loads(
        (DATASET.parents[1] / "manifest.json").read_text(encoding="utf-8")
    )
    registered = manifest["designed_datasets"]["stereo_perception_conformance"]

    assert registered["path"].endswith("perception_conformance_cases.json")
    assert registered["records"] == 40
    assert registered["benchmark_status"] == "data_contract_only"
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == registered["sha256"]


def _integer_references(value: Any) -> list[int]:
    if type(value) is int:
        return [value]
    if isinstance(value, list):
        return [
            reference
            for item in value
            for reference in _integer_references(item)
        ]
    if isinstance(value, dict):
        return [
            reference
            for item in value.values()
            for reference in _integer_references(item)
        ]
    return []


def test_case_matrix_covers_every_family_and_information_state_once() -> None:
    payload = _dataset()
    cases = payload["cases"]

    assert payload["schema"] == "synkit.stereo-perception-conformance/1"
    assert payload["records"] == len(cases) == 40
    assert set(payload["families"]) == FAMILIES
    assert set(payload["case_types"]) == CASE_TYPES
    assert len({case["id"] for case in cases}) == len(cases)
    assert Counter(case["family"] for case in cases) == {
        family: 4 for family in FAMILIES
    }
    assert Counter(case["case_type"] for case in cases) == {
        case_type: 10 for case_type in CASE_TYPES
    }
    assert {
        (case["family"], case["case_type"]) for case in cases
    } == {
        (family, case_type)
        for family in FAMILIES
        for case_type in CASE_TYPES
    }


def test_every_structure_is_used_and_has_a_valid_representation() -> None:
    payload = _dataset()
    structures = payload["structures"]
    used = {case["structure"] for case in payload["cases"]}

    assert used == set(structures)
    for name, structure in structures.items():
        if structure["format"] == "smiles":
            assert Chem.MolFromSmiles(structure["value"]) is not None, name
            continue
        assert structure["format"] == "formal_graph"
        assert structure["chemical_validation"] is False
        node_ids = [node["id"] for node in structure["nodes"]]
        assert len(node_ids) == len(set(node_ids))
        assert node_ids
        assert all(
            len(edge) == 3 and edge[0] in node_ids and edge[1] in node_ids
            for edge in structure["edges"]
        )


def test_positive_supports_reference_atoms_present_in_the_input() -> None:
    payload = _dataset()
    structures = payload["structures"]
    for case in payload["cases"]:
        expected = case["expected"]
        support = expected["support"]
        if support is None:
            continue
        structure = structures[case["structure"]]
        if structure["format"] == "smiles":
            molecule = Chem.MolFromSmiles(structure["value"])
            assert molecule is not None
            atom_count = molecule.GetNumAtoms()
            assert all(
                0 <= reference < atom_count
                for reference in _integer_references(support)
            ), case["id"]
        else:
            node_ids = {node["id"] for node in structure["nodes"]}
            assert set(_integer_references(support)) <= node_ids, case["id"]


def test_configured_cases_restore_as_the_declared_descriptor_family() -> None:
    for case in _dataset()["cases"]:
        expected = case["expected"]
        descriptor_payload = expected["configured_descriptor"]
        if case["case_type"] == "configured_positive":
            assert expected["configuration_state"] == "specified"
            assert descriptor_payload is not None
            descriptor = stereo_from_dict(descriptor_payload)
            assert descriptor.descriptor_class == DESCRIPTOR_CLASSES[case["family"]]
        else:
            assert descriptor_payload is None


def test_outcome_and_reason_contracts_are_fail_closed() -> None:
    for case in _dataset()["cases"]:
        expected = case["expected"]
        if case["case_type"] in {
            "configured_positive",
            "unconfigured_positive",
        }:
            assert expected["outcome"] == "carrier_present"
            assert expected["support"] is not None
        elif case["case_type"] == "negative_near_miss":
            assert expected["outcome"] == "carrier_absent"
            assert expected["support"] is None
            assert expected["reason_code"]
        else:
            assert expected["outcome"] in {"unsupported", "ambiguous"}
            assert expected["support"] is None
            assert expected["reason_code"]


def test_dataset_contains_no_benchmark_scores_or_stability_claims() -> None:
    payload = _dataset()
    rendered = json.dumps(payload, sort_keys=True)

    assert payload["provenance"]["stability_claims"] is False
    assert payload["provenance"]["whole_molecule_chirality_claims"] is False
    for forbidden in (
        '"accuracy"',
        '"f1"',
        '"precision"',
        '"predicted"',
        '"recall"',
        '"stable": true',
    ):
        assert forbidden not in rendered


def test_task_aware_runner_scores_every_family_scope() -> None:
    report = benchmark_perception_conformance(DATASET)

    assert report["summary"] == {
        "records": 40,
        "scored_records": 40,
        "passed": 40,
        "failed": 0,
        "not_applicable": 0,
        "errors": 0,
        "strict_conformance": 1.0,
        "configuration_checks": 0,
        "configuration_checks_passed": 0,
        "configuration_conformance": None,
        "reason_checks": 20,
        "reason_checks_passed": 20,
        "reason_conformance": 1.0,
    }
    assert report["failure_ids"] == []
    assert report["by_family"]["tetrahedral"]["conformance"] == 1.0
    assert report["by_family"]["double_bond"]["conformance"] == 1.0
    assert report["by_family"]["cumulene_axis"]["conformance"] == 1.0
    assert report["by_family"]["extended_cis_trans"]["conformance"] == 1.0
    assert report["by_family"]["atrop_axis"]["conformance"] == 1.0
    assert report["by_family"]["square_planar"]["conformance"] == 1.0
    assert report["by_family"]["trigonal_bipyramidal"]["conformance"] == 1.0
    assert report["by_family"]["octahedral"]["conformance"] == 1.0
    assert report["by_family"]["helical"]["conformance"] == 1.0
    assert report["by_family"]["planar_chirality"]["conformance"] == 1.0


def test_perception_runner_erases_configuration_before_carrier_detection() -> None:
    tetrahedral = neutralize_rdkit_configuration(
        Chem.MolFromSmiles("F[C@H](Cl)Br")
    )
    double_bond = neutralize_rdkit_configuration(
        Chem.MolFromSmiles("F/C(Cl)=C(/Br)I")
    )
    assert detect_potential_stereo_elements(
        tetrahedral
    )[0].configuration_state is StereoConfigurationState.UNSPECIFIED
    assert detect_potential_stereo_elements(
        double_bond
    )[0].configuration_state is StereoConfigurationState.UNSPECIFIED

    for smiles in (
        "F[Pt@SP1](Cl)(Br)I",
        "F[P@TB1](Cl)(Br)(I)N",
        "F[Co@OH1](Cl)(Br)(I)(N)P",
    ):
        neutral = neutralize_rdkit_configuration(Chem.MolFromSmiles(smiles))
        descriptor = next(
            iter(descriptors_from_rdkit(neutral, require_atom_maps=False).values())
        )
        assert descriptor.parity is None


def test_frozen_conformance_report_matches_the_live_stable_result() -> None:
    frozen = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    live = benchmark_perception_conformance(DATASET)

    for key in ("schema", "dataset", "protocol", "summary", "failure_ids", "by_family"):
        assert frozen[key] == live[key]
    assert [
        {
            key: value
            for key, value in record.items()
            if key not in {"seconds"}
        }
        for record in frozen["records"]
    ] == live["records"]
