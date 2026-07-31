"""Integrity checks for the exhaustive three-dataset carrier benchmark."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from rdkit import Chem

from Experiment.Stereo.datasets import ROTA, load_rota
from Experiment.Stereo.Perception.family_executors import (
    neutralize_rdkit_configuration,
)
from Experiment.Stereo.Perception.full_detection import (
    _cip_molecule_with_local_orientation,
    _cip_reference_positions,
    _cip_reference_positions_by_case,
    _detect,
    _path_text,
)
from synkit.Chem.Molecule.stereo_perception import (
    StereoCarrierStatus,
    StereoElementType,
    detect_potential_stereo_elements,
)

ROOT = Path(__file__).resolve().parents[4]
REPORT = (
    ROOT
    / "Experiment"
    / "Stereo"
    / "Data"
    / "Perception"
    / "full_detection_report.json"
)
MANIFEST = ROOT / "Experiment" / "Stereo" / "Data" / "manifest.json"


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_cip_reference_mapping_preserves_family_and_descriptor_case() -> None:
    record = {
        "recommended_labels": ("1R", "2s", "3E", "4z", "5P", "6m"),
    }

    assert _cip_reference_positions(record) == {
        "tetrahedral:1",
        "tetrahedral:2",
        "planar:3",
        "planar:4",
        "axial:5",
        "axial:6",
    }
    upper, lower = _cip_reference_positions_by_case(record)
    assert upper == {"tetrahedral:1", "planar:3", "axial:5"}
    assert lower == {"tetrahedral:2", "planar:4", "axial:6"}


def test_expanded_axis_paths_are_direction_independent() -> None:
    assert _path_text((8, 9, 10)) == "8-9-10"
    assert _path_text((10, 9, 8)) == "8-9-10"


def test_large_rota_symmetry_case_completes_with_exact_statuses() -> None:
    row = load_rota(ROTA)[292]
    molecule = neutralize_rdkit_configuration(Chem.MolFromSmiles(row["SMILES"]))
    axes = [
        element
        for element in detect_potential_stereo_elements(molecule)
        if element.element_type is StereoElementType.ATROP_AXIS
    ]

    confirmed = {
        tuple(element.support.path)
        for element in axes
        if element.carrier_status is StereoCarrierStatus.CONFIRMED
    }
    symmetry_related = {
        tuple(element.support.path)
        for element in axes
        if element.carrier_status is StereoCarrierStatus.SYMMETRY_RELATED
    }

    assert {(9, 10), (57, 80)} <= confirmed
    assert {(12, 20), (65, 66), (7, 105), (83, 91)} <= symmetry_related


def test_neighbor_configuration_mode_never_uses_the_focal_tag() -> None:
    molecule = Chem.MolFromSmiles("F[C@](Cl)([C@H](Br)I)[C@@H](Br)I")
    assert molecule is not None

    neutral, _, _ = _detect(
        molecule,
        check_renumbering=False,
        timeout_seconds=5.0,
    )
    assisted, _, _ = _detect(
        molecule,
        check_renumbering=False,
        timeout_seconds=5.0,
        retain_neighbor_configuration=True,
    )

    neutral_centers = {
        element.support.center
        for element in neutral
        if element.element_type is StereoElementType.TETRAHEDRAL
    }
    assisted_centers = {
        element.support.center
        for element in assisted
        if element.element_type is StereoElementType.TETRAHEDRAL
    }
    assert neutral_centers == {3, 6}
    assert assisted_centers == {1, 3, 6}


def test_preserved_focal_tag_cannot_prove_repeated_ligands_distinct() -> None:
    molecule = _cip_molecule_with_local_orientation("F[C@](F)(Cl)Br")
    assert molecule is not None
    assert molecule.GetAtomWithIdx(1).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED

    assisted, _, _ = _detect(
        molecule,
        check_renumbering=False,
        timeout_seconds=5.0,
        retain_neighbor_configuration=True,
    )

    assert not any(
        element.element_type is StereoElementType.TETRAHEDRAL
        and element.support.center == 1
        for element in assisted
    )


def test_manifest_registers_full_detection_for_all_source_datasets() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for dataset in manifest["datasets"]:
        assert dataset["protocols"]["full_configuration_neutral_carrier_detection"]
        assert dataset["full_detection_report"].endswith(
            "Perception/full_detection_report.json"
        )
    benchmark = manifest["empirical_benchmarks"][
        "full_configuration_neutral_carrier_detection"
    ]
    assert benchmark["source_rows"] == 1208
    assert benchmark["sampling"] == "none"
    assert not benchmark["pooled_accuracy"]
    digest = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    assert digest == benchmark["sha256"]


def test_frozen_report_covers_every_row_and_every_source_annotation() -> None:
    report = _report()
    datasets = report["datasets"]
    acs = datasets["acs_stereomolgraph_molecular_chirality"]
    rota = datasets["chiralfinder_rota"]
    cip = datasets["cip_validation_suite"]

    assert report["schema"] == "synkit.full-stereo-carrier-detection/4"
    assert report["coverage"] == {
        "dataset_rows": 1208,
        "acs_rows": 258,
        "rota_rows": 650,
        "cip_rows": 300,
        "unique_configuration_neutral_constitutions": 903,
        "acs_cip_shared_ids": 258,
        "acs_cip_identical_smiles": 258,
        "pooled_accuracy_reported": False,
    }
    assert len(acs["records"]) == acs["dataset_records"] == 258
    assert acs["dataset_reference_supplied_loci"] == 1007
    assert acs["by_reference_family"]["tetrahedral"]["broad"]["reference_loci"] == 916
    assert acs["by_reference_family"]["double_bond"]["broad"]["reference_loci"] == 91
    assert acs["broad"]["recovered_loci"] == 743
    assert (
        sum(len(record["broad_missed_supplied_loci"]) for record in acs["records"])
        == 264
    )
    assert (
        acs["two_setting_comparison"]["configuration_erased"]["true_positive_loci"]
        == 743
    )
    assert (
        acs["two_setting_comparison"]["local_neighbor_orientation_retained"][
            "true_positive_loci"
        ]
        == 1007
    )
    assert (
        acs["two_setting_comparison"]["local_neighbor_orientation_retained"][
            "additional_detected_loci"
        ]
        == 109
    )
    assert acs["configuration_erased_miss_structure"] == {
        "tetrahedral_misses_audited": 264,
        "by_configuration_erased_status": {"symmetry_related": 264},
        "by_atom_element": {"C": 264},
        "by_ring_membership": {"acyclic": 134, "ring": 130},
        "by_available_local_neighbor_frame_families": {
            "planar": 5,
            "tetrahedral": 249,
            "tetrahedral+planar": 10,
        },
    }
    assert not acs["local_neighbor_orientation_retained_renumbering"]["failures"]

    assert len(rota["records"]) == rota["dataset_records"] == 650
    assert rota["dataset_reference_loci"] == 698
    assert rota["dataset_expanded_reference_paths"] == 15
    assert rota["annotation_shape"] == {
        "two_atom_loci": 686,
        "single_atom_loci": 12,
    }
    assert rota["broad"]["true_positive_loci"] == 698
    assert rota["broad"]["recall"] == 1.0
    assert rota["broad_expanded_paths"]["true_positive_loci"] == 15
    assert not any(record["broad_missed_pairs"] for record in rota["records"])
    assert not any(record["broad_missed_expanded_paths"] for record in rota["records"])
    for setting in (
        "configuration_erased",
        "local_neighbor_orientation_retained",
    ):
        assert rota["two_setting_comparison"][setting]["true_positive_loci"] == 698
        assert rota["two_setting_comparison"][setting]["missed_loci"] == 0
        assert (
            rota["two_setting_comparison"][setting]["additional_detected_loci"] == 1562
        )
    assert not rota["local_neighbor_orientation_retained_renumbering"]["failures"]

    assert len(cip["records"]) == cip["dataset_records"] == 300
    assert cip["dataset_reference_positions"] == 1252
    assert cip["by_reference_class"]["tetrahedral"]["broad"]["reference_loci"] == 980
    assert cip["by_reference_class"]["planar"]["broad"]["reference_loci"] == 232
    assert cip["by_reference_class"]["axial"]["broad"]["reference_loci"] == 40
    assert cip["by_reference_case"]["uppercase"]["broad"]["reference_loci"] == 1044
    assert cip["by_reference_case"]["lowercase"]["broad"]["reference_loci"] == 208
    assert cip["broad"]["true_positive_loci"] == 969
    assert cip["confirmed"]["true_positive_loci"] == 959
    assert cip["by_reference_class"]["axial"]["broad"]["recovered_loci"] == 40
    assert cip["detected_carriers_by_family"]["helical"] == 2
    assert Counter(
        position.split(":")[0]
        for record in cip["records"]
        for position in record["broad_missed_positions"]
    ) == {"tetrahedral": 283}
    assert cip["neighbor_configuration_assisted"]["true_positive_loci"] == 1252
    assert (
        cip["by_reference_class"]["tetrahedral"]["neighbor_configuration_assisted"][
            "recovered_loci"
        ]
        == 980
    )
    assert not Counter(
        position.split(":")[0]
        for record in cip["records"]
        for position in record["neighbor_configuration_assisted_missed_positions"]
    )
    assert not cip["neighbor_configuration_assisted_renumbering"]["failures"]
    assert (
        cip["two_setting_comparison"]["configuration_erased"][
            "additional_detected_loci"
        ]
        == 19
    )
    assert (
        cip["two_setting_comparison"]["local_neighbor_orientation_retained"][
            "additional_detected_loci"
        ]
        == 57
    )
    assert cip["configuration_erased_miss_structure"] == {
        "tetrahedral_misses_audited": 283,
        "by_configuration_erased_status": {"symmetry_related": 283},
        "by_atom_element": {"C": 283},
        "by_ring_membership": {"acyclic": 137, "ring": 146},
        "by_available_local_neighbor_frame_families": {
            "cumulene": 1,
            "planar": 9,
            "tetrahedral": 261,
            "tetrahedral+planar": 12,
        },
    }


def test_report_contains_no_cip_structures_or_configuration_predictions() -> None:
    cip = _report()["datasets"]["cip_validation_suite"]

    assert cip["structures_vendored"] is False
    for record in cip["records"]:
        assert "smiles" not in record
        assert "structure" not in record
        for field in (
            "detected_carriers",
            "neighbor_configuration_assisted_detected_carriers",
        ):
            for carrier in record.get(field, ()):
                assert set(carrier) == {
                    "family",
                    "carrier_status",
                    "carrier_reason",
                    "support",
                }
