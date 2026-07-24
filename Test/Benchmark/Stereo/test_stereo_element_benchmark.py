"""Stereo-element audit tests; CIP labels are reference positions only."""

import json
from pathlib import Path

from rdkit import Chem

from Experiment.Stereo.Perception.stereo_elements import (
    _record_result,
    _reference_rs_positions,
)

ROOT = Path(__file__).resolve().parents[3]
FROZEN_REPORT = (
    ROOT
    / "Experiment"
    / "Stereo"
    / "Data"
    / "Perception"
    / "stereo_element_report.json"
)


def _record(
    labels: tuple[str, ...],
    *,
    identifier: str = "TEST001",
    units: str = "TH",
) -> dict[str, object]:
    return {
        "ID": identifier,
        "recommended_labels": labels,
        "stereo_units": units,
    }


def _molecule(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return molecule


def test_audit_detects_canonicalizes_and_attaches_configured_tetrahedron() -> None:
    result = _record_result(
        _record(("2R",)),
        _molecule("F[C@H](Cl)Br"),
    )

    assert result["reference_rs_positions"] == [2]
    assert result["confirmed_centers"] == [2]
    assert result["primary_centers"] == [2]
    assert result["stereo_dependent_centers"] == []
    assert result["canonical_frames"] == [2]
    assert result["supplied_tetrahedral_centers"] == [2]
    assert result["attached_configurations"] == [2]
    assert result["supplied_unsupported_geometry"] == []
    assert result["supplied_constitutionally_unresolved"] == []
    assert result["supplied_unresolved_key_collision"] == []
    assert result["reference_rs_confirmed"] == [2]
    assert result["reference_rs_canonicalized"] == [2]
    assert result["supplied_configuration_attached"] == [2]
    assert result["renumbering_invariant"]
    assert result["renumbering_issues"] == []


def test_audit_retains_symmetry_related_carrier_without_promoting_it() -> None:
    result = _record_result(
        _record((), units="none"),
        _molecule("CC(C)C"),
    )

    assert result["broad_carriers"] == [1, 2, 3, 4]
    assert result["symmetry_related_carriers"] == [1, 2, 3, 4]
    assert result["confirmed_centers"] == []
    assert result["canonical_frames"] == []
    assert result["attached_configurations"] == []
    assert result["renumbering_invariant"]


def test_audit_separates_primary_and_stereo_dependent_centers() -> None:
    result = _record_result(
        _record(("2r", "4R", "7S")),
        _molecule("F[C@](Cl)([C@H](Br)I)[C@@H](Br)I"),
    )

    assert result["primary_centers"] == [4, 7]
    assert result["stereo_dependent_centers"] == [2]
    assert result["canonical_frames"] == [4, 7]
    assert result["dependency_iterations"] == 1
    assert result["renumbering_invariant"]


def test_reference_position_filter_is_not_a_cip_prediction() -> None:
    record = _record(
        ("1R", "2S", "3r", "4s", "5E", "6Z", "7Ra", "8M"),
        units="TH,CT,AT,HE",
    )

    assert _reference_rs_positions(record) == (1, 2, 3, 4)


def test_audit_does_not_read_injected_cip_properties() -> None:
    molecule = _molecule("F[C@H](Cl)Br")
    baseline = _record_result(_record(("2R",)), molecule)
    for index, atom in enumerate(molecule.GetAtoms()):
        atom.SetProp("_CIPRank", str(index + 1000))
        atom.SetProp("_CIPCode", "S" if index % 2 else "R")

    modified = _record_result(_record(("2R",)), molecule)

    ignored = {"renumbering_issues"}
    assert {key: value for key, value in baseline.items() if key not in ignored} == {
        key: value for key, value in modified.items() if key not in ignored
    }


def test_skipped_renumbering_is_not_reported_as_a_pass() -> None:
    result = _record_result(
        _record(("2R",)),
        _molecule("F[C@H](Cl)Br"),
        check_renumbering=False,
    )

    assert result["renumbering_invariant"] is None
    assert result["renumbering_issues"] == []


def test_frozen_audit_covers_all_300_records_without_promoting_ties() -> None:
    report = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))

    assert report["schema"] == "synkit.stereo-element-audit/1"
    assert report["dataset"] == {
        "audited_sha256": (
            "df178635c00b6c41fad820d2609fc4ff18403c63dec4e5c3e1756a6db5858059"
        ),
        "records": 300,
        "structures_vendored": False,
    }
    assert report["parse_failures"] == []
    assert report["frame_statuses"] == {
        "canonical_primary": 660,
        "canonical_stereo_dependent": 135,
        "neighborhood_key_collision": 0,
        "symmetry_related": 1892,
    }
    assert report["totals"] == {
        "broad_carriers": 2687,
        "canonical_frames": 795,
        "confirmed_centers": 795,
        "primary_centers": 660,
        "reference_rs_canonicalized": 795,
        "reference_rs_confirmed": 795,
        "reference_rs_positions": 980,
        "renumbering_checked_records": 300,
        "renumbering_invariant_records": 300,
        "stereo_dependent_centers": 135,
        "supplied_configuration_attached": 795,
        "supplied_constitutionally_unresolved": 132,
        "supplied_tetrahedral_centers": 943,
        "supplied_unresolved_key_collision": 0,
        "supplied_unsupported_geometry": 16,
    }
    assert report["renumbering_failures"] == []


def test_frozen_audit_does_not_redistribute_benchmark_structures() -> None:
    report_text = FROZEN_REPORT.read_text(encoding="utf-8")

    assert '"SMILES"' not in report_text
