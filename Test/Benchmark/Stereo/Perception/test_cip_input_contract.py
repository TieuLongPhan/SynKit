"""Information-identifiability gates for the pinned CIP suite."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Geometry import Point3D

from Experiment.Stereo.Perception.cip_input_contract import (
    SCHEMA,
    _normalized_oriented_volume,
    audit_cip_input_contract,
)

ROOT = Path(__file__).resolve().parents[4]
CIP = Path("/tmp/cip-validation-suite-compounds.smi")
CIP_3D = Path("/tmp/cip-validation-suite-compounds-3d.sdf")
FROZEN = ROOT / "Experiment" / "Stereo" / "Data" / "CIP" / "input_contract_audit.json"


def test_pinned_cip_input_contract_has_a_proven_299_record_ceiling() -> None:
    if not CIP.exists():
        return
    report = audit_cip_input_contract(CIP)

    assert report["schema"] == SCHEMA
    assert report["maximum_exact_records_from_supplied_smiles"] == 299
    assert report["complete_300_record_assignment_identifiable"] is False
    assert report["conflicting_records"] == 2
    assert report["conflicting_input_groups"] == [
        {
            "record_ids": ["VS010", "VS011"],
            "stereo_units": ["HE"],
            "expected_label_sets": [
                {"labels": ["19M", "26M"], "record_ids": ["VS011"]},
                {"labels": ["19P", "26P"], "record_ids": ["VS010"]},
            ],
            "reason": (
                "Byte-identical molecular input has mutually exclusive "
                "configured label sets."
            ),
        }
    ]


def test_frozen_contract_audit_contains_no_benchmark_structure() -> None:
    report_text = FROZEN.read_text(encoding="utf-8")
    report = json.loads(report_text)

    assert report["schema"] == SCHEMA
    assert "SMILES" not in report_text
    assert report["dataset"]["structures_vendored"] is False
    assert report["complete_300_record_input_identifiable_with_coordinates"]
    assert report["coordinate_contract_extension"]["all_smiles_conflicts_distinguished"]

    manifest = json.loads(
        (ROOT / "Experiment" / "Stereo" / "Data" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    registered = manifest["designed_datasets"]["cip_input_contract_audit"]
    assert registered["sha256"] == hashlib.sha256(FROZEN.read_bytes()).hexdigest()


def test_pinned_3d_extension_resolves_the_smiles_conflict() -> None:
    if not CIP.exists() or not CIP_3D.exists():
        return
    report = audit_cip_input_contract(CIP, coordinate_path=CIP_3D)
    extension = report["coordinate_contract_extension"]

    assert report["complete_300_record_input_identifiable_with_coordinates"]
    assert extension["all_smiles_conflicts_distinguished"]
    assert extension["conflict_orientation_witnesses"] == [
        {
            "record_ids": ["VS010", "VS011"],
            "atom_positions_one_based": [1, 2, 3, 4],
            "orientations": [
                {"record_id": "VS010", "orientation": "positive"},
                {"record_id": "VS011", "orientation": "negative"},
            ],
            "distinguished": True,
            "invariance": "translation_rotation_and_positive_scale",
            "reflection_sensitive": True,
        }
    ]


def test_oriented_volume_is_proper_motion_invariant_and_reflection_sensitive() -> None:
    molecule = Chem.MolFromSmiles("CCCC")
    assert molecule is not None
    points = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def with_coordinates(
        transform: Callable[
            [tuple[float, float, float]],
            tuple[float, float, float],
        ],
    ) -> Chem.Mol:
        result = Chem.Mol(molecule)
        conformer = Chem.Conformer(result.GetNumAtoms())
        for index, point in enumerate(points):
            x, y, z = transform(point)
            conformer.SetAtomPosition(index, Point3D(x, y, z))
        result.AddConformer(conformer)
        return result

    identity = with_coordinates(lambda point: point)
    moved = with_coordinates(
        lambda point: (
            -3.0 * point[1] + 7.0,
            3.0 * point[0] - 2.0,
            3.0 * point[2] + 5.0,
        )
    )
    reflected = with_coordinates(lambda point: (-point[0], point[1], point[2]))
    atom_positions = (0, 1, 2, 3)

    reference = _normalized_oriented_volume(identity, atom_positions)
    assert _normalized_oriented_volume(moved, atom_positions) == reference
    assert _normalized_oriented_volume(reflected, atom_positions) == -reference
