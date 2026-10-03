"""Guard frozen source conventions and independent geometry certification."""

from copy import deepcopy
import hashlib
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from Experiment.Stereo.ChemicalValidation.coordination import (
    ChemicalInputUnresolved, IDEAL_OH, _pose_key, fit_octahedron,
    proper_oh_rotations, read_coordination, signed_tetrahedral_volume, source_molecule,
)
from Experiment.Stereo.Canonicalization.independent_model import assignment_key, mirror

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/chemical_coordination_selection.json"


@pytest.fixture
def record():
    return deepcopy(json.loads(MANIFEST.read_text())["records"][0])


def test_geometry_partition_has_thirty_proper_orbits():
    rotations = proper_oh_rotations()
    assert len(set(rotations)) == 24
    classes = {}
    for pose in permutations(range(6)):
        classes.setdefault(_pose_key(pose), []).append(pose)
    assert len(classes) == 30
    assert {len(values) for values in classes.values()} == {24}


def test_fit_is_invariant_under_arbitrary_proper_rotation_and_distance_scaling():
    axis = np.array((2, 3, -4), dtype=float)
    axis /= np.linalg.norm(axis)
    skew = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    angle = 0.713
    rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew
    before, audit = fit_octahedron(IDEAL_OH)
    after, moved = fit_octahedron((IDEAL_OH @ rotation) * np.arange(1, 7)[:, None])
    assert before == after
    assert audit["normalized_rmsd"] < 1e-12
    assert moved["normalized_rmsd"] < 1e-12


def test_coordinate_reflection_is_a_different_proper_octahedral_pose():
    pose, _audit = fit_octahedron(IDEAL_OH)
    points = IDEAL_OH.copy()
    points[:, 0] *= -1
    reflected, _audit = fit_octahedron(points)
    assert _pose_key(pose) != _pose_key(reflected)


@pytest.mark.parametrize("vectors,reason", [
    (np.zeros((6, 3)), "zero_donor_vector"),
    (np.ones((6, 3)), "residual_exceeds"),
    (np.zeros((5, 3)), "invalid_six"),
    (np.full((6, 3), np.nan), "invalid_six"),
])
def test_bad_or_degenerate_geometry_is_refused(vectors, reason):
    with pytest.raises(ChemicalInputUnresolved, match=reason):
        fit_octahedron(vectors)


def test_tetrahedral_volume_flips_on_reflection_and_refuses_coplanarity():
    points = np.array(((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)), dtype=float)
    volume, parity = signed_tetrahedral_volume(points)
    points[:, 0] *= -1
    reflected_volume, reflected_parity = signed_tetrahedral_volume(points)
    assert reflected_volume == -volume
    assert reflected_parity == -parity
    points[:, 2] = 0
    with pytest.raises(ChemicalInputUnresolved, match="degenerate"):
        signed_tetrahedral_volume(points)


def test_parser_preserves_source_nitrogen_tag_and_explicit_hydride(record):
    metadata = json.loads((ROOT / record["path"]).read_text().splitlines()[1])
    molecule = source_molecule(metadata["smiles"])
    rows = {atom.GetAtomMapNum(): atom for atom in molecule.GetAtoms()}
    assert rows[9].GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    assert rows[4].GetSymbol() == "H"
    assert rows[4].GetIsotope() == 4  # A source donor slot, not a physical isotope.


def test_source_implicit_h_has_a_coordinate_witness_and_all_frames_survive(record):
    result = read_coordination(record, ROOT)
    assert [state.family for state in result.states] == ["octahedral", "tetrahedral"]
    assert result.states[1].atoms[-1] == "@H:9"
    assert result.audit["tetrahedral_annotations"][0]["reference_coordinate_rows"][-1] == 23
    assert result.graph.nodes[4]["color"][1] == 0
    assert assignment_key(tuple(mirror(state) for state in result.states)) == assignment_key(result.coordinate_states)


@pytest.mark.parametrize("mutation,reason", [
    ("digest", "hash_mismatch"), ("annotation", "frame_inventory_mismatch"),
    ("slot", "donor_label_mismatch"), ("duplicate_donor", "six_unique"),
])
def test_corrupted_frozen_source_metadata_is_rejected(record, mutation, reason):
    if mutation == "digest":
        record["sha256"] = "0" * 64
    elif mutation == "annotation":
        record["source_local_orientation_annotations"] = []
    elif mutation == "slot":
        record["source_configuration_labels"][0]["element"] = "C"
    else:
        record["source_configuration_labels"][1]["source_coordinate_index"] = 1
    with pytest.raises(ChemicalInputUnresolved, match=reason):
        read_coordination(record, ROOT)


def test_reflected_xyz_with_unchanged_source_tag_is_rejected(record, tmp_path):
    lines = (ROOT / record["path"]).read_text().splitlines()
    for index in range(2, 2 + int(lines[0])):
        entries = lines[index].split()
        entries[1] = str(-float(entries[1]))
        lines[index] = " ".join(entries)
    path = tmp_path / "reflected.xyz"
    path.write_text("\n".join(lines) + "\n")
    record["path"] = str(path)
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ChemicalInputUnresolved, match="tag_disagrees"):
        read_coordination(record, tmp_path)
