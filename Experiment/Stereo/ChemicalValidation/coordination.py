"""Read MACE source conventions and certify geometry without production helpers.

The local OH frame is inferred from all 720 donor/slot assignments under
proper rotations. Supplied tetrahedral annotations are retained and checked
against signed coordinate volumes. This module never imports SynKit.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from itertools import permutations, product
import json
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem

from Experiment.Stereo.Canonicalization.independent_model import State

IDEAL_OH = np.array(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)), dtype=float)
REGISTERED_BASIS = (4, 5, 0, 2, 1, 3)
RMSD_LIMIT = 0.35
POSE_GAP_MINIMUM = 0.02


class ChemicalInputUnresolved(ValueError):
    """The frozen chemical input does not certify a configured state."""


@dataclass
class CoordinationInput:
    graph: nx.Graph
    states: tuple[State, ...]
    coordinate_states: tuple[State, ...]
    audit: dict


@lru_cache(maxsize=1)
def proper_oh_rotations():
    """Derive the 24 signed permutation rotations from integer geometry."""
    rotations = []
    for axes in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.eye(3, dtype=int)[:, axes] * signs
            if round(np.linalg.det(matrix)) != 1:
                continue
            image = IDEAL_OH @ matrix
            rotations.append(tuple(next(j for j, other in enumerate(IDEAL_OH)
                                        if np.array_equal(point, other)) for point in image))
    return tuple(sorted(rotations))


def _pose_key(pose):
    return min(tuple(pose[i] for i in rotation) for rotation in proper_oh_rotations())


def fit_octahedron(vectors):
    """Certify a nearest proper pose or refuse the fixed residual/gap limits."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.shape != (6, 3) or not np.all(np.isfinite(vectors)):
        raise ChemicalInputUnresolved("invalid_six_donor_vectors")
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-12):
        raise ChemicalInputUnresolved("zero_donor_vector")
    vectors = vectors / lengths[:, None]
    poses = {}
    for pose in permutations(range(6)):
        target = vectors[list(pose)]
        left, _values, right = np.linalg.svd(IDEAL_OH.T @ target)
        proper = left @ np.diag((1, 1, np.linalg.det(left @ right))) @ right
        residual = float(np.sqrt(np.mean(np.sum((IDEAL_OH @ proper - target) ** 2, axis=1))))
        key = _pose_key(pose)
        poses[key] = min(residual, poses.get(key, float("inf")))
    ordered = sorted((residual, pose) for pose, residual in poses.items())
    best, pose = ordered[0]
    gap = ordered[1][0] - best
    audit = {"proper_rotation_count": len(proper_oh_rotations()), "raw_assignments": 720,
             "inequivalent_pose_count": len(poses), "normalized_rmsd": best,
             "next_inequivalent_pose_gap": gap, "rmsd_limit": RMSD_LIMIT,
             "pose_gap_minimum": POSE_GAP_MINIMUM, "slot_to_donor": list(pose)}
    if best > RMSD_LIMIT:
        raise ChemicalInputUnresolved(f"oh_residual_exceeds_frozen_limit:{best}")
    if gap < POSE_GAP_MINIMUM:
        raise ChemicalInputUnresolved(f"oh_pose_not_separated_by_frozen_gap:{gap}")
    return tuple(pose[i] for i in REGISTERED_BASIS), audit


def signed_tetrahedral_volume(points):
    """Use an ordered tetrahedron, with a dimensionless degeneracy check."""
    points = np.asarray(points, dtype=float)
    if points.shape != (4, 3) or not np.all(np.isfinite(points)):
        raise ChemicalInputUnresolved("invalid_tetrahedral_coordinate_witness")
    edges = points[:3] - points[3]
    scale = float(np.prod(np.linalg.norm(edges, axis=1)))
    volume = float(np.linalg.det(edges))
    if scale <= 1e-12 or abs(volume) / scale <= 1e-6:
        raise ChemicalInputUnresolved("degenerate_tetrahedral_coordinate_witness")
    return volume, -1 if volume > 0 else 1


def source_molecule(smiles):
    """Keep source tags and all explicit H; no chirality cleanup is allowed."""
    params = Chem.SmilesParserParams()
    params.sanitize = False
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None:
        raise ChemicalInputUnresolved("source_smiles_parse_failure")
    options = Chem.SanitizeFlags.SANITIZE_ALL & ~Chem.SanitizeFlags.SANITIZE_CLEANUPCHIRALITY
    Chem.SanitizeMol(molecule, sanitizeOps=options)
    return molecule


def _source_rows(molecule, elements):
    rows = {atom.GetAtomMapNum(): atom for atom in molecule.GetAtoms()}
    if len(rows) != molecule.GetNumAtoms():
        raise ChemicalInputUnresolved("source_coordinate_map_not_injective")
    if any(row < 0 or row >= len(elements) or atom.GetSymbol() != elements[row]
           for row, atom in rows.items()):
        raise ChemicalInputUnresolved("source_coordinate_map_element_mismatch")
    return rows


def _source_graph(molecule, donor_rows):
    """Freeze a declared chemical profile, erasing source slot labels only.

    Every source DATIVE bond goes to the unique Mn/Ru atom; its orientation
    can therefore be reconstructed from endpoint types in this dataset.
    """
    graph = nx.Graph()
    for atom in molecule.GetAtoms():
        row = atom.GetAtomMapNum()
        isotope = 0 if row in donor_rows else atom.GetIsotope()
        graph.add_node(row, color=(atom.GetAtomicNum(), isotope, atom.GetFormalCharge(),
                                   atom.GetTotalNumHs(), atom.GetIsAromatic(), atom.GetNumRadicalElectrons()))
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            raise ChemicalInputUnresolved("source_bond_orientation_not_in_frozen_inventory")
        if bond.GetBondType() == Chem.BondType.DATIVE and right.GetSymbol() not in {"Mn", "Ru"}:
            raise ChemicalInputUnresolved("dative_direction_not_recoverable_from_declared_types")
        graph.add_edge(left.GetAtomMapNum(), right.GetAtomMapNum(),
                       color=(str(bond.GetBondType()), bond.GetIsAromatic()))
    return graph


def _validate_coordinate_constitution(rows, full_rows):
    """The explicit-H coordinate export must describe the same constitution."""
    def properties(atom):
        return (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
                atom.GetIsAromatic(), atom.GetNumRadicalElectrons())

    for row, atom in rows.items():
        full = full_rows[row]
        if properties(atom) != properties(full):
            raise ChemicalInputUnresolved("coordinate_export_atom_attributes_disagree")
        compact_neighbors = {neighbor.GetAtomMapNum() for neighbor in atom.GetNeighbors()}
        full_neighbors = {neighbor.GetAtomMapNum() for neighbor in full.GetNeighbors()}
        if compact_neighbors != full_neighbors.intersection(rows):
            raise ChemicalInputUnresolved("coordinate_export_connectivity_disagrees")
        hidden = full_neighbors.difference(rows)
        if len(hidden) != atom.GetTotalNumHs() or any(full_rows[node].GetSymbol() != "H" for node in hidden):
            raise ChemicalInputUnresolved("coordinate_export_hidden_hydrogen_count_disagrees")
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom).GetAtomMapNum()
            matching = full.GetOwningMol().GetBondBetweenAtoms(full.GetIdx(), full_rows[other].GetIdx())
            if matching.GetBondType() != bond.GetBondType() or matching.GetIsAromatic() != bond.GetIsAromatic():
                raise ChemicalInputUnresolved("coordinate_export_bond_attributes_disagree")


def _tetrahedral_frames(annotations, rows, full_rows, coordinates, states, coordinate_mirrors):
    actual = {row: str(atom.GetChiralTag()) for row, atom in rows.items()
              if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED}
    expected = {value["source_coordinate_index"]: value["source_chiral_tag"] for value in annotations}
    if actual != expected:
        raise ChemicalInputUnresolved("source_local_frame_inventory_mismatch")
    tetrahedral = []
    for annotation in annotations:
        owner = annotation["source_coordinate_index"]
        atom = rows[owner]
        if atom.GetSymbol() != annotation["element"]:
            raise ChemicalInputUnresolved("source_annotation_element_mismatch")
        refs = [neighbor.GetAtomMapNum() for neighbor in atom.GetNeighbors()]
        coordinate_refs = refs.copy()
        if len(refs) == 3 and atom.GetTotalNumHs() == 1:
            hydrogens = [neighbor.GetAtomMapNum() for neighbor in full_rows[owner].GetNeighbors()
                         if neighbor.GetSymbol() == "H" and neighbor.GetAtomMapNum() not in rows]
            if len(hydrogens) != 1:
                raise ChemicalInputUnresolved("owner_hydrogen_coordinate_witness_not_unique")
            refs.append(f"@H:{owner}")
            coordinate_refs.extend(hydrogens)
        if len(refs) != 4:
            raise ChemicalInputUnresolved("source_tetrahedral_frame_does_not_have_four_ports")
        volume, parity = signed_tetrahedral_volume(coordinates[coordinate_refs])
        tag_parity = {"CHI_TETRAHEDRAL_CW": 1, "CHI_TETRAHEDRAL_CCW": -1}.get(actual[owner])
        if parity != tag_parity:
            raise ChemicalInputUnresolved("source_tetrahedral_tag_disagrees_with_coordinate_volume")
        points = coordinates[coordinate_refs].copy()
        points[:, 0] *= -1
        mirror_volume, mirror_parity = signed_tetrahedral_volume(points)
        states.append(State("tetrahedral", (owner, *refs), parity))
        coordinate_mirrors.append(State("tetrahedral", (owner, *refs), mirror_parity))
        tetrahedral.append({"center_coordinate_row": owner, "reference_coordinate_rows": coordinate_refs,
                            "references": refs, "signed_volume": volume, "parity": parity,
                            "source_tag": actual[owner], "reflected_signed_volume": mirror_volume})
    return tetrahedral


def read_coordination(record, root):
    """Replay immutable XYZ bytes and infer all supplied configured frames."""
    raw = (Path(root) / record["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ChemicalInputUnresolved("source_xyz_hash_mismatch")
    lines = raw.decode("utf-8").splitlines()
    count = int(lines[0])
    metadata = json.loads(lines[1])
    entries = [line.split() for line in lines[2:2 + count]]
    if len(entries) != count or any(len(entry) != 4 for entry in entries):
        raise ChemicalInputUnresolved("first_xyz_conformer_incomplete")
    elements = [entry[0] for entry in entries]
    coordinates = np.array([list(map(float, entry[1:])) for entry in entries])
    molecule = source_molecule(metadata["smiles"])
    full = source_molecule(metadata["smiles3D"])
    rows, full_rows = _source_rows(molecule, elements), _source_rows(full, elements)
    if set(full_rows) != set(range(count)):
        raise ChemicalInputUnresolved("all_coordinate_rows_need_source_atom_witnesses")
    _validate_coordinate_constitution(rows, full_rows)
    labels = record["source_configuration_labels"]
    donor_rows = tuple(value["source_coordinate_index"] for value in labels)
    if len(set(donor_rows)) != 6 or sorted(value["source_slot_label"] for value in labels) != list(range(1, 7)):
        raise ChemicalInputUnresolved("source_donor_inventory_not_six_unique_slots")
    for value in labels:
        atom = rows.get(value["source_coordinate_index"])
        if (atom is None or atom.GetSymbol() != value["element"] or
                atom.GetIsotope() != value["source_slot_label"]):
            raise ChemicalInputUnresolved("source_donor_label_mismatch")
    centers = [row for row, atom in rows.items() if atom.GetSymbol() in {"Mn", "Ru"}]
    if len(centers) != 1:
        raise ChemicalInputUnresolved("coordination_input_needs_one_metal_center")
    center = centers[0]
    if {neighbor.GetAtomMapNum() for neighbor in rows[center].GetNeighbors()} != set(donor_rows):
        raise ChemicalInputUnresolved("supplied_donors_do_not_equal_metal_neighbors")
    vectors = coordinates[list(donor_rows)] - coordinates[center]
    pose, geometry = fit_octahedron(vectors)
    reflected = vectors.copy()
    reflected[:, 0] *= -1
    mirror_pose, mirror_geometry = fit_octahedron(reflected)
    states = [State("octahedral", (center, *(donor_rows[i] for i in pose)), 1)]
    coordinate_mirrors = [State("octahedral", (center, *(donor_rows[i] for i in mirror_pose)), 1)]
    tetrahedral = _tetrahedral_frames(
        record["source_local_orientation_annotations"], rows, full_rows, coordinates, states, coordinate_mirrors,
    )
    return CoordinationInput(_source_graph(molecule, set(donor_rows)), tuple(states), tuple(coordinate_mirrors),
                             {"id": record["id"], "source_sha256": record["sha256"],
                              "first_conformer_only": True, "donor_slot_isotopes_erased": True,
                              "atom_maps_are_coordinate_rows": True, "geometry": geometry,
                              "reflected_geometry": mirror_geometry, "tetrahedral_annotations": tetrahedral,
                              "supplied_local_frames_retained": len(tetrahedral), "carrier_count": len(states)})
