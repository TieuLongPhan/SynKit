"""Exact local-orientation constraints for molecular graph automorphisms.

The constraints encode only relative local input orientation. They do not
contain CIP labels, ranks, or a configuration at the focal center.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Mapping

import networkx as nx
from rdkit import Chem

from synkit.Graph.Stereo.descriptors import Reference, virtual_reference

from ._chirality_loci import _potential_cumulene_supports


_TETRAHEDRAL_TAGS = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
}
_PLANAR_STEREO = {
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
}
_PLANAR_TRANS = {
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOTRANS,
}
_PLANAR_FRAME_SYMMETRIES = (
    (0, 1, 2, 3, 4, 5),
    (1, 0, 2, 3, 5, 4),
    (4, 5, 3, 2, 0, 1),
    (5, 4, 3, 2, 1, 0),
)
_NODE_ATTRIBUTES = (
    "atomic_number",
    "isotope",
    "formal_charge",
    "radical_electrons",
    "hcount",
    "aromatic",
    "stereo_anchor",
    "stereo_marker",
    "ligand_probe",
)
_EDGE_ATTRIBUTES = ("bond_type", "aromatic")
_NODE_DEFAULTS = (0, 0, 0, 0, 0, False, False, None, False)
_EDGE_DEFAULTS = ("", False)


def _hidden_hydrogens(atom: Chem.Atom) -> int:
    return int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())


def _tetrahedral_lone_pair(atom: Chem.Atom) -> bool:
    coordination = atom.GetDegree() + _hidden_hydrogens(atom)
    return (
        atom.GetAtomicNum() in {7, 15, 16}
        and coordination == 3
        and atom.GetFormalCharge() <= 0
    )


def _relabel_reference(
    reference: Reference,
    mapping: Mapping[int, int],
) -> Reference:
    if type(reference) is int:
        return mapping[reference]
    prefix, owner = reference.rsplit(":", 1)
    return f"{prefix}:{mapping[int(owner)]}"


def _permutation_sign(
    values: tuple[Reference, ...],
    ordered: tuple[Reference, ...],
) -> int:
    positions = {value: index for index, value in enumerate(ordered)}
    permutation = tuple(positions[value] for value in values)
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


@dataclass(frozen=True)
class LocalTetrahedralOrientation:
    """One oriented four-neighbor frame from the molecular input."""

    center: int
    references: tuple[Reference, Reference, Reference, Reference]


@dataclass(frozen=True)
class LocalPlanarOrientation:
    """One ordered six-slot local alkene/imine orientation frame."""

    bond: frozenset[int]
    frame: tuple[
        Reference,
        Reference,
        int,
        int,
        Reference,
        Reference,
    ]


@dataclass(frozen=True)
class LocalCumuleneOrientation:
    """One oriented neighbor frame over an even-bond cumulene path."""

    path: tuple[int, ...]
    terminal_frames: tuple[
        tuple[Reference, Reference],
        tuple[Reference, Reference],
    ]


@dataclass(frozen=True)
class LocalOrientationConstraints:
    """Local input orientations that restrict graph automorphisms."""

    tetrahedral: tuple[LocalTetrahedralOrientation, ...] = ()
    planar: tuple[LocalPlanarOrientation, ...] = ()
    cumulene: tuple[LocalCumuleneOrientation, ...] = ()


@dataclass(frozen=True)
class TetrahedralStabilizerResult:
    """Parity conclusion for the center stabilizer acting on four slots."""

    is_stereogenic: bool
    automorphism_checks: int
    orientation_preserving_checks: int
    odd_permutation_witness: tuple[tuple[int, int], ...] | None = None


def _virtual_planar_reference(atom: Chem.Atom) -> Reference | None:
    owner = atom.GetIdx()
    if _hidden_hydrogens(atom):
        return virtual_reference("H", owner)
    if atom.GetAtomicNum() in {7, 15} and atom.GetFormalCharge() <= 0:
        return virtual_reference("LP", owner)
    return None


def _tetrahedral_constraints(
    molecule: Chem.Mol,
    excluded_center: int | None,
) -> tuple[LocalTetrahedralOrientation, ...]:
    constraints = []
    for atom in molecule.GetAtoms():
        center = atom.GetIdx()
        tag = atom.GetChiralTag()
        if tag not in _TETRAHEDRAL_TAGS or center == excluded_center:
            continue
        references: list[Reference] = [
            neighbor.GetIdx() for neighbor in atom.GetNeighbors()
        ]
        references.extend(
            virtual_reference("H", center)
            for _ in range(_hidden_hydrogens(atom))
        )
        if _tetrahedral_lone_pair(atom):
            references.append(virtual_reference("LP", center))
        if len(references) != 4 or len(set(references)) != 4:
            continue
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            references[0], references[1] = references[1], references[0]
        constraints.append(
            LocalTetrahedralOrientation(
                center,
                tuple(references),  # type: ignore[arg-type]
            )
        )
    return tuple(constraints)


def _planar_constraints(
    molecule: Chem.Mol,
) -> tuple[LocalPlanarOrientation, ...]:
    constraints = []
    for bond in molecule.GetBonds():
        stereo = bond.GetStereo()
        if (
            bond.GetBondType() != Chem.BondType.DOUBLE
            or stereo not in _PLANAR_STEREO
        ):
            continue
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        selected = tuple(bond.GetStereoAtoms())
        if len(selected) != 2:
            continue
        selected_left, selected_right = selected
        left_references: list[Reference] = [selected_left]
        left_references.extend(
            neighbor.GetIdx()
            for neighbor in molecule.GetAtomWithIdx(left).GetNeighbors()
            if neighbor.GetIdx() not in {right, selected_left}
        )
        right_references: list[Reference] = [selected_right]
        right_references.extend(
            neighbor.GetIdx()
            for neighbor in molecule.GetAtomWithIdx(right).GetNeighbors()
            if neighbor.GetIdx() not in {left, selected_right}
        )
        for references, owner in (
            (left_references, left),
            (right_references, right),
        ):
            if len(references) == 1:
                virtual = _virtual_planar_reference(
                    molecule.GetAtomWithIdx(owner)
                )
                if virtual is not None:
                    references.append(virtual)
        if len(left_references) != 2 or len(right_references) != 2:
            continue
        if stereo in _PLANAR_TRANS:
            right_references.reverse()
        constraints.append(
            LocalPlanarOrientation(
                frozenset((left, right)),
                (
                    *left_references,
                    left,
                    right,
                    *right_references,
                ),  # type: ignore[arg-type]
            )
        )
    return tuple(constraints)


def _cumulene_constraints(
    molecule: Chem.Mol,
) -> tuple[LocalCumuleneOrientation, ...]:
    constraints = []
    for support in _potential_cumulene_supports(molecule):
        midpoint = len(support.path) // 2
        center = support.path[midpoint]
        atom = molecule.GetAtomWithIdx(center)
        tag = atom.GetChiralTag()
        if tag not in _TETRAHEDRAL_TAGS or atom.GetDegree() != 2:
            continue
        path = support.path
        terminal_frames = support.terminal_frames
        local_axis_neighbors = tuple(
            neighbor.GetIdx() for neighbor in atom.GetNeighbors()
        )
        if path[midpoint - 1] != local_axis_neighbors[0]:
            path = tuple(reversed(path))
            terminal_frames = (
                terminal_frames[1],
                terminal_frames[0],
            )
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            terminal_frames = (
                tuple(reversed(terminal_frames[0])),
                terminal_frames[1],
            )
        constraints.append(
            LocalCumuleneOrientation(path, terminal_frames)
        )
    return tuple(constraints)


def extract_local_orientation_constraints(
    molecule: Chem.Mol,
    *,
    excluded_tetrahedral_center: int | None = None,
) -> LocalOrientationConstraints:
    """Extract label-free local orientation relations from molecular input."""
    return LocalOrientationConstraints(
        _tetrahedral_constraints(molecule, excluded_tetrahedral_center),
        _planar_constraints(molecule),
        _cumulene_constraints(molecule),
    )


def _preserves_tetrahedral_constraints(
    mapping: Mapping[int, int],
    constraints: tuple[LocalTetrahedralOrientation, ...],
) -> bool:
    by_center = {constraint.center: constraint for constraint in constraints}
    for source in constraints:
        if source.center not in mapping:
            continue
        target = by_center.get(mapping[source.center])
        if target is None:
            return False
        mapped = tuple(
            _relabel_reference(reference, mapping)
            for reference in source.references
        )
        if set(mapped) != set(target.references):
            return False
        if _permutation_sign(mapped, target.references) != 1:
            return False
    return True


def _preserves_planar_constraints(
    mapping: Mapping[int, int],
    constraints: tuple[LocalPlanarOrientation, ...],
) -> bool:
    by_bond = {constraint.bond: constraint for constraint in constraints}
    for source in constraints:
        if not source.bond <= mapping.keys():
            continue
        mapped_bond = frozenset(mapping[atom] for atom in source.bond)
        target = by_bond.get(mapped_bond)
        if target is None:
            return False
        mapped_frame = tuple(
            _relabel_reference(reference, mapping)
            for reference in source.frame
        )
        preserving_frames = {
            tuple(target.frame[index] for index in permutation)
            for permutation in _PLANAR_FRAME_SYMMETRIES
        }
        if mapped_frame not in preserving_frames:
            return False
    return True


def _cumulene_normalized_frame(
    constraint: LocalCumuleneOrientation,
) -> tuple[Reference, Reference, int, int, Reference, Reference]:
    left, right = constraint.terminal_frames
    frame: tuple[
        Reference,
        Reference,
        int,
        int,
        Reference,
        Reference,
    ] = (*left, constraint.path[0], constraint.path[-1], *right)
    return frame


def _cumulene_variants(
    constraint: LocalCumuleneOrientation,
) -> set[tuple[tuple[int, ...], tuple[Reference, ...]]]:
    frame = _cumulene_normalized_frame(constraint)
    return {
        (
            (
                tuple(reversed(constraint.path))
                if permutation[2] == 3
                else constraint.path
            ),
            tuple(frame[index] for index in permutation),
        )
        for permutation in _PLANAR_FRAME_SYMMETRIES
    }


def _preserves_cumulene_constraints(
    mapping: Mapping[int, int],
    constraints: tuple[LocalCumuleneOrientation, ...],
) -> bool:
    by_atoms = {
        frozenset(constraint.path): constraint
        for constraint in constraints
    }
    for source in constraints:
        if not set(source.path) <= mapping.keys():
            continue
        mapped_atoms = frozenset(mapping[atom] for atom in source.path)
        target = by_atoms.get(mapped_atoms)
        if target is None:
            return False
        source_path, source_frame = min(
            _cumulene_variants(source),
            key=repr,
        )
        mapped = (
            tuple(mapping[atom] for atom in source_path),
            tuple(
                _relabel_reference(reference, mapping)
                for reference in source_frame
            ),
        )
        if mapped not in _cumulene_variants(target):
            return False
    return True


def mapping_preserves_local_orientations(
    mapping: Mapping[int, int],
    constraints: LocalOrientationConstraints,
) -> bool:
    """Return whether one graph map preserves every local orientation."""
    return _preserves_tetrahedral_constraints(
        mapping,
        constraints.tetrahedral,
    ) and _preserves_planar_constraints(
        mapping,
        constraints.planar,
    ) and _preserves_cumulene_constraints(mapping, constraints.cumulene)


def _vf2pp_labeled_graph(
    graph: nx.Graph,
    node_attributes: tuple[str, ...],
    node_defaults: tuple[object, ...],
) -> nx.Graph:
    """Encode node and edge attributes as one VF2++ node label."""
    labeled = nx.Graph()
    for node, attributes in graph.nodes(data=True):
        label = (
            "atom",
            *(
                attributes.get(name, default)
                for name, default in zip(node_attributes, node_defaults)
            ),
        )
        labeled.add_node(node, _synkit_label=label)
    for sequence, (left, right, attributes) in enumerate(
        graph.edges(data=True)
    ):
        edge_node = ("_synkit_edge", sequence)
        label = (
            "bond",
            *(
                attributes.get(name, default)
                for name, default in zip(
                    _EDGE_ATTRIBUTES,
                    _EDGE_DEFAULTS,
                )
            ),
        )
        labeled.add_node(edge_node, _synkit_label=label)
        labeled.add_edge(left, edge_node)
        labeled.add_edge(edge_node, right)
    return labeled


def tetrahedral_stabilizer_result(
    graph: nx.Graph,
    references: tuple[Reference, Reference, Reference, Reference],
    *,
    constraints: LocalOrientationConstraints | None = None,
) -> TetrahedralStabilizerResult:
    """Test whether the center stabilizer contains an odd slot permutation.

    A tetrahedral configuration is stereogenic exactly when every allowed
    center-fixing automorphism induces an even permutation of its four slots.
    """
    if len(set(references)) != 4:
        return TetrahedralStabilizerResult(False, 0, 0)
    active = constraints or LocalOrientationConstraints()
    checks = 0
    preserving_checks = 0
    slot_attribute = "_synkit_tetrahedral_slot"
    node_attributes = (*_NODE_ATTRIBUTES, slot_attribute)
    node_defaults = (*_NODE_DEFAULTS, None)
    for image_indices in permutations(range(4)):
        mapped_references = tuple(
            references[index] for index in image_indices
        )
        if _permutation_sign(mapped_references, references) != -1:
            continue
        left_graph = graph.copy()
        right_graph = graph.copy()
        nx.set_node_attributes(left_graph, None, slot_attribute)
        nx.set_node_attributes(right_graph, None, slot_attribute)
        compatible = True
        for slot, (source, target) in enumerate(
            zip(references, mapped_references)
        ):
            if type(source) is int and type(target) is int:
                left_graph.nodes[source][slot_attribute] = slot
                right_graph.nodes[target][slot_attribute] = slot
            elif source != target:
                compatible = False
                break
        if not compatible:
            continue
        left_labeled = _vf2pp_labeled_graph(
            left_graph,
            node_attributes,
            node_defaults,
        )
        right_labeled = _vf2pp_labeled_graph(
            right_graph,
            node_attributes,
            node_defaults,
        )
        for labeled_mapping in nx.vf2pp_all_isomorphisms(
            left_labeled,
            right_labeled,
            node_label="_synkit_label",
        ):
            mapping = {
                source: target
                for source, target in labeled_mapping.items()
                if type(source) is int and type(target) is int
            }
            checks += 1
            if not mapping_preserves_local_orientations(mapping, active):
                continue
            preserving_checks += 1
            return TetrahedralStabilizerResult(
                False,
                checks,
                preserving_checks,
                tuple(sorted(mapping.items())),
            )
    return TetrahedralStabilizerResult(True, checks, preserving_checks)


__all__ = [
    "LocalOrientationConstraints",
    "TetrahedralStabilizerResult",
    "extract_local_orientation_constraints",
    "mapping_preserves_local_orientations",
    "tetrahedral_stabilizer_result",
]
