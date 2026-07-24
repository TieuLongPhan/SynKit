"""Topology-only detection of potential molecular stereo axes."""

from __future__ import annotations

import networkx as nx
from rdkit import Chem

from synkit.Graph.Stereo.descriptors import Reference, virtual_reference
from synkit.Graph.Stereo.supports import AxisStereoSupport


def _cumulene_end_references(
    atom: Chem.Atom,
    axis_neighbor: int,
) -> tuple[Reference, Reference] | None:
    """Return zero-based terminal references for one cumulene end."""
    center = atom.GetIdx()
    references: list[Reference] = [
        neighbor.GetIdx()
        for neighbor in atom.GetNeighbors()
        if neighbor.GetIdx() != axis_neighbor
    ]
    hidden_hydrogens = int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())
    references.extend(virtual_reference("H", center) for _ in range(hidden_hydrogens))
    if len(references) == 1 and atom.GetAtomicNum() == 15:
        references.append(virtual_reference("LP", center))
    if len(references) != 2 or references[0] == references[1]:
        return None
    return references[0], references[1]


def _potential_cumulene_supports(
    molecule: Chem.Mol,
) -> tuple[AxisStereoSupport, ...]:
    """Detect even-cumulene axes without assigning their orientation."""
    double_graph = nx.Graph()
    double_graph.add_edges_from(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    supports = []
    for component in nx.connected_components(double_graph):
        axis = double_graph.subgraph(component)
        edge_count = axis.number_of_edges()
        if edge_count < 2 or edge_count % 2 != 0:
            continue
        if edge_count != axis.number_of_nodes() - 1:
            continue
        degrees = dict(axis.degree())
        if any(degree > 2 for degree in degrees.values()):
            continue
        ends = sorted(node for node, degree in degrees.items() if degree == 1)
        if len(ends) != 2:
            continue
        path = nx.shortest_path(axis, ends[0], ends[1])
        if any(molecule.GetAtomWithIdx(node).GetDegree() != 2 for node in path[1:-1]):
            continue
        left_refs = _cumulene_end_references(
            molecule.GetAtomWithIdx(path[0]),
            path[1],
        )
        right_refs = _cumulene_end_references(
            molecule.GetAtomWithIdx(path[-1]),
            path[-2],
        )
        if left_refs is None or right_refs is None:
            continue
        supports.append(AxisStereoSupport(tuple(path), (left_refs, right_refs)))
    return tuple(supports)


def _potential_biaryl_supports(
    molecule: Chem.Mol,
    *,
    include_extended_ring_axes: bool,
) -> tuple[AxisStereoSupport, ...]:
    """Detect ring-associated bond axes without orientation or barrier claims."""
    supports = []
    for bond in molecule.GetBonds():
        if bond.GetBondType() not in {
            Chem.BondType.SINGLE,
            Chem.BondType.AROMATIC,
        }:
            continue
        left_atom = bond.GetBeginAtom()
        right_atom = bond.GetEndAtom()
        hybridizations = {
            left_atom.GetHybridization(),
            right_atom.GetHybridization(),
        }
        containing_ring_sizes = tuple(
            len(ring)
            for ring in molecule.GetRingInfo().BondRings()
            if bond.GetIdx() in ring
        )
        aromatic_ring_axis = (
            bond.GetIsAromatic()
            and left_atom.GetIsAromatic()
            and right_atom.GetIsAromatic()
            and any(size > 6 for size in containing_ring_sizes)
        )
        linked_ring_axis = (
            not bond.GetIsAromatic()
            and (
                left_atom.IsInRing()
                or right_atom.IsInRing()
            )
            and Chem.HybridizationType.SP2 in hybridizations
            and hybridizations
            <= {
                Chem.HybridizationType.SP2,
                Chem.HybridizationType.SP3,
            }
        )
        standard_biaryl_axis = (
            not bond.GetIsAromatic()
            and left_atom.GetIsAromatic()
            and right_atom.GetIsAromatic()
        )
        if not standard_biaryl_axis and not (
            include_extended_ring_axes
            and (aromatic_ring_axis or linked_ring_axis)
        ):
            continue
        left, right = left_atom.GetIdx(), right_atom.GetIdx()
        left_refs = tuple(
            neighbor.GetIdx()
            for neighbor in left_atom.GetNeighbors()
            if neighbor.GetIdx() != right_atom.GetIdx()
        )
        right_refs = tuple(
            neighbor.GetIdx()
            for neighbor in right_atom.GetNeighbors()
            if neighbor.GetIdx() != left_atom.GetIdx()
        )
        if len(left_refs) != 2 or len(right_refs) != 2:
            continue
        supports.append(
            AxisStereoSupport(
                (left, right),
                (left_refs, right_refs),
            )
        )
    return tuple(supports)


def detect_potential_axis_supports(
    molecule: Chem.Mol,
    *,
    include_extended_ring_axes: bool = False,
) -> tuple[tuple[str, AxisStereoSupport], ...]:
    """Return typed topology supports for potential molecular stereo axes."""
    if molecule is None:
        raise ValueError("Potential stereo-locus detection requires a molecule.")
    working = Chem.Mol(molecule)
    return (
        *(
            ("cumulene_axis", support)
            for support in _potential_cumulene_supports(working)
        ),
        *(
            ("atrop_axis", support)
            for support in _potential_biaryl_supports(
                working,
                include_extended_ring_axes=include_extended_ring_axes,
            )
        ),
    )
