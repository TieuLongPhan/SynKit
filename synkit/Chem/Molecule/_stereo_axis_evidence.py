"""Exact constitutional evidence for broad molecular stereo-axis candidates."""

from __future__ import annotations

from enum import Enum

import networkx as nx
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)
from rdkit import Chem

from synkit.Graph.Stereo.descriptors import Reference, virtual_reference
from synkit.Graph.Stereo.supports import AxisStereoSupport


class StereoCarrierStatus(str, Enum):
    """Constitutional conclusion for a perceived carrier.

    ``CONFIRMED`` means only that the represented constitution has
    distinguishable carrier references. It is not a configurational-stability
    claim. ``SYMMETRY_RELATED`` retains a broad topology candidate while
    recording why it must not be promoted to a confirmed stereo carrier.
    """

    CONFIRMED = "confirmed"
    SYMMETRY_RELATED = "symmetry_related"


def validated_carrier_status(
    value: StereoCarrierStatus | str,
    reason: str | None,
) -> StereoCarrierStatus:
    """Normalize a carrier status and enforce its diagnostic invariant."""
    status = StereoCarrierStatus(value)
    if status is StereoCarrierStatus.SYMMETRY_RELATED and not reason:
        raise ValueError(
            "Symmetry-related stereo carriers require a diagnostic reason."
        )
    return status


_NODE_ATTRIBUTES = (
    "atomic_number",
    "isotope",
    "formal_charge",
    "radical_electrons",
    "hcount",
    "aromatic",
    "stereo_anchor",
    "stereo_marker",
    "axis_anchor",
    "ligand_probe",
)
_EDGE_ATTRIBUTES = ("bond_type", "aromatic")
_NODE_DEFAULTS = (0, 0, 0, 0, 0, False, False, None, None, False)
_EDGE_DEFAULTS = ("", False)


def _refinement_fingerprint(graph: nx.Graph) -> str:
    """Return an isomorphism-invariant rejection fingerprint.

    Unequal Weisfeiler-Lehman fingerprints prove that the rooted attributed
    graphs cannot be isomorphic. Equal fingerprints remain inconclusive and
    therefore continue to the exact VF2 witness check.
    """
    node_label = "_synkit_symmetry_node_label"
    edge_label = "_synkit_symmetry_edge_label"
    for _node, attributes in graph.nodes(data=True):
        attributes[node_label] = repr(
            tuple(
                attributes.get(name, default)
                for name, default in zip(_NODE_ATTRIBUTES, _NODE_DEFAULTS)
            )
        )
    for _left, _right, attributes in graph.edges(data=True):
        attributes[edge_label] = repr(
            tuple(
                attributes.get(name, default)
                for name, default in zip(_EDGE_ATTRIBUTES, _EDGE_DEFAULTS)
            )
        )
    return nx.weisfeiler_lehman_graph_hash(
        graph,
        node_attr=node_label,
        edge_attr=edge_label,
        iterations=max(1, graph.number_of_nodes()),
    )


def cumulene_terminal_references(
    molecule: Chem.Mol,
    owner: int,
    path_neighbor: int,
) -> tuple[Reference, Reference] | None:
    """Return two represented references at one cumulene endpoint."""
    atom = molecule.GetAtomWithIdx(owner)
    references: list[Reference] = [
        neighbor.GetIdx()
        for neighbor in atom.GetNeighbors()
        if neighbor.GetIdx() != path_neighbor
    ]
    hidden_hydrogens = int(atom.GetNumExplicitHs()) + int(
        atom.GetNumImplicitHs()
    )
    references.extend(
        virtual_reference("H", owner) for _ in range(hidden_hydrogens)
    )
    if len(references) == 1 and atom.GetAtomicNum() == 15:
        references.append(virtual_reference("LP", owner))
    if len(references) != 2 or references[0] == references[1]:
        return None
    return references[0], references[1]


def _axis_fixed_reference_symmetry_witness(
    graph: nx.Graph,
    support: AxisStereoSupport,
    left: Reference,
    right: Reference,
) -> tuple[tuple[int, int], ...] | None:
    """Return an exact terminal-reference automorphism witness, if one exists."""
    if type(left) is not int or type(right) is not int:
        return None
    left_rooted = graph.copy()
    right_rooted = graph.copy()
    for rooted, probe in ((left_rooted, left), (right_rooted, right)):
        nx.set_node_attributes(rooted, None, "axis_anchor")
        nx.set_node_attributes(rooted, False, "ligand_probe")
        for position, atom in enumerate(support.path):
            rooted.nodes[atom]["axis_anchor"] = position
        rooted.nodes[probe]["ligand_probe"] = True
    if _refinement_fingerprint(left_rooted) != _refinement_fingerprint(
        right_rooted
    ):
        return None
    matcher = GraphMatcher(
        left_rooted,
        right_rooted,
        node_match=categorical_node_match(
            _NODE_ATTRIBUTES,
            _NODE_DEFAULTS,
        ),
        edge_match=categorical_edge_match(
            _EDGE_ATTRIBUTES,
            _EDGE_DEFAULTS,
        ),
    )
    mapping = next(matcher.isomorphisms_iter(), None)
    if mapping is None:
        return None
    return tuple(sorted((int(source), int(target)) for source, target in mapping.items()))


def axis_terminal_symmetry_witnesses(
    graph: nx.Graph,
    support: AxisStereoSupport,
) -> tuple[tuple[tuple[int, int], ...] | None, ...]:
    """Return one exact symmetry witness for each terminal reference pair.

    Axis atoms are fixed in path order and one terminal ligand probe is mapped
    to the other. A non-``None`` witness therefore proves that terminal frame
    constitution cannot support a configured axis. This is a constitutional
    result only; it says nothing about rotational barriers.
    """
    return tuple(
        _axis_fixed_reference_symmetry_witness(
            graph,
            support,
            frame[0],
            frame[1],
        )
        for frame in support.terminal_frames
    )


def axis_carrier_status(
    graph: nx.Graph,
    element_type: str,
    support: AxisStereoSupport,
) -> tuple[StereoCarrierStatus, str | None]:
    """Return exact terminal-symmetry evidence for a broad axis candidate."""
    if any(
        witness is not None
        for witness in axis_terminal_symmetry_witnesses(graph, support)
    ):
        reason = (
            "symmetry_related_terminal_paths"
            if element_type == "atrop_axis"
            else "duplicate_terminal_ligands"
        )
        return StereoCarrierStatus.SYMMETRY_RELATED, reason
    return StereoCarrierStatus.CONFIRMED, None


__all__ = [
    "StereoCarrierStatus",
    "axis_carrier_status",
    "axis_terminal_symmetry_witnesses",
    "cumulene_terminal_references",
    "validated_carrier_status",
]
