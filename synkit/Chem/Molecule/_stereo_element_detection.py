"""Internal non-tetrahedral stereo-element detection helpers."""

from __future__ import annotations

import networkx as nx
from rdkit import Chem

from synkit.Graph.Stereo.supports import (
    AxisStereoSupport,
    BondStereoSupport,
    PathStereoSupport,
)

from ._helical_loci import detect_helicene_supports
from ._stereo_axis_evidence import (
    StereoCarrierStatus,
    axis_carrier_status,
    cumulene_terminal_references,
)
from .stereo_perception import (
    PotentialStereoElement,
    StereoConfigurationState,
    StereoElementType,
    _configuration_state,
    _constitutional_graph,
    _cumulated_double_bonds,
)


def rdkit_double_bond_elements(
    molecule: Chem.Mol,
) -> tuple[PotentialStereoElement, ...]:
    """Project and complete double-bond perception as typed support evidence."""
    cumulated_bonds = _cumulated_double_bonds(molecule)
    elements = []
    perceived_bonds = set()
    for info in Chem.FindPotentialStereo(molecule):
        centered_on = int(info.centeredOn)
        if info.type != Chem.StereoType.Bond_Double:
            continue
        if centered_on in cumulated_bonds:
            continue
        perceived_bonds.add(centered_on)
        bond = molecule.GetBondWithIdx(centered_on)
        elements.append(
            PotentialStereoElement(
                StereoElementType.DOUBLE_BOND,
                BondStereoSupport(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                ),
                _configuration_state(info),
                "rdkit_find_potential_stereo",
                f"Bond_Double:{centered_on}",
            )
        )
    for bond in molecule.GetBonds():
        bond_index = bond.GetIdx()
        if (
            bond.GetBondType() != Chem.BondType.DOUBLE
            or bond_index in cumulated_bonds
            or bond_index in perceived_bonds
        ):
            continue
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left_refs = cumulene_terminal_references(molecule, left, right)
        right_refs = cumulene_terminal_references(molecule, right, left)
        if left_refs is None or right_refs is None:
            continue
        if not any(
            type(reference) is int
            and molecule.GetAtomWithIdx(reference).GetIsotope() != 0
            for frame in (left_refs, right_refs)
            for reference in frame
        ):
            continue
        support = AxisStereoSupport((left, right), (left_refs, right_refs))
        status, _reason = axis_carrier_status(
            _constitutional_graph(molecule, left),
            StereoElementType.DOUBLE_BOND.value,
            support,
        )
        if status is not StereoCarrierStatus.CONFIRMED:
            continue
        elements.append(
            PotentialStereoElement(
                StereoElementType.DOUBLE_BOND,
                BondStereoSupport(left, right),
                StereoConfigurationState.UNSPECIFIED,
                "synkit_double_bond_exact_terminal_symmetry",
                f"Bond_Double:{bond_index}",
            )
        )
    return tuple(elements)


def extended_cis_trans_elements(
    molecule: Chem.Mol,
) -> tuple[PotentialStereoElement, ...]:
    """Detect odd-bond extended cumulene E/Z supports without inventing state."""
    double_graph = nx.Graph()
    double_graph.add_edges_from(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    elements = []
    for component in nx.connected_components(double_graph):
        path_graph = double_graph.subgraph(component)
        edge_count = path_graph.number_of_edges()
        if edge_count < 3 or edge_count % 2 != 1:
            continue
        if edge_count != path_graph.number_of_nodes() - 1:
            continue
        if any(degree > 2 for _, degree in path_graph.degree()):
            continue
        ends = sorted(node for node, degree in path_graph.degree() if degree == 1)
        if len(ends) != 2:
            continue
        path = tuple(nx.shortest_path(path_graph, ends[0], ends[1]))
        if any(
            molecule.GetAtomWithIdx(node).GetDegree() != 2
            for node in path[1:-1]
        ):
            continue
        left = cumulene_terminal_references(molecule, path[0], path[1])
        right = cumulene_terminal_references(molecule, path[-1], path[-2])
        if left is None or right is None:
            continue
        support = AxisStereoSupport(path, (left, right))
        carrier_status, carrier_reason = axis_carrier_status(
            _constitutional_graph(molecule, support.path[0]),
            StereoElementType.EXTENDED_CIS_TRANS.value,
            support,
        )
        elements.append(
            PotentialStereoElement(
                StereoElementType.EXTENDED_CIS_TRANS,
                support,
                StereoConfigurationState.UNSPECIFIED,
                "synkit_odd_cumulene_path_perception",
                "Extended_CisTrans:" + "-".join(str(atom) for atom in path),
                carrier_status=carrier_status,
                carrier_reason=carrier_reason,
            )
        )
    return tuple(elements)


def axis_elements(
    molecule: Chem.Mol,
    *,
    include_extended_ring_axes: bool,
) -> tuple[PotentialStereoElement, ...]:
    """Adapt the conservative axial-locus detector to typed elements."""
    from .chirality import detect_potential_stereo_loci

    elements = []
    for locus in detect_potential_stereo_loci(
        molecule,
        include_extended_ring_axes=include_extended_ring_axes,
    ):
        element_type = StereoElementType(locus.locus_type.value)
        carrier_status, carrier_reason = axis_carrier_status(
            _constitutional_graph(molecule, locus.support.path[0]),
            element_type.value,
            locus.support,
        )
        elements.append(
            PotentialStereoElement(
                element_type,
                locus.support,
                StereoConfigurationState.UNSPECIFIED,
                locus.evidence_provenance,
                locus.identifier,
                carrier_status=carrier_status,
                carrier_reason=carrier_reason,
            )
        )
    return tuple(elements)


def helical_elements(
    molecule: Chem.Mol,
) -> tuple[PotentialStereoElement, ...]:
    """Return helicene-like path carriers without orientation claims."""
    return tuple(
        PotentialStereoElement(
            StereoElementType.HELICAL,
            PathStereoSupport(support.path, support.cyclic),
            StereoConfigurationState.UNSPECIFIED,
            "synkit_angular_fused_aromatic_helicene_topology",
            "Helical:" + "-".join(str(atom) for atom in support.path),
        )
        for support in detect_helicene_supports(molecule)
    )
