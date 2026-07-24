"""Sprint 38 configured organic stereograph gates."""

from __future__ import annotations

from itertools import permutations

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    AtropBondStereo,
    PlanarBondStereo,
    TetrahedralStereo,
    canonicalize_rdkit_stereograph,
    canonicalize_rdkit_tetrahedral_stereograph,
    canonicalize_stereo_registry,
    canonicalize_stereograph,
    canonicalize_tetrahedral_stereograph,
    expand_stereograph,
    virtual_reference,
)
from synkit.Graph.Stereo.canonical import (
    STEREOGRAPH_SCHEMA,
    StereoPortVertex,
)
from synkit.Graph.Stereo.orbits import SHAPE_DEFINITIONS


def _alkene(colors: tuple[str, str, str, str]) -> nx.Graph:
    graph = nx.Graph()
    for node, color in {
        0: colors[0],
        1: colors[1],
        2: "alkene-carbon",
        3: "alkene-carbon",
        4: colors[2],
        5: colors[3],
    }.items():
        graph.add_node(node, color=color)
    for left, right in ((0, 2), (1, 2), (3, 4), (3, 5)):
        graph.add_edge(left, right, color="single")
    graph.add_edge(2, 3, color="double")
    return graph


def _planar(
    graph: nx.Graph,
    descriptor: PlanarBondStereo,
):
    return canonicalize_stereograph(
        graph,
        (descriptor,),
        atom_color="color",
        bond_color="color",
    )


def test_planar_bond_gadget_projects_exactly_v4() -> None:
    graph = _alkene(("same", "same", "same", "same"))
    descriptor = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)

    result = _planar(graph, descriptor)
    images = {
        tuple(
            witness.as_dict()[StereoPortVertex(0, position)].position
            for position in range(4)
        )
        for witness in result.port_automorphisms
    }

    assert len(result.auxiliary.canonical_order) == 44
    assert images == {
        (0, 1, 2, 3),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 1, 0),
    }
    assert result.schema == STEREOGRAPH_SCHEMA
    assert result.complete and result.exact


def test_planar_preserving_frames_share_one_code_and_opposite_differs() -> None:
    graph = _alkene(("fluorine", "chlorine", "bromine", "iodine"))
    descriptor = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)
    definition = SHAPE_DEFINITIONS["planar_bond"]
    equivalent_codes = {
        _planar(
            graph,
            PlanarBondStereo(permutation.apply(descriptor.atoms), 0),
        ).canonical_code
        for permutation in definition.preserving_group.elements
    }

    assert equivalent_codes == {_planar(graph, descriptor).canonical_code}
    assert _planar(graph, descriptor.invert()).canonical_code not in equivalent_codes


def test_one_end_exchange_inverts_but_two_end_exchange_preserves() -> None:
    graph = _alkene(("fluorine", "chlorine", "bromine", "iodine"))
    descriptor = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)
    one_end = PlanarBondStereo((1, 0, 2, 3, 4, 5), 0)
    two_ends = PlanarBondStereo((1, 0, 2, 3, 5, 4), 0)

    reference = _planar(graph, descriptor).canonical_code

    assert _planar(graph, one_end).canonical_code != reference
    assert _planar(graph, two_ends).canonical_code == reference


def test_all_720_atom_relabelings_preserve_planar_code() -> None:
    graph = _alkene(("fluorine", "chlorine", "bromine", "iodine"))
    descriptor = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)
    expected = _planar(graph, descriptor).canonical_code

    for images in permutations((10, 20, 30, 40, 50, 60)):
        mapping = dict(zip(graph, images))
        relabeled = nx.relabel_nodes(graph, mapping, copy=True)
        assert (
            _planar(relabeled, descriptor.relabel(mapping)).canonical_code == expected
        )


def test_planar_hidden_hydrogens_are_owner_scoped_resources() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(
        (
            (0, {"color": "methyl"}),
            (2, {"color": "alkene-carbon"}),
            (3, {"color": "alkene-carbon"}),
            (4, {"color": "ethyl"}),
        )
    )
    graph.add_edge(0, 2, color="single")
    graph.add_edge(2, 3, color="double")
    graph.add_edge(3, 4, color="single")
    descriptor = PlanarBondStereo(
        (
            0,
            virtual_reference("H", 2),
            2,
            3,
            4,
            virtual_reference("H", 3),
        ),
        0,
    )

    expansion = expand_stereograph(
        graph,
        (descriptor,),
        atom_color="color",
        bond_color="color",
    )

    assert (
        sum(
            attributes["color"] == ("virtual_resource", "H")
            for _, attributes in expansion.nodes(data=True)
        )
        == 2
    )
    assert (
        _planar(graph, descriptor).canonical_code
        != _planar(graph, descriptor.invert()).canonical_code
    )


def test_planar_support_must_match_the_central_bond_and_endpoint_owners() -> None:
    graph = _alkene(("fluorine", "chlorine", "bromine", "iodine"))

    with pytest.raises(ValueError, match="not a base-graph bond"):
        _planar(graph, PlanarBondStereo((1, 2, 0, 3, 4, 5), 0))

    with pytest.raises(ValueError, match="not bonded to owner"):
        _planar(graph, PlanarBondStereo((4, 1, 2, 3, 0, 5), 0))


def test_mixed_locus_order_is_nonsemantic_and_projections_are_complete() -> None:
    graph = _alkene(("fluorine", "chlorine", "bromine", "iodine"))
    graph.add_node(10, color="tetra-center")
    for node, color in {
        11: "hydrogen",
        12: "fluorine",
        13: "chlorine",
        14: "bromine",
    }.items():
        graph.add_node(node, color=color)
        graph.add_edge(10, node, color="single")
    descriptors = (
        TetrahedralStereo((10, 11, 12, 13, 14), 1),
        PlanarBondStereo((0, 1, 2, 3, 4, 5), 0),
    )
    options = {"atom_color": "color", "bond_color": "color"}

    first = canonicalize_stereograph(graph, descriptors, **options)
    second = canonicalize_stereograph(graph, tuple(reversed(descriptors)), **options)

    assert first.canonical_code == second.canonical_code
    assert len(first.atom_order) == len(graph)
    assert len(first.bond_order) == graph.number_of_edges()
    assert len(first.locus_order) == 2
    assert len(first.port_order) == 8
    assert len(first.locus_automorphisms) >= 1


def test_version_one_registry_accepts_tetrahedral_and_planar_only() -> None:
    graph = _alkene(("fluorine", "chlorine", "bromine", "iodine"))
    planar = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)
    expected = _planar(graph, planar)

    observed = canonicalize_stereo_registry(
        graph,
        {"bond:2-3": planar},
        atom_color="color",
        bond_color="color",
    )

    assert observed.same_stereograph(expected)
    with pytest.raises(TypeError, match="atrop_bond"):
        canonicalize_stereo_registry(
            graph,
            {"bond:2-3": AtropBondStereo(planar.atoms, 1)},
        )


def test_tetrahedral_compatibility_apis_remain_strict() -> None:
    graph = _alkene(("fluorine", "chlorine", "bromine", "iodine"))
    descriptor = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)

    with pytest.raises(TypeError, match="only TetrahedralStereo"):
        canonicalize_tetrahedral_stereograph(graph, (descriptor,))


def test_unknown_and_duplicate_planar_loci_fail_closed() -> None:
    graph = _alkene(("fluorine", "chlorine", "bromine", "iodine"))
    fixed = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)

    with pytest.raises(ValueError, match="requires fixed planar-bond"):
        _planar(graph, PlanarBondStereo(fixed.atoms, None))

    with pytest.raises(ValueError, match="Duplicate configured stereo locus"):
        canonicalize_stereograph(
            graph,
            (fixed, PlanarBondStereo((5, 4, 3, 2, 1, 0), 0)),
            atom_color="color",
            bond_color="color",
        )


def test_rdkit_e_and_z_receive_distinct_exact_codes() -> None:
    e_molecule = Chem.MolFromSmiles("F/C(Cl)=C(Br)/I")
    z_molecule = Chem.MolFromSmiles(r"F/C(Cl)=C(Br)\I")
    assert e_molecule is not None and z_molecule is not None

    e_result = canonicalize_rdkit_stereograph(e_molecule)
    z_result = canonicalize_rdkit_stereograph(z_molecule)

    assert e_result.canonical_code != z_result.canonical_code
    assert e_result.schema == z_result.schema == STEREOGRAPH_SCHEMA


def test_rdkit_planar_code_survives_atom_renumbering() -> None:
    molecule = Chem.MolFromSmiles("F/C(Cl)=C(Br)/I")
    assert molecule is not None
    expected = canonicalize_rdkit_stereograph(molecule)

    for order in (
        (5, 4, 3, 2, 1, 0),
        (2, 0, 5, 3, 1, 4),
        (1, 3, 0, 5, 2, 4),
    ):
        renumbered = Chem.RenumberAtoms(molecule, order)
        assert canonicalize_rdkit_stereograph(renumbered).same_stereograph(expected)


def test_rdkit_mixed_tetrahedral_and_planar_stereo_share_one_code() -> None:
    molecule = Chem.MolFromSmiles("F[C@H](Cl)/C=C/Br")
    assert molecule is not None
    renumbered = Chem.RenumberAtoms(
        molecule,
        tuple(reversed(range(molecule.GetNumAtoms()))),
    )

    original = canonicalize_rdkit_stereograph(molecule)
    transported = canonicalize_rdkit_stereograph(renumbered)

    assert original.canonical_code == transported.canonical_code
    assert len(original.locus_order) == 2


def test_old_rdkit_entry_point_refuses_to_omit_planar_stereo() -> None:
    molecule = Chem.MolFromSmiles("F/C(Cl)=C(Br)/I")
    assert molecule is not None

    with pytest.raises(TypeError, match="planar_bond"):
        canonicalize_rdkit_tetrahedral_stereograph(molecule)


def test_planar_constitutional_symmetry_can_identify_apparent_opposites() -> None:
    graph = _alkene(("same", "same", "same", "same"))
    descriptor = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)

    assert (
        _planar(graph, descriptor).canonical_code
        == _planar(graph, descriptor.invert()).canonical_code
    )


def test_planar_provenance_and_rdkit_stereo_flags_are_not_atom_colours() -> None:
    graph = _alkene(("fluorine", "chlorine", "bromine", "iodine"))
    descriptor = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0, "first")
    modified = graph.copy()
    for node in modified:
        modified.nodes[node]["_CIPCode"] = "R"
        modified.nodes[node]["coordinates"] = (node, node + 1, node + 2)

    assert (
        _planar(graph, descriptor).canonical_code
        == _planar(
            modified,
            PlanarBondStereo(descriptor.atoms, 0, "second"),
        ).canonical_code
    )
