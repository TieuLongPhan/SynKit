"""Faithful tetrahedral orbit-tuple stereograph canonicalization."""

from __future__ import annotations

import inspect
from itertools import permutations

import networkx as nx
import pytest
from rdkit import Chem

import synkit.Graph.Stereo.canonical as canonical_module
from synkit.Graph.Stereo import PlanarBondStereo, TetrahedralStereo
from synkit.Graph.Stereo import virtual_reference
from synkit.Graph.Stereo.canonical import (
    STEREOGRAPH_SCHEMA,
    StereoPortVertex,
    canonicalize_rdkit_tetrahedral_stereograph,
    canonicalize_tetrahedral_registry,
    canonicalize_tetrahedral_stereograph,
    expand_tetrahedral_stereograph,
)


def _permutation_sign(values: tuple[int, ...]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return -1 if inversions % 2 else 1


def _star(colors: tuple[str, str, str, str]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, color="center")
    for reference, color in enumerate(colors, start=1):
        graph.add_node(reference, color=color)
        graph.add_edge(0, reference, color="single")
    return graph


def _canonical(
    graph: nx.Graph,
    descriptor: TetrahedralStereo,
):
    return canonicalize_tetrahedral_stereograph(
        graph,
        (descriptor,),
        atom_color="color",
        bond_color="color",
    )


def test_tetrahedral_gadget_projects_exactly_a4() -> None:
    graph = _star(("same", "same", "same", "same"))
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)

    result = _canonical(graph, descriptor)
    images = {
        tuple(
            witness.as_dict()[StereoPortVertex(0, position)].position
            for position in range(4)
        )
        for witness in result.port_automorphisms
    }

    assert len(result.auxiliary.canonical_order) == 74
    assert len(images) == 12
    assert all(_permutation_sign(image) == 1 for image in images)
    assert result.schema == STEREOGRAPH_SCHEMA
    assert result.complete and result.exact


def test_all_equivalent_raw_frames_have_one_canonical_code() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    reference = TetrahedralStereo((0, 1, 2, 3, 4), 1)
    observed = set()

    for order in permutations((1, 2, 3, 4)):
        candidates = (
            TetrahedralStereo((0, *order), 1),
            TetrahedralStereo((0, *order), -1),
        )
        equivalent = next(item for item in candidates if item == reference)
        observed.add(_canonical(graph, equivalent).canonical_code)

    assert len(observed) == 1


def test_distinct_ligand_enantiomers_have_distinct_codes() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    fixed = TetrahedralStereo((0, 1, 2, 3, 4), 1)

    assert (
        _canonical(graph, fixed).canonical_code
        != _canonical(
            graph,
            fixed.opposite(),
        ).canonical_code
    )


def test_all_even_port_actions_preserve_and_all_odd_actions_invert() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    fixed = _canonical(
        graph,
        TetrahedralStereo((0, 1, 2, 3, 4), 1),
    ).canonical_code
    opposite = _canonical(
        graph,
        TetrahedralStereo((0, 1, 2, 3, 4), -1),
    ).canonical_code
    observed = {1: set(), -1: set()}

    for order in permutations((1, 2, 3, 4)):
        observed[_permutation_sign(order)].add(
            _canonical(
                graph,
                TetrahedralStereo((0, *order), 1),
            ).canonical_code
        )

    assert observed[1] == {fixed}
    assert observed[-1] == {opposite}
    assert fixed != opposite


def test_odd_constitutional_symmetry_identifies_apparent_opposites() -> None:
    graph = _star(("same", "same", "same", "same"))
    fixed = TetrahedralStereo((0, 1, 2, 3, 4), 1)

    assert (
        _canonical(graph, fixed).canonical_code
        == _canonical(
            graph,
            fixed.opposite(),
        ).canonical_code
    )


def test_atom_relabeling_preserves_code_and_transports_atom_order() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)
    mapping = {0: 10, 1: 40, 2: 30, 3: 20, 4: 50}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)

    original = _canonical(graph, descriptor)
    transported = _canonical(relabeled, descriptor.relabel(mapping))

    assert transported.canonical_code == original.canonical_code
    assert set(transported.atom_order) == set(mapping.values())
    assert len(transported.atom_order) == len(original.atom_order)


def test_cip_coordinates_and_input_order_do_not_enter_identity() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)
    modified = nx.Graph()
    for node in reversed(tuple(graph.nodes())):
        attributes = dict(graph.nodes[node])
        attributes.update(
            {
                "_CIPCode": "R" if node % 2 else "S",
                "_CIPRank": 100 - node,
                "coordinates": (node, node + 1, node + 2),
            }
        )
        modified.add_node(node, **attributes)
    for left, right, attributes in reversed(tuple(graph.edges(data=True))):
        modified.add_edge(left, right, **attributes)

    assert (
        _canonical(graph, descriptor).canonical_code
        == _canonical(
            modified,
            descriptor,
        ).canonical_code
    )


def test_descriptor_reference_must_be_a_real_owner_incidence() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    graph.add_node(5, color="iodine")

    with pytest.raises(ValueError, match="not bonded"):
        _canonical(
            graph,
            TetrahedralStereo((0, 1, 2, 3, 5), 1),
        )


def test_expansion_uses_coloured_bond_incidence_vertices() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    expansion = expand_tetrahedral_stereograph(
        graph,
        (TetrahedralStereo((0, 1, 2, 3, 4), 1),),
        atom_color="color",
        bond_color="color",
    )

    colors = [attributes["color"][0] for _, attributes in expansion.nodes(data=True)]
    assert colors.count("atom") == 5
    assert colors.count("bond_resource") == 4
    assert colors.count("stereo_locus") == 1
    assert colors.count("stereo_port") == 4
    assert colors.count("stereo_tuple") == 12
    assert colors.count("stereo_slot") == 48


def test_owner_scoped_hidden_hydrogen_is_an_explicit_resource() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(
        (
            (0, {"color": "center"}),
            (1, {"color": "fluorine"}),
            (2, {"color": "chlorine"}),
            (3, {"color": "bromine"}),
        )
    )
    graph.add_edges_from(
        (
            (0, 1, {"color": "single"}),
            (0, 2, {"color": "single"}),
            (0, 3, {"color": "single"}),
        )
    )
    fixed = TetrahedralStereo(
        (0, 1, 2, 3, virtual_reference("H", 0)),
        1,
    )

    result = _canonical(graph, fixed)
    opposite = _canonical(graph, fixed.opposite())
    expansion = expand_tetrahedral_stereograph(
        graph,
        (fixed,),
        atom_color="color",
        bond_color="color",
    )

    assert result.canonical_code != opposite.canonical_code
    assert (
        sum(
            attributes["color"] == ("virtual_resource", "H")
            for _, attributes in expansion.nodes(data=True)
        )
        == 1
    )


def test_two_center_meso_control_equals_its_global_mirror() -> None:
    graph = nx.Graph()
    for node, color in {
        0: "center",
        5: "center",
        1: "A",
        6: "A",
        2: "B",
        7: "B",
        3: "C",
        8: "C",
    }.items():
        graph.add_node(node, color=color)
    for left, right in (
        (0, 5),
        (0, 1),
        (0, 2),
        (0, 3),
        (5, 6),
        (5, 7),
        (5, 8),
    ):
        graph.add_edge(left, right, color="single")

    meso = (
        TetrahedralStereo((0, 1, 2, 3, 5), 1),
        TetrahedralStereo((5, 6, 7, 8, 0), -1),
    )
    chiral = (
        TetrahedralStereo((0, 1, 2, 3, 5), 1),
        TetrahedralStereo((5, 6, 7, 8, 0), 1),
    )
    options = {"atom_color": "color", "bond_color": "color"}

    meso_code = canonicalize_tetrahedral_stereograph(
        graph,
        meso,
        **options,
    ).canonical_code
    meso_mirror = canonicalize_tetrahedral_stereograph(
        graph,
        tuple(descriptor.opposite() for descriptor in meso),
        **options,
    ).canonical_code
    chiral_code = canonicalize_tetrahedral_stereograph(
        graph,
        chiral,
        **options,
    ).canonical_code
    chiral_mirror = canonicalize_tetrahedral_stereograph(
        graph,
        tuple(descriptor.opposite() for descriptor in chiral),
        **options,
    ).canonical_code

    assert meso_code == meso_mirror
    assert chiral_code != chiral_mirror
    assert (
        canonicalize_tetrahedral_stereograph(
            graph,
            tuple(reversed(meso)),
            **options,
        ).canonical_code
        == meso_code
    )


def test_registry_refuses_to_silently_omit_other_stereo_families() -> None:
    graph = nx.path_graph(6)
    descriptor = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)

    with pytest.raises(TypeError, match="planar_bond"):
        canonicalize_tetrahedral_registry(
            graph,
            {"bond:2-3": descriptor},
        )


@pytest.mark.parametrize(
    "left,right",
    (
        ("F[C@H](Cl)Br", "F[C@H](Cl)Br"),
        ("[F:1][C@H:2]([Cl:3])[Br:4]", "[F:91][C@H:72]([Cl:53])[Br:34]"),
        ("[F:1][C@H]([Cl:3])[Br:4]", "[F:9][C@H]([Cl:7])[Br:6]"),
    ),
)
def test_rdkit_entry_point_uses_one_consistent_identifier_namespace(
    left: str,
    right: str,
) -> None:
    first = Chem.MolFromSmiles(left)
    second = Chem.MolFromSmiles(right)
    assert first is not None and second is not None

    left_result = canonicalize_rdkit_tetrahedral_stereograph(first)
    right_result = canonicalize_rdkit_tetrahedral_stereograph(second)

    assert left_result.same_stereograph(right_result)
    assert left_result.canonical_code == right_result.canonical_code


def test_rdkit_renumbering_is_invariant_and_enantiomer_is_distinct() -> None:
    molecule = Chem.MolFromSmiles("F[C@H](Cl)Br")
    opposite = Chem.MolFromSmiles("F[C@@H](Cl)Br")
    assert molecule is not None and opposite is not None
    renumbered = Chem.RenumberAtoms(molecule, (3, 1, 0, 2))

    original = canonicalize_rdkit_tetrahedral_stereograph(molecule)
    transported = canonicalize_rdkit_tetrahedral_stereograph(renumbered)
    mirror = canonicalize_rdkit_tetrahedral_stereograph(opposite)

    assert original.same_stereograph(transported)
    assert original.canonical_code != mirror.canonical_code


def test_deep_ligand_distinctions_are_resolved_by_global_refinement() -> None:
    graph = nx.Graph()
    graph.add_node(0, color="center")
    graph.add_node(1, color="carbon")
    graph.add_node(2, color="carbon")
    graph.add_node(3, color="bromine")
    graph.add_node(4, color="iodine")
    graph.add_edges_from(
        (
            (0, 1, {"color": "single"}),
            (0, 2, {"color": "single"}),
            (0, 3, {"color": "single"}),
            (0, 4, {"color": "single"}),
        )
    )
    for branch, terminal in ((1, "fluorine"), (2, "chlorine")):
        previous = branch
        for offset in range(3):
            node = 10 * branch + offset
            graph.add_node(node, color="carbon")
            graph.add_edge(previous, node, color="single")
            previous = node
        node = 10 * branch + 3
        graph.add_node(node, color=terminal)
        graph.add_edge(previous, node, color="single")

    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)

    assert (
        _canonical(graph, descriptor).canonical_code
        != _canonical(
            graph,
            descriptor.opposite(),
        ).canonical_code
    )


def test_stereograph_encoder_has_no_crn_dependency() -> None:
    assert "synkit.CRN" not in inspect.getsource(canonical_module)
