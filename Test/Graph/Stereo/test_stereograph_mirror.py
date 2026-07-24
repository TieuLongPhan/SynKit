"""Sprint 39 exact configured-stereograph mirror integration."""

from __future__ import annotations

from itertools import permutations

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    AtropBondStereo,
    PlanarBondStereo,
    StereographMirrorStatus,
    TetrahedralStereo,
    classify_rdkit_stereograph_mirror,
    classify_stereograph_mirror,
    mirror_stereo_descriptor,
    mirror_stereo_registry,
)


def _star(colors: tuple[str, str, str, str]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, color="center")
    for reference, color in enumerate(colors, start=1):
        graph.add_node(reference, color=color)
        graph.add_edge(0, reference, color="single")
    return graph


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


def _two_center() -> tuple[nx.Graph, tuple[TetrahedralStereo, ...]]:
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
    return graph, meso


def _classify(
    graph: nx.Graph,
    registry: dict[str, object],
    **diagnostics: object,
):
    return classify_stereograph_mirror(
        graph,
        registry,  # type: ignore[arg-type]
        atom_color="color",
        bond_color="color",
        **diagnostics,
    )


def test_geometry_specific_mirror_action_is_involutive() -> None:
    tetrahedral = TetrahedralStereo((0, 1, 2, 3, 4), 1, "source")
    planar = PlanarBondStereo((1, 2, 3, 4, 5, 6), 0, "source")
    unknown_tetrahedral = TetrahedralStereo(tetrahedral.atoms, None)
    unknown_planar = PlanarBondStereo(planar.atoms, None)

    tetrahedral_mirror = mirror_stereo_descriptor(tetrahedral)

    assert tetrahedral_mirror == tetrahedral.opposite()
    assert tetrahedral_mirror.provenance == "source"
    assert mirror_stereo_descriptor(tetrahedral_mirror) == tetrahedral
    assert mirror_stereo_descriptor(planar) is planar
    assert mirror_stereo_descriptor(unknown_tetrahedral) is unknown_tetrahedral
    assert mirror_stereo_descriptor(unknown_planar) is unknown_planar


def test_mirror_transform_refuses_an_unimplemented_geometry() -> None:
    descriptor = AtropBondStereo((0, 1, 2, 3, 4, 5), 1)

    with pytest.raises(TypeError, match="atrop_bond"):
        mirror_stereo_descriptor(descriptor)


def test_distinct_ligand_tetrahedral_center_is_globally_chiral() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)

    result = _classify(graph, {"atom:0": descriptor})

    assert result.status is StereographMirrorStatus.CHIRAL
    assert result.is_definitive and result.is_chiral is True
    assert result.original is not None and result.mirror is not None
    assert result.original.canonical_code != result.mirror.canonical_code
    assert result.atom_mirror_isomorphism is None
    assert result.mirrored_descriptors == (descriptor.opposite(),)


def test_two_center_meso_stereograph_has_an_exact_mirror_isomorphism() -> None:
    graph, descriptors = _two_center()

    result = _classify(
        graph,
        {f"atom:{descriptor.center}": descriptor for descriptor in descriptors},
    )

    assert result.status is StereographMirrorStatus.ACHIRAL
    assert result.is_chiral is False
    assert result.original is not None and result.mirror is not None
    assert result.original.same_stereograph(result.mirror)
    mapping = dict(result.atom_mirror_isomorphism or ())
    assert mapping[0] == 5 and mapping[5] == 0
    assert {
        graph.nodes[source]["color"]: graph.nodes[target]["color"]
        for source, target in mapping.items()
    } == {
        color: color for color in set(nx.get_node_attributes(graph, "color").values())
    }
    assert all(
        graph.has_edge(mapping[left], mapping[right]) for left, right in graph.edges
    )


def test_constitutional_symmetry_can_make_one_center_achiral() -> None:
    graph = _star(("same", "same", "same", "same"))
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)

    result = _classify(graph, {"atom:0": descriptor})

    assert result.status is StereographMirrorStatus.ACHIRAL
    assert result.atom_mirror_isomorphism is not None


def test_e_and_z_are_each_achiral_under_spatial_reflection() -> None:
    graph = _alkene(("fluorine", "chlorine", "bromine", "iodine"))
    e_or_z = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)

    first = _classify(graph, {"bond:2-3": e_or_z})
    opposite = _classify(graph, {"bond:2-3": e_or_z.invert()})

    assert first.status is StereographMirrorStatus.ACHIRAL
    assert opposite.status is StereographMirrorStatus.ACHIRAL
    assert first.mirrored_descriptors == (e_or_z,)
    assert opposite.mirrored_descriptors == (e_or_z.invert(),)


def test_mixed_tetrahedral_planar_identity_uses_only_tetrahedral_mirror_action() -> (
    None
):
    graph = nx.disjoint_union(
        _star(("fluorine", "chlorine", "bromine", "hydrogen")),
        _alkene(("fluorine", "chlorine", "bromine", "iodine")),
    )
    tetrahedral = TetrahedralStereo((0, 1, 2, 3, 4), 1)
    planar = PlanarBondStereo((5, 6, 7, 8, 9, 10), 0)
    registry = {"atom:0": tetrahedral, "bond:7-8": planar}

    result = _classify(graph, registry)
    reversed_result = _classify(
        graph,
        dict(reversed(tuple(registry.items()))),
    )

    assert result.status is reversed_result.status is StereographMirrorStatus.CHIRAL
    assert result.original is not None and reversed_result.original is not None
    assert result.original.canonical_code == reversed_result.original.canonical_code
    mirrored = mirror_stereo_registry(registry)
    assert mirrored["atom:0"] == tetrahedral.opposite()
    assert mirrored["bond:7-8"] is planar


def test_all_120_atom_relabelings_preserve_mirror_verdict() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)

    for images in permutations((10, 20, 30, 40, 50)):
        mapping = dict(zip(graph, images))
        relabeled = nx.relabel_nodes(graph, mapping, copy=True)
        result = _classify(
            relabeled,
            {f"atom:{mapping[0]}": descriptor.relabel(mapping)},
        )
        assert result.status is StereographMirrorStatus.CHIRAL


def test_unknown_supported_descriptor_returns_incomplete_without_codes() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), None)

    result = _classify(graph, {"atom:0": descriptor})

    assert result.status is StereographMirrorStatus.INCOMPLETE
    assert result.is_chiral is None and not result.is_definitive
    assert result.incomplete_loci == ("atom:0",)
    assert result.original is None and result.mirror is None


def test_declared_missing_configuration_returns_incomplete() -> None:
    result = _classify(
        nx.path_graph(2),
        {},
        incomplete_loci=("tetrahedral:1",),
    )

    assert result.status is StereographMirrorStatus.INCOMPLETE
    assert result.incomplete_loci == ("tetrahedral:1",)
    assert result.descriptor_count == 0


def test_unsupported_geometry_takes_precedence_over_incomplete() -> None:
    graph = _alkene(("fluorine", "chlorine", "bromine", "iodine"))
    descriptor = AtropBondStereo((0, 1, 2, 3, 4, 5), 1)

    result = _classify(
        graph,
        {"bond:2-3": descriptor},
        incomplete_loci=("tetrahedral:9",),
    )

    assert result.status is StereographMirrorStatus.UNSUPPORTED
    assert result.unsupported_loci == ("bond:2-3",)
    assert result.unsupported_families == ("atrop_bond",)
    assert result.incomplete_loci == ()
    assert result.original is None and result.mirror is None


def test_empty_declared_complete_stereograph_is_achiral() -> None:
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, "carbon", "color")
    nx.set_edge_attributes(graph, "single", "color")

    result = _classify(graph, {})

    assert result.status is StereographMirrorStatus.ACHIRAL
    assert result.descriptor_count == 0
    assert result.original is not None and result.mirror is not None


@pytest.mark.parametrize(
    ("smiles", "expected"),
    (
        ("F[C@H](Cl)Br", StereographMirrorStatus.CHIRAL),
        ("C[C@H](Cl)[C@H](Cl)C", StereographMirrorStatus.ACHIRAL),
        ("F/C(Cl)=C(Br)/I", StereographMirrorStatus.ACHIRAL),
        (r"F/C(Cl)=C(Br)\I", StereographMirrorStatus.ACHIRAL),
        ("CC", StereographMirrorStatus.ACHIRAL),
    ),
)
def test_rdkit_complete_configured_examples(
    smiles: str,
    expected: StereographMirrorStatus,
) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None

    result = classify_rdkit_stereograph_mirror(molecule)

    assert result.status is expected
    assert result.is_definitive


@pytest.mark.parametrize(
    ("smiles", "locus"),
    (
        ("FC(Cl)Br", "tetrahedral:1"),
        ("FC=CCl", "double_bond:1-2"),
    ),
)
def test_rdkit_unconfigured_supported_loci_are_incomplete(
    smiles: str,
    locus: str,
) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None

    result = classify_rdkit_stereograph_mirror(molecule)

    assert result.status is StereographMirrorStatus.INCOMPLETE
    assert result.incomplete_loci == (locus,)
    assert result.original is None and result.mirror is None


@pytest.mark.parametrize(
    ("smiles", "family_or_locus"),
    (
        ("[H][Pt@SP1](F)(Cl)Br", "square_planar"),
        ("ClC=C=CCl", "cumulene_axis:1-2-3"),
    ),
)
def test_rdkit_later_families_are_unsupported(
    smiles: str,
    family_or_locus: str,
) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None

    result = classify_rdkit_stereograph_mirror(molecule)

    assert result.status is StereographMirrorStatus.UNSUPPORTED
    assert (
        family_or_locus in result.unsupported_families
        or family_or_locus in result.unsupported_loci
    )


def test_rdkit_enhanced_stereo_group_is_not_flattened_to_one_isomer() -> None:
    molecule = Chem.MolFromSmiles("F[C@H](Cl)[C@H](Br)I |&1:1,3|")
    assert molecule is not None

    result = classify_rdkit_stereograph_mirror(molecule)

    assert result.status is StereographMirrorStatus.UNSUPPORTED
    assert result.unsupported_loci == ("enhanced_stereo_group:0:STEREO_AND",)
    assert result.original is None and result.mirror is None


def test_rdkit_mirror_verdict_and_codes_survive_atom_renumbering() -> None:
    molecule = Chem.MolFromSmiles("C[C@H](Cl)[C@H](Cl)C")
    assert molecule is not None
    expected = classify_rdkit_stereograph_mirror(molecule)

    for order in (
        tuple(reversed(range(molecule.GetNumAtoms()))),
        (2, 0, 5, 3, 1, 4),
        (1, 3, 0, 5, 2, 4),
    ):
        result = classify_rdkit_stereograph_mirror(Chem.RenumberAtoms(molecule, order))
        assert result.status is expected.status
        assert result.original is not None and expected.original is not None
        assert result.original.canonical_code == expected.original.canonical_code


def test_rdkit_mirror_classification_does_not_mutate_input() -> None:
    molecule = Chem.MolFromSmiles("F/C(Cl)=C(Br)/I")
    assert molecule is not None
    before = Chem.MolToSmiles(molecule, canonical=False, isomericSmiles=True)

    classify_rdkit_stereograph_mirror(molecule)

    assert (
        Chem.MolToSmiles(
            molecule,
            canonical=False,
            isomericSmiles=True,
        )
        == before
    )
