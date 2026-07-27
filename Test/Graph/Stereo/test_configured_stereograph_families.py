"""Sprint 40 complete configured-family canonicalization gates."""

from __future__ import annotations

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    CONFIGURED_STEREOGRAPH_SCHEMA,
    STEREOGRAPH_SCHEMA,
    AtropBondStereo,
    CumuleneAxisStereo,
    HelicalStereo,
    OctahedralStereo,
    PlanarChiralityStereo,
    SquarePlanarStereo,
    StereographMirrorStatus,
    TrigonalBipyramidalStereo,
    canonicalize_configured_stereograph,
    canonicalize_rdkit_configured_stereograph,
    classify_configured_stereograph_mirror,
    classify_rdkit_configured_stereograph_mirror,
    descriptor_id,
    expand_configured_stereograph,
    mirror_configured_descriptor,
    stereo_from_dict,
)
from synkit.Graph.Stereo.canonical import StereoPortVertex
from synkit.Graph.Stereo.orbits import SHAPE_DEFINITIONS


def _star(arity: int, *, symmetric: bool = False) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, color="center")
    for reference in range(1, arity + 1):
        graph.add_node(
            reference,
            color="same" if symmetric else f"ligand:{reference}",
        )
        graph.add_edge(0, reference, color="single")
    return graph


def _canonical(graph: nx.Graph, descriptor):
    return canonicalize_configured_stereograph(
        graph,
        (descriptor,),
        atom_color="color",
        bond_color="color",
    )


def test_configured_schema_name_is_a_compatibility_alias() -> None:
    assert CONFIGURED_STEREOGRAPH_SCHEMA == STEREOGRAPH_SCHEMA


def test_existing_chiral_certificate_is_definitive_with_unresolved_amine() -> None:
    molecule = Chem.MolFromSmiles("FC(CN1[C@@H](C2=CC=CS2)CCC1)(F)F")
    assert molecule is not None

    result = classify_rdkit_configured_stereograph_mirror(molecule)

    assert result.status is StereographMirrorStatus.CHIRAL
    assert result.is_definitive
    assert result.incomplete_loci == ("tetrahedral:3",)
    assert result.original is not None and result.mirror is not None
    assert "monotone_chiral_mirror_proof" in result.method
    assert result.method.endswith(":chemical")


def test_supplied_configuration_mode_does_not_require_latent_loci() -> None:
    molecule = Chem.MolFromSmiles("C/C=C/1\\C/C(/C1)=C\\C")
    assert molecule is not None

    strict = classify_rdkit_configured_stereograph_mirror(molecule)
    supplied = classify_rdkit_configured_stereograph_mirror(
        molecule,
        require_complete=False,
    )

    assert strict.status is StereographMirrorStatus.INCOMPLETE
    assert supplied.status is StereographMirrorStatus.ACHIRAL
    assert supplied.incomplete_loci == ()


@pytest.mark.parametrize(
    "smiles",
    (
        "C1=C2[C@@H]3C(C=NC4=CC=CC(=N1)[C@@H]43)=CC=C2",
        "C1=C2[C@H]3C(C=NC4=CC=CC(=N1)[C@H]43)=CC=C2",
    ),
)
def test_resonance_family_recognizes_vs215_vs216_as_achiral(
    smiles: str,
) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None

    chemical = classify_rdkit_configured_stereograph_mirror(
        molecule,
        require_complete=False,
    )
    lewis = classify_rdkit_configured_stereograph_mirror(
        molecule,
        require_complete=False,
        identity_profile="lewis_state",
    )

    assert chemical.status is StereographMirrorStatus.ACHIRAL
    assert chemical.method.endswith(":chemical")
    assert lewis.status is StereographMirrorStatus.CHIRAL
    assert lewis.method.endswith(":lewis_state")


def test_vs170_preserves_genuine_bond_order_difference() -> None:
    molecule = Chem.MolFromSmiles("[C@H](O)(SI)S#I")
    assert molecule is not None

    chemical = classify_rdkit_configured_stereograph_mirror(
        molecule,
        require_complete=False,
    )
    acs_topology = classify_rdkit_configured_stereograph_mirror(
        molecule,
        require_complete=False,
        identity_profile="acs_topology",
    )

    assert chemical.status is StereographMirrorStatus.CHIRAL
    assert chemical.method.endswith(":chemical")
    assert acs_topology.status is StereographMirrorStatus.ACHIRAL
    assert acs_topology.method.endswith(":acs_topology")


@pytest.mark.parametrize(
    "smiles",
    (
        "CC1=C(C=CC=C1)/N=N/P(=O)([O-])OC",
        "COP(=O)(/N=N/C1=CC(=CC=C1)F)[O-]",
    ),
)
def test_terminal_phosphate_resonance_pair_is_not_a_missing_p_center(
    smiles: str,
) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None

    result = classify_rdkit_configured_stereograph_mirror(molecule)

    assert result.status is StereographMirrorStatus.ACHIRAL
    assert result.incomplete_loci == ()


@pytest.mark.parametrize(
    ("descriptor", "group_size", "vertex_count"),
    (
        (SquarePlanarStereo((0, 1, 2, 3, 4), 0), 8, 54),
        (TrigonalBipyramidalStereo((0, 1, 2, 3, 4, 5), 1), 6, 53),
        (OctahedralStereo((0, 1, 2, 3, 4, 5, 6), 1), 24, 188),
    ),
)
def test_coordination_gadgets_project_exact_rotation_groups(
    descriptor,
    group_size: int,
    vertex_count: int,
) -> None:
    graph = _star(len(descriptor.atoms) - 1, symmetric=True)

    result = _canonical(graph, descriptor)
    images = {
        tuple(
            witness.as_dict()[StereoPortVertex(0, position)].position
            for position in range(len(descriptor.atoms) - 1)
        )
        for witness in result.port_automorphisms
    }

    assert len(images) == group_size
    assert len(result.auxiliary.canonical_order) == vertex_count
    assert result.schema == CONFIGURED_STEREOGRAPH_SCHEMA


@pytest.mark.parametrize(
    "descriptor",
    (
        SquarePlanarStereo((0, 1, 2, 3, 4), 0),
        TrigonalBipyramidalStereo((0, 1, 2, 3, 4, 5), 1),
        OctahedralStereo((0, 1, 2, 3, 4, 5, 6), 1),
    ),
)
def test_every_authoritative_coordination_rotation_preserves_code(
    descriptor,
) -> None:
    graph = _star(len(descriptor.atoms) - 1)
    definition = SHAPE_DEFINITIONS[descriptor.descriptor_class]
    expected = _canonical(graph, descriptor).canonical_code

    observed = {
        _canonical(
            graph,
            type(descriptor)(
                permutation.apply(descriptor.configuration.frame),
                0 if isinstance(descriptor, SquarePlanarStereo) else 1,
            ),
        ).canonical_code
        for permutation in definition.preserving_group.elements
    }

    assert observed == {expected}


def test_coordination_mirror_actions_are_geometry_specific() -> None:
    square = SquarePlanarStereo((0, 1, 2, 3, 4), 0)
    tbp = TrigonalBipyramidalStereo((0, 1, 2, 3, 4, 5), 1)
    octahedral = OctahedralStereo((0, 1, 2, 3, 4, 5, 6), 1)

    assert mirror_configured_descriptor(square) is square
    assert mirror_configured_descriptor(tbp) == tbp.invert()
    assert mirror_configured_descriptor(octahedral) == octahedral.invert()
    assert mirror_configured_descriptor(tbp.invert()) == tbp
    assert mirror_configured_descriptor(octahedral.invert()) == octahedral

    for descriptor in (tbp, octahedral):
        graph = _star(len(descriptor.atoms) - 1)
        result = classify_configured_stereograph_mirror(
            graph,
            {"atom:0": descriptor},
            atom_color="color",
            bond_color="color",
        )
        assert result.status is StereographMirrorStatus.CHIRAL


def _axis_graph(*, cumulene: bool = False) -> nx.Graph:
    graph = nx.Graph()
    for node in range(7 if cumulene else 6):
        graph.add_node(node, color=f"atom:{node}")
    if cumulene:
        edges = ((0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6))
        for edge in edges:
            graph.add_edge(
                *edge,
                color="double" if edge in {(2, 3), (3, 4)} else "single",
            )
    else:
        for edge in ((0, 2), (1, 2), (2, 3), (3, 4), (3, 5)):
            graph.add_edge(*edge, color="single")
    return graph


def test_atrop_and_cumulene_axes_receive_exact_opposite_codes() -> None:
    atrop = AtropBondStereo((0, 1, 2, 3, 4, 5), 1)
    cumulene = CumuleneAxisStereo((2, 3, 4), ((0, 1), (5, 6)), 1)

    for graph, descriptor in (
        (_axis_graph(), atrop),
        (_axis_graph(cumulene=True), cumulene),
    ):
        original = _canonical(graph, descriptor)
        opposite = _canonical(graph, descriptor.invert())
        verdict = classify_configured_stereograph_mirror(
            graph,
            {descriptor_id(descriptor): descriptor},
            atom_color="color",
            bond_color="color",
        )
        assert original.canonical_code != opposite.canonical_code
        assert verdict.status is StereographMirrorStatus.CHIRAL
        assert mirror_configured_descriptor(descriptor) == descriptor.invert()


def test_cumulene_axis_path_reversal_and_relabeling_are_nonsemantic() -> None:
    graph = _axis_graph(cumulene=True)
    descriptor = CumuleneAxisStereo((2, 3, 4), ((0, 1), (5, 6)), 1)
    mapping = {node: node + 20 for node in graph}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)

    expected = _canonical(graph, descriptor).canonical_code

    assert _canonical(graph, descriptor.reversed()).canonical_code == expected
    assert (
        _canonical(
            relabeled,
            descriptor.relabel(mapping),
        ).canonical_code
        == expected
    )


def _path_graph(offset: int = 0) -> nx.Graph:
    graph = nx.path_graph(range(offset, offset + 5))
    nx.set_node_attributes(
        graph,
        {node: f"path:{node - offset}" for node in graph},
        "color",
    )
    nx.set_edge_attributes(graph, "single", "color")
    return graph


def test_helical_path_reversal_preserves_and_mirror_changes_code() -> None:
    graph = _path_graph()
    descriptor = HelicalStereo((0, 1, 2, 3, 4), 1)

    original = _canonical(graph, descriptor)

    assert _canonical(graph, descriptor.reversed()).canonical_code == (
        original.canonical_code
    )
    assert _canonical(graph, descriptor.invert()).canonical_code != (
        original.canonical_code
    )


def test_opposite_symmetric_helices_form_an_achiral_pair() -> None:
    graph = nx.disjoint_union(_path_graph(), _path_graph())
    positive = HelicalStereo((0, 1, 2, 3, 4), 1)
    negative = HelicalStereo((5, 6, 7, 8, 9), -1)

    result = classify_configured_stereograph_mirror(
        graph,
        {"path:left": positive, "path:right": negative},
        atom_color="color",
        bond_color="color",
    )

    assert result.status is StereographMirrorStatus.ACHIRAL
    assert result.atom_mirror_isomorphism is not None


def _plane_graph() -> nx.Graph:
    graph = nx.cycle_graph(4)
    graph.add_node(4, color="pilot")
    nx.set_node_attributes(
        graph,
        {
            0: "plane:A",
            1: "plane:B",
            2: "plane:C",
            3: "plane:D",
            4: "pilot",
        },
        "color",
    )
    nx.set_edge_attributes(graph, "plane", "color")
    return graph


def test_planar_chirality_has_rotation_orbit_and_reflected_opposite() -> None:
    graph = _plane_graph()
    descriptor = PlanarChiralityStereo((0, 1, 2, 3), 4, 1, "sidecar")
    rotated = PlanarChiralityStereo((2, 3, 0, 1), 4, 1)

    original = _canonical(graph, descriptor)

    assert descriptor == rotated == descriptor.reversed()
    assert stereo_from_dict(descriptor.to_dict()) == descriptor
    assert descriptor_id(descriptor).startswith("plane:")
    assert _canonical(graph, rotated).canonical_code == original.canonical_code
    assert _canonical(graph, descriptor.invert()).canonical_code != (
        original.canonical_code
    )
    assert mirror_configured_descriptor(descriptor) == descriptor.invert()


def test_planar_chirality_requires_a_closed_plane_support() -> None:
    graph = _plane_graph()
    graph.remove_edge(3, 0)

    with pytest.raises(ValueError, match="not continuous"):
        _canonical(
            graph,
            PlanarChiralityStereo((0, 1, 2, 3), 4, 1),
        )


@pytest.mark.parametrize(
    "smiles",
    (
        "[H][Pt@SP1](F)(Cl)Br",
        "S[As@TB1](F)(Cl)(Br)N",
        "O[Co@OH1](Cl)(C)(N)(F)P",
    ),
)
def test_rdkit_coordination_entry_point_survives_atom_renumbering(
    smiles: str,
) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    expected = canonicalize_rdkit_configured_stereograph(molecule)
    renumbered = Chem.RenumberAtoms(
        molecule,
        tuple(reversed(range(molecule.GetNumAtoms()))),
    )

    observed = canonicalize_rdkit_configured_stereograph(renumbered)

    assert observed.schema == CONFIGURED_STEREOGRAPH_SCHEMA
    assert observed.canonical_code == expected.canonical_code


def test_unknown_extended_configuration_fails_before_exact_identity() -> None:
    graph = _star(5)
    descriptor = TrigonalBipyramidalStereo((0, 1, 2, 3, 4, 5), None)

    with pytest.raises(ValueError, match="must be fixed"):
        _canonical(graph, descriptor)


def test_expansion_contains_every_configured_family_in_one_graph() -> None:
    graph = nx.disjoint_union_all(
        (
            _star(4),
            _star(4),
            _star(5),
            _star(6),
            _axis_graph(),
            _axis_graph(cumulene=True),
            _path_graph(),
            _plane_graph(),
        )
    )
    descriptors = (
        # Offsets are cumulative component sizes: 0, 5, 10, 16, 23, 29, 36, 41.
        SquarePlanarStereo((5, 6, 7, 8, 9), 0),
        TrigonalBipyramidalStereo((10, 11, 12, 13, 14, 15), 1),
        OctahedralStereo((16, 17, 18, 19, 20, 21, 22), 1),
        AtropBondStereo((23, 24, 25, 26, 27, 28), 1),
        CumuleneAxisStereo((31, 32, 33), ((29, 30), (34, 35)), 1),
        HelicalStereo((36, 37, 38, 39, 40), 1),
        PlanarChiralityStereo((41, 42, 43, 44), 45, 1),
    )

    expansion = expand_configured_stereograph(
        graph,
        descriptors,
        atom_color="color",
        bond_color="color",
    )
    locus_colors = {
        attributes["color"][1]
        for _, attributes in expansion.nodes(data=True)
        if attributes["color"][0] == "stereo_locus"
    }

    assert locus_colors == {
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
        "atrop_bond",
        "cumulene_axis",
        "helical",
        "planar_chirality",
    }
