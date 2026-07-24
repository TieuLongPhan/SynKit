"""Sprint 41 exact focal omission and assignment enumeration."""

from __future__ import annotations

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    OctahedralStereo,
    PlanarBondStereo,
    PlanarChiralityStereo,
    SquarePlanarStereo,
    StereoEnumerationLimitError,
    StereographMirrorStatus,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    enumerate_rdkit_stereographs,
    enumerate_stereograph_assignments,
    focal_stereo_evidence,
    local_configuration_classes,
)


def _star(*, symmetric: bool = False) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, color="center")
    for reference, color in enumerate(
        ("fluorine", "chlorine", "bromine", "hydrogen"),
        start=1,
    ):
        graph.add_node(reference, color="same" if symmetric else color)
        graph.add_edge(0, reference, color="single")
    return graph


def _enumerate(graph, registry, **options):
    return enumerate_stereograph_assignments(
        graph,
        registry,
        atom_color="color",
        bond_color="color",
        **options,
    )


@pytest.mark.parametrize(
    ("descriptor", "expected"),
    (
        (TetrahedralStereo((0, 1, 2, 3, 4), None), 2),
        (SquarePlanarStereo((0, 1, 2, 3, 4), None), 3),
        (TrigonalBipyramidalStereo((0, 1, 2, 3, 4, 5), None), 20),
        (OctahedralStereo((0, 1, 2, 3, 4, 5, 6), None), 30),
        (PlanarBondStereo((0, 1, 2, 3, 4, 5), None), 2),
        (AtropBondStereo((0, 1, 2, 3, 4, 5), None), 2),
        (CumuleneAxisStereo((2, 3, 4), ((0, 1), (5, 6)), None), 2),
        (
            ExtendedCisTransStereo(
                (2, 3, 4, 5),
                ((0, 1), (6, 7)),
                None,
            ),
            2,
        ),
        (HelicalStereo((0, 1, 2, 3, 4), None), 2),
        (PlanarChiralityStereo((0, 1, 2, 3), 4, None), 2),
    ),
)
def test_local_classes_are_geometry_group_quotients(
    descriptor,
    expected: int,
) -> None:
    classes = local_configuration_classes(descriptor)

    assert len(classes) == expected
    assert all(item.parity is not None for item in classes)
    assert len({item.canonical_form() for item in classes}) == expected


def test_focal_omission_finds_four_distinct_ligand_orbits() -> None:
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), None)

    evidence = focal_stereo_evidence(
        _star(),
        {"atom:0": descriptor},
        "atom:0",
        atom_color="color",
        bond_color="color",
    )

    assert evidence.port_orbits == ((0,), (1,), (2,), (3,))
    assert evidence.all_ports_distinct
    assert evidence.auxiliary.complete and evidence.auxiliary.exact


def test_focal_omission_retains_constitutional_symmetry() -> None:
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), None)

    evidence = focal_stereo_evidence(
        _star(symmetric=True),
        {"atom:0": descriptor},
        "atom:0",
        atom_color="color",
        bond_color="color",
    )

    assert evidence.port_orbits == ((0, 1, 2, 3),)
    assert not evidence.all_ports_distinct


def test_focal_supplied_orientation_cannot_prove_itself() -> None:
    positive = TetrahedralStereo((0, 1, 2, 3, 4), 1)
    negative = positive.invert()

    first = focal_stereo_evidence(
        _star(),
        {"atom:0": positive},
        "atom:0",
        atom_color="color",
        bond_color="color",
    )
    second = focal_stereo_evidence(
        _star(),
        {"atom:0": negative},
        "atom:0",
        atom_color="color",
        bond_color="color",
    )

    assert first.canonical_code == second.canonical_code
    assert first.port_orbits == second.port_orbits


def test_distinct_tetrahedral_support_enumerates_one_enantiomer_pair() -> None:
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), None)

    result = _enumerate(_star(), {"atom:0": descriptor})

    assert result.theoretical_assignment_count == 2
    assert result.exact_assignment_count == 2
    assert result.enantiomer_class_count == 1
    assert {assignment.mirror_status for assignment in result.assignments} == {
        StereographMirrorStatus.CHIRAL
    }
    assert result.complete and result.exact


def test_symmetric_tetrahedral_support_collapses_both_raw_assignments() -> None:
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), None)

    result = _enumerate(
        _star(symmetric=True),
        {"atom:0": descriptor},
    )

    assert result.theoretical_assignment_count == 2
    assert result.exact_assignment_count == 1
    assert result.enantiomer_class_count == 1
    assert result.assignments[0].mirror_status is StereographMirrorStatus.ACHIRAL


def _two_center_graph() -> nx.Graph:
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
    for edge in (
        (0, 5),
        (0, 1),
        (0, 2),
        (0, 3),
        (5, 6),
        (5, 7),
        (5, 8),
    ):
        graph.add_edge(*edge, color="single")
    return graph


def test_two_symmetric_unknown_centers_yield_rr_ss_and_meso_classes() -> None:
    registry = {
        "atom:0": TetrahedralStereo((0, 1, 2, 3, 5), None),
        "atom:5": TetrahedralStereo((5, 6, 7, 8, 0), None),
    }

    result = _enumerate(_two_center_graph(), registry)

    assert result.theoretical_assignment_count == 4
    assert result.exact_assignment_count == 3
    assert result.enantiomer_class_count == 2
    assert sorted(
        assignment.mirror_status.value for assignment in result.assignments
    ) == ["achiral", "chiral", "chiral"]


def _alkene() -> nx.Graph:
    graph = nx.Graph()
    for node, color in {
        0: "F",
        1: "Cl",
        2: "C",
        3: "C",
        4: "Br",
        5: "I",
    }.items():
        graph.add_node(node, color=color)
    for edge in ((0, 2), (1, 2), (3, 4), (3, 5)):
        graph.add_edge(*edge, color="single")
    graph.add_edge(2, 3, color="double")
    return graph


def test_planar_bond_enumerates_distinct_achiral_e_and_z() -> None:
    descriptor = PlanarBondStereo((0, 1, 2, 3, 4, 5), None)

    result = _enumerate(_alkene(), {"bond:2-3": descriptor})

    assert result.theoretical_assignment_count == 2
    assert result.exact_assignment_count == 2
    assert result.enantiomer_class_count == 2
    assert {assignment.mirror_status for assignment in result.assignments} == {
        StereographMirrorStatus.ACHIRAL
    }


def test_rdkit_odd_cumulene_enumerates_two_achiral_extended_states() -> None:
    molecule = Chem.MolFromSmiles("FC=C=C=CCl")
    assert molecule is not None

    result = enumerate_rdkit_stereographs(molecule)

    assert result.unresolved_loci == ("extended_bond:2-3-4-5",)
    assert result.theoretical_assignment_count == 2
    assert result.exact_assignment_count == 2
    assert {assignment.mirror_status for assignment in result.assignments} == {
        StereographMirrorStatus.ACHIRAL
    }


def test_square_planar_three_local_classes_remain_three_global_classes() -> None:
    graph = _star()
    descriptor = SquarePlanarStereo((0, 1, 2, 3, 4), None)

    result = _enumerate(graph, {"atom:0": descriptor})

    assert result.theoretical_assignment_count == 3
    assert result.exact_assignment_count == 3
    assert result.enantiomer_class_count == 3
    assert all(
        assignment.mirror_status is StereographMirrorStatus.ACHIRAL
        for assignment in result.assignments
    )


def test_complete_enumeration_cap_fails_before_partial_results() -> None:
    descriptor = OctahedralStereo((0, 1, 2, 3, 4, 5, 6), None)

    with pytest.raises(StereoEnumerationLimitError) as error:
        _enumerate(
            nx.complete_graph(7),
            {"atom:0": descriptor},
            max_assignments=29,
        )

    assert error.value.theoretical_count == 30
    assert error.value.limit == 29


def test_invalid_enumeration_caps_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _enumerate(nx.Graph(), {}, max_assignments=0)


def test_assignment_codes_survive_atom_relabeling() -> None:
    graph = _star()
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), None)
    mapping = {0: 50, 1: 40, 2: 30, 3: 20, 4: 10}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)

    original = _enumerate(graph, {"atom:0": descriptor})
    transported = _enumerate(
        relabeled,
        {"atom:50": descriptor.relabel(mapping)},
    )

    assert {assignment.canonical_code for assignment in original.assignments} == {
        assignment.canonical_code for assignment in transported.assignments
    }


@pytest.mark.parametrize(
    ("smiles", "raw_count", "exact_count", "mirror_classes"),
    (
        ("FC(Cl)Br", 2, 2, 1),
        ("FC=CCl", 2, 2, 2),
        ("CC(Cl)C(Cl)C", 4, 3, 2),
    ),
)
def test_rdkit_perception_feeds_exact_assignment_quotient(
    smiles: str,
    raw_count: int,
    exact_count: int,
    mirror_classes: int,
) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None

    result = enumerate_rdkit_stereographs(molecule)

    assert result.theoretical_assignment_count == raw_count
    assert result.exact_assignment_count == exact_count
    assert result.enantiomer_class_count == mirror_classes


def test_opposite_configured_ligands_resolve_a_dependent_focal_center() -> None:
    molecule = Chem.MolFromSmiles("F[C](Cl)([C@H](Br)I)[C@@H](Br)I")
    assert molecule is not None

    result = enumerate_rdkit_stereographs(molecule)

    assert result.unresolved_loci == ("atom:2",)
    assert result.focal_evidence[0].all_ports_distinct
    assert result.theoretical_assignment_count == 2
    assert result.exact_assignment_count == 2


def test_equal_configured_ligands_do_not_create_a_focal_element() -> None:
    molecule = Chem.MolFromSmiles("F[C](Cl)([C@H](Br)I)[C@H](Br)I")
    assert molecule is not None

    result = enumerate_rdkit_stereographs(molecule)

    assert result.unresolved_loci == ()
    assert result.theoretical_assignment_count == 1
    assert result.focal_evidence == ()


def test_rdkit_enumeration_rejects_enhanced_population_semantics() -> None:
    molecule = Chem.MolFromSmiles("F[C@H](Cl)[C@H](Br)I |o1:1,3|")
    assert molecule is not None

    with pytest.raises(TypeError, match="population enumeration"):
        enumerate_rdkit_stereographs(molecule)
