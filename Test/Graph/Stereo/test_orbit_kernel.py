"""Executable finite-group laws for the Sprint 16 stereo-orbit kernel."""

from itertools import permutations

import pytest

from synkit.Graph.Stereo import (
    Permutation,
    PermutationGroup,
    SHAPE_DEFINITIONS,
    StereoConfiguration,
    StereoRelationKind,
    StereoSpecification,
)


EXPECTED_GROUP_ORDERS = {
    "tetrahedral": (12, 24),
    "square_planar": (8, 24),
    "trigonal_bipyramidal": (6, 120),
    "octahedral": (24, 720),
    "planar_bond": (4, 8),
    "atrop_bond": (4, 8),
}


@pytest.mark.parametrize("shape,orders", EXPECTED_GROUP_ORDERS.items())
def test_shape_groups_have_exact_orders_and_laws(
    shape: str,
    orders: tuple[int, int],
) -> None:
    definition = SHAPE_DEFINITIONS[shape]
    fixed, unspecified = definition.preserving_group, definition.unspecified_group

    assert (len(fixed.elements), len(unspecified.elements)) == orders
    assert set(fixed.elements) <= set(unspecified.elements)
    for group in (fixed, unspecified):
        identity = Permutation.identity(group.degree)
        assert identity in group
        images = {element.image for element in group.elements}
        for element in group.elements:
            assert element.inverse() in group
            assert element.then(element.inverse()) == identity
            assert element.inverse().then(element) == identity
        for left in group.elements:
            for right in group.elements:
                composed = tuple(left.image[index] for index in right.image)
                assert composed in images


def test_invalid_permutation_groups_are_rejected() -> None:
    identity = Permutation.identity(3)
    swap = Permutation((1, 0, 2))
    cycle = Permutation((1, 2, 0))

    with pytest.raises(ValueError, match="closed|inverse"):
        PermutationGroup("not-a-group", 3, (identity, swap, cycle))


@pytest.mark.parametrize(
    "shape,reference_count,expected_orbits",
    (
        ("tetrahedral", 4, 2),
        ("square_planar", 4, 3),
        ("trigonal_bipyramidal", 5, 20),
        ("octahedral", 6, 30),
    ),
)
def test_all_atom_frame_permutations_partition_into_expected_fixed_orbits(
    shape: str,
    reference_count: int,
    expected_orbits: int,
) -> None:
    forms = {
        StereoConfiguration(shape, (0, *references)).canonical_frame
        for references in permutations(range(1, reference_count + 1))
    }

    assert len(forms) == expected_orbits


@pytest.mark.parametrize(
    "shape",
    ("tetrahedral", "square_planar", "trigonal_bipyramidal", "octahedral"),
)
def test_symmetric_group_canonical_shortcut_equals_exact_orbit_minimum(
    shape: str,
) -> None:
    group = SHAPE_DEFINITIONS[shape].unspecified_group
    frame = (0, *reversed(range(1, group.degree)))

    assert group.canonical(frame) == group.orbit(frame)[0]


@pytest.mark.parametrize("shape", ("planar_bond", "atrop_bond"))
def test_bond_frame_fixed_and_unspecified_orbit_sizes(shape: str) -> None:
    fixed = StereoConfiguration(shape, (1, 2, 3, 4, 5, 6))
    unknown = StereoConfiguration(
        shape,
        fixed.frame,
        StereoSpecification.UNSPECIFIED,
    )

    assert len(fixed.definition.preserving_group.orbit(fixed.frame)) == 4
    assert len(unknown.definition.unspecified_group.orbit(unknown.frame)) == 8


def test_tetrahedral_24_permutations_split_into_equivalent_and_opposite() -> None:
    source = StereoConfiguration("tetrahedral", (0, 1, 2, 3, 4))
    counts = {kind: 0 for kind in StereoRelationKind}

    for references in permutations((1, 2, 3, 4)):
        target = StereoConfiguration("tetrahedral", (0, *references))
        relation = source.relation_to(target)
        counts[relation.kind] += 1
        assert relation.witness is not None
        assert relation.witness.apply(source.frame) == target.frame

    assert counts[StereoRelationKind.EQUIVALENT] == 12
    assert counts[StereoRelationKind.OPPOSITE] == 12
    assert counts[StereoRelationKind.RECONFIGURED] == 0


@pytest.mark.parametrize(
    "shape,source,target,expected",
    (
        (
            "square_planar",
            (0, 1, 2, 3, 4),
            (0, 2, 1, 3, 4),
            StereoRelationKind.RECONFIGURED,
        ),
        (
            "trigonal_bipyramidal",
            (0, 1, 2, 3, 4, 5),
            (0, 1, 2, 3, 5, 4),
            StereoRelationKind.RECONFIGURED,
        ),
        (
            "octahedral",
            (0, 1, 2, 3, 4, 5, 6),
            (0, 2, 1, 3, 4, 5, 6),
            StereoRelationKind.RECONFIGURED,
        ),
        (
            "planar_bond",
            (1, 2, 3, 4, 5, 6),
            (2, 1, 3, 4, 5, 6),
            StereoRelationKind.OPPOSITE,
        ),
        (
            "atrop_bond",
            (1, 2, 3, 4, 5, 6),
            (2, 1, 3, 4, 5, 6),
            StereoRelationKind.OPPOSITE,
        ),
    ),
)
def test_geometry_specific_nonidentity_relations(
    shape: str,
    source: tuple[int, ...],
    target: tuple[int, ...],
    expected: StereoRelationKind,
) -> None:
    relation = StereoConfiguration(shape, source).relation_to(
        StereoConfiguration(shape, target)
    )

    assert relation.kind is expected
    assert relation.class_id is not None
    assert relation.witness is not None
    assert relation.witness.apply(source) == target


@pytest.mark.parametrize("shape", EXPECTED_GROUP_ORDERS)
def test_relation_class_is_independent_of_fixed_representatives(shape: str) -> None:
    definition = SHAPE_DEFINITIONS[shape]
    source_frame = tuple(range(definition.frame_arity))
    target_frame = definition.unspecified_group.elements[-1].apply(source_frame)
    source = StereoConfiguration(shape, source_frame)
    target = StereoConfiguration(shape, target_frame)
    expected = source.relation_to(target)

    for left in definition.preserving_group.elements:
        for right in definition.preserving_group.elements:
            observed = StereoConfiguration(shape, left.apply(source_frame)).relation_to(
                StereoConfiguration(shape, right.apply(target_frame))
            )
            assert observed.kind is expected.kind
            assert observed.class_id == expected.class_id


@pytest.mark.parametrize("shape", EXPECTED_GROUP_ORDERS)
def test_cached_tuple_double_coset_matches_explicit_group_composition(
    shape: str,
) -> None:
    definition = SHAPE_DEFINITIONS[shape]
    frame = tuple(range(definition.frame_arity))
    witnesses = definition.unspecified_group.elements
    for witness in (witnesses[0], witnesses[len(witnesses) // 2], witnesses[-1]):
        target = witness.apply(frame)
        relation = StereoConfiguration(shape, frame).relation_to(
            StereoConfiguration(shape, target)
        )
        expected = min(
            left.then(witness).then(right).image
            for left in definition.preserving_group.elements
            for right in definition.preserving_group.elements
        )
        assert relation.class_id == expected


@pytest.mark.parametrize("shape", EXPECTED_GROUP_ORDERS)
def test_witness_composition_replays_direct_transport(shape: str) -> None:
    definition = SHAPE_DEFINITIONS[shape]
    frame_a = tuple(range(definition.frame_arity))
    first = definition.unspecified_group.elements[-1]
    second = definition.unspecified_group.elements[len(definition.unspecified_group.elements) // 2]
    frame_b = first.apply(frame_a)
    frame_c = second.apply(frame_b)
    relation_ab = StereoConfiguration(shape, frame_a).relation_to(
        StereoConfiguration(shape, frame_b)
    )
    relation_bc = StereoConfiguration(shape, frame_b).relation_to(
        StereoConfiguration(shape, frame_c)
    )

    assert relation_ab.witness is not None
    assert relation_bc.witness is not None
    composed = relation_ab.witness.then(relation_bc.witness)
    assert composed.apply(frame_a) == frame_c


@pytest.mark.parametrize("shape", ("tetrahedral", "planar_bond", "atrop_bond"))
def test_binary_opposite_is_an_involution(shape: str) -> None:
    frame = tuple(range(SHAPE_DEFINITIONS[shape].frame_arity))
    configuration = StereoConfiguration(shape, frame)

    opposite = configuration.opposite()
    assert configuration.relation_to(opposite).kind is StereoRelationKind.OPPOSITE
    assert opposite.opposite() == configuration


def test_relabeling_and_reference_replacement_are_equivariant() -> None:
    configuration = StereoConfiguration("tetrahedral", (1, 2, 3, 4, 5))
    equivalent = StereoConfiguration("tetrahedral", (1, 3, 4, 2, 5))
    relabeling = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15}

    assert configuration.relabel(relabeling) == equivalent.relabel(relabeling)
    assert configuration.replace_reference(2, 99) == (
        equivalent.replace_reference(2, 99)
    )
    with pytest.raises(ValueError, match="loci"):
        configuration.replace_reference(1, 99)


def test_unspecified_and_unrelated_relations_are_explicit() -> None:
    fixed = StereoConfiguration("tetrahedral", (1, 2, 3, 4, 5))
    unspecified = StereoConfiguration(
        "tetrahedral",
        (1, 3, 2, 4, 5),
        StereoSpecification.UNSPECIFIED,
    )
    other_shape = StereoConfiguration("square_planar", (1, 2, 3, 4, 5))

    relation = fixed.relation_to(unspecified)
    assert relation.kind is StereoRelationKind.UNSPECIFIED
    assert relation.witness is not None
    assert fixed.relation_to(other_shape).kind is StereoRelationKind.UNRELATED


def test_mixed_material_and_virtual_references_have_deterministic_identity() -> None:
    first = StereoConfiguration("tetrahedral", (1, 2, 3, "@H:1", "@LP:1"))
    equivalent = StereoConfiguration(
        "tetrahedral",
        (1, 3, "@H:1", 2, "@LP:1"),
    )

    assert first == equivalent
    assert hash(first) == hash(equivalent)
