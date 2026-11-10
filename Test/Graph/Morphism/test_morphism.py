"""Law tests for embeddings with guarded wildcard obligations."""

from itertools import permutations, product

import pytest

from synkit.Graph.Morphism import (
    GraphMorphism,
    GraphMorphismError,
    MorphismIssueCode,
    WildcardConstraint,
    WildcardRole,
)


def _chain() -> tuple[GraphMorphism, GraphMorphism, GraphMorphism]:
    first = GraphMorphism(
        "A",
        "B",
        {1, 2},
        {10, 20},
        {1: 10, 2: 20},
        {
            1: WildcardConstraint(
                WildcardRole.ATTACHMENT_PORT,
                elements={"C", "N", "O"},
                owner=2,
                capacity=3,
                resource_budget=3,
            )
        },
    )
    second = GraphMorphism(
        "B",
        "C",
        {10, 20},
        {100, 200},
        {10: 100, 20: 200},
        {
            10: WildcardConstraint(
                WildcardRole.ATTACHMENT_PORT,
                elements={"C", "N"},
                owner=20,
                capacity=2,
                resource_budget=2,
            )
        },
    )
    third = GraphMorphism(
        "C",
        "D",
        {100, 200},
        {1000, 2000},
        {100: 1000, 200: 2000},
        {
            100: WildcardConstraint(
                WildcardRole.ATTACHMENT_PORT,
                elements={"C"},
                owner=200,
                capacity=1,
                resource_budget=1,
            )
        },
    )
    return first, second, third


def test_material_mapping_is_total_and_injective() -> None:
    with pytest.raises(GraphMorphismError) as noninjective:
        GraphMorphism("A", "B", {1, 2}, {10}, {1: 10, 2: 10})
    assert noninjective.value.issues[0].code is MorphismIssueCode.NON_INJECTIVE

    with pytest.raises(GraphMorphismError) as partial:
        GraphMorphism("A", "B", {1, 2}, {10, 20}, {1: 10})
    assert any(
        issue.code is MorphismIssueCode.PARTIAL_MAPPING
        for issue in partial.value.issues
    )


def test_identity_is_neutral_on_both_sides() -> None:
    first, _, _ = _chain()

    assert GraphMorphism.identity("A", {1, 2}).then(first) == first
    assert first.then(GraphMorphism.identity("B", {10, 20})) == first


def test_composition_is_associative_and_refines_resources() -> None:
    first, second, third = _chain()

    left = first.then(second).then(third)
    right = first.then(second.then(third))
    assert left == right
    constraint = left.substitutions[1]
    assert constraint.elements == frozenset({"C"})
    assert constraint.owner == 2
    assert constraint.capacity == 1
    assert constraint.resource_budget == 1


def test_contradictions_never_compose() -> None:
    first = GraphMorphism(
        "A",
        "B",
        {1},
        {2},
        {1: 2},
        {1: WildcardConstraint(WildcardRole.QUERY_ATOM, elements={"C"})},
    )
    second = GraphMorphism(
        "B",
        "C",
        {2},
        {3},
        {2: 3},
        {2: WildcardConstraint(WildcardRole.QUERY_ATOM, elements={"O"})},
    )

    with pytest.raises(GraphMorphismError) as contradiction:
        first.then(second)
    assert contradiction.value.issues[0].code is (
        MorphismIssueCode.CONSTRAINT_CONTRADICTION
    )


def test_endpoint_object_identity_is_required_for_composition() -> None:
    first, second, _ = _chain()
    incompatible = GraphMorphism(
        "not-B", "C", second.source_nodes, second.target_nodes, second.f, second.theta
    )
    with pytest.raises(GraphMorphismError) as mismatch:
        first.then(incompatible)
    assert mismatch.value.issues[0].code is MorphismIssueCode.ENDPOINT_MISMATCH


def test_owner_incidence_cannot_escape_source_or_composed_image() -> None:
    with pytest.raises(GraphMorphismError) as outside_source:
        GraphMorphism(
            "A",
            "B",
            {1},
            {10},
            {1: 10},
            {1: WildcardConstraint(WildcardRole.ATTACHMENT_PORT, owner=999)},
        )
    assert outside_source.value.issues[0].code is (
        MorphismIssueCode.OWNER_OUTSIDE_SOURCE
    )

    first = GraphMorphism("A", "B", {1}, {10, 20}, {1: 10})
    second = GraphMorphism(
        "B",
        "C",
        {10, 20},
        {100, 200},
        {10: 100, 20: 200},
        {10: WildcardConstraint(WildcardRole.ATTACHMENT_PORT, owner=20)},
    )
    with pytest.raises(GraphMorphismError) as outside_image:
        first.then(second)
    assert outside_image.value.issues[0].code is (MorphismIssueCode.OWNER_OUTSIDE_IMAGE)


def test_consistent_relabeling_preserves_canonical_signature() -> None:
    first, _, _ = _chain()
    relabeled = first.relabel(
        {1: "left-port", 2: "left-owner"},
        {10: "right-port", 20: "right-owner"},
        source="A-prime",
        target="B-prime",
    )

    assert relabeled.substitutions["left-port"].owner == "left-owner"
    assert relabeled.canonical_signature() == first.canonical_signature()


def test_constraints_outside_embedding_image_cannot_disappear() -> None:
    """An omitted obligation must not hide an intermediate contradiction."""
    first = GraphMorphism("A", "B", {"a"}, {"b", "c"}, {"a": "b"})
    second = GraphMorphism(
        "B", "C", {"b", "c"}, {"x", "y"}, {"b": "x", "c": "y"},
        {"c": WildcardConstraint(WildcardRole.QUERY_ATOM, elements={"C"})},
    )
    third = GraphMorphism(
        "C", "D", {"x", "y"}, {"p", "q"}, {"x": "p", "y": "q"},
        {"y": WildcardConstraint(WildcardRole.QUERY_ATOM, elements={"N"})},
    )

    with pytest.raises(GraphMorphismError) as left:
        first.then(second).then(third)
    assert left.value.issues[0].code is MorphismIssueCode.CONSTRAINT_OUTSIDE_IMAGE
    with pytest.raises(GraphMorphismError) as right:
        first.then(second.then(third))
    assert right.value.issues[0].code is MorphismIssueCode.CONSTRAINT_CONTRADICTION


def test_proper_embeddings_compose_when_all_obligations_are_representable() -> None:
    first = GraphMorphism(
        "A", "B", {1, 2}, {10, 20, 30}, {1: 10, 2: 20},
        {1: WildcardConstraint(WildcardRole.ATTACHMENT_PORT, owner=2,
                               elements={"C", "N", "O"})},
    )
    second = GraphMorphism(
        "B", "C", {10, 20, 30}, {100, 200, 300, 400},
        {10: 100, 20: 200, 30: 300},
        {10: WildcardConstraint(WildcardRole.ATTACHMENT_PORT, owner=20,
                                elements={"C", "N"})},
    )
    third = GraphMorphism(
        "C", "D", {100, 200, 300, 400}, {1000, 2000, 3000, 4000, 5000},
        {100: 1000, 200: 2000, 300: 3000, 400: 4000},
        {100: WildcardConstraint(WildcardRole.ATTACHMENT_PORT, owner=200,
                                 elements={"N"})},
    )

    composed = first.then(second).then(third)
    assert composed == first.then(second.then(third))
    assert composed.mapping == {1: 1000, 2: 2000}
    assert composed.substitutions == {
        1: WildcardConstraint(WildcardRole.ATTACHMENT_PORT, owner=2, elements={"N"})
    }
    assert GraphMorphism.identity("A", {1, 2}).then(composed) == composed
    assert composed.then(GraphMorphism.identity("D", third.target_nodes)) == composed


def test_finite_guarded_composition_associativity_including_definedness() -> None:
    """Check all 20,216 representative chains, including proper embeddings.

    Each arrow has at most one obligation, with every source position, owner
    position (or none), and nonempty domain over {C, N}.  Every injective map
    occurs for the carrier sizes 1 -> 2 -> 2 -> 2.
    """
    def arrows(source, target, source_size, target_size):
        source_nodes, target_nodes = set(range(source_size)), set(range(target_size))
        environments = [{}]
        for node, owner, domain in product(
            source_nodes, (None, *sorted(source_nodes)), ({"C"}, {"N"}, {"C", "N"})
        ):
            environments.append({
                node: WildcardConstraint(WildcardRole.QUERY_ATOM,
                                         elements=domain, owner=owner)
            })
        return tuple(
            GraphMorphism(source, target, source_nodes, target_nodes,
                          dict(zip(sorted(source_nodes), image)), environment)
            for image, environment in product(
                permutations(sorted(target_nodes), source_size), environments
            )
        )

    def outcome(first, second, third, left):
        try:
            return (
                first.then(second).then(third)
                if left else first.then(second.then(third))
            )
        except GraphMorphismError:
            return None

    checked, successful = 0, 0
    for first, second, third in product(
        arrows("A", "B", 1, 2),
        arrows("B", "C", 2, 2),
        arrows("C", "D", 2, 2),
    ):
        left = outcome(first, second, third, True)
        right = outcome(first, second, third, False)
        assert left == right, (first, second, third)
        checked += 1
        successful += left is not None
    assert checked == 20_216
    assert 0 < successful < checked
