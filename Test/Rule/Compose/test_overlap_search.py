"""Exhaustive bounded overlap search and exact quotient contracts."""

from __future__ import annotations

import networkx as nx
import pytest

from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph
from synkit.Rule.Apply import RuleSpan
from synkit.Rule.Compose import (
    CompositionIssueCode,
    OverlapSearchError,
    OverlapSearchIssueCode,
    OverlapSearchLimits,
    canonical_rule_identity,
    rule_spans_isomorphic,
    search_compositions,
)

SCHEMA = LabelSchema(
    node_identity=("kind",),
    node_state=("state",),
    edge_state=("weight",),
    name="overlap-search-test/1",
)


def _llg(
    nodes: list[tuple[object, int]],
    edges: tuple[tuple[object, object, int], ...] = (),
) -> LewisLabelledGraph:
    graph = nx.Graph()
    for node, state in nodes:
        graph.add_node(node, kind="X", state=state)
    for left, right, weight in edges:
        graph.add_edge(left, right, weight=weight)
    return LewisLabelledGraph.from_networkx(graph, SCHEMA)


def _identity_two(
    left_nodes: tuple[object, object], right_nodes: tuple[object, object]
) -> RuleSpan:
    return RuleSpan.from_mapping(
        _llg([(left_nodes[0], 0), (left_nodes[1], 0)]),
        _llg([(right_nodes[0], 0), (right_nodes[1], 0)]),
        dict(zip(left_nodes, right_nodes)),
        name="identity-two",
    )


def _update_one(left: object, right: object) -> RuleSpan:
    return RuleSpan.from_mapping(
        _llg([(left, 0)]),
        _llg([(right, 1)]),
        {left: right},
        name="update-one",
    )


def _update_two(
    left_nodes: tuple[object, object], right_nodes: tuple[object, object]
) -> RuleSpan:
    return RuleSpan.from_mapping(
        _llg([(left_nodes[0], 0), (left_nodes[1], 0)]),
        _llg([(right_nodes[0], 1), (right_nodes[1], 1)]),
        dict(zip(left_nodes, right_nodes)),
        name="update-two",
    )


def test_repeated_component_witnesses_survive_exact_quotienting() -> None:
    first = _identity_two((1, 2), (10, 20))
    second = _update_one("a", "b")

    result = search_compositions(first, second)

    assert result.raw_overlap_count == 3
    assert result.accepted_count == 3
    assert result.exact_class_count == 2
    assert not result.rejected
    witness_counts = sorted(len(group.witnesses) for group in result.classes)
    assert witness_counts == [1, 2]
    shared = next(group for group in result.classes if len(group.witnesses) == 2)
    assert len({witness.overlap_digest for witness in shared.witnesses}) == 1
    assert {len(witness.overlap.interface.node_ids) for witness in shared.witnesses} == {
        1
    }


def test_automorphic_partial_injections_are_all_material_witnesses() -> None:
    first = _identity_two((1, 2), (10, 20))
    second = _update_two(("a", "b"), ("x", "y"))

    result = search_compositions(first, second)

    assert result.raw_overlap_count == 7
    assert result.accepted_count == 7
    assert result.exact_class_count == 3
    assert sorted(len(group.witnesses) for group in result.classes) == [1, 2, 4]


def test_counts_and_class_identities_ignore_insertion_order_and_carrier_names() -> None:
    first = _identity_two((1, 2), (10, 20))
    second = _update_two(("a", "b"), ("x", "y"))
    baseline = search_compositions(first, second)

    reordered_first = RuleSpan.from_mapping(
        _llg([(2, 0), (1, 0)]),
        _llg([(20, 0), (10, 0)]),
        {1: 10, 2: 20},
        name="identity-two",
    )
    reordered_second = RuleSpan.from_mapping(
        _llg([("b", 0), ("a", 0)]),
        _llg([("y", 1), ("x", 1)]),
        {"a": "x", "b": "y"},
        name="update-two",
    )
    reordered = search_compositions(reordered_first, reordered_second)

    relabeled_first = _identity_two(("l1", "l2"), ("r1", "r2"))
    relabeled_second = _update_two((100, 200), (300, 400))
    relabeled = search_compositions(relabeled_first, relabeled_second)

    summaries = {
        (
            result.raw_overlap_count,
            result.accepted_count,
            tuple(group.canonical_id for group in result.classes),
            tuple(sorted(len(group.witnesses) for group in result.classes)),
        )
        for result in (baseline, reordered, relabeled)
    }
    assert len(summaries) == 1


def test_exact_rule_isomorphism_not_hash_or_carrier_equality_defines_classes() -> None:
    first = _update_one(1, 2)
    relabeled = _update_one("left", "right")
    different = RuleSpan.from_mapping(
        _llg([("left", 0)]),
        _llg([("right", 2)]),
        {"left": "right"},
    )

    assert rule_spans_isomorphic(first, relabeled)
    assert canonical_rule_identity(first, permutation_limit=10) == (
        canonical_rule_identity(relabeled, permutation_limit=10)
    )
    assert not rule_spans_isomorphic(first, different)


def test_composition_refusals_are_retained_with_their_failed_premise() -> None:
    first = RuleSpan.from_mapping(
        _llg([(1, 0)]),
        _llg([(10, 0), (20, 0)]),
        {1: 10},
        name="create-node",
    )
    second = RuleSpan.from_mapping(
        _llg([("a", 0), ("b", 0)], (("a", "b", 1),)),
        _llg([("x", 0), ("y", 0)], (("x", "y", 2),)),
        {"a": "x", "b": "y"},
        name="edge-context",
    )

    result = search_compositions(first, second)

    assert result.raw_overlap_count == 5
    assert result.accepted_count + len(result.rejected) == 5
    assert result.rejected
    assert {
        rejection.issues[0].code for rejection in result.rejected
    } == {CompositionIssueCode.PULLBACK_COMPLEMENT}


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        (OverlapSearchLimits(max_states=1), OverlapSearchIssueCode.STATE_LIMIT),
        (
            OverlapSearchLimits(max_overlaps=2),
            OverlapSearchIssueCode.OVERLAP_LIMIT,
        ),
        (
            OverlapSearchLimits(max_overlap_nodes=0),
            OverlapSearchIssueCode.NODE_LIMIT,
        ),
        (
            OverlapSearchLimits(max_canonical_permutations=1),
            OverlapSearchIssueCode.CANONICAL_LIMIT,
        ),
    ],
)
def test_resource_exhaustion_is_typed_and_never_silent(
    limits: OverlapSearchLimits, expected: OverlapSearchIssueCode
) -> None:
    first = _identity_two((1, 2), (10, 20))
    second = _update_two(("a", "b"), ("x", "y"))

    with pytest.raises(OverlapSearchError) as error:
        search_compositions(first, second, limits=limits)
    assert error.value.issue.code is expected


def test_search_does_not_mutate_source_rules() -> None:
    first = _identity_two((1, 2), (10, 20))
    second = _update_one("a", "b")
    before = (
        first.left.to_networkx(),
        first.right.to_networkx(),
        second.left.to_networkx(),
        second.right.to_networkx(),
    )

    search_compositions(first, second)

    after = (
        first.left.to_networkx(),
        first.right.to_networkx(),
        second.left.to_networkx(),
        second.right.to_networkx(),
    )
    assert all(
        nx.utils.graphs_equal(old, new) for old, new in zip(before, after)
    )
