"""Certified binary composition for the admitted finite-LLG overlap class."""

from __future__ import annotations

import networkx as nx
import pytest

from synkit.Graph.Morphism import (
    ELECTRON_LLG_SCHEMA,
    LLGError,
    LLGIssueCode,
    LabelSchema,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Rule.Apply import DPOIssueCode, EnvironmentToken, RuleSpan, SystemBoundary
from synkit.Rule.Compose import (
    CompositionError,
    CompositionIssueCode,
    ProvenanceRef,
    RuleOverlap,
    compose_rules,
)

SCHEMA = LabelSchema(
    node_identity=("kind",),
    node_state=("state",),
    edge_state=("weight",),
    name="composition-test/1",
)


def _llg(
    nodes: dict[object, int], edges: tuple[tuple[object, object, int], ...] = ()
) -> LewisLabelledGraph:
    graph = nx.Graph()
    for node, state in nodes.items():
        graph.add_node(node, kind="X", state=state)
    for left, right, weight in edges:
        graph.add_edge(left, right, weight=weight)
    return LewisLabelledGraph.from_networkx(graph, SCHEMA)


def _state_rule(before: int, after: int, name: str) -> RuleSpan:
    return RuleSpan.from_mapping(
        _llg({f"{name}:L": before}),
        _llg({f"{name}:R": after}),
        {f"{name}:L": f"{name}:R"},
        name=name,
    )


def _electron_llg(
    nodes: dict[object, tuple[str, int, int, int, int]],
) -> LewisLabelledGraph:
    graph = nx.Graph()
    for node, (element, hcount, radical, lone_pairs, valence) in nodes.items():
        graph.add_node(
            node,
            element=element,
            aromatic=False,
            hcount=hcount,
            radical=radical,
            lone_pairs=lone_pairs,
            valence_electrons=valence,
        )
    return LewisLabelledGraph.from_networkx(
        derive_electron_labeled_graph(graph), ELECTRON_LLG_SCHEMA
    )


def test_state_updates_compose_and_replay_as_one_outer_dpo_square() -> None:
    first = _state_rule(0, 1, "first")
    second = _state_rule(1, 2, "second")
    overlap = RuleOverlap.from_mapping(
        first.right, second.left, {"first:R": "second:L"}
    )

    result = compose_rules(first, second, overlap)

    assert result.rule.left.node_labels("first:L", semantic=True)["state"] == 0
    right_node = next(iter(result.rule.right.node_ids))
    assert result.rule.right.node_labels(right_node, semantic=True)["state"] == 2
    assert result.certificate.replay().valid
    assert result.certificate.second_application.result.is_isomorphic(
        result.certificate.outer_application.result
    )


def test_edge_creation_then_update_derives_the_expected_outer_rule() -> None:
    first = RuleSpan.from_mapping(
        _llg({1: 0, 2: 0}),
        _llg({10: 0, 20: 0}, ((10, 20, 1),)),
        {1: 10, 2: 20},
        name="add-edge",
    )
    second = RuleSpan.from_mapping(
        _llg({"a": 0, "b": 0}, (("a", "b", 1),)),
        _llg({"x": 0, "y": 0}, (("x", "y", 2),)),
        {"a": "x", "b": "y"},
        name="update-edge",
    )
    overlap = RuleOverlap.from_mapping(first.right, second.left, {10: "a", 20: "b"})

    result = compose_rules(first, second, overlap)

    assert not result.rule.left.edge_keys
    assert len(result.rule.right.edge_keys) == 1
    edge = next(iter(result.rule.right.edge_keys))
    assert result.rule.right.edge_labels(edge, semantic=True)["weight"] == 2
    assert result.certificate.replay().valid


def test_transient_edge_disappears_from_the_composite_boundary() -> None:
    first = RuleSpan.from_mapping(
        _llg({1: 0, 2: 0}),
        _llg({10: 0, 20: 0}, ((10, 20, 1),)),
        {1: 10, 2: 20},
    )
    second = RuleSpan.from_mapping(
        _llg({"a": 0, "b": 0}, (("a", "b", 1),)),
        _llg({"x": 0, "y": 0}),
        {"a": "x", "b": "y"},
    )
    overlap = RuleOverlap.from_mapping(first.right, second.left, {10: "a", 20: "b"})

    result = compose_rules(first, second, overlap)

    assert not result.rule.left.edge_keys
    assert not result.rule.right.edge_keys
    assert result.rule.left.is_isomorphic(result.rule.right)


def test_identity_and_empty_overlap_are_admitted() -> None:
    identity_left = _llg({1: 0})
    identity = RuleSpan.from_mapping(identity_left, _llg({10: 0}), {1: 10})
    update = _state_rule(0, 1, "update")
    identity_overlap = RuleOverlap.from_mapping(
        identity.right, update.left, {10: "update:L"}
    )

    with_identity = compose_rules(identity, update, identity_overlap)
    assert with_identity.rule.left.is_isomorphic(update.left)
    assert with_identity.rule.right.is_isomorphic(update.right)

    disjoint_first = _state_rule(0, 1, "disjoint-first")
    disjoint_second = _state_rule(2, 3, "disjoint-second")
    empty = RuleOverlap.from_mapping(disjoint_first.right, disjoint_second.left, {})
    disjoint = compose_rules(disjoint_first, disjoint_second, empty)
    assert len(disjoint.rule.left.node_ids) == 2
    assert len(disjoint.rule.right.node_ids) == 2
    assert disjoint.certificate.replay().valid


def test_open_resource_tokens_compose_by_signed_addition() -> None:
    first = RuleSpan.from_mapping(
        _electron_llg({1: ("C", 4, 0, 0, 4)}),
        _electron_llg({10: ("C", 4, 0, 0, 4), 20: ("H", 0, 1, 0, 1)}),
        {1: 10},
        boundary=SystemBoundary.OPEN,
        environment=EnvironmentToken("H input", (("H", 1),), 1),
    )
    second = RuleSpan.from_mapping(
        _electron_llg({"c": ("C", 4, 0, 0, 4), "h": ("H", 0, 1, 0, 1)}),
        _electron_llg({"out": ("C", 4, 0, 0, 4)}),
        {"c": "out"},
        boundary=SystemBoundary.OPEN,
        environment=EnvironmentToken("H output", (("H", -1),), -1),
    )
    overlap = RuleOverlap.from_mapping(first.right, second.left, {10: "c", 20: "h"})

    result = compose_rules(first, second, overlap)

    assert result.rule.boundary is SystemBoundary.OPEN
    assert result.rule.environment is not None
    assert result.rule.environment.element_delta == ()
    assert result.rule.environment.electron_delta == 0
    assert result.certificate.replay().valid


def test_overlap_rejects_incompatible_labels_and_noninjective_maps() -> None:
    first_right = _llg({1: 0, 2: 0})
    second_left = _llg({"a": 1, "b": 0})

    with pytest.raises(CompositionError) as labels:
        RuleOverlap.from_mapping(first_right, second_left, {1: "a"})
    assert labels.value.issues[0].code is CompositionIssueCode.OVERLAP_NODE_LABEL

    with pytest.raises(CompositionError) as injectivity:
        RuleOverlap.from_mapping(first_right, second_left, {1: "b", 2: "b"})
    assert (
        injectivity.value.issues[0].code is CompositionIssueCode.OVERLAP_NON_INJECTIVE
    )


def test_overlap_must_be_edge_induced_on_its_selected_nodes() -> None:
    first = RuleSpan.from_mapping(
        _llg({1: 0, 2: 0}, ((1, 2, 1),)),
        _llg({10: 0, 20: 0}, ((10, 20, 1),)),
        {1: 10, 2: 20},
    )
    second = RuleSpan.from_mapping(
        _llg({"a": 0, "b": 0}, (("a", "b", 1),)),
        _llg({"x": 0, "y": 0}, (("x", "y", 1),)),
        {"a": "x", "b": "y"},
    )
    overlap = RuleOverlap.from_mapping(
        first.right,
        second.left,
        {10: "a", 20: "b"},
        overlap_edges=set(),
    )

    with pytest.raises(CompositionError) as error:
        compose_rules(first, second, overlap)
    assert error.value.issues[0].code is CompositionIssueCode.OVERLAP_EDGE_CLOSURE


def test_created_overlap_node_cannot_require_preexisting_external_context() -> None:
    first = RuleSpan.from_mapping(
        _llg({1: 0}),
        _llg({10: 0, 20: 0}),
        {1: 10},
        name="create-node",
    )
    second = RuleSpan.from_mapping(
        _llg({"created": 0, "context": 0}, (("created", "context", 1),)),
        _llg({"x": 0, "y": 0}, (("x", "y", 2),)),
        {"created": "x", "context": "y"},
        name="needs-context",
    )
    overlap = RuleOverlap.from_mapping(first.right, second.left, {20: "created"})

    with pytest.raises(CompositionError) as error:
        compose_rules(first, second, overlap)
    assert error.value.issues[0].code is CompositionIssueCode.PULLBACK_COMPLEMENT


def test_second_rule_cannot_add_an_edge_already_supplied_by_first_rule() -> None:
    first = RuleSpan.from_mapping(
        _llg({1: 0, 2: 0}, ((1, 2, 1),)),
        _llg({10: 0, 20: 0}, ((10, 20, 1),)),
        {1: 10, 2: 20},
    )
    second = RuleSpan.from_mapping(
        _llg({"a": 0, "b": 0}),
        _llg({"x": 0, "y": 0}, (("x", "y", 1),)),
        {"a": "x", "b": "y"},
    )
    overlap = RuleOverlap.from_mapping(first.right, second.left, {10: "a", 20: "b"})

    with pytest.raises(CompositionError) as error:
        compose_rules(first, second, overlap)

    assert error.value.issues[0].code is CompositionIssueCode.SECOND_APPLICATION
    nested = error.value.issues[0].context["issues"]
    assert nested[0]["code"] == DPOIssueCode.RESULT_LABEL.value


def test_provenance_is_total_and_shared_overlap_carriers_have_both_origins() -> None:
    first_left = _llg({1: 0, 2: 0}, ((1, 2, 1),))
    first = RuleSpan.from_mapping(
        first_left,
        _llg({10: 0, 20: 0}, ((10, 20, 1),)),
        {1: 10, 2: 20},
    )
    second = RuleSpan.from_mapping(
        _llg({"a": 0, "b": 0}, (("a", "b", 1),)),
        _llg({"x": 1, "y": 0}, (("x", "y", 1),)),
        {"a": "x", "b": "y"},
    )
    overlap = RuleOverlap.from_mapping(first.right, second.left, {10: "a", 20: "b"})

    certificate = compose_rules(first, second, overlap).certificate

    assert certificate.replay().valid
    left_edge_sources = [
        refs for side, _, refs in certificate.provenance.edge_sources if side == "left"
    ]
    assert any(
        ProvenanceRef(1, "left", frozenset((1, 2))) in refs
        for refs in left_edge_sources
    )
    assert any(
        ProvenanceRef(2, "left", frozenset(("a", "b"))) in refs
        for refs in left_edge_sources
    )


def test_composition_is_covariant_under_independent_carrier_relabeling() -> None:
    first = _state_rule(0, 1, "first")
    second = _state_rule(1, 2, "second")
    original = compose_rules(
        first,
        second,
        RuleOverlap.from_mapping(first.right, second.left, {"first:R": "second:L"}),
    )

    relabeled_first = RuleSpan.from_mapping(
        _llg({1: 0}), _llg({2: 1}), {1: 2}, name="first"
    )
    relabeled_second = RuleSpan.from_mapping(
        _llg({3: 1}), _llg({4: 2}), {3: 4}, name="second"
    )
    relabeled = compose_rules(
        relabeled_first,
        relabeled_second,
        RuleOverlap.from_mapping(relabeled_first.right, relabeled_second.left, {2: 3}),
    )

    assert original.rule.left.is_isomorphic(relabeled.rule.left)
    assert original.rule.right.is_isomorphic(relabeled.rule.right)


def test_loops_are_refused_at_the_llg_object_boundary() -> None:
    graph = nx.Graph()
    graph.add_node(1, kind="X", state=0)
    graph.add_edge(1, 1, weight=1)

    with pytest.raises(LLGError) as error:
        LewisLabelledGraph.from_networkx(graph, SCHEMA)
    assert error.value.issues[0].code is LLGIssueCode.SELF_LOOP


def test_source_rules_are_not_mutated() -> None:
    first = _state_rule(0, 1, "first")
    second = _state_rule(1, 2, "second")
    before = (
        first.left.to_networkx(),
        first.right.to_networkx(),
        second.left.to_networkx(),
        second.right.to_networkx(),
    )

    compose_rules(
        first,
        second,
        RuleOverlap.from_mapping(first.right, second.left, {"first:R": "second:L"}),
    )

    after = (
        first.left.to_networkx(),
        first.right.to_networkx(),
        second.left.to_networkx(),
        second.right.to_networkx(),
    )
    assert all(nx.utils.graphs_equal(old, new) for old, new in zip(before, after))
