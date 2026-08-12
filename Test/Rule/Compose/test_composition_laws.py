"""Named premises and replay witnesses for composition and concurrency laws."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import pytest

from synkit.Graph.Morphism import (
    ELECTRON_LLG_SCHEMA,
    LabelSchema,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Rule.Apply import EnvironmentToken, RuleSpan, SystemBoundary, apply_dpo
from synkit.Rule.Compose import (
    LawError,
    LawIssueCode,
    RuleOverlap,
    check_parallel_independence,
    commute_independent,
    compose_rules,
    find_rule_span_isomorphism,
    find_llg_isomorphism,
    identity_rule,
    reverse_rule,
)

SCHEMA = LabelSchema(
    node_identity=("kind",),
    node_state=("state",),
    edge_state=("weight",),
    name="composition-laws-test/1",
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


def _state_rule(before: int, after: int, prefix: str) -> RuleSpan:
    return RuleSpan.from_mapping(
        _llg({f"{prefix}:L": before}),
        _llg({f"{prefix}:R": after}),
        {f"{prefix}:L": f"{prefix}:R"},
        name=prefix,
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


def _full_overlap(first: RuleSpan, second: RuleSpan) -> RuleOverlap:
    first_nodes = sorted(first.right.node_ids, key=repr)
    second_nodes = sorted(second.left.node_ids, key=repr)
    return RuleOverlap.from_mapping(
        first.right, second.left, dict(zip(first_nodes, second_nodes))
    )


def test_left_and_right_identity_have_replayable_rule_isomorphisms() -> None:
    rule = _state_rule(0, 1, "p")
    left_identity = identity_rule(rule.left, name="left-identity")
    right_identity = identity_rule(rule.right, name="right-identity")

    left_composite = compose_rules(
        left_identity, rule, _full_overlap(left_identity, rule)
    )
    right_composite = compose_rules(
        rule, right_identity, _full_overlap(rule, right_identity)
    )
    left_witness = find_rule_span_isomorphism(left_composite.rule, rule)
    right_witness = find_rule_span_isomorphism(right_composite.rule, rule)

    assert left_witness is not None and left_witness.replay()
    assert right_witness is not None and right_witness.replay()
    assert left_composite.certificate.replay().valid
    assert right_composite.certificate.replay().valid


def test_reversal_is_involutive_and_replays_on_the_exact_result_domain() -> None:
    rule = RuleSpan.from_mapping(
        _llg({1: 0, 2: 0}),
        _llg({10: 1, 30: 0}, ((10, 30, 1),)),
        {1: 10},
        name="replace-node",
    )
    reverse = reverse_rule(rule)
    involution = find_rule_span_isomorphism(reverse_rule(reverse), rule)
    forward = apply_dpo(rule, rule.left, {1: 1, 2: 2})
    reverse_match = dict(forward.certificate.right_to_result)
    backward = apply_dpo(reverse, forward.result, reverse_match)

    assert involution is not None and involution.replay()
    assert backward.result.is_isomorphic(rule.left)
    assert backward.certificate.replay().valid


def test_reversal_negates_an_open_environment_resource_token() -> None:
    rule = RuleSpan.from_mapping(
        _electron_llg({1: ("C", 4, 0, 0, 4)}),
        _electron_llg({10: ("C", 4, 0, 0, 4), 20: ("H", 0, 1, 0, 1)}),
        {1: 10},
        boundary=SystemBoundary.OPEN,
        environment=EnvironmentToken("hydrogen reservoir", (("H", 1),), 1),
    )

    reverse = reverse_rule(rule)

    assert reverse.environment is not None
    assert reverse.environment.element_delta == (("H", -1),)
    assert reverse.environment.electron_delta == -1
    witness = find_rule_span_isomorphism(reverse_rule(reverse), rule)
    assert witness is not None and witness.replay()


def test_reversal_duality_reverses_composition_order() -> None:
    first = _state_rule(0, 1, "p1")
    second = _state_rule(1, 2, "p2")
    forward = compose_rules(first, second, _full_overlap(first, second)).rule
    reversed_forward = reverse_rule(forward)
    reverse_second = reverse_rule(second)
    reverse_first = reverse_rule(first)
    backward = compose_rules(
        reverse_second,
        reverse_first,
        _full_overlap(reverse_second, reverse_first),
    ).rule

    witness = find_rule_span_isomorphism(reversed_forward, backward)
    assert witness is not None and witness.replay()


def test_sequential_and_composed_application_agree_in_larger_context() -> None:
    first = _state_rule(0, 1, "p1")
    second = _state_rule(1, 2, "p2")
    composition = compose_rules(first, second, _full_overlap(first, second))
    host = _llg({"target": 0, "context": 9})

    first_application = apply_dpo(first, host, {"p1:L": "target"})
    second_application = apply_dpo(second, first_application.result, {"p2:L": "target"})
    composed_application = apply_dpo(composition.rule, host, {"p1:L": "target"})
    witness = find_llg_isomorphism(
        second_application.result, composed_application.result
    )

    assert first_application.certificate.replay().valid
    assert second_application.certificate.replay().valid
    assert composed_application.certificate.replay().valid
    assert witness is not None and witness.is_isomorphism


def test_associativity_holds_up_to_a_replayable_rule_isomorphism() -> None:
    first = _state_rule(0, 1, "p1")
    second = _state_rule(1, 2, "p2")
    third = _state_rule(2, 3, "p3")

    first_second = compose_rules(first, second, _full_overlap(first, second)).rule
    left_grouped = compose_rules(
        first_second, third, _full_overlap(first_second, third)
    ).rule
    second_third = compose_rules(second, third, _full_overlap(second, third)).rule
    right_grouped = compose_rules(
        first, second_third, _full_overlap(first, second_third)
    ).rule

    witness = find_rule_span_isomorphism(left_grouped, right_grouped)
    assert witness is not None and witness.replay()


def test_rule_isomorphism_witness_replay_detects_a_tampered_layer_map() -> None:
    rule = _state_rule(0, 1, "p")
    relabeled = RuleSpan.from_mapping(_llg({10: 0}), _llg({20: 1}), {10: 20}, name="p")
    witness = find_rule_span_isomorphism(rule, relabeled)
    assert witness is not None

    tampered = replace(witness, left_map=(("p:L", "missing"),))
    assert not tampered.replay()


def test_disjoint_steps_commute_with_an_explicit_endpoint_isomorphism() -> None:
    host = _llg({"a": 0, "b": 0})
    first = _state_rule(0, 1, "first")
    second = _state_rule(0, 2, "second")

    independence = check_parallel_independence(
        first,
        {"first:L": "a"},
        second,
        {"second:L": "b"},
        host,
    )
    certificate = commute_independent(
        first,
        {"first:L": "a"},
        second,
        {"second:L": "b"},
        host,
    )

    assert independence.independent
    assert certificate.replay()
    assert certificate.result_isomorphism.is_isomorphism


def test_delete_use_conflict_names_nodes_and_edges() -> None:
    host = _llg({"a": 0, "b": 0}, (("a", "b", 1),))
    delete = RuleSpan.from_mapping(
        _llg({1: 0, 2: 0}, ((1, 2, 1),)),
        _llg({10: 0}),
        {1: 10},
        name="delete-b",
    )
    use = identity_rule(host, name="use-b")

    decision = check_parallel_independence(
        delete, {1: "a", 2: "b"}, use, {"a": "a", "b": "b"}, host
    )

    assert not decision.independent
    assert {issue.code for issue in decision.issues} >= {
        LawIssueCode.DELETE_USE_NODE,
        LawIssueCode.DELETE_USE_EDGE,
    }


def test_label_write_conflict_is_not_normalized_to_a_commuting_result() -> None:
    host = _llg({"a": 0})
    first = _state_rule(0, 1, "first")
    second = _state_rule(0, 2, "second")
    decision = check_parallel_independence(
        first,
        {"first:L": "a"},
        second,
        {"second:L": "a"},
        host,
    )

    assert not decision.independent
    assert {issue.code for issue in decision.issues} == {LawIssueCode.LABEL_WRITE_NODE}
    with pytest.raises(LawError) as error:
        commute_independent(
            first,
            {"first:L": "a"},
            second,
            {"second:L": "a"},
            host,
        )
    assert error.value.issues[0].code is LawIssueCode.LABEL_WRITE_NODE


def test_edge_label_write_conflict_is_typed() -> None:
    host = _llg({"a": 0, "b": 0}, (("a", "b", 0),))
    first = RuleSpan.from_mapping(
        _llg({1: 0, 2: 0}, ((1, 2, 0),)),
        _llg({10: 0, 20: 0}, ((10, 20, 1),)),
        {1: 10, 2: 20},
    )
    second = RuleSpan.from_mapping(
        _llg({3: 0, 4: 0}, ((3, 4, 0),)),
        _llg({30: 0, 40: 0}, ((30, 40, 2),)),
        {3: 30, 4: 40},
    )

    decision = check_parallel_independence(
        first, {1: "a", 2: "b"}, second, {3: "a", 4: "b"}, host
    )

    assert not decision.independent
    assert {issue.code for issue in decision.issues} == {LawIssueCode.LABEL_WRITE_EDGE}


def test_two_rules_cannot_both_add_the_same_simple_edge() -> None:
    host = _llg({"a": 0, "b": 0})
    first = RuleSpan.from_mapping(
        _llg({1: 0, 2: 0}),
        _llg({10: 0, 20: 0}, ((10, 20, 1),)),
        {1: 10, 2: 20},
    )
    second = RuleSpan.from_mapping(
        _llg({3: 0, 4: 0}),
        _llg({30: 0, 40: 0}, ((30, 40, 1),)),
        {3: 30, 4: 40},
    )

    decision = check_parallel_independence(
        first, {1: "a", 2: "b"}, second, {3: "a", 4: "b"}, host
    )

    assert not decision.independent
    assert {issue.code for issue in decision.issues} == {LawIssueCode.ADD_ADD_EDGE}


def test_component_applicability_is_a_named_independence_premise() -> None:
    host = _llg({"a": 1, "b": 0})
    first = _state_rule(0, 1, "first")
    second = _state_rule(0, 2, "second")

    decision = check_parallel_independence(
        first,
        {"first:L": "a"},
        second,
        {"second:L": "b"},
        host,
    )

    assert not decision.independent
    assert decision.issues[0].code is LawIssueCode.COMPONENT_APPLICATION


def test_nonisomorphic_linearizations_are_a_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import synkit.Rule.Compose.laws as laws_module

    host = _llg({"a": 0, "b": 0})
    first = _state_rule(0, 1, "first")
    second = _state_rule(0, 2, "second")
    monkeypatch.setattr(laws_module, "find_llg_isomorphism", lambda *_: None)

    with pytest.raises(LawError) as error:
        commute_independent(
            first,
            {"first:L": "a"},
            second,
            {"second:L": "b"},
            host,
        )
    assert error.value.issues[0].code is LawIssueCode.NONCOMMUTATIVE
