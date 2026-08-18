"""DPO rule construction, gluing, resource, and replay laws."""

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
from synkit.IO import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Rule.Apply import (
    DPOError,
    DPOIssueCode,
    EnvironmentToken,
    RuleSpan,
    SystemBoundary,
    apply_dpo,
    rule_from_its,
    rule_from_synrule,
)

SCHEMA = LabelSchema(
    node_identity=("kind",),
    node_state=("state",),
    edge_state=("weight",),
    name="dpo-test/1",
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


def test_interface_is_partially_labeled_and_allows_endpoint_state_change() -> None:
    left = _llg({1: 0})
    right = _llg({10: 1})
    rule = RuleSpan.from_mapping(left, right, {1: 10}, name="state-change")

    assert rule.interface.schema.node_state == ()
    assert rule.left_arm.mapping == {0: 1}
    assert rule.right_arm.mapping == {0: 10}

    application = apply_dpo(rule, left, {1: 1})
    assert application.result.node_labels(1, semantic=True) == {
        "kind": "X",
        "state": 1,
    }
    assert application.certificate.replay().valid


def test_dpo_deletes_and_adds_edges_with_two_commuting_squares() -> None:
    left = _llg({1: 0, 2: 0, 3: 0}, ((1, 2, 1),))
    right = _llg({10: 0, 20: 0, 30: 0}, ((20, 30, 1),))
    rule = RuleSpan.from_mapping(left, right, {1: 10, 2: 20, 3: 30})

    application = apply_dpo(rule, left, {1: 1, 2: 2, 3: 3})

    assert frozenset((1, 2)) not in application.result.edge_keys
    assert frozenset((2, 3)) in application.result.edge_keys
    assert application.certificate.deleted_edges == (frozenset((1, 2)),)
    assert application.certificate.added_edges == (frozenset((2, 3)),)
    assert application.certificate.replay().valid


def test_dangling_condition_reports_the_external_incident_edge() -> None:
    left = _llg({1: 0, 2: 0}, ((1, 2, 1),))
    right = _llg({10: 0})
    rule = RuleSpan.from_mapping(left, right, {1: 10})
    host = _llg({1: 0, 2: 0, 3: 0}, ((1, 2, 1), (2, 3, 1)))

    with pytest.raises(DPOError) as error:
        apply_dpo(rule, host, {1: 1, 2: 2})
    assert error.value.issues[0].code is DPOIssueCode.DANGLING


def test_added_edge_cannot_collapse_with_unmatched_host_context() -> None:
    left = _llg({1: 0, 2: 0})
    right = _llg({10: 0, 20: 0}, ((10, 20, 1),))
    rule = RuleSpan.from_mapping(left, right, {1: 10, 2: 20})
    host = _llg({1: 0, 2: 0}, ((1, 2, 1),))

    with pytest.raises(DPOError) as error:
        apply_dpo(rule, host, {1: 1, 2: 2})
    assert error.value.issues[0].code is DPOIssueCode.RESULT_LABEL


def test_identification_partial_incidence_and_label_failures_are_typed() -> None:
    left = _llg({1: 0, 2: 0}, ((1, 2, 1),))
    right = _llg({10: 0, 20: 0}, ((10, 20, 1),))
    rule = RuleSpan.from_mapping(left, right, {1: 10, 2: 20})
    host = _llg({3: 0, 4: 0}, ((3, 4, 1),))

    with pytest.raises(DPOError) as identification:
        apply_dpo(rule, host, {1: 3, 2: 3})
    assert identification.value.issues[0].code is DPOIssueCode.IDENTIFICATION

    with pytest.raises(DPOError) as partial:
        apply_dpo(rule, host, {1: 3})
    assert partial.value.issues[0].code is DPOIssueCode.MATCH_PARTIAL

    no_edge = _llg({3: 0, 4: 0})
    with pytest.raises(DPOError) as incidence:
        apply_dpo(rule, no_edge, {1: 3, 2: 4})
    assert incidence.value.issues[0].code is DPOIssueCode.MATCH_INCIDENCE

    changed = _llg({3: 1, 4: 0}, ((3, 4, 1),))
    with pytest.raises(DPOError) as labels:
        apply_dpo(rule, changed, {1: 3, 2: 4})
    assert labels.value.issues[0].code is DPOIssueCode.MATCH_LABEL


def test_node_creation_and_deletion_have_complete_provenance() -> None:
    left = _llg({1: 0, 2: 0})
    right = _llg({10: 0, 30: 0}, ((10, 30, 1),))
    rule = RuleSpan.from_mapping(left, right, {1: 10})

    application = apply_dpo(rule, left, {1: 1, 2: 2})

    assert application.certificate.deleted_nodes == (2,)
    assert len(application.certificate.added_nodes) == 1
    added = application.certificate.added_nodes[0]
    assert application.result.node_labels(added, semantic=True)["kind"] == "X"
    assert frozenset((1, added)) in application.result.edge_keys


def test_source_objects_are_not_mutated_and_relabeling_is_covariant() -> None:
    left = _llg({1: 0, 2: 0}, ((1, 2, 1),))
    right = _llg({10: 0, 20: 0}, ((10, 20, 2),))
    rule = RuleSpan.from_mapping(left, right, {1: 10, 2: 20})
    left_before = left.to_networkx()
    right_before = right.to_networkx()

    first = apply_dpo(rule, left, {1: 1, 2: 2})

    relabeled_left = left.relabel({1: "a", 2: "b"})
    relabeled_right = right.relabel({10: "x", 20: "y"})
    relabeled_rule = RuleSpan.from_mapping(
        relabeled_left, relabeled_right, {"a": "x", "b": "y"}
    )
    second = apply_dpo(relabeled_rule, relabeled_left, {"a": "a", "b": "b"})

    assert first.result.is_isomorphic(second.result)
    assert nx.utils.graphs_equal(left.to_networkx(), left_before)
    assert nx.utils.graphs_equal(right.to_networkx(), right_before)


def test_certificate_replay_detects_a_noncommuting_square() -> None:
    left = _llg({1: 0})
    right = _llg({10: 1})
    application = apply_dpo(RuleSpan.from_mapping(left, right, {1: 10}), left, {1: 1})
    tampered = replace(
        application.certificate,
        interface_to_context=((0, "not-the-match"),),
    )

    replay = tampered.replay()
    assert not replay.valid
    assert replay.issues[0].code is DPOIssueCode.NONCOMMUTATIVE


def _electron_llg(
    nodes: dict[int, tuple[str, int, int, int, int]],
    edges: tuple[tuple[int, int, float, float], ...] = (),
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
    for left, right, sigma, pi in edges:
        graph.add_edge(left, right, sigma_order=sigma, pi_order=pi)
    return LewisLabelledGraph.from_networkx(
        derive_electron_labeled_graph(graph), ELECTRON_LLG_SCHEMA
    )


def test_closed_system_conserves_material_and_electrons() -> None:
    left = _electron_llg(
        {1: ("C", 3, 0, 0, 4), 2: ("C", 3, 0, 0, 4), 3: ("C", 4, 0, 0, 4)},
        ((1, 2, 1.0, 0.0),),
    )
    right = _electron_llg(
        {10: ("C", 4, 0, 0, 4), 20: ("C", 3, 0, 0, 4), 30: ("C", 3, 0, 0, 4)},
        ((20, 30, 1.0, 0.0),),
    )
    rule = RuleSpan.from_mapping(
        left,
        right,
        {1: 10, 2: 20, 3: 30},
        boundary=SystemBoundary.CLOSED,
    )

    application = apply_dpo(rule, left, {1: 1, 2: 2, 3: 3})

    assert application.certificate.resource_delta is not None
    assert application.certificate.resource_delta.element_delta == ()
    assert application.certificate.resource_delta.electron_delta == 0
    assert application.result.is_isomorphic(right)


def test_closed_imbalance_fails_and_open_token_is_exact() -> None:
    left = _electron_llg({1: ("C", 4, 0, 0, 4)})
    right = _electron_llg({10: ("C", 4, 0, 0, 4), 20: ("H", 0, 1, 0, 1)})

    with pytest.raises(DPOError) as closed:
        RuleSpan.from_mapping(left, right, {1: 10}, boundary=SystemBoundary.CLOSED)
    assert closed.value.issues[0].code is DPOIssueCode.CLOSED_MATERIAL

    with pytest.raises(DPOError) as absent:
        RuleSpan.from_mapping(left, right, {1: 10}, boundary=SystemBoundary.OPEN)
    assert absent.value.issues[0].code is DPOIssueCode.OPEN_TOKEN_REQUIRED

    token = EnvironmentToken("H radical input", (("H", 1),), 1)
    rule = RuleSpan.from_mapping(
        left,
        right,
        {1: 10},
        boundary=SystemBoundary.OPEN,
        environment=token,
    )
    assert rule.resource_delta is not None
    assert rule.resource_delta.element_delta == (("H", 1),)
    assert rule.resource_delta.electron_delta == 1.0


def test_tuple_its_adapter_builds_a_span_not_a_paired_label_rule() -> None:
    its = rsmi_to_its("[CH3:1][OH:2]>>[CH2:1]=[O:2]", format="tuple")
    rule = rule_from_its(its, electron_complete=True)

    assert rule.left.schema is ELECTRON_LLG_SCHEMA
    assert rule.interface.schema.node_state == ()
    assert set(rule.left_arm.mapping.values()) == set(rule.left.node_ids)
    assert set(rule.right_arm.mapping.values()) == set(rule.right.node_ids)


def test_synrule_adapter_uses_explicit_endpoint_span() -> None:
    synrule = SynRule.from_smart(
        "[CH3:1][OH:2]>>[CH2:1]=[O:2]", format="tuple", implicit_h=False
    )
    rule = rule_from_synrule(synrule)

    assert rule.name == "rule"
    assert rule.interface.schema.node_state == ()
    assert rule.left.schema.name == "chemical-common/1"
