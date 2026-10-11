"""Regression witnesses for natural contexts and forged DPO certificates."""

from dataclasses import replace

import networkx as nx
import pytest

from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph
from synkit.Rule.Apply import DPOError, RuleSpan, apply_dpo
from synkit.Rule.Compose import RuleOverlap, compose_rules

SCHEMA = LabelSchema(
    node_identity=("kind",), node_state=("state", "fixed"), edge_state=("weight",)
)


def graph(nodes=(1, 2), *, state=0, edge=False):
    value = nx.Graph()
    value.add_nodes_from(
        (node, {"kind": "X", "state": state, "fixed": 7}) for node in nodes
    )
    if edge:
        value.add_edge(*nodes[:2], weight=1)
    return LewisLabelledGraph.from_networkx(value, SCHEMA)


def certificate():
    left = graph(edge=True)
    right = graph(state=1, edge=True)
    rule = RuleSpan.from_mapping(left, right, {1: 1, 2: 2})
    return apply_dpo(rule, left, {1: 1, 2: 2}).certificate


def test_context_erases_changed_coordinates_and_retains_common_ones():
    c = certificate()
    assert c.context.partial and c.rule.interface.partial
    assert c.context.schema == c.host.schema == c.rule.interface.schema
    assert c.context.node_labels(1, semantic=True) == {"kind": "X", "fixed": 7}
    assert c.rule.interface.node_labels(0, semantic=True) == {"kind": "X", "fixed": 7}
    assert c.context.edge_labels(frozenset((1, 2)), semantic=True) == {"weight": 1}
    assert c.replay().valid


def test_context_erases_unchanged_coordinate_if_explicit_interface_leaves_it_undefined():
    c = certificate()
    k_graph = c.rule.interface.to_networkx()
    for _, attrs in k_graph.nodes(data=True):
        attrs.pop("fixed")
    interface = LewisLabelledGraph.from_networkx(k_graph, SCHEMA, partial=True)
    from synkit.Rule.Apply import RuleEmbedding

    rule = replace(
        c.rule,
        interface=interface,
        left_arm=RuleEmbedding(interface, c.rule.left, c.rule.left_arm.f),
        right_arm=RuleEmbedding(interface, c.rule.right, c.rule.right_arm.f),
    )
    application = apply_dpo(rule, c.host, dict(c.match))
    assert "fixed" not in application.certificate.context.node_labels(1, semantic=True)
    assert application.result.node_labels(1)["fixed"] == 7
    assert application.certificate.replay().valid


def test_preserved_identity_cannot_be_omitted_from_a_manual_interface():
    from synkit.Rule.Apply import RuleEmbedding

    c = certificate()
    value = c.rule.interface.to_networkx()
    value.nodes[0].pop("kind")
    interface = LewisLabelledGraph.from_networkx(value, SCHEMA, partial=True)
    with pytest.raises(DPOError, match="omits identity"):
        RuleEmbedding(interface, c.rule.left, c.rule.left_arm.f)


@pytest.mark.parametrize("field", ["result", "host", "context"])
def test_unjustified_extra_carrier_is_rejected(field):
    c = certificate()
    original = getattr(c, field)
    changed = original.to_networkx()
    changed.add_node("unjustified", kind="X", state=0, fixed=7)
    altered = LewisLabelledGraph.from_networkx(
        changed, SCHEMA, partial=original.partial
    )
    assert not replace(c, **{field: altered}).replay().valid


@pytest.mark.parametrize(
    "field",
    [
        "match",
        "interface_to_context",
        "context_to_host",
        "context_to_result",
        "right_to_result",
    ],
)
@pytest.mark.parametrize(
    "corruption", ["duplicate", "noninjective", "outside", "missing"]
)
def test_corrupt_maps_are_rejected_without_exception(field, corruption):
    c = certificate()
    pairs = getattr(c, field)
    if corruption == "duplicate":
        changed = (*pairs, pairs[0])
    elif corruption == "noninjective":
        changed = tuple((node, pairs[0][1]) for node, _ in pairs)
    elif corruption == "outside":
        changed = ((pairs[0][0], "outside"), *pairs[1:])
    else:
        changed = pairs[1:]
    assert not replace(c, **{field: changed}).replay().valid


def test_old_total_context_is_rejected_even_for_a_commuting_carrier_square():
    c = certificate()
    wrong = LewisLabelledGraph.from_networkx(c.host.to_networkx(), SCHEMA, partial=True)
    assert not replace(c, context=wrong).replay().valid


@pytest.mark.parametrize(
    "field", ["deleted_nodes", "deleted_edges", "added_nodes", "added_edges"]
)
def test_forged_edit_provenance_is_rejected(field):
    c = certificate()
    assert not replace(c, **{field: ("forged",)}).replay().valid


def test_extra_result_edge_fails_coverage_and_edge_label_corruption_fails_preservation():
    left = graph(nodes=(1, 2, 3))
    right = graph(nodes=(1, 2, 3), state=1)
    c = apply_dpo(
        RuleSpan.from_mapping(left, right, {n: n for n in left.node_ids}),
        left,
        {n: n for n in left.node_ids},
    ).certificate
    extra = c.result.to_networkx()
    extra.add_edge(1, 3, weight=1)
    assert (
        not replace(c, result=LewisLabelledGraph.from_networkx(extra, SCHEMA))
        .replay()
        .valid
    )
    c = certificate()
    wrong = c.result.to_networkx()
    wrong.edges[1, 2]["weight"] = 2
    assert (
        not replace(c, result=LewisLabelledGraph.from_networkx(wrong, SCHEMA))
        .replay()
        .valid
    )


def test_composite_retains_transient_edge_absence_on_larger_and_relabeled_hosts():
    first = RuleSpan.from_mapping(graph(), graph((10, 20), edge=True), {1: 10, 2: 20})
    second = RuleSpan.from_mapping(
        graph(("a", "b"), edge=True), graph(("x", "y")), {"a": "x", "b": "y"}
    )
    composed = compose_rules(
        first,
        second,
        RuleOverlap.from_mapping(first.right, second.left, {10: "a", 20: "b"}),
    )
    host = graph(("p", "q", "spectator"))
    application = apply_dpo(composed.rule, host, {1: "p", 2: "q"})
    assert application.result == host
    blocked = host.to_networkx()
    blocked.add_edge("p", "q", weight=1)
    with pytest.raises(DPOError, match="retained sequential"):
        apply_dpo(
            composed.rule,
            LewisLabelledGraph.from_networkx(blocked, SCHEMA),
            {1: "p", 2: "q"},
        )


def test_composite_transports_fresh_carriers_to_second_step():
    first = RuleSpan.from_mapping(graph((1,)), graph((10, 20), edge=True), {1: 10})
    second = RuleSpan.from_mapping(
        graph(("a", "b"), edge=True), graph(("x",)), {"a": "x"}
    )
    composed = compose_rules(
        first,
        second,
        RuleOverlap.from_mapping(first.right, second.left, {10: "a", 20: "b"}),
    )
    host = graph(("p", "spectator"))
    application = apply_dpo(composed.rule, host, {1: "p"})
    assert application.result == host
    assert application.certificate.replay().valid


def test_deleted_then_recreated_edge_has_new_identity_in_the_composite():
    first = RuleSpan.from_mapping(graph(edge=True), graph((10, 20)), {1: 10, 2: 20})
    second = RuleSpan.from_mapping(
        graph(("a", "b")), graph(("x", "y"), edge=True), {"a": "x", "b": "y"}
    )
    composed = compose_rules(
        first,
        second,
        RuleOverlap.from_mapping(first.right, second.left, {10: "a", 20: "b"}),
    )
    assert composed.rule.left.edge_keys == composed.rule.right.edge_keys
    assert not composed.rule.interface.edge_keys
    assert composed.certificate.outer_application.certificate.deleted_edges == (
        frozenset((1, 2)),
    )
    assert composed.certificate.outer_application.certificate.added_edges == (
        frozenset((1, 2)),
    )
    right_refs = [
        refs
        for side, _, refs in composed.certificate.provenance.edge_sources
        if side == "right"
    ]
    assert all(
        not any(ref.rule == 1 and ref.side == "left" for ref in refs)
        for refs in right_refs
    )
    assert composed.certificate.replay().valid


def test_annotations_survive_repeated_roundtrips_without_prefix_growth():
    value = graph().to_networkx()
    value.nodes[1]["custom"] = "provenance"
    frozen = LewisLabelledGraph.from_networkx(value, SCHEMA)
    again = LewisLabelledGraph.from_networkx(frozen.to_networkx(), SCHEMA)
    assert again == frozen
