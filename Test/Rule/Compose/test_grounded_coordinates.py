"""Grounded composition retains natural coordinate and frame provenance."""

from dataclasses import replace

import networkx as nx
import pytest

from synkit.Graph.Morphism import (
    ISOTOPE_ELECTRON_LLG_SCHEMA,
    LabelSchema,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Rule.Apply import RuleEmbedding, RuleSpan, apply_dpo
from synkit.Rule.Compose import (
    CompositionError,
    RuleOverlap,
    compose_rules,
    search_compositions,
)

SCHEMA = LabelSchema(node_identity=("kind",), node_state=("x",), edge_state=("b",))


def graph(state=0, bond=None):
    value = nx.Graph()
    value.add_nodes_from(
        [(0, {"kind": "X", "x": state}), (1, {"kind": "X", "x": state})]
    )
    if bond is not None:
        value.add_edge(0, 1, b=bond)
    return LewisLabelledGraph.from_networkx(value, SCHEMA)


def rule(left, right):
    return RuleSpan.from_mapping(left, right, {0: 0, 1: 1})


def compose(first, second):
    return compose_rules(
        first, second, RuleOverlap.from_mapping(first.right, second.left, {0: 0, 1: 1})
    )


def test_returned_node_coordinate_is_recreated_in_outer_interface():
    composed = compose(rule(graph(0), graph(1)), rule(graph(1), graph(0)))
    assert all(
        composed.rule.interface.node_labels(n, semantic=True) == {"kind": "X"}
        for n in composed.rule.interface.node_ids
    )
    assert composed.certificate.replay().valid
    assert all(
        "x"
        not in composed.certificate.outer_application.certificate.context.node_labels(n)
        for n in composed.rule.left.node_ids
    )


def test_returned_bond_coordinate_is_recreated_without_deleting_the_bond():
    composed = compose(
        rule(graph(bond=0), graph(bond=1)), rule(graph(bond=1), graph(bond=0))
    )
    assert len(composed.rule.interface.edge_keys) == 1
    assert (
        composed.rule.interface.edge_labels(
            next(iter(composed.rule.interface.edge_keys)), semantic=True
        )
        == {}
    )
    assert not composed.certificate.outer_application.certificate.deleted_edges
    assert composed.certificate.replay().valid


def test_intentionally_erased_equal_coordinate_stays_erased_through_identity():
    first = rule(graph(0), graph(0))
    pattern = first.interface.to_networkx()
    for n in pattern:
        pattern.nodes[n].pop("x")
    interface = LewisLabelledGraph.from_networkx(pattern, SCHEMA, partial=True)
    first = replace(
        first,
        interface=interface,
        left_arm=RuleEmbedding(interface, first.left, first.left_arm.f),
        right_arm=RuleEmbedding(interface, first.right, first.right_arm.f),
    )
    composed = compose(first, rule(graph(0), graph(0)))
    assert all(
        "x" not in composed.rule.interface.node_labels(n)
        for n in composed.rule.interface.node_ids
    )
    assert composed.certificate.replay().valid


def test_second_rule_frame_bond_between_shared_atoms_is_retained():
    first = rule(graph(0), graph(1))
    second = rule(graph(1, bond=0), graph(1, bond=1))
    composed = compose(first, second)
    assert composed.rule.left.edge_labels(frozenset((0, 1)), semantic=True) == {"b": 0}
    assert composed.rule.right.edge_labels(frozenset((0, 1)), semantic=True) == {"b": 1}
    assert not composed.certificate.overlap.interface.edge_keys
    assert composed.certificate.replay().valid
    host = graph(0, bond=0)
    after_first = apply_dpo(first, host, {0: 0, 1: 1})
    after_second = apply_dpo(second, after_first.result, {0: 0, 1: 1})
    outer = apply_dpo(composed.rule, host, {0: 0, 1: 1})
    assert outer.result == after_second.result


def test_second_rule_cannot_read_a_bond_deleted_by_first():
    first = rule(graph(bond=0), graph())
    second = rule(graph(bond=0), graph(bond=1))
    with pytest.raises(CompositionError):
        compose(first, second)


def test_search_finds_the_second_rule_frame_bond_on_shared_atoms():
    first = rule(graph(0), graph(1))
    second = rule(graph(1, bond=0), graph(1, bond=1))
    result = search_compositions(first, second)
    assert result.raw_overlap_count == result.accepted_count == 7
    full = [
        witness
        for group in result.classes
        for witness in group.witnesses
        if len(witness.overlap.interface.node_ids) == 2
    ]
    assert len(full) == 2
    host = graph(0, bond=0)
    after_first = apply_dpo(first, host, {0: 0, 1: 1})
    sequential = apply_dpo(second, after_first.result, {0: 0, 1: 1})
    for witness in full:
        assert witness.composition.certificate.replay().valid
        assert not witness.overlap.interface.edge_keys
        outer = apply_dpo(witness.composition.rule, host, {0: 0, 1: 1})
        assert outer.result == sequential.result


def test_search_excludes_conflicting_shared_bond_values():
    first = rule(graph(0, bond=0), graph(1, bond=0))
    second = rule(graph(1, bond=1), graph(1, bond=2))
    result = search_compositions(first, second)
    assert result.raw_overlap_count == 5
    assert all(
        len(witness.overlap.interface.node_ids) < 2
        for group in result.classes
        for witness in group.witnesses
    )


def test_search_keeps_deleted_bond_candidates_as_composition_refusals():
    first = rule(graph(bond=0), graph())
    second = rule(graph(bond=0), graph(bond=1))
    result = search_compositions(first, second)
    assert result.raw_overlap_count == 7
    assert all(
        len(witness.overlap.interface.node_ids) < 2
        for group in result.classes
        for witness in group.witnesses
    )
    assert (
        len(
            [
                refusal
                for refusal in result.rejected
                if len(refusal.overlap.interface.node_ids) == 2
            ]
        )
        == 2
    )


def test_isotope_frame_bond_recomputes_minimal_host_charges():
    def oxygen_graph(pairs, bond=False):
        value = nx.Graph()
        for node, lone_pairs in enumerate(pairs):
            value.add_node(
                node,
                element="O",
                isotope=18 if node == 0 else 16,
                aromatic=False,
                hcount=0,
                radical=0,
                lone_pairs=lone_pairs,
                valence_electrons=6,
            )
        if bond:
            value.add_edge(0, 1, sigma_order=1, pi_order=0)
        return LewisLabelledGraph.from_networkx(
            derive_electron_labeled_graph(value), ISOTOPE_ELECTRON_LLG_SCHEMA
        )

    first = RuleSpan.from_mapping(
        oxygen_graph((3, 2)), oxygen_graph((2, 3)), {0: 0, 1: 1}, boundary="closed"
    )
    host = oxygen_graph((3, 2), bond=True)
    application = apply_dpo(first, host, {0: 0, 1: 1})
    second = RuleSpan.from_mapping(
        application.result, application.result, {0: 0, 1: 1}, boundary="closed"
    )
    composed = compose(first, second)
    assert composed.certificate.minimal_host == host
    assert composed.certificate.replay().valid
    assert composed.rule.resource_delta.isotope_delta == ()
    assert [composed.rule.left.node_labels(n)["charge"] for n in (0, 1)] == [-1, 1]
    assert [composed.rule.right.node_labels(n)["isotope"] for n in (0, 1)] == [18, 16]
