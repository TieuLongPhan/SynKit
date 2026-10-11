"""Cached graph indexes preserve fresh labels and authoritative tuple edits."""

from dataclasses import replace
import pickle

import networkx as nx
import pytest

from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph
from synkit.Rule.Apply import RuleSpan, apply_dpo

SCHEMA = LabelSchema(node_identity=("kind",), node_state=("x",), edge_state=("b",))


def graph():
    value = nx.Graph()
    value.add_node(0, kind="X", x=0)
    value.add_node(1, kind="X", x=1)
    value.add_edge(0, 1, b=2)
    return LewisLabelledGraph.from_networkx(value, SCHEMA)


def test_cached_owner_sets_and_labels_equal_uncached_tuple_oracle():
    value = graph()
    for _ in range(3):
        assert value.node_ids == frozenset(node for node, _ in value.nodes)
        assert value.edge_keys == frozenset(edge for edge, _ in value.edges)
        assert value.node_labels(1) == dict(dict(value.nodes)[1])
        assert value.edge_labels(frozenset((0, 1))) == dict(
            dict(value.edges)[frozenset((0, 1))]
        )
    assert value.node_ids is value.node_ids
    assert value.edge_keys is value.edge_keys


def test_returned_label_dictionaries_remain_fresh_and_isolated():
    value = graph()
    nodes, edges = value.node_labels(0), value.edge_labels(frozenset((0, 1)))
    nodes["x"] = 999
    edges["b"] = 999
    assert value.node_labels(0)["x"] == 0
    assert value.edge_labels(frozenset((0, 1)))["b"] == 2
    assert value.node_labels(0) is not value.node_labels(0)
    assert value.edge_labels(frozenset((0, 1))) is not value.edge_labels(
        frozenset((0, 1))
    )


def test_replaced_graph_does_not_inherit_indexes_from_its_previous_values():
    value = graph()
    assert value.node_labels(0)["x"] == 0
    nodes = tuple(
        (
            node,
            tuple((key, 5 if node == 0 and key == "x" else raw) for key, raw in labels),
        )
        for node, labels in value.nodes
    )
    updated = replace(value, nodes=nodes)
    assert updated.node_labels(0)["x"] == 5
    assert value.node_labels(0)["x"] == 0
    assert updated != value


def test_bypassed_frozen_node_tuple_change_invalidates_topology_and_labels():
    value = graph()
    assert value.node_ids == {0, 1}
    assert value.node_labels(0)["x"] == 0
    object.__setattr__(value, "nodes", ((2, (("kind", "X"), ("x", 3))),))
    assert value.node_ids == {2}
    assert value.node_labels(2)["x"] == 3
    with pytest.raises(KeyError):
        value.node_labels(0)


def test_bypassed_frozen_edge_tuple_change_invalidates_topology_and_labels():
    value = graph()
    pair = frozenset((0, 1))
    assert value.edge_keys == {pair}
    assert value.edge_labels(pair)["b"] == 2
    object.__setattr__(value, "edges", ((pair, (("b", 4),)),))
    assert value.edge_labels(pair)["b"] == 4
    object.__setattr__(value, "edges", ())
    assert value.edge_keys == frozenset()
    with pytest.raises(KeyError):
        value.edge_labels(pair)


def test_indexed_graph_pickle_roundtrip_retains_declared_data():
    value = graph()
    value.node_labels(0)
    value.edge_labels(frozenset((0, 1)))
    restored = pickle.loads(pickle.dumps(value))
    assert restored == value
    assert restored.node_ids == value.node_ids
    assert restored.edge_keys == value.edge_keys
    assert restored.node_labels(0) == value.node_labels(0)
    assert restored.edge_labels(frozenset((0, 1))) == value.edge_labels(
        frozenset((0, 1))
    )


def test_warmed_indexes_do_not_hide_tampered_certificate_coordinates():
    initial = graph()
    rule = RuleSpan.from_mapping(initial, initial, {0: 0, 1: 1})
    certificate = apply_dpo(rule, initial, {0: 0, 1: 1}).certificate
    assert certificate.replay().valid
    certificate.context.node_labels(0)
    nodes = tuple(
        (
            node,
            tuple(
                (key, 999 if node == 0 and key == "x" else raw) for key, raw in labels
            ),
        )
        for node, labels in certificate.context.nodes
    )
    object.__setattr__(certificate.context, "nodes", nodes)
    assert not certificate.replay().valid
