import networkx as nx
import pytest

from synkit.Graph.Matcher.multi_turbo_iso import MultiTurboISO
from synkit.Graph.Matcher.turbo_iso import TurboISO


def test_empty_query_has_one_empty_mapping():
    matcher = TurboISO(nx.path_graph(2), node_label=[])

    assert matcher.search(nx.Graph()) == [{}]
    assert matcher.search(nx.Graph(), prune=True) is True


def test_attribute_signatures_do_not_have_delimiter_collisions():
    host = nx.Graph()
    host.add_node(0, first="x|y", second="z")
    query = nx.Graph()
    query.add_node(10, first="x", second="y|z")

    matcher = TurboISO(host, node_label=["first", "second"])

    assert matcher.search(query) == []


def test_attribute_signatures_preserve_value_types():
    host = nx.Graph()
    host.add_node(0, label=1)
    query = nx.Graph()
    query.add_node(10, label="1")

    matcher = TurboISO(host, node_label="label")

    assert matcher.search(query) == []


def test_directed_incoming_edges_are_checked():
    host = nx.DiGraph()
    host.add_edge(0, 1)
    host.nodes[0]["element"] = "C"
    host.nodes[1]["element"] = "O"
    query = nx.DiGraph()
    query.add_edge(10, 11)
    query.nodes[10]["element"] = "O"
    query.nodes[11]["element"] = "C"

    matcher = TurboISO(host, node_label="element")

    assert matcher.search(query) == []


def test_query_self_loop_must_exist_in_host():
    host = nx.path_graph(3)
    query = nx.Graph()
    query.add_edge(10, 10)

    matcher = TurboISO(host, node_label=[])

    assert matcher.search(query) == []


def test_multigraphs_are_rejected_explicitly():
    with pytest.raises(ValueError, match="multigraph"):
        TurboISO(nx.MultiGraph(), node_label=[])


def test_multi_turbo_empty_query_matches_every_host():
    matcher = MultiTurboISO([nx.path_graph(2), nx.cycle_graph(3)], node_label=[])

    assert matcher.search_one(nx.Graph()) == {0: [{}], 1: [{}]}
    assert matcher.search_one(nx.Graph(), prune=True) == {0: True, 1: True}
