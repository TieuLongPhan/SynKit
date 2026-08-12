import networkx as nx

from synkit.Graph.Matcher.wl_sel import WLSel


def test_combined_wl_attributes_do_not_mutate_input_graphs():
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(
            element="C",
            charge=0,
            __wl_node_temp__=f"node-{node}",
        )
    graph.edges[0, 1].update(
        order=1,
        phase="sigma",
        __wl_edge_temp__="original-edge-value",
    )
    expected_nodes = {node: dict(attrs) for node, attrs in graph.nodes(data=True)}
    expected_edges = {
        (left, right): dict(attrs) for left, right, attrs in graph.edges(data=True)
    }

    WLSel(
        [graph],
        [graph],
        node_attrs=["element", "charge"],
        edge_attrs=["order", "phase"],
    ).build_signatures()

    assert {
        node: dict(attrs) for node, attrs in graph.nodes(data=True)
    } == expected_nodes
    assert {
        (left, right): dict(attrs) for left, right, attrs in graph.edges(data=True)
    } == expected_edges
