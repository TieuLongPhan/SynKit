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


def test_bounded_scoring_matches_full_prefix_and_counts_all_pairs():
    forward = [nx.path_graph(size) for size in (2, 3, 4)]
    backward = [nx.path_graph(size) for size in (2, 3, 4)]
    for graph in (*forward, *backward):
        nx.set_node_attributes(graph, "C", "element")
        nx.set_edge_attributes(graph, 1.0, "order")

    full = WLSel(forward, backward, min_score=0.0).score_pairs()
    bounded = WLSel(forward, backward, min_score=0.0).score_pairs(top_k=3)

    assert bounded.pair_indices == full.pair_indices[:3]
    assert bounded.pair_candidate_count == 9
    assert len(bounded.pair_scores) == 3
