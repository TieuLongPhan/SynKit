"""Cache hits must preserve exact colored-graph classification."""

import networkx as nx

from synkit.Chem.Mapper import spectrum


def colored(graph):
    nx.set_node_attributes(graph, "atom", "color")
    nx.set_edge_attributes(graph, "bond", "color")
    return graph


def code(cache, graph):
    value, reason = cache.code(graph, timeout_seconds=1, max_search_nodes=100000)
    assert reason is None
    return value


def test_isomorphic_relabeling_hits_but_histogram_collision_does_not():
    cache = spectrum._ExactCodeCache()
    cycle = colored(nx.cycle_graph(6))
    triangles = colored(nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3)))
    assert cache._key(cycle) == cache._key(triangles)
    first = code(cache, cycle)
    assert code(cache, nx.relabel_nodes(cycle, {i: 20 - i for i in range(6)})) == first
    assert cache.hits == 1
    assert code(cache, triangles) != first
    assert cache.hits == 1


def test_color_changes_cannot_reuse_code_even_when_bucket_keys_collide(monkeypatch):
    monkeypatch.setattr(spectrum._ExactCodeCache, "_key", staticmethod(lambda graph: 0))
    cache = spectrum._ExactCodeCache()
    original = colored(nx.path_graph(4))
    first = code(cache, original)
    changed = original.copy()
    changed.nodes[0]["color"] = "different"
    assert code(cache, changed) != first
    changed = original.copy()
    changed.edges[0, 1]["color"] = "different"
    assert code(cache, changed) != first
    assert cache.hits == 0


def test_lookup_budget_exit_falls_back_to_exact_canonicalization(monkeypatch):
    cache = spectrum._ExactCodeCache()
    graph = colored(nx.path_graph(4))
    first = code(cache, graph)

    def exhausted(*args):
        raise spectrum._CacheLookupBudget

    monkeypatch.setattr(spectrum, "_quick_color_isomorphism", lambda *args: False)
    monkeypatch.setattr(
        spectrum._BudgetedColorMatcher, "syntactic_feasibility", exhausted
    )
    assert code(cache, nx.relabel_nodes(graph, {i: i + 10 for i in graph})) == first
    assert cache.hits == 0 and cache.budget_exits == 1


def test_cache_is_bounded_and_retains_an_unmutated_graph_snapshot():
    cache = spectrum._ExactCodeCache(max_entries=2)
    graph = colored(nx.path_graph(4))
    first = code(cache, graph)
    graph.nodes[0]["color"] = "changed"
    assert code(cache, colored(nx.path_graph(4))) == first
    code(cache, colored(nx.path_graph(5)))
    code(cache, colored(nx.path_graph(6)))
    assert len(cache.entries) == 2


def test_connected_regular_graph_collision_requires_actual_isomorphism():
    cache = spectrum._ExactCodeCache()
    prism = colored(nx.circular_ladder_graph(3))
    bipartite = colored(nx.complete_bipartite_graph(3, 3))
    assert cache._key(prism) == cache._key(bipartite)
    assert code(cache, prism) != code(cache, bipartite)
    assert cache.hits == 0


def test_component_matching_is_invariant_to_component_order_and_node_ids():
    cache = spectrum._ExactCodeCache()
    graph = colored(nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(4)))
    first = code(cache, graph)
    permuted = nx.relabel_nodes(graph, {i: 20 - i for i in graph})
    assert code(cache, permuted) == first
    assert cache.hits == 1


def test_quick_transport_checks_edges_even_when_refinement_collides(monkeypatch):
    monkeypatch.setattr(
        spectrum, "_cache_refinement",
        lambda graph, rounds=3: {node: 0 for node in graph},
    )
    cycle = colored(nx.cycle_graph(6))
    triangles = colored(nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3)))
    assert not spectrum._quick_color_isomorphism(cycle, triangles)
    relabeled = nx.relabel_nodes(cycle, {i: 40 - i for i in cycle})
    assert spectrum._quick_color_isomorphism(cycle, relabeled)
    relabeled.nodes[40]["color"] = "different"
    assert not spectrum._quick_color_isomorphism(cycle, relabeled)
