"""Robust additive stock, exact integer cuts, and independent schedule checks."""

import itertools
import random

import networkx as nx
import pytest

from synkit.Graph.MTG import minimum_initial_stock


def _schedule_requirement(order, demands, products):
    prefix = 0
    needed = 0
    for event in order:
        needed = max(needed, demands[event] - prefix)
        prefix += products[event] - demands[event]
    return needed


def test_sequence_deficit_counts_each_consumption_before_its_products():
    graph = nx.DiGraph((("a", "b"), ("b", "c")))
    result = minimum_initial_stock(graph, {e: 1 for e in graph}, {e: 0 for e in graph})
    assert result.initial_stock == 3
    assert result.witness.next_event == "c"
    assert result.witness.completed_events == frozenset({"a", "b"})
    assert result.witness.required_stock == result.initial_stock


def test_all_schedule_bound_differs_from_a_favourable_schedule():
    graph = nx.DiGraph()
    graph.add_nodes_from(("produce", "consume"))
    demands = {"produce": 0, "consume": 1}
    products = {"produce": 1, "consume": 0}
    result = minimum_initial_stock(graph, demands, products)
    assert _schedule_requirement(("produce", "consume"), demands, products) == 0
    assert result.initial_stock == 1
    assert result.witness.required_stock == 1
    assert (
        result.witness.demand
        - sum(
            products[event] - demands[event]
            for event in result.witness.completed_events
        )
        == result.initial_stock
    )


def test_precedence_restricts_enabling_ideals():
    graph = nx.DiGraph((("produce", "consume"),))
    result = minimum_initial_stock(
        graph, {"produce": 0, "consume": 1}, {"produce": 1, "consume": 0}
    )
    assert result.initial_stock == 0
    assert result.witness is None


def test_arbitrary_event_keys_cannot_collide_with_internal_cut_terminals():
    source_key, sink_key = ("source",), ("sink",)
    graph = nx.DiGraph(((source_key, sink_key),))
    result = minimum_initial_stock(
        graph, {source_key: 1, sink_key: 1}, {source_key: 0, sink_key: 0}
    )
    assert result.initial_stock == 2
    assert result.witness.next_event == sink_key


def test_large_integer_resources_are_exact():
    count = 10**30 + 7
    graph = nx.DiGraph((("a", "b"),))
    result = minimum_initial_stock(graph, {"a": count, "b": count}, {"a": 0, "b": 0})
    assert result.initial_stock == 2 * count
    assert isinstance(result.initial_stock, int)


def test_no_schedule_or_ideal_enumeration_and_input_graph_is_preserved(monkeypatch):
    graph = nx.DiGraph((("a", "b"), ("a", "d"), ("c", "d")))
    before = graph.copy()

    def refuse(*args, **kwargs):
        raise AssertionError("The closure algorithm must not enumerate schedules.")

    monkeypatch.setattr(nx, "all_topological_sorts", refuse)
    result = minimum_initial_stock(graph, {e: 1 for e in graph}, {e: 0 for e in graph})
    assert result.initial_stock == 4
    assert nx.utils.graphs_equal(graph, before)


def test_empty_poset_needs_no_stock():
    result = minimum_initial_stock(nx.DiGraph(), {}, {})
    assert result.initial_stock == 0 and result.witness is None


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
@pytest.mark.parametrize("side", ["demand", "product"])
def test_invalid_resource_counts_are_refused(invalid, side):
    graph = nx.DiGraph()
    graph.add_node("e")
    demands, products = {"e": 1}, {"e": 0}
    (demands if side == "demand" else products)["e"] = invalid
    with pytest.raises(ValueError, match="nonnegative integers"):
        minimum_initial_stock(graph, demands, products)


@pytest.mark.parametrize("profile", [{}, {"e": 1, "extra": 1}])
def test_missing_or_extra_profile_events_are_refused(profile):
    graph = nx.DiGraph()
    graph.add_node("e")
    with pytest.raises(ValueError, match="exactly one count"):
        minimum_initial_stock(graph, profile, {"e": 0})


@pytest.mark.parametrize(
    "graph",
    [nx.Graph(((0, 1),)), nx.MultiDiGraph(((0, 1),)), nx.DiGraph(((0, 1), (1, 0)))],
)
def test_invalid_orders_are_refused(graph):
    with pytest.raises(ValueError):
        minimum_initial_stock(graph, {e: 0 for e in graph}, {e: 0 for e in graph})


def test_seeded_small_posets_match_every_schedule_and_witness_prefix():
    randomizer = random.Random(20261003)
    for _ in range(32):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(5))
        graph.add_edges_from(
            pair
            for pair in itertools.combinations(range(5), 2)
            if randomizer.random() < 0.25
        )
        demands = {event: randomizer.randrange(4) for event in graph}
        products = {event: randomizer.randrange(4) for event in graph}
        schedules = list(nx.all_topological_sorts(graph))
        result = minimum_initial_stock(graph, demands, products)
        assert result.initial_stock == max(
            _schedule_requirement(order, demands, products) for order in schedules
        )
        if result.witness is not None:
            witness = result.witness
            assert witness.required_stock == result.initial_stock
            assert nx.ancestors(graph, witness.next_event) <= witness.completed_events
            assert witness.next_event not in witness.completed_events
            assert all(
                nx.ancestors(graph, node) <= witness.completed_events
                for node in witness.completed_events
            )
            assert any(
                set(order[: len(witness.completed_events)]) == witness.completed_events
                and order[len(witness.completed_events)] == witness.next_event
                for order in schedules
            )
