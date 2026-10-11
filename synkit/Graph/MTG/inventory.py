"""Exact all-schedule initial stock for a fixed additive event poset.

This uses constrained maximum closure, an established minimum-cut reduction.
Species counts alone do not prove a graph match or a material-copy assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Hashable, Mapping

import networkx as nx


@dataclass(frozen=True)
class StockWitness:
    """An enabling ideal attaining a positive initial-stock requirement."""

    completed_events: frozenset[Hashable]
    next_event: Hashable
    demand: int
    prefix_delta: int

    @property
    def required_stock(self) -> int:
        return self.demand - self.prefix_delta


@dataclass(frozen=True)
class StockBound:
    """Least nonnegative initial stock sufficient for every linear extension.

    The witness proves the necessity of a positive bound by one reachable
    enabling ideal. It is not a certificate for physical reaction admission.
    """

    initial_stock: int
    witness: StockWitness | None = None


def _profile(
    graph: nx.DiGraph, values: Mapping[Hashable, int], name: str
) -> dict[Hashable, int]:
    if set(values) != set(graph):
        raise ValueError(f"{name} must provide exactly one count per event.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        for value in values.values()
    ):
        raise ValueError(
            f"{name} counts must be nonnegative integers, excluding booleans."
        )
    return {event: int(values[event]) for event in graph}


def _enabling_ideal(
    graph: nx.DiGraph, weights: dict[Hashable, int], event: Hashable
) -> frozenset[Hashable]:
    source, sink = object(), object()
    network = nx.DiGraph()
    network.add_nodes_from((source, sink, *graph))
    bound = 1 + sum(abs(weight) for weight in weights.values())
    for node, weight in weights.items():
        if weight > 0:
            network.add_edge(source, node, capacity=weight)
        elif weight < 0:
            network.add_edge(node, sink, capacity=-weight)
    for first, second in graph.edges:
        network.add_edge(second, first, capacity=bound)
    for node in nx.ancestors(graph, event):
        network.add_edge(source, node, capacity=bound)
    for node in {event} | nx.descendants(graph, event):
        network.add_edge(node, sink, capacity=bound)
    _, (selected, _) = nx.minimum_cut(network, source, sink)
    return frozenset(set(selected) - {source, sink})


def minimum_initial_stock(
    event_order: nx.DiGraph,
    demands: Mapping[Hashable, int],
    products: Mapping[Hashable, int],
) -> StockBound:
    """Compute exact robust initial stock for one additive resource.

    At each event its demand must be available before its products are added.
    The returned stock suffices for every linear extension of the supplied
    poset. It need not be minimal for a particular schedule. No schedules or
    ideals are enumerated; at most one minimum-cut solve is needed per event.
    Conditions, graph matches and environment routing require separate checks.

    :param event_order: Finite simple directed acyclic event graph.
    :param demands: Nonnegative integer input count for every event.
    :param products: Nonnegative integer output count for every event.
    :return: Least robust stock and an attaining enabling ideal when positive.
    :raises ValueError: If the graph or resource profiles violate the contract.
    """
    if not isinstance(event_order, nx.DiGraph) or event_order.is_multigraph():
        raise ValueError("The event order must be a simple directed graph.")
    graph = event_order.copy()
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The event order must be acyclic; unfold network cycles.")
    inputs = _profile(graph, demands, "Demand")
    outputs = _profile(graph, products, "Product")
    deltas = {event: outputs[event] - inputs[event] for event in graph}
    weights = {event: -deltas[event] for event in graph}
    best = StockBound(0)
    for event in graph:
        ideal = _enabling_ideal(graph, weights, event)
        witness = StockWitness(
            ideal, event, inputs[event], sum(deltas[node] for node in ideal)
        )
        if witness.required_stock > best.initial_stock:
            best = StockBound(witness.required_stock, witness)
    return best


__all__ = ["StockBound", "StockWitness", "minimum_initial_stock"]
