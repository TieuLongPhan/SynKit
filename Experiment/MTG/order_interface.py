"""Finite order interfaces for an already validated, bound occurrence model.

This experimental order operation does not construct chemical rules, validate
material ports, transport atoms or replace source execution certificates.
"""

from dataclasses import dataclass
from typing import Iterable

import networkx as nx


@dataclass(frozen=True)
class OrderInterface:
    """Observed events and connection events with their inherited strict order."""

    visible: frozenset[str]
    ports: frozenset[str]
    pairs: tuple[tuple[str, str], ...]

    @property
    def graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(sorted(self.visible | self.ports))
        graph.add_edges_from(self.pairs)
        return graph

    @property
    def observation_order(self) -> nx.DiGraph:
        """Hide connection events only after inheriting complete reachability."""
        return summarize_order(self.graph, self.visible, ()).graph


def _event_set(events: Iterable[str]) -> frozenset[str]:
    result = frozenset(events)
    if any(not isinstance(event, str) or not event for event in result):
        raise ValueError("Events must have nonempty string identifiers.")
    return result


def summarize_order(graph, visible, ports) -> OrderInterface:
    """Retain visible writers and every endpoint allowed for external edges."""
    if not isinstance(graph, nx.DiGraph) or not nx.is_directed_acyclic_graph(graph):
        raise ValueError("An order interface requires an acyclic directed graph.")
    nodes = _event_set(graph)
    visible, ports = _event_set(visible), _event_set(ports)
    if not visible | ports <= nodes:
        raise ValueError("A visible or connection event is unknown.")
    retained = visible | ports
    pairs = tuple(
        sorted(
            (first, second)
            for first in retained
            for second in nx.descendants(graph, first) & retained
        )
    )
    return OrderInterface(visible, ports, pairs)


def _require_summary(summary):
    if (
        not isinstance(summary, OrderInterface)
        or summarize_order(summary.graph, summary.visible, summary.ports) != summary
    ):
        raise ValueError("An interface must store exactly its inherited strict order.")


def compose_order_interfaces(left, right, connections) -> OrderInterface:
    """Compose disjoint event namespaces through declared connection endpoints.

    All connection events remain available for later composition. The returned
    observation order hides them while retaining paths through them. Cycles are
    refused even when every event participating in the cycle is unobserved.
    """
    _require_summary(left)
    _require_summary(right)
    left_nodes, right_nodes = set(left.graph), set(right.graph)
    if left_nodes & right_nodes:
        raise ValueError("Fragment event namespaces must be disjoint.")
    edges = set()
    for pair in connections:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each connection must have two event identifiers.")
        first, second = pair
        if not (
            (first in left.ports and second in right.ports)
            or (first in right.ports and second in left.ports)
        ):
            raise ValueError(
                "Connection endpoints must be ports in opposite fragments."
            )
        edges.add((first, second))
    graph = nx.compose(left.graph, right.graph)
    graph.add_edges_from(edges)
    return summarize_order(
        graph, left.visible | right.visible, left.ports | right.ports
    )
