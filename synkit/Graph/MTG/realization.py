"""Realize fixed-carrier material-port histories as coordinate histories."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .history import OccurrenceMTG
from .local import (
    GroundedLocalEvent,
    LocalCoordinateHistory,
    LocalHistoryError,
    build_local_coordinate_history,
)


def _same_raw_world(first, second):
    return (
        first.schema == second.schema
        and first.node_ids == second.node_ids
        and first.edge_keys == second.edge_keys
        and all(
            first.node_labels(node, semantic=True)
            == second.node_labels(node, semantic=True)
            for node in first.node_ids
        )
        and all(
            first.edge_labels(pair, semantic=True)
            == second.edge_labels(pair, semantic=True)
            for pair in first.edge_keys
        )
    )


@dataclass(frozen=True)
class CoordinateRealization:
    """Original copy bindings and their exact fixed-carrier source realization."""

    source: OccurrenceMTG
    history: LocalCoordinateHistory

    def replay(self) -> bool:
        """Revalidate bindings, transported matches, order and realized prefix worlds."""
        try:
            return realize_coordinate_history(self.source) == self
        except (ValueError, TypeError, KeyError):
            return False


def realize_coordinate_history(source: OccurrenceMTG) -> CoordinateRealization:
    """Transport a supported material-port history into its persistent atom universe.

    Every primitive rule must meet the unguarded fixed-carrier locality profile.
    Declared port precedence is retained, and every extracted access dependency
    must already be implied by that order. No copy binding, physical independence
    or alternative mechanism is inferred. All canonical prefix worlds are checked
    against the original occurrence construction.

    :param source: A replayable material-port occurrence history.
    :return: Original history and its anchored coordinate-poset realization.
    :raises LocalHistoryError: If profile, order or prefix-world agreement fails.
    """
    if not source.replay().valid:
        raise LocalHistoryError("The material-port source history is not replayable.")
    graph = source.to_process().causal_graph
    order = tuple(nx.lexicographical_topological_sort(graph))
    initial = source.configuration_cut(()).world
    events = []
    for event_id in order:
        event = source.event_by_id[event_id]
        match = {
            rule_node: source.node_lineage_by_carrier[
                binding.material_id, material_node
            ]
            for binding in event.inputs
            for rule_node, material_node in binding.mapping
        }
        events.append(GroundedLocalEvent(event_id, event.rule, match))
    local = build_local_coordinate_history(initial, events, precedence=graph.edges)
    if any(not nx.has_path(graph, a, b) for a, b in local.causal_pairs):
        raise LocalHistoryError(
            "The port order does not cover every grounded coordinate dependency."
        )
    completed = set()
    if not _same_raw_world(initial, local.initial):
        raise LocalHistoryError("The initial anchored worlds disagree.")
    for record in local.records:
        completed.add(record.event.event_id)
        original = source.configuration_cut(completed).world
        if not _same_raw_world(record.application.result, original):
            raise LocalHistoryError(
                "A realized source prefix has a different raw world."
            )
    return CoordinateRealization(source, local)
