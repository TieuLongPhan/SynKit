"""Observation projections of certified fixed-carrier coordinate histories.

These diagrams answer questions on declared raw coordinates and their versions.
They retain the complete source history for validation and lifting; deleting a
hidden event does not authorize executing the remaining source rules alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx

from .local import (
    Coordinate,
    LocalCoordinateHistory,
    LocalHistoryError,
    _coordinate_state,
)


@dataclass(frozen=True)
class ObservationCut:
    """Observed values and last writers at a reduced configuration."""

    executed_events: frozenset[str]
    values: tuple[tuple[Coordinate, Any], ...]
    versions: tuple[tuple[Coordinate, str | None], ...]


def _ideal(graph, completed):
    completed = frozenset(completed)
    if not completed <= set(graph) or any(
        second in completed and first not in completed for first, second in graph.edges
    ):
        raise LocalHistoryError("The observation cut must be a known event ideal.")
    return completed


def _extension(graph, order):
    order = tuple(order)
    positions = {event: index for index, event in enumerate(order)}
    if (
        len(positions) != len(order)
        or set(positions) != set(graph)
        or any(positions[a] >= positions[b] for a, b in graph.edges)
    ):
        raise LocalHistoryError(
            "An observation extension must contain all retained events once."
        )
    return order


@dataclass(frozen=True)
class ObservationProjection:
    """A coordinate-labelled induced order with its certified full-history lift.

    Retained events write at least one selected coordinate, including equal-value
    recreations. Causal pairs are the full inherited reachability relation, so
    paths through hidden events are preserved. This finite trace-specific object
    does not infer rates, alternative mechanisms or a standalone source program.
    """

    history: LocalCoordinateHistory
    coordinates: frozenset[Coordinate]
    event_ids: tuple[str, ...]
    causal_pairs: tuple[tuple[str, str], ...]

    @property
    def causal_graph(self) -> nx.DiGraph:
        """Return a fresh graph of the inherited order on retained writers."""
        graph = nx.DiGraph()
        graph.add_nodes_from(self.event_ids)
        graph.add_edges_from(self.causal_pairs)
        return graph

    def replay(self) -> bool:
        """Revalidate the full source construction and the observer's exact order."""
        try:
            return build_observation_projection(self.history, self.coordinates) == self
        except (ValueError, TypeError, KeyError):
            return False

    def _require_valid(self):
        if not self.replay():
            raise LocalHistoryError("The observation projection is not replayable.")

    def project(self, completed: Iterable[str]) -> frozenset[str]:
        """Map a full-history ideal to its retained writer events.

        :param completed: A known ideal of the complete coordinate history.
        :return: The corresponding ideal of the observation order.
        :raises LocalHistoryError: If replay or full-ideal validation fails.
        """
        self._require_valid()
        return _ideal(self.history.causal_graph, completed) & frozenset(self.event_ids)

    def lift(self, completed: Iterable[str]) -> frozenset[str]:
        """Return the least complete-history ideal above an observation ideal.

        :param completed: A known ideal of the retained writer order.
        :return: These events and all their full-history causal ancestors.
        :raises LocalHistoryError: If replay or reduced-ideal validation fails.
        """
        self._require_valid()
        completed = _ideal(self.causal_graph, completed)
        graph = self.history.causal_graph
        return completed | frozenset(
            ancestor for event in completed for ancestor in nx.ancestors(graph, event)
        )

    def configuration(self, completed: Iterable[str]) -> ObservationCut:
        """Project an ideal by observed writers without replaying a lifted schedule.

        Full source validation runs first and has a separate cost. Missing world
        coordinates are not imputed. Bond values retain their definedness tags.

        :param completed: A known ideal of the observation order.
        :return: Exactly the requested raw values and last-writer versions.
        :raises LocalHistoryError: If replay or reduced-ideal validation fails.
        """
        self._require_valid()
        completed = _ideal(self.causal_graph, completed)
        initial = _coordinate_state(self.history.initial, self.history.bond_pairs)
        values = {key: initial[key] for key in self.coordinates}
        versions = {key: None for key in self.coordinates}
        for record in self.history.records:
            if record.event.event_id in completed:
                for key, value in record.assignments:
                    if key in self.coordinates:
                        values[key] = value
                        versions[key] = record.event.event_id
        return ObservationCut(
            completed,
            tuple(sorted(values.items(), key=repr)),
            tuple(sorted(versions.items(), key=repr)),
        )

    def lift_extension(self, order: Iterable[str]) -> tuple[str, ...]:
        """Lift a reduced full extension by inserting required hidden ancestors.

        Remaining hidden events are appended after the visible extension. The
        returned word is a full-history extension and can be passed to its
        ``execute`` method for actual primitive DPO execution.

        :param order: All retained events once in a reduced linear extension.
        :return: A complete source extension projecting to exactly that word.
        :raises LocalHistoryError: If replay or extension validation fails.
        """
        self._require_valid()
        order = _extension(self.causal_graph, order)
        graph = self.history.causal_graph
        original = tuple(record.event.event_id for record in self.history.records)
        completed = set()
        lifted = []
        for event in order:
            missing = nx.ancestors(graph, event) - completed
            for predecessor in original:
                if predecessor in missing:
                    lifted.append(predecessor)
                    completed.add(predecessor)
            lifted.append(event)
            completed.add(event)
        lifted.extend(event for event in original if event not in completed)
        return _extension(graph, lifted)


def build_observation_projection(
    history: LocalCoordinateHistory, coordinates: Iterable[Coordinate]
) -> ObservationProjection:
    """Retain observed writers and their complete inherited causal order.

    Predicates must declare every authoritative coordinate they depend on.
    Derived charge, connectivity and isotope roles require their corresponding
    raw dependencies; selecting one field does not authorize a whole-world
    claim. Fixed atom IDs remain the observation's origin anchors.

    :param history: A replayable fixed-carrier coordinate history.
    :param coordinates: Known raw semantic coordinates to observe, possibly empty.
    :return: A surjective ideal projection with least lifts and extension lifting.
    :raises LocalHistoryError: If replay fails or a coordinate is unknown.
    """
    if not history.replay():
        raise LocalHistoryError("The full source coordinate history is not replayable.")
    try:
        coordinates = frozenset(coordinates)
    except TypeError as error:
        raise LocalHistoryError(
            "Observation coordinates must be hashable coordinate keys."
        ) from error
    universe = _coordinate_state(history.initial, history.bond_pairs)
    if not coordinates <= universe.keys():
        raise LocalHistoryError(
            "An observation coordinate is outside the stored raw universe."
        )
    event_ids = tuple(
        record.event.event_id
        for record in history.records
        if record.writes & coordinates
    )
    graph = history.causal_graph
    causal_pairs = tuple(
        (first, second)
        for first in event_ids
        for second in event_ids
        if first != second and nx.has_path(graph, first, second)
    )
    return ObservationProjection(history, coordinates, event_ids, causal_pairs)
