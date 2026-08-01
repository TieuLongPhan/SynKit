"""Optional exact series/parallel decomposition of an occurrence poset."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx

from .process import OccurrenceProcess, ProcessError, ProcessIssue, ProcessIssueCode


@dataclass(frozen=True)
class SeriesParallelDecomposition:
    """A detected decomposition; absence never changes process semantics."""

    kind: str
    events: frozenset[str]
    children: tuple["SeriesParallelDecomposition", ...] = ()


def detect_series_parallel(
    process: OccurrenceProcess,
    *,
    max_partition_states: int = 100_000,
) -> SeriesParallelDecomposition | None:
    """Return an exact recursive decomposition or ``None`` for a general poset."""
    if max_partition_states <= 0:
        raise ValueError("max_partition_states must be positive.")
    events = frozenset(process.event_by_id)
    relation = set(process.causal_pairs)
    explored = 0

    def comparable(left: str, right: str) -> bool:
        return (left, right) in relation or (right, left) in relation

    def visit(current: frozenset[str]) -> SeriesParallelDecomposition | None:
        nonlocal explored
        explored += 1
        if explored > max_partition_states:
            raise ProcessError(
                ProcessIssue(
                    ProcessIssueCode.SERIES_PARALLEL_LIMIT,
                    "Series/parallel detection exceeded its explicit partition bound.",
                    {"limit": max_partition_states},
                )
            )
        if len(current) <= 1:
            return SeriesParallelDecomposition("event", current)

        comparability = nx.Graph()
        comparability.add_nodes_from(current)
        comparability.add_edges_from(
            (left, right)
            for left in current
            for right in current
            if repr(left) < repr(right) and comparable(left, right)
        )
        components = tuple(
            frozenset(component) for component in nx.connected_components(comparability)
        )
        if len(components) > 1:
            children = tuple(visit(component) for component in components)
            if all(child is not None for child in children):
                return SeriesParallelDecomposition(
                    "parallel",
                    current,
                    tuple(child for child in children if child is not None),
                )

        ordered = tuple(sorted(current, key=repr))
        anchor = ordered[0]
        for size in range(1, len(ordered)):
            for selected in itertools.combinations(ordered[1:], size - 1):
                explored += 1
                if explored > max_partition_states:
                    raise ProcessError(
                        ProcessIssue(
                            ProcessIssueCode.SERIES_PARALLEL_LIMIT,
                            "Series/parallel detection exceeded its partition bound.",
                            {"limit": max_partition_states},
                        )
                    )
                first = frozenset((anchor, *selected))
                second = current - first
                orientations = (
                    all((left, right) in relation for left in first for right in second),
                    all((right, left) in relation for left in first for right in second),
                )
                if not any(orientations):
                    continue
                lower, upper = (first, second) if orientations[0] else (second, first)
                lower_tree, upper_tree = visit(lower), visit(upper)
                if lower_tree is not None and upper_tree is not None:
                    return SeriesParallelDecomposition(
                        "series", current, (lower_tree, upper_tree)
                    )
        return None

    return visit(events)


__all__ = ["SeriesParallelDecomposition", "detect_series_parallel"]
