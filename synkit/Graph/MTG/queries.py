"""Structural coexistence queries on explicitly bound occurrence histories."""

from __future__ import annotations

from typing import Iterable

from .history import (
    ConfigurationCut,
    HistoryError,
    HistoryIssue,
    HistoryIssueCode,
    OccurrenceMTG,
)


def cooccurrence_configuration(
    history: OccurrenceMTG, material_ids: Iterable[str]
) -> ConfigurationCut | None:
    """Find the least structural ideal containing every requested material copy.

    The producer downset is necessary for presence. If it contains a requested
    copy's consumer, coexistence is impossible; otherwise that downset is the
    least witnessing ideal. No schedules or ideals are enumerated. This tests
    structural presence, not arbitrary guards, rates or physical feasibility.

    :param history: Retained occurrence history, revalidated before querying.
    :param material_ids: Known material occurrence IDs; duplicates are ignored.
    :return: Least witnessing cut, or ``None`` if no structural cut exists.
    :raises HistoryError: If the retained history or a material ID is invalid.
    """
    replay = history.replay()
    if not replay.valid:
        raise HistoryError(*replay.issues)
    requested = frozenset(material_ids)
    unknown = requested - set(history.material_by_id)
    if unknown:
        raise HistoryError(
            HistoryIssue(
                HistoryIssueCode.FLOW_LOOKUP,
                "A coexistence query must name known material occurrence IDs.",
                {"unknown": tuple(sorted(unknown, key=repr))},
            )
        )
    producers = {
        binding.material_id: event.occurrence_id
        for event in history.events
        for binding in event.outputs
    }
    consumers = {
        binding.material_id: event.occurrence_id
        for event in history.events
        for binding in event.inputs
    }
    predecessors: dict[str, set[str]] = {}
    for first, second in history.causal_pairs:
        predecessors.setdefault(second, set()).add(first)
    pending = [producers[copy] for copy in requested if copy in producers]
    completed: set[str] = set()
    while pending:
        event = pending.pop()
        if event not in completed:
            completed.add(event)
            pending.extend(predecessors.get(event, ()))
    if any(consumers.get(copy) in completed for copy in requested):
        return None
    return history.configuration_cut(completed)
