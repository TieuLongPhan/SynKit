"""Coexistence answers against independently enumerated configuration worlds."""

from dataclasses import replace
import itertools

import networkx as nx
import pytest

from synkit.Graph.MTG import (
    HistoryError,
    HistoryIssueCode,
    NodeLineage,
    OccurrenceMTG,
    OccurrenceProcess,
    cooccurrence_configuration,
)

from .test_occurrence_history import _chain, _independent, _temporary_bond_process


@pytest.mark.parametrize("factory", [_chain, _independent, _temporary_bond_process])
def test_every_copy_subset_agrees_with_exhaustive_schedule_cuts(factory):
    process = factory()
    history = OccurrenceMTG.from_process(process)
    cuts = {}
    for order in process.linear_extensions():
        active = set(history.initial_material_ids)
        cuts[frozenset()] = frozenset(active)
        for index, event_id in enumerate(order, start=1):
            event = process.event_by_id[event_id]
            active.difference_update(binding.material_id for binding in event.inputs)
            active.update(binding.material_id for binding in event.outputs)
            cuts[frozenset(order[:index])] = frozenset(active)
    materials = tuple(process.material_by_id)
    for count in range(len(materials) + 1):
        for subset in itertools.combinations(materials, count):
            candidates = [
                ideal for ideal, active in cuts.items() if set(subset) <= active
            ]
            result = cooccurrence_configuration(history, subset)
            assert (result is not None) == bool(candidates)
            if result is not None:
                assert set(subset) <= result.active_material_ids
                assert all(result.executed_events <= ideal for ideal in candidates)


def test_independent_intermediates_coexist_without_schedule_enumeration(monkeypatch):
    history = OccurrenceMTG.from_process(_independent())

    def forbidden(*args, **kwargs):
        raise AssertionError("A coexistence query must not enumerate schedules.")

    monkeypatch.setattr(OccurrenceProcess, "linear_extensions", forbidden)
    monkeypatch.setattr(OccurrenceMTG, "full_history", forbidden)
    monkeypatch.setattr(nx, "all_topological_sorts", forbidden)
    cut = cooccurrence_configuration(history, iter(("a1", "b0", "a1")))
    assert cut.executed_events == frozenset({"E1"})
    assert cut.active_material_ids == frozenset({"a1", "b0"})
    assert cooccurrence_configuration(history, ("a0", "a1")) is None
    assert cooccurrence_configuration(history, ()).executed_events == frozenset()


def test_unknown_copies_are_errors_rather_than_negative_answers():
    history = OccurrenceMTG.from_process(_chain())
    with pytest.raises(HistoryError) as caught:
        cooccurrence_configuration(history, ("not-a-copy",))
    assert caught.value.issues[0].code is HistoryIssueCode.FLOW_LOOKUP


def test_impossible_queries_still_refuse_corrupt_retained_lineages():
    history = OccurrenceMTG.from_process(_chain())
    corrupt = replace(
        history,
        node_lineages=tuple(
            NodeLineage(f"false-split:{index}", (member,))
            for index, member in enumerate(history.node_lineages[0].members)
        ),
    )
    with pytest.raises(HistoryError):
        cooccurrence_configuration(corrupt, ("m0", "m1"))
