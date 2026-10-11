"""Binding-preserving bridges and additional declared ordering constraints."""

from dataclasses import replace

import networkx as nx
import pytest

from synkit.Graph.MTG import (
    GroundedLocalEvent,
    LocalHistoryError,
    OccurrenceProcess,
    OccurrenceMTG,
    RuleOccurrence,
    build_local_coordinate_history,
    build_observation_projection,
    realize_coordinate_history,
    compose_histories,
)
from synkit.Rule.Apply import RuleSpan

from .test_history_gluing import _port, _step
from .test_local_coordinates import local_event, world
from .test_occurrence_history import (
    SCHEMA,
    _chain,
    _independent,
    _llg,
    _material,
    _temporary_bond_process,
    _whole_binding,
)


@pytest.mark.parametrize("factory", [_chain, _independent, _temporary_bond_process])
def test_every_original_configuration_matches_its_anchored_realization(factory):
    source = OccurrenceMTG.from_process(factory())
    realized = realize_coordinate_history(source)
    assert realized.replay()
    graph = source.to_process().causal_graph
    assert set(realized.history.declared_precedence) == set(graph.edges)
    assert (
        nx.transitive_closure_dag(realized.history.causal_graph).edges
        == nx.transitive_closure_dag(graph).edges
    )
    for order in nx.all_topological_sorts(graph):
        applications = realized.history.execute(order)
        for index in range(len(order) + 1):
            ideal = frozenset(order[:index])
            original = source.configuration_cut(ideal).world
            coordinate = realized.history.configuration(ideal).world
            assert original == coordinate
            if index:
                assert applications[index - 1].result == original


def test_read_only_material_reuse_order_is_not_lost_in_local_accesses():
    fragment = _step(0, 0)
    source = compose_histories(fragment, fragment, [_port()]).history
    realized = realize_coordinate_history(source)
    assert not any(record.writes for record in realized.history.records)
    # Retained declared precedence still carries the source material binding.
    assert nx.has_path(realized.history.causal_graph, "left:E", "right:E")
    with pytest.raises(LocalHistoryError):
        realized.history.configuration({"right:E"})


def test_additional_order_can_refine_independent_read_events_without_a_write():
    initial = world(1)
    events = tuple(local_event(initial, name, (0,), {}) for name in ("a", "b"))
    unrestricted = build_local_coordinate_history(initial, events)
    assert unrestricted.causal_pairs == ()
    history = build_local_coordinate_history(initial, events, precedence=(("a", "b"),))
    assert history.replay()
    assert history.declared_precedence == (("a", "b"),)
    with pytest.raises(LocalHistoryError):
        history.configuration({"b"})
    with pytest.raises(LocalHistoryError):
        history.execute(("b", "a"))
    assert len(history.execute(("a", "b"))) == 2


def test_hidden_read_only_binding_constraint_survives_observer_order():
    initial = world(2)
    first = local_event(initial, "a", (0,), {0: 1})
    hidden = local_event(initial, "hidden", (1,), {})
    last = local_event(initial, "c", (1,), {1: 1})
    history = build_local_coordinate_history(
        initial, (first, hidden, last), precedence=(("a", "hidden"), ("hidden", "c"))
    )
    observer = build_observation_projection(
        history, [("node", 0, "x"), ("node", 1, "x")]
    )
    assert observer.event_ids == ("a", "c")
    assert observer.causal_pairs == (("a", "c"),)
    assert observer.lift_extension(("a", "c")) == ("a", "hidden", "c")


@pytest.mark.parametrize(
    "precedence",
    [
        (("a", "unknown"),),
        (("b", "a"),),
        (("a", "a"),),
        (("a",),),
        (None,),
        (([], "b"),),
    ],
)
def test_invalid_declared_order_is_refused(precedence):
    initial = world(1)
    events = tuple(local_event(initial, name, (0,), {}) for name in ("a", "b"))
    with pytest.raises(LocalHistoryError):
        build_local_coordinate_history(initial, events, precedence=precedence)


def test_tampered_declared_order_is_refused_on_replay():
    initial = world(1)
    events = tuple(local_event(initial, name, (0,), {}) for name in ("a", "b"))
    history = build_local_coordinate_history(initial, events, precedence=(("a", "b"),))
    corrupted = replace(history, declared_precedence=())
    assert not corrupted.replay()
    with pytest.raises(LocalHistoryError):
        corrupted.configuration(())


def test_nonlocal_guarded_material_history_is_not_claimed_local():
    source = _step(0, 1, guard=lambda host, match: True)
    with pytest.raises(LocalHistoryError):
        realize_coordinate_history(source)


def test_tampered_realization_is_refused():
    realized = realize_coordinate_history(OccurrenceMTG.from_process(_chain()))
    corrupted = replace(realized, history=replace(realized.history, causal_pairs=()))
    assert not corrupted.replay()


def test_declared_constraints_accept_one_pass_iterables_and_normalize_duplicates():
    initial = world(1)
    rule = local_event(initial, "a", (0,), {}).rule
    events = [GroundedLocalEvent(name, rule, {0: 0}) for name in ("a", "b")]
    history = build_local_coordinate_history(
        initial, events, precedence=iter((("a", "b"), ("a", "b")))
    )
    assert history.declared_precedence == (("a", "b"),)
    assert history.replay()


def test_atom_allocation_history_requires_another_realization_profile():
    empty = _llg({})
    material = _material("new", _llg({"atom": 0}))
    rule = RuleSpan.from_mapping(empty, material.value, {})
    event = RuleOccurrence("birth", rule, (), (_whole_binding(material, rule.right),))
    source = OccurrenceMTG.from_process(
        OccurrenceProcess("birth", (material,), (event,))
    )
    with pytest.raises(LocalHistoryError):
        realize_coordinate_history(source)


def test_empty_typed_history_has_an_empty_coordinate_realization():
    realized = realize_coordinate_history(OccurrenceMTG.empty(SCHEMA))
    assert realized.replay()
    assert realized.history.records == ()
    assert realized.history.declared_precedence == ()
    assert (
        realized.history.configuration(()).world
        == realized.source.configuration_cut(()).world
    )
