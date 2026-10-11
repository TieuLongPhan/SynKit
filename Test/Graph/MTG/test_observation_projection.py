"""Independent source execution checks for coordinate observation projections."""

from dataclasses import replace
from itertools import combinations, permutations

import networkx as nx
import pytest

from synkit.Graph.Morphism import LewisLabelledGraph
from synkit.Graph.MTG import (
    GroundedLocalEvent,
    LocalHistoryError,
    build_local_coordinate_history,
    build_observation_projection,
)
from synkit.Rule.Apply import DPOError, RuleEmbedding, RuleSpan, apply_dpo

from .test_local_coordinates import SCHEMA, local_event, n_poset_history, world


def subsets(items):
    items = tuple(items)
    return [
        frozenset(subset)
        for size in range(len(items) + 1)
        for subset in combinations(items, size)
    ]


def source_cuts(history):
    events = {record.event.event_id: record.event for record in history.records}
    cuts = {}
    orders = []
    for order in permutations(events):
        current = history.initial
        prefix = [(frozenset(), current)]
        try:
            for index, event_id in enumerate(order, 1):
                event = events[event_id]
                current = apply_dpo(event.rule, current, dict(event.match)).result
                prefix.append((frozenset(order[:index]), current))
        except DPOError:
            continue
        orders.append(order)
        for ideal, current in prefix:
            if ideal in cuts:
                assert cuts[ideal] == current
            cuts[ideal] = current
    return cuts, orders


def chain_history():
    initial = current = world(3)
    events = []
    for name, nodes, updates in (
        ("a", (0,), {0: 1}),
        ("hidden", (0, 1), {1: 1}),
        ("c", (1, 2), {2: 1}),
    ):
        event = local_event(current, name, nodes, updates)
        events.append(event)
        current = apply_dpo(event.rule, current, dict(event.match)).result
    return build_local_coordinate_history(initial, events)


def test_every_n_poset_observation_has_exact_full_source_fibers_and_words():
    history = n_poset_history()
    cuts, orders = source_cuts(history)
    for coordinates in subsets(
        ("node", node, "x") for node in history.initial.node_ids
    ):
        projection = build_observation_projection(history, coordinates)
        reduced = {}
        for ideal, actual in cuts.items():
            projected = projection.project(ideal)
            observed = projection.configuration(projected)
            assert dict(observed.values) == {
                key: actual.node_labels(key[1], semantic=True)[key[2]]
                for key in coordinates
            }
            versions = dict(history.configuration(ideal).versions)
            assert dict(observed.versions) == {
                key: versions[key] for key in coordinates
            }
            reduced[projected] = observed
            lift = projection.lift(projected)
            assert lift <= ideal
            assert projection.project(lift) == projected
            assert projection.configuration(projection.project(lift)) == observed
        for candidate in subsets(projection.event_ids):
            if candidate in reduced:
                assert projection.configuration(candidate) == reduced[candidate]
            else:
                with pytest.raises(LocalHistoryError):
                    projection.configuration(candidate)
        reduced_orders = {
            tuple(order) for order in nx.all_topological_sorts(projection.causal_graph)
        }
        assert {
            tuple(e for e in order if e in projection.event_ids) for order in orders
        } == reduced_orders
        for order in reduced_orders:
            lift = projection.lift_extension(order)
            assert lift in orders
            assert tuple(e for e in lift if e in projection.event_ids) == order
            assert history.execute(lift)[-1].result == cuts[frozenset(lift)]


def test_hidden_order_paths_are_retained_and_lifts_supply_source_preconditions():
    history = chain_history()
    assert set(history.causal_pairs) == {("a", "hidden"), ("hidden", "c")}
    projection = build_observation_projection(
        history, (("node", 0, "x"), ("node", 2, "x"))
    )
    assert projection.event_ids == ("a", "c")
    assert projection.causal_pairs == (("a", "c"),)
    assert projection.lift({"a", "c"}) == {"a", "hidden", "c"}
    assert projection.lift_extension(("a", "c")) == ("a", "hidden", "c")
    # A projected diagram is not an executable source subsequence.
    a, _, c = (record.event for record in history.records)
    after_a = apply_dpo(a.rule, history.initial, dict(a.match)).result
    with pytest.raises(DPOError):
        apply_dpo(c.rule, after_a, dict(c.match))
    assert (
        history.execute(projection.lift_extension(("a", "c")))[-1]
        .certificate.replay()
        .valid
    )
    with pytest.raises(LocalHistoryError):
        projection.lift_extension(("c", "a"))


def test_every_hidden_move_stutters_and_each_visible_move_lifts_from_every_fiber():
    history = n_poset_history()
    cuts, _ = source_cuts(history)
    for coordinates in subsets(
        ("node", node, "x") for node in history.initial.node_ids
    ):
        projection = build_observation_projection(history, coordinates)
        graph = history.causal_graph
        for ideal in cuts:
            reduced = projection.project(ideal)
            for event in graph:
                target = ideal | {event}
                if event in ideal or target not in cuts:
                    continue
                if event not in projection.event_ids:
                    assert projection.configuration(
                        reduced
                    ) == projection.configuration(projection.project(target))
            for event in projection.event_ids:
                target = reduced | {event}
                if event in reduced:
                    continue
                try:
                    projection.configuration(target)
                except LocalHistoryError:
                    continue
                # Each pending ancestor is hidden and can precede the visible event.
                pending = nx.ancestors(graph, event) - ideal
                assert not pending & set(projection.event_ids)
                prefix = set(ideal)
                for record in history.records:
                    if record.event.event_id in pending:
                        prefix.add(record.event.event_id)
                        assert frozenset(prefix) in cuts
                        assert projection.project(prefix) == reduced
                prefix.add(event)
                assert frozenset(prefix) in cuts
                assert projection.project(prefix) == target


def test_equal_value_recreation_retains_a_visible_version_change():
    initial = world(1)
    rule = RuleSpan.from_mapping(initial, initial, {0: 0})
    graph = rule.interface.to_networkx()
    graph.nodes[0].pop("x")
    interface = LewisLabelledGraph.from_networkx(graph, SCHEMA, partial=True)
    rule = replace(
        rule,
        interface=interface,
        left_arm=RuleEmbedding(interface, initial, {0: 0}),
        right_arm=RuleEmbedding(interface, initial, {0: 0}),
    )
    history = build_local_coordinate_history(
        initial, [GroundedLocalEvent("recreate", rule, {0: 0})]
    )
    projection = build_observation_projection(history, [("node", 0, "x")])
    before, after = projection.configuration(()), projection.configuration({"recreate"})
    assert before.values == after.values
    assert dict(before.versions)[("node", 0, "x")] is None
    assert dict(after.versions)[("node", 0, "x")] == "recreate"
    assert projection.event_ids == ("recreate",)


def test_empty_vocabulary_has_one_observation_and_a_complete_source_lift():
    history = n_poset_history()
    projection = build_observation_projection(history, ())
    assert projection.event_ids == ()
    assert projection.lift(()) == frozenset()
    assert projection.configuration(()).values == ()
    assert projection.configuration(()).versions == ()
    lifted = projection.lift_extension(())
    assert set(lifted) == set(history.causal_graph)
    assert history.execute(lifted)[-1].certificate.replay().valid


def test_defined_none_bond_coordinate_is_distinct_from_absence():
    graph = world(2).to_networkx()
    graph.edges[0, 1]["b"] = None
    initial = LewisLabelledGraph.from_networkx(graph, SCHEMA)
    absent_graph = graph.copy()
    absent_graph.remove_edge(0, 1)
    absent = LewisLabelledGraph.from_networkx(absent_graph, SCHEMA)
    rule = RuleSpan.from_mapping(initial, absent, {0: 0, 1: 1})
    history = build_local_coordinate_history(
        initial, [GroundedLocalEvent("delete", rule, {0: 0, 1: 1})]
    )
    key = ("bond", frozenset((0, 1)), "b")
    projection = build_observation_projection(history, [key])
    assert dict(projection.configuration(()).values)[key] == (True, None)
    assert dict(projection.configuration({"delete"}).values)[key] == (False, None)


@pytest.mark.parametrize(
    "coordinate",
    [("node", 999, "x"), ("node", 0, "charge"), ("presence", frozenset((0, 3)), None)],
)
def test_unknown_or_derived_coordinates_are_refused(coordinate):
    with pytest.raises(LocalHistoryError):
        build_observation_projection(n_poset_history(), [coordinate])


@pytest.mark.parametrize("completed", [{"c"}, {"unknown"}])
def test_unknown_and_nonideal_observer_cuts_are_refused(completed):
    projection = build_observation_projection(
        chain_history(), [("node", 0, "x"), ("node", 2, "x")]
    )
    with pytest.raises(LocalHistoryError):
        projection.configuration(completed)
    with pytest.raises(LocalHistoryError):
        projection.lift(completed)


def test_unknown_or_nonideal_full_source_projection_is_refused():
    projection = build_observation_projection(chain_history(), [("node", 2, "x")])
    for completed in ({"c"}, {"unknown"}):
        with pytest.raises(LocalHistoryError):
            projection.project(completed)


@pytest.mark.parametrize(
    "mutation",
    [{"causal_pairs": ()}, {"event_ids": ("c",)}, {"coordinates": frozenset()}],
)
def test_tampered_observation_projection_is_refused(mutation):
    projection = build_observation_projection(
        chain_history(), [("node", 0, "x"), ("node", 2, "x")]
    )
    corrupted = replace(projection, **mutation)
    assert not corrupted.replay()
    with pytest.raises(LocalHistoryError):
        corrupted.configuration(())


def test_tampered_full_history_cannot_authorize_an_observer():
    history = replace(chain_history(), causal_pairs=())
    with pytest.raises(LocalHistoryError):
        build_observation_projection(history, [("node", 2, "x")])


@pytest.mark.parametrize("order", [("a", "a"), ("a",), ("a", "c", "unknown")])
def test_incomplete_duplicate_or_unknown_extensions_are_refused(order):
    projection = build_observation_projection(
        chain_history(), [("node", 0, "x"), ("node", 2, "x")]
    )
    with pytest.raises(LocalHistoryError):
        projection.lift_extension(order)
