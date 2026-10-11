"""Whole-world source oracles for shared-context coordinate histories."""

from dataclasses import replace
from itertools import permutations

import networkx as nx
import pytest

from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph
from synkit.Graph.MTG import (
    GroundedLocalEvent,
    LocalHistoryError,
    build_local_coordinate_history,
)
from synkit.Rule.Apply import DPOError, RuleEmbedding, RuleSpan, apply_dpo

SCHEMA = LabelSchema(node_identity=("kind",), node_state=("x",), edge_state=("b",))


def world(size=4):
    graph = nx.path_graph(size)
    nx.set_node_attributes(graph, "X", "kind")
    nx.set_node_attributes(graph, 0, "x")
    nx.set_edge_attributes(graph, 0, "b")
    return LewisLabelledGraph.from_networkx(graph, SCHEMA)


def local_event(host, event_id, nodes, updates):
    left = host.to_networkx().subgraph(nodes).copy()
    right = left.copy()
    for node, value in updates.items():
        right.nodes[node]["x"] = value
    rule = RuleSpan.from_mapping(
        LewisLabelledGraph.from_networkx(left, SCHEMA),
        LewisLabelledGraph.from_networkx(right, SCHEMA),
        {node: node for node in nodes},
    )
    return GroundedLocalEvent(event_id, rule, {node: node for node in nodes})


def n_poset_history():
    initial = current = world()
    events = []
    for event_id, nodes, updates in (
        ("a", (0,), {0: 1}),
        ("b", (1,), {1: 1}),
        ("c", (0, 2), {2: 1}),
        ("d", (0, 1, 3), {3: 1}),
    ):
        event = local_event(current, event_id, nodes, updates)
        events.append(event)
        current = apply_dpo(event.rule, current, dict(event.match)).result
    return build_local_coordinate_history(initial, events)


def test_connected_n_poset_cuts_equal_all_admitted_primitive_permutations():
    history = n_poset_history()
    assert nx.is_connected(history.initial.to_networkx())
    assert set(history.causal_pairs) == {("a", "c"), ("a", "d"), ("b", "d")}
    records = {record.event.event_id: record for record in history.records}
    oracle = {}
    admitted = set()
    for order in permutations(records):
        current = history.initial
        cuts = [(frozenset(), current)]
        try:
            for index, event_id in enumerate(order, 1):
                event = records[event_id].event
                application = apply_dpo(event.rule, current, dict(event.match))
                assert application.certificate.replay().valid
                current = application.result
                cuts.append((frozenset(order[:index]), current))
        except DPOError:
            continue
        admitted.add(order)
        for ideal, current in cuts:
            if ideal in oracle:
                assert oracle[ideal] == current
            oracle[ideal] = current
            assert history.configuration(ideal).world == current
        assert history.execute(order)[-1].result == current
    assert len(admitted) == 5
    assert len(oracle) == 8
    assert admitted == {
        tuple(order) for order in nx.all_topological_sorts(history.causal_graph)
    }
    assert records["c"].reads & records["d"].reads
    assert not records["c"].writes & records["d"].reads
    assert not records["d"].writes & records["c"].reads


def test_shared_atom_reads_do_not_consume_the_whole_component():
    initial = world(1)
    events = [
        local_event(initial, event_id, (0,), {}) for event_id in ("read1", "read2")
    ]
    history = build_local_coordinate_history(initial, events)
    assert history.causal_pairs == ()
    assert history.records[0].reads == history.records[1].reads
    assert history.configuration({"read2"}).world == initial
    assert all(
        writer is None
        for _, writer in history.configuration({"read1", "read2"}).versions
    )


def test_reader_precedes_a_later_overwriting_event():
    initial = world(2)
    read = local_event(initial, "read", (0, 1), {1: 1})
    after = apply_dpo(read.rule, initial, dict(read.match)).result
    overwrite = local_event(after, "write", (0,), {0: 1})
    history = build_local_coordinate_history(initial, (read, overwrite))
    assert history.causal_pairs == (("read", "write"),)
    with pytest.raises(LocalHistoryError):
        history.configuration({"write"})
    with pytest.raises(LocalHistoryError):
        history.execute(("write", "read"))
    after_write = apply_dpo(overwrite.rule, initial, dict(overwrite.match)).result
    with pytest.raises(DPOError):
        apply_dpo(read.rule, after_write, dict(read.match))


def test_creation_absence_and_shared_bond_read_order_are_retained():
    initial_nx = world(2).to_networkx()
    initial_nx.remove_edge(0, 1)
    initial = LewisLabelledGraph.from_networkx(initial_nx, SCHEMA)
    bonded = world(2)
    rules = (
        RuleSpan.from_mapping(initial, bonded, {0: 0, 1: 1}),
        RuleSpan.from_mapping(bonded, bonded, {0: 0, 1: 1}),
        RuleSpan.from_mapping(bonded, initial, {0: 0, 1: 1}),
    )
    history = build_local_coordinate_history(
        initial,
        [
            GroundedLocalEvent(name, rule, {0: 0, 1: 1})
            for name, rule in zip(("add", "read", "delete"), rules)
        ],
    )
    pair = frozenset((0, 1))
    assert ("presence", pair, None) in history.records[0].reads
    assert nx.has_path(history.causal_graph, "add", "read")
    assert nx.has_path(history.causal_graph, "read", "delete")
    assert history.configuration({"add", "read", "delete"}).world == initial
    with pytest.raises(DPOError):
        apply_dpo(rules[0], bonded, {0: 0, 1: 1})


def test_erased_equal_coordinate_remains_a_write_and_changes_its_version():
    initial = world(1)
    identity = RuleSpan.from_mapping(initial, initial, {0: 0})
    graph = identity.interface.to_networkx()
    graph.nodes[0].pop("x")
    interface = LewisLabelledGraph.from_networkx(graph, SCHEMA, partial=True)
    recreate = replace(
        identity,
        interface=interface,
        left_arm=RuleEmbedding(interface, initial, {0: 0}),
        right_arm=RuleEmbedding(interface, initial, {0: 0}),
    )
    history = build_local_coordinate_history(
        initial, [GroundedLocalEvent("recreate", recreate, {0: 0})]
    )
    cut = history.configuration({"recreate"})
    assert cut.world == initial
    assert ("node", 0, "x") in history.records[0].writes
    assert dict(cut.versions)[("node", 0, "x")] == "recreate"


def test_bond_owner_recreation_is_recorded_when_endpoint_values_agree():
    initial = world(2)
    identity = RuleSpan.from_mapping(initial, initial, {0: 0, 1: 1})
    graph = identity.interface.to_networkx()
    graph.remove_edge(0, 1)
    interface = LewisLabelledGraph.from_networkx(graph, SCHEMA, partial=True)
    recreate = replace(
        identity,
        interface=interface,
        left_arm=RuleEmbedding(interface, initial, {0: 0, 1: 1}),
        right_arm=RuleEmbedding(interface, initial, {0: 0, 1: 1}),
    )
    history = build_local_coordinate_history(
        initial, [GroundedLocalEvent("recreate", recreate, {0: 0, 1: 1})]
    )
    pair = frozenset((0, 1))
    cut = history.configuration({"recreate"})
    assert cut.world == initial
    assert dict(cut.versions)[("presence", pair, None)] == "recreate"


def test_renamed_rule_endpoints_preserve_supplied_world_atom_anchors():
    initial = world(1)
    left = initial.relabel({0: "left"})
    right_nx = left.to_networkx()
    right_nx.nodes["left"]["x"] = 1
    right = LewisLabelledGraph.from_networkx(right_nx, SCHEMA).relabel(
        {"left": "right"}
    )
    rule = RuleSpan.from_mapping(left, right, {"left": "right"})
    history = build_local_coordinate_history(
        initial, [GroundedLocalEvent("change", rule, {"left": 0})]
    )
    cut = history.configuration({"change"})
    assert cut.world.node_ids == {0}
    assert cut.world.node_labels(0)["x"] == 1
    assert history.execute(("change",))[-1].result == cut.world


def test_missing_accesses_or_dependencies_in_retained_history_fail_replay():
    history = n_poset_history()
    record = replace(history.records[0], writes=frozenset())
    corrupt = replace(history, records=(record,) + history.records[1:])
    assert not corrupt.replay()
    assert not replace(history, causal_pairs=()).replay()
    with pytest.raises(LocalHistoryError):
        corrupt.configuration(())


@pytest.mark.parametrize("completed", [{"missing"}, {"c"}, {"d", "a"}])
def test_unknown_or_nonideal_event_sets_refuse(completed):
    with pytest.raises(LocalHistoryError):
        n_poset_history().configuration(completed)


def test_nonlocal_guard_is_refused_instead_of_receiving_a_locality_claim():
    class Guard:
        def validate(self, host, match):
            return None

    initial = world(1)
    event = local_event(initial, "guarded", (0,), {0: 1})
    event = replace(event, rule=replace(event.rule, condition=Guard()))
    with pytest.raises(LocalHistoryError, match="guards"):
        build_local_coordinate_history(initial, [event])


def test_atom_allocation_requires_a_separate_profile():
    initial = world(1)
    graph = initial.to_networkx()
    graph.add_node(1, kind="X", x=0)
    right = LewisLabelledGraph.from_networkx(graph, SCHEMA)
    rule = RuleSpan.from_mapping(initial, right, {0: 0})
    with pytest.raises(LocalHistoryError, match="every rule atom"):
        build_local_coordinate_history(
            initial, [GroundedLocalEvent("fresh", rule, {0: 0})]
        )


def test_empty_trace_has_the_initial_world_and_no_versions_written():
    initial = world()
    history = build_local_coordinate_history(initial, [])
    assert history.replay()
    assert history.execute(()) == ()
    assert history.configuration(()).world == initial


def test_empty_trace_still_requires_a_total_initial_world():
    partial = LewisLabelledGraph.from_networkx(nx.Graph(), SCHEMA, partial=True)
    with pytest.raises(LocalHistoryError, match="total"):
        build_local_coordinate_history(partial, [])


def test_defined_none_bond_value_is_not_confused_with_bond_absence():
    graph = world(2).to_networkx()
    graph.edges[0, 1]["b"] = None
    initial = LewisLabelledGraph.from_networkx(graph, SCHEMA)
    history = build_local_coordinate_history(initial, [])
    assert history.configuration(()).world.edge_labels(frozenset((0, 1)))["b"] is None


def test_corrupt_initial_electron_charge_is_refused_even_without_events():
    from synkit.Graph.Morphism import (
        ELECTRON_LLG_SCHEMA,
        LLGError,
        derive_electron_labeled_graph,
    )

    graph = nx.Graph()
    graph.add_node(
        0,
        element="C",
        aromatic=False,
        hcount=4,
        radical=0,
        lone_pairs=0,
        valence_electrons=4,
    )
    graph = derive_electron_labeled_graph(graph)
    graph.nodes[0]["charge"] = 9
    initial = LewisLabelledGraph.from_networkx(graph, ELECTRON_LLG_SCHEMA)
    with pytest.raises(LLGError):
        build_local_coordinate_history(initial, [])


def test_raw_aromatic_rule_conservatively_reads_the_complete_world():
    schema = LabelSchema(
        node_identity=("kind",), node_state=("x", "aromatic"), edge_state=("b",)
    )
    graph = world(2).to_networkx()
    graph.nodes[0]["aromatic"] = True
    graph.nodes[1]["aromatic"] = False
    initial = LewisLabelledGraph.from_networkx(graph, schema)
    left = LewisLabelledGraph.from_networkx(graph.subgraph([0]).copy(), schema)
    right_graph = left.to_networkx()
    right_graph.nodes[0]["x"] = 1
    right = LewisLabelledGraph.from_networkx(right_graph, schema)
    rule = RuleSpan.from_mapping(left, right, {0: 0})
    history = build_local_coordinate_history(
        initial, [GroundedLocalEvent("aromatic", rule, {0: 0})]
    )
    assert ("node", 1, "x") in history.records[0].reads
    assert ("presence", frozenset((0, 1)), None) in history.records[0].reads
    assert history.configuration({"aromatic"}).world.node_labels(0)["x"] == 1
