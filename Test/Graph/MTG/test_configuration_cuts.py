"""Order-ideal cuts, carrier decoding, and corrupt-provenance refusals."""

from dataclasses import replace

import networkx as nx
import pytest

from synkit.Graph.Morphism import ISOTOPE_ELECTRON_LLG_SCHEMA, LewisLabelledGraph
from synkit.Graph.MTG import (
    ConfigurationCut,
    EdgeLineage,
    HistoryError,
    HistoryIssueCode,
    NodeLineage,
    OccurrenceMTG,
    OccurrenceProcess,
    ProcessError,
    ProcessIssueCode,
    IndependenceWitness,
    RuleOccurrence,
)
from synkit.Rule.Apply import RuleSpan
from synkit.Rule.Compose import commute_independent

from .test_occurrence_history import (
    _chain,
    _independent,
    _llg,
    _material,
    _temporary_bond_process,
    _whole_binding,
)


def _no_schedule(*args, **kwargs):
    raise AssertionError("A configuration query must not materialize a schedule.")


def test_cut_and_boundary_projection_do_not_select_or_enumerate_schedules(monkeypatch):
    history = OccurrenceMTG.from_process(_independent())
    monkeypatch.setattr(OccurrenceProcess, "linear_extensions", _no_schedule)
    monkeypatch.setattr(OccurrenceMTG, "full_history", _no_schedule)
    monkeypatch.setattr(nx, "all_topological_sorts", _no_schedule)
    monkeypatch.setattr(nx, "lexicographical_topological_sort", _no_schedule)

    cut = history.configuration_cut({"E2"})
    boundary = history.composed_change_graph()

    assert isinstance(cut, ConfigurationCut)
    assert cut.active_material_ids == frozenset({"a0", "b2"})
    assert cut.executed_events == frozenset({"E2"})
    observed = {
        state.material_id: dict(state.labels)["state"]
        for _, state in cut.node_states
        if state.present
    }
    assert observed == {"a0": 0, "b2": 2}
    assert "event_order" not in cut.lineage_graph().graph
    assert boundary.number_of_nodes() == 2


def test_cut_world_uses_lineage_ids_and_retains_original_material_correspondence():
    history = OccurrenceMTG.from_process(_chain())
    cut = history.configuration_cut({"E1"})
    lineage, state = cut.node_states[0]

    assert cut.world.node_ids == frozenset({lineage.lineage_id})
    assert cut.world.node_labels(lineage.lineage_id)["state"] == 1
    assert state.material_id == "m1" and state.carrier == "m1"
    assert cut.active_material_ids == frozenset({"m1"})
    assert cut.lineage_graph().nodes[lineage.lineage_id]["state"] == state


def test_configuration_is_a_set_and_accepts_one_pass_iterables():
    history = OccurrenceMTG.from_process(_independent())
    first = history.configuration_cut(iter(("E2", "E1", "E2")))
    second = history.configuration_cut({"E1", "E2"})
    assert first.executed_events == second.executed_events
    assert first.node_states == second.node_states
    assert first.world == second.world


@pytest.mark.parametrize("events", [{"E2"}, {"not-an-event"}, {"E1", "not-an-event"}])
def test_unknown_events_and_nonideals_are_refused(events):
    history = OccurrenceMTG.from_process(_chain())
    with pytest.raises(HistoryError) as caught:
        history.configuration_cut(events)
    assert caught.value.issues[0].code is HistoryIssueCode.INVALID_CONFIGURATION


def test_temporary_bond_presence_and_inert_material_are_decoded_at_the_ideal():
    history = OccurrenceMTG.from_process(_temporary_bond_process())
    initial = history.configuration_cut(())
    middle = history.configuration_cut({"add"})
    final = history.configuration_cut({"add", "remove"})

    assert [
        cut.world.to_networkx().number_of_edges() for cut in (initial, middle, final)
    ] == [0, 1, 0]
    assert "inert" in middle.active_material_ids
    assert (
        dict(
            middle.world.node_labels(
                history.node_lineage_by_carrier[("inert", "spectator")]
            )
        )["state"]
        == 9
    )
    assert initial.edge_states[0][1].present is False
    assert middle.edge_states[0][1].present is True
    assert final.edge_states[0][1].present is False
    assert initial.lineage_graph().number_of_edges() == 1
    assert initial.lineage_graph(include_absent=False).number_of_edges() == 0


def test_birth_and_death_have_explicit_absence_and_empty_boundary_worlds():
    empty = _llg({})
    value = _material("born", _llg({"v": 1}))
    birth = RuleSpan.from_mapping(empty, value.value, {}, name="birth")
    death = RuleSpan.from_mapping(value.value, empty, {}, name="death")
    process = OccurrenceProcess(
        "birth-death",
        (value,),
        (
            RuleOccurrence("birth", birth, (), (_whole_binding(value, birth.right),)),
            RuleOccurrence("death", death, (_whole_binding(value, death.left),), ()),
        ),
    )
    history = OccurrenceMTG.from_process(process)
    cuts = [
        history.configuration_cut(events)
        for events in ((), {"birth"}, {"birth", "death"})
    ]

    assert [len(cut.world.node_ids) for cut in cuts] == [0, 1, 0]
    assert [cut.node_states[0][1].present for cut in cuts] == [False, True, False]
    assert cuts[0].lineage_graph(include_absent=False).number_of_nodes() == 0
    assert cuts[0].lineage_graph().number_of_nodes() == 1


def test_world_equals_independent_material_union_at_every_schedule_prefix():
    process = _independent()
    history = OccurrenceMTG.from_process(process)
    observations = {}
    for order in process.linear_extensions():
        timeline = history.full_history(order)
        active = set(history.initial_material_ids)
        for index in range(len(order) + 1):
            if index:
                event = process.event_by_id[order[index - 1]]
                active.difference_update(
                    binding.material_id for binding in event.inputs
                )
                active.update(binding.material_id for binding in event.outputs)
            ideal = frozenset(order[:index])
            cut = history.configuration_cut(ideal)
            assert cut.active_material_ids == active
            expected = nx.Graph()
            for material_id in active:
                material = process.material_by_id[material_id]
                renaming = {
                    node: history.node_lineage_by_carrier[(material_id, node)]
                    for node in material.value.node_ids
                }
                expected = nx.compose(
                    expected, nx.relabel_nodes(material.value.to_networkx(), renaming)
                )
            assert nx.utils.graphs_equal(cut.world.to_networkx(), expected)
            for lineage, state in cut.node_states:
                assert state == timeline.nodes[lineage.lineage_id]["history"][index]
            if ideal in observations:
                assert cut.node_states == observations[ideal]
            observations[ideal] = cut.node_states
    assert len(observations) == 4


@pytest.mark.parametrize("corruption", ["merge", "split", "duplicate-id"])
def test_exact_carrier_coverage_does_not_hide_corrupt_lineage_partitions(corruption):
    history = OccurrenceMTG.from_process(
        _independent() if corruption != "split" else _chain()
    )
    if corruption == "merge":
        members = tuple(
            member for node in history.node_lineages for member in node.members
        )
        nodes = (NodeLineage("merged", members),)
    elif corruption == "split":
        nodes = tuple(
            NodeLineage(f"split:{index}", (member,))
            for index, member in enumerate(history.node_lineages[0].members)
        )
    else:
        nodes = tuple(
            replace(node, lineage_id="duplicate") for node in history.node_lineages
        )
    tampered = replace(history, node_lineages=nodes)
    assert not tampered.replay().valid
    with pytest.raises(HistoryError, match="Lineage partitions"):
        tampered.configuration_cut(())


def test_wrong_edge_lineage_endpoints_are_refused_before_decoding():
    history = OccurrenceMTG.from_process(_temporary_bond_process())
    edge = history.edge_lineages[0]
    wrong = EdgeLineage(
        edge.lineage_id, frozenset({"missing", "endpoint"}), edge.members
    )
    tampered = replace(history, edge_lineages=(wrong,))
    assert not tampered.replay().valid
    with pytest.raises(HistoryError, match="preserved carrier endpoints"):
        tampered.configuration_cut({"add"})


def test_consistent_lineage_renaming_preserves_world_and_observations():
    history = OccurrenceMTG.from_process(_temporary_bond_process())
    renaming = {
        node.lineage_id: f"renamed:{node.lineage_id}" for node in history.node_lineages
    }
    changed = replace(
        history,
        node_lineages=tuple(
            replace(node, lineage_id=renaming[node.lineage_id])
            for node in history.node_lineages
        ),
        edge_lineages=tuple(
            replace(
                edge,
                lineage_id=f"renamed:{edge.lineage_id}",
                endpoints=frozenset(renaming[node] for node in edge.endpoints),
            )
            for edge in history.edge_lineages
        ),
    )
    assert changed.replay().valid
    before = history.configuration_cut({"add"})
    after = changed.configuration_cut({"add"})
    assert before.world.relabel(renaming) == after.world
    assert [state for _, state in before.node_states] == [
        state for _, state in after.node_states
    ]


def test_isotope_world_derives_missing_charge_and_refuses_contradictory_material_state():
    graph = nx.Graph()
    graph.add_node(
        "oxygen",
        element="O",
        isotope=18,
        aromatic=False,
        hcount=0,
        radical=0,
        lone_pairs=3,
        valence_electrons=6,
    )
    material = _material(
        "labelled-oxygen",
        LewisLabelledGraph.from_networkx(graph, ISOTOPE_ELECTRON_LLG_SCHEMA),
    )
    history = OccurrenceMTG.from_process(
        OccurrenceProcess("isotope-identity", (material,), ())
    )
    cut = history.configuration_cut(())
    lineage = cut.node_states[0][0].lineage_id
    assert cut.world.node_labels(lineage)["isotope"] == 18
    assert cut.world.node_labels(lineage)["charge"] == 0
    graph.nodes["oxygen"]["charge"] = 5
    invalid = replace(
        material,
        value=LewisLabelledGraph.from_networkx(graph, ISOTOPE_ELECTRON_LLG_SCHEMA),
    )
    corrupted = replace(history, materials=(invalid,))
    assert not corrupted.replay().valid
    with pytest.raises(HistoryError) as caught:
        corrupted.configuration_cut(())
    assert caught.value.issues[0].code is HistoryIssueCode.INVALID_STATE


class _AllowedGuard:
    def validate(self, host, match):
        return None


def test_bare_span_witness_cannot_stand_in_for_a_guarded_event():
    process = _independent()
    first, second = process.events
    changed = replace(first, rule=replace(first.rule, condition=_AllowedGuard()))
    with pytest.raises(ProcessError) as caught:
        replace(process, events=(changed, second))
    assert ProcessIssueCode.INVALID_INDEPENDENCE in {
        issue.code for issue in caught.value.issues
    }


def test_guarded_event_accepts_its_actual_executed_commutation_witness():
    process = _independent()
    first, second = process.events
    changed = replace(first, rule=replace(first.rule, condition=_AllowedGuard()))
    certificate = commute_independent(
        changed.rule,
        {"first:L": "a"},
        second.rule,
        {"second:L": "b"},
        _llg({"a": 0, "b": 0}),
    )
    guarded = replace(
        process,
        events=(changed, second),
        independence=(IndependenceWitness("E1", "E2", certificate),),
    )
    cut = OccurrenceMTG.from_process(guarded).configuration_cut({"E1"})
    assert cut.active_material_ids == frozenset({"a1", "b0"})
