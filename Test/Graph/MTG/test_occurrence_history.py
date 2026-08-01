"""Lossless occurrence-MTG construction, projections, and reconstruction."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import pytest

from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph
from synkit.Graph.MTG import (
    ChoiceWitness,
    HistoryError,
    HistoryIssueCode,
    IndependenceWitness,
    MaterialBinding,
    MaterialOccurrence,
    OccurrenceMTG,
    OccurrenceMTGFamily,
    OccurrenceProcess,
    OccurrenceProcessFamily,
    ProcessAlternative,
    RuleOccurrence,
)
from synkit.Rule.Apply import RuleSpan
from synkit.Rule.Compose import commute_independent, rule_spans_isomorphic

SCHEMA = LabelSchema(
    node_identity=("kind",),
    node_state=("state",),
    edge_state=("weight",),
    name="occurrence-history-test/1",
)


def _llg(
    nodes: dict[object, int], edges: tuple[tuple[object, object, int], ...] = ()
) -> LewisLabelledGraph:
    graph = nx.Graph()
    for node, state in nodes.items():
        graph.add_node(node, kind="X", state=state)
    for left, right, weight in edges:
        graph.add_edge(left, right, weight=weight)
    return LewisLabelledGraph.from_networkx(graph, SCHEMA)


def _state_rule(before: int, after: int, prefix: str) -> RuleSpan:
    return RuleSpan.from_mapping(
        _llg({f"{prefix}:L": before}),
        _llg({f"{prefix}:R": after}),
        {f"{prefix}:L": f"{prefix}:R"},
        name=prefix,
    )


def _material(identifier: str, value: LewisLabelledGraph) -> MaterialOccurrence:
    return MaterialOccurrence(identifier, value)


def _whole_binding(
    material: MaterialOccurrence,
    endpoint: LewisLabelledGraph,
    mapping: dict[object, object] | None = None,
) -> MaterialBinding:
    if mapping is None:
        endpoint_nodes = sorted(endpoint.node_ids, key=repr)
        material_nodes = sorted(material.value.node_ids, key=repr)
        mapping = dict(zip(endpoint_nodes, material_nodes))
    return MaterialBinding(
        material.occurrence_id,
        frozenset(mapping),
        tuple(mapping.items()),
    )


def _single_event(
    identifier: str,
    rule: RuleSpan,
    input_material: MaterialOccurrence,
    output_material: MaterialOccurrence,
) -> RuleOccurrence:
    return RuleOccurrence(
        identifier,
        rule,
        (_whole_binding(input_material, rule.left),),
        (_whole_binding(output_material, rule.right),),
    )


def _chain(final_state: int = 2) -> OccurrenceProcess:
    first = _state_rule(0, 1, "first")
    second = _state_rule(1, final_state, "second")
    initial = _material("m0", _llg({"m0": 0}))
    intermediate = _material("m1", _llg({"m1": 1}))
    final = _material("m2", _llg({"m2": final_state}))
    return OccurrenceProcess(
        "chain",
        (initial, intermediate, final),
        (
            _single_event("E1", first, initial, intermediate),
            _single_event("E2", second, intermediate, final),
        ),
    )


def _independent() -> OccurrenceProcess:
    first = _state_rule(0, 1, "first")
    second = _state_rule(0, 2, "second")
    a0 = _material("a0", _llg({"a0": 0}))
    a1 = _material("a1", _llg({"a1": 1}))
    b0 = _material("b0", _llg({"b0": 0}))
    b2 = _material("b2", _llg({"b2": 2}))
    certificate = commute_independent(
        first,
        {"first:L": "a"},
        second,
        {"second:L": "b"},
        _llg({"a": 0, "b": 0}),
    )
    return OccurrenceProcess(
        "independent",
        (a0, a1, b0, b2),
        (
            _single_event("E1", first, a0, a1),
            _single_event("E2", second, b0, b2),
        ),
        (IndependenceWitness("E1", "E2", certificate),),
    )


def test_linear_history_roundtrips_rules_bindings_flow_and_causality() -> None:
    process = _chain()

    history = OccurrenceMTG.from_process(process)
    restored = history.to_process()

    assert history.replay().valid
    assert restored.events == process.events
    assert restored.materials == process.materials
    assert restored.causal_pairs == (("E1", "E2"),)
    assert len(history.node_lineages) == 1
    assert len(history.node_lineages[0].members) == 3


def test_full_timeline_has_explicit_states_and_outer_transformation() -> None:
    history = OccurrenceMTG.from_process(_chain())

    graph = history.full_history(("E1", "E2"))
    states = next(iter(graph.nodes(data=True)))[1]["history"]
    outer = history.outer_rule()
    composed = history.composed_change_graph()

    assert tuple(dict(state.labels)["state"] for state in states) == (0, 1, 2)
    assert all(state.present for state in states)
    assert len(outer.left.node_ids) == len(outer.right.node_ids) == 1
    assert next(iter(composed.nodes(data=True)))[1]["left"].present
    assert next(iter(composed.nodes(data=True)))[1]["right"].present


def test_independent_extensions_share_outer_result_but_keep_event_order() -> None:
    process = _independent()
    history = OccurrenceMTG.from_process(process)

    first = history.full_history(("E1", "E2"))
    second = history.full_history(("E2", "E1"))
    equivalence = process.extension_equivalence(("E1", "E2"), ("E2", "E1"))

    assert first.graph["event_order"] == ("E1", "E2")
    assert second.graph["event_order"] == ("E2", "E1")
    assert equivalence.replay(process)
    assert rule_spans_isomorphic(history.outer_rule(), history.outer_rule())
    assert history.replay().valid


def _temporary_bond_process() -> OccurrenceProcess:
    add = RuleSpan.from_mapping(
        _llg({"a": 0, "b": 0}),
        _llg({"x": 0, "y": 0}, (("x", "y", 1),)),
        {"a": "x", "b": "y"},
        name="add-bond",
    )
    remove = RuleSpan.from_mapping(
        _llg({"p": 0, "q": 0}, (("p", "q", 1),)),
        _llg({"r": 0, "s": 0}),
        {"p": "r", "q": "s"},
        name="remove-bond",
    )
    a0 = _material("a0", _llg({"a0": 0}))
    b0 = _material("b0", _llg({"b0": 0}))
    bonded = _material("bonded", _llg({"u": 0, "v": 0}, (("u", "v", 1),)))
    a2 = _material("a2", _llg({"a2": 0}))
    b2 = _material("b2", _llg({"b2": 0}))
    inert = _material("inert", _llg({"spectator": 9}))
    add_event = RuleOccurrence(
        "add",
        add,
        (
            _whole_binding(a0, add.left, {"a": "a0"}),
            _whole_binding(b0, add.left, {"b": "b0"}),
        ),
        (_whole_binding(bonded, add.right, {"x": "u", "y": "v"}),),
    )
    remove_event = RuleOccurrence(
        "remove",
        remove,
        (_whole_binding(bonded, remove.left, {"p": "u", "q": "v"}),),
        (
            _whole_binding(a2, remove.right, {"r": "a2"}),
            _whole_binding(b2, remove.right, {"s": "b2"}),
        ),
    )
    return OccurrenceProcess(
        "temporary-bond",
        (a0, b0, bonded, a2, b2, inert),
        (add_event, remove_event),
    )


def test_temporary_edge_absence_is_explicit_and_inert_context_is_separate() -> None:
    history = OccurrenceMTG.from_process(_temporary_bond_process())

    full = history.full_history(("add", "remove"))
    edge_history = next(iter(full.edges(data=True)))[2]["history"]
    composed = history.composed_change_graph()
    minimal = history.minimal_changed_core(("add", "remove"))
    outer = history.outer_rule()

    assert tuple(state.present for state in edge_history) == (False, True, False)
    assert composed.number_of_edges() == 0
    assert minimal.number_of_edges() == 1
    assert minimal.number_of_nodes() == 2
    assert full.number_of_nodes() == 3
    assert outer.left.is_isomorphic(outer.right)


def test_catalyst_return_is_outer_identity_but_not_empty_history() -> None:
    history = OccurrenceMTG.from_process(_chain(final_state=0))

    outer = history.outer_rule()
    minimal = history.minimal_changed_core(("E1", "E2"))
    states = next(iter(minimal.nodes(data=True)))[1]["history"]

    assert outer.left.is_isomorphic(outer.right)
    assert minimal.number_of_nodes() == 1
    assert tuple(dict(state.labels)["state"] for state in states) == (0, 1, 0)


def test_invalid_extension_and_tampered_causality_are_detected() -> None:
    history = OccurrenceMTG.from_process(_chain())

    with pytest.raises(HistoryError) as error:
        history.full_history(("E2", "E1"))
    assert error.value.issues[0].code is HistoryIssueCode.INVALID_EXTENSION

    tampered = replace(history, causal_pairs=())
    replay = tampered.replay()
    assert not replay.valid
    assert replay.issues[0].code is HistoryIssueCode.ROUNDTRIP


def test_process_alternatives_emit_separate_mtg_candidates() -> None:
    rule = _state_rule(1, 2, "consume")
    direct = _material("ga3p-direct", _llg({"direct": 1}))
    converted = _material("ga3p-converted", _llg({"converted": 1}))
    direct_product = _material("product-direct", _llg({"pd": 2}))
    converted_product = _material("product-converted", _llg({"pc": 2}))
    direct_process = OccurrenceProcess(
        "direct-process",
        (direct, direct_product),
        (_single_event("TKT", rule, direct, direct_product),),
    )
    converted_process = OccurrenceProcess(
        "converted-process",
        (converted, converted_product),
        (_single_event("TKT", rule, converted, converted_product),),
    )
    family = OccurrenceProcessFamily(
        (
            ProcessAlternative("direct", direct_process),
            ProcessAlternative("converted", converted_process),
        ),
        (
            ChoiceWitness(
                "direct",
                "converted",
                "material_assignment",
                "Two isomorphic GA3P copies remain distinct.",
            ),
        ),
    )

    histories = OccurrenceMTGFamily.from_process_family(family)
    restored = histories.to_process_family()

    assert tuple(item.alternative_id for item in histories.alternatives) == (
        "direct",
        "converted",
    )
    assert (
        histories.alternatives[0]
        .history.events[0]
        .inputs[0]
        .material_id
        == "ga3p-direct"
    )
    assert (
        histories.alternatives[1]
        .history.events[0]
        .inputs[0]
        .material_id
        == "ga3p-converted"
    )
    assert restored.choices == family.choices


def test_material_carrier_relabeling_preserves_semantic_histories() -> None:
    process = _chain()
    baseline = OccurrenceMTG.from_process(process)
    relabeled_materials = tuple(
        MaterialOccurrence(
            material.occurrence_id,
            material.value.relabel(
                {node: ("renamed", material.occurrence_id) for node in material.value.node_ids}
            ),
        )
        for material in process.materials
    )
    by_id = {material.occurrence_id: material for material in relabeled_materials}
    relabeled_events = tuple(
        RuleOccurrence(
            event.occurrence_id,
            event.rule,
            tuple(
                _whole_binding(by_id[binding.material_id], event.rule.left)
                for binding in event.inputs
            ),
            tuple(
                _whole_binding(by_id[binding.material_id], event.rule.right)
                for binding in event.outputs
            ),
        )
        for event in process.events
    )
    relabeled = OccurrenceMTG.from_process(
        OccurrenceProcess(
            "relabeled", relabeled_materials, relabeled_events, process.independence
        )
    )

    baseline_states = tuple(
        (state.present, state.labels)
        for _, attrs in baseline.full_history(("E1", "E2")).nodes(data=True)
        for state in attrs["history"]
    )
    relabeled_states = tuple(
        (state.present, state.labels)
        for _, attrs in relabeled.full_history(("E1", "E2")).nodes(data=True)
        for state in attrs["history"]
    )
    assert baseline_states == relabeled_states
