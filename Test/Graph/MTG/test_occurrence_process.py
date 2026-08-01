"""Occurrence identity, material flow, partial order, and choice semantics."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pytest

from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph
from synkit.Graph.MTG import (
    ChoiceWitness,
    IndependenceWitness,
    MaterialBinding,
    MaterialOccurrence,
    OccurrenceProcess,
    OccurrenceProcessFamily,
    ProcessAlternative,
    ProcessError,
    ProcessIssueCode,
    RuleOccurrence,
    detect_series_parallel,
)
from synkit.Rule.Apply import RuleSpan
from synkit.Rule.Compose import (
    LawIssueCode,
    check_parallel_independence,
    commute_independent,
    rule_spans_isomorphic,
)

SCHEMA = LabelSchema(
    node_identity=("kind",),
    node_state=("state",),
    edge_state=("weight",),
    name="occurrence-process-test/1",
)


def _llg(nodes: dict[object, int]) -> LewisLabelledGraph:
    graph = nx.Graph()
    for node, state in nodes.items():
        graph.add_node(node, kind="X", state=state)
    return LewisLabelledGraph.from_networkx(graph, SCHEMA)


def _state_rule(before: int, after: int, prefix: str) -> RuleSpan:
    return RuleSpan.from_mapping(
        _llg({f"{prefix}:L": before}),
        _llg({f"{prefix}:R": after}),
        {f"{prefix}:L": f"{prefix}:R"},
        name=prefix,
    )


def _material(identifier: str, state: int) -> MaterialOccurrence:
    return MaterialOccurrence(identifier, _llg({f"{identifier}:node": state}))


def _binding(
    material: MaterialOccurrence, endpoint: LewisLabelledGraph
) -> MaterialBinding:
    endpoint_node = next(iter(endpoint.node_ids))
    material_node = next(iter(material.value.node_ids))
    return MaterialBinding(
        material.occurrence_id,
        frozenset((endpoint_node,)),
        ((endpoint_node, material_node),),
    )


def _event(
    identifier: str,
    rule: RuleSpan,
    input_material: MaterialOccurrence,
    output_material: MaterialOccurrence,
) -> RuleOccurrence:
    return RuleOccurrence(
        identifier,
        rule,
        (_binding(input_material, rule.left),),
        (_binding(output_material, rule.right),),
    )


def _independent_process() -> OccurrenceProcess:
    first = _state_rule(0, 1, "first")
    second = _state_rule(0, 2, "second")
    first_in, first_out = _material("a0", 0), _material("a1", 1)
    second_in, second_out = _material("b0", 0), _material("b2", 2)
    host = _llg({"a": 0, "b": 0})
    certificate = commute_independent(
        first,
        {"first:L": "a"},
        second,
        {"second:L": "b"},
        host,
    )
    return OccurrenceProcess(
        "independent",
        (first_in, first_out, second_in, second_out),
        (
            _event("E1", first, first_in, first_out),
            _event("E2", second, second_in, second_out),
        ),
        (IndependenceWitness("E1", "E2", certificate),),
    )


def _causal_chain() -> OccurrenceProcess:
    first = _state_rule(0, 1, "first")
    second = _state_rule(1, 2, "second")
    initial, intermediate, final = (
        _material("m0", 0),
        _material("m1", 1),
        _material("m2", 2),
    )
    return OccurrenceProcess(
        "chain",
        (initial, intermediate, final),
        (
            _event("E1", first, initial, intermediate),
            _event("E2", second, intermediate, final),
        ),
    )


def test_causality_is_derived_from_material_occurrence_flow() -> None:
    process = _causal_chain()

    assert process.causal_pairs == (("E1", "E2"),)
    assert process.cover_pairs == (("E1", "E2"),)
    assert process.incomparable_pairs == ()
    assert process.linear_extensions() == (("E1", "E2"),)
    assert process.is_linear_extension(("E1", "E2"))
    assert not process.is_linear_extension(("E2", "E1"))


def test_independent_interleavings_share_one_process_and_swap_proof() -> None:
    process = _independent_process()

    assert process.causal_pairs == ()
    assert process.incomparable_pairs == (frozenset(("E1", "E2")),)
    assert set(process.linear_extensions()) == {("E1", "E2"), ("E2", "E1")}
    equivalence = process.extension_equivalence(("E1", "E2"), ("E2", "E1"))
    assert equivalence.swaps == (("E1", "E2"),)
    assert equivalence.replay(process)


def test_process_relations_ignore_storage_order_and_material_carrier_names() -> None:
    process = _independent_process()
    reordered = OccurrenceProcess(
        "reordered",
        tuple(reversed(process.materials)),
        tuple(reversed(process.events)),
        process.independence,
    )
    relabeled_materials = tuple(
        MaterialOccurrence(
            material.occurrence_id,
            material.value.relabel(
                {
                    node: ("renamed", material.occurrence_id, index)
                    for index, node in enumerate(material.value.node_ids)
                }
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
                _binding(by_id[binding.material_id], event.rule.left)
                for binding in event.inputs
            ),
            tuple(
                _binding(by_id[binding.material_id], event.rule.right)
                for binding in event.outputs
            ),
        )
        for event in process.events
    )
    relabeled = OccurrenceProcess(
        "relabeled",
        relabeled_materials,
        relabeled_events,
        process.independence,
    )

    summaries = {
        (item.causal_pairs, item.incomparable_pairs, item.linear_extensions())
        for item in (process, reordered, relabeled)
    }
    assert len(summaries) == 1


def test_incomparable_events_without_commutation_evidence_are_rejected() -> None:
    first = _state_rule(0, 1, "first")
    second = _state_rule(0, 2, "second")
    materials = (
        _material("a0", 0),
        _material("a1", 1),
        _material("b0", 0),
        _material("b2", 2),
    )

    with pytest.raises(ProcessError) as error:
        OccurrenceProcess(
            "unsupported-incomparability",
            materials,
            (
                _event("E1", first, materials[0], materials[1]),
                _event("E2", second, materials[2], materials[3]),
            ),
        )
    assert error.value.issues[-1].code is ProcessIssueCode.MISSING_INDEPENDENCE


def test_linear_extension_limit_is_typed_and_never_returns_a_prefix() -> None:
    process = _independent_process()

    with pytest.raises(ProcessError) as error:
        process.linear_extensions(max_extensions=1)
    assert error.value.issues[0].code is ProcessIssueCode.EXTENSION_LIMIT


def test_material_bindings_are_component_isomorphisms_not_names() -> None:
    rule = _state_rule(0, 1, "p")
    wrong_input = _material("wrong", 9)
    output = _material("out", 1)

    with pytest.raises(ProcessError) as error:
        OccurrenceProcess(
            "bad-binding",
            (wrong_input, output),
            (_event("E", rule, wrong_input, output),),
        )
    assert (
        error.value.issues[0].code
        is ProcessIssueCode.BINDING_NOT_ISOMORPHIC
    )


def test_finite_unfolding_uses_distinct_catalyst_occurrences() -> None:
    forward = _state_rule(0, 1, "forward")
    backward = _state_rule(1, 0, "backward")
    catalyst_0 = _material("catalyst@0", 0)
    intermediate = _material("intermediate@1", 1)
    catalyst_2 = _material("catalyst@2", 0)
    unfolded = OccurrenceProcess(
        "unfolded-cycle",
        (catalyst_0, intermediate, catalyst_2),
        (
            _event("forward@1", forward, catalyst_0, intermediate),
            _event("backward@2", backward, intermediate, catalyst_2),
        ),
    )

    assert catalyst_0.value.is_isomorphic(catalyst_2.value)
    assert catalyst_0.occurrence_id != catalyst_2.occurrence_id
    assert unfolded.causal_pairs == (("forward@1", "backward@2"),)

    with pytest.raises(ProcessError) as cyclic:
        OccurrenceProcess(
            "bad-cycle",
            (catalyst_0, intermediate),
            (
                _event("forward@1", forward, catalyst_0, intermediate),
                _event("backward@2", backward, intermediate, catalyst_0),
            ),
        )
    assert ProcessIssueCode.CAUSAL_CYCLE in {
        issue.code for issue in cyclic.value.issues
    }


def _ga3p_alternatives() -> OccurrenceProcessFamily:
    fba = _state_rule(0, 1, "fba")
    tpi = _state_rule(0, 1, "tpi")
    tkt = _state_rule(1, 2, "tkt")
    a0, b0 = _material("fbp-fragment", 0), _material("dhap", 0)
    direct = _material("ga3p-direct", 1)
    converted = _material("ga3p-from-tpi", 1)
    product_direct = _material("tkt-product-direct", 2)
    product_converted = _material("tkt-product-converted", 2)

    producer_commutation = commute_independent(
        fba, {"fba:L": "a"}, tpi, {"tpi:L": "b"}, _llg({"a": 0, "b": 0})
    )
    tpi_tkt = commute_independent(
        tpi, {"tpi:L": "b"}, tkt, {"tkt:L": "a"}, _llg({"a": 1, "b": 0})
    )
    fba_tkt = commute_independent(
        fba, {"fba:L": "a"}, tkt, {"tkt:L": "b"}, _llg({"a": 0, "b": 1})
    )
    shared_materials = (a0, b0, direct, converted)
    direct_process = OccurrenceProcess(
        "consume-direct",
        shared_materials + (product_direct,),
        (
            _event("FBA", fba, a0, direct),
            _event("TPI", tpi, b0, converted),
            _event("TKT", tkt, direct, product_direct),
        ),
        (
            IndependenceWitness("FBA", "TPI", producer_commutation),
            IndependenceWitness("TPI", "TKT", tpi_tkt),
        ),
    )
    converted_process = OccurrenceProcess(
        "consume-converted",
        shared_materials + (product_converted,),
        (
            _event("FBA", fba, a0, direct),
            _event("TPI", tpi, b0, converted),
            _event("TKT", tkt, converted, product_converted),
        ),
        (
            IndependenceWitness("FBA", "TPI", producer_commutation),
            IndependenceWitness("FBA", "TKT", fba_tkt),
        ),
    )
    return OccurrenceProcessFamily(
        (
            ProcessAlternative("direct", direct_process),
            ProcessAlternative("converted", converted_process),
        ),
        (
            ChoiceWitness(
                "direct",
                "converted",
                "material_assignment",
                "TKT consumes one of two isomorphic GA3P occurrences.",
            ),
        ),
    )


def test_repeated_ga3p_material_assignments_remain_distinct_processes() -> None:
    family = _ga3p_alternatives()
    direct, converted = (item.process for item in family.alternatives)

    assert direct.material_by_id["ga3p-direct"].value.is_isomorphic(
        direct.material_by_id["ga3p-from-tpi"].value
    )
    assert direct.event_by_id["TKT"].inputs[0].material_id == "ga3p-direct"
    assert converted.event_by_id["TKT"].inputs[0].material_id == "ga3p-from-tpi"
    assert ("FBA", "TKT") in direct.causal_pairs
    assert ("TPI", "TKT") in converted.causal_pairs
    assert family.choices[0].kind == "material_assignment"


def test_conflicting_rule_choices_are_alternative_processes_not_a_poset() -> None:
    first = _state_rule(0, 1, "first")
    second = _state_rule(0, 2, "second")
    host = _llg({"a": 0})
    decision = check_parallel_independence(
        first, {"first:L": "a"}, second, {"second:L": "a"}, host
    )
    input_one, output_one = _material("input-one", 0), _material("output-one", 1)
    input_two, output_two = _material("input-two", 0), _material("output-two", 2)
    first_process = OccurrenceProcess(
        "first-choice",
        (input_one, output_one),
        (_event("E-first", first, input_one, output_one),),
    )
    second_process = OccurrenceProcess(
        "second-choice",
        (input_two, output_two),
        (_event("E-second", second, input_two, output_two),),
    )
    family = OccurrenceProcessFamily(
        (
            ProcessAlternative("first", first_process),
            ProcessAlternative("second", second_process),
        ),
        (
            ChoiceWitness(
                "first",
                "second",
                "conflict",
                "Both events write the same material state.",
                decision.issues,
            ),
        ),
    )

    assert family.choices[0].conflicts[0].code is LawIssueCode.LABEL_WRITE_NODE


def test_repeated_isomorphic_rule_values_do_not_identify_event_occurrences() -> None:
    family = _ga3p_alternatives()
    process = family.alternatives[0].process
    fba, tpi = process.event_by_id["FBA"], process.event_by_id["TPI"]

    assert rule_spans_isomorphic(fba.rule, tpi.rule)
    assert fba.occurrence_id != tpi.occurrence_id
    assert set(process.event_by_id) == {"FBA", "TPI", "TKT"}


def test_series_parallel_is_detected_but_not_assumed_for_general_posets() -> None:
    parallel = detect_series_parallel(_independent_process())
    series = detect_series_parallel(_causal_chain())
    assert parallel is not None and parallel.kind == "parallel"
    assert series is not None and series.kind == "series"

    @dataclass
    class GeneralPoset:
        event_by_id: dict[str, None]
        causal_pairs: tuple[tuple[str, str], ...]

    n_poset = GeneralPoset(
        {key: None for key in "abcd"},
        (("a", "c"), ("b", "c"), ("b", "d")),
    )
    assert detect_series_parallel(n_poset) is None  # type: ignore[arg-type]

    with pytest.raises(ProcessError) as exhausted:
        detect_series_parallel(n_poset, max_partition_states=1)  # type: ignore[arg-type]
    assert (
        exhausted.value.issues[0].code
        is ProcessIssueCode.SERIES_PARALLEL_LIMIT
    )
