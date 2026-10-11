"""Port composition laws, origin transport and refused execution premises."""

from dataclasses import replace

import networkx as nx
import pytest

from synkit.Graph.Morphism import (
    ISOTOPE_ELECTRON_LLG_SCHEMA,
    LabelSchema,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Graph.MTG import (
    CrossIndependenceWitness,
    GluingError,
    GluingIssueCode,
    HistoryError,
    OccurrenceMTG,
    OccurrenceProcess,
    MaterialOccurrence,
    PortGluing,
    ProcessError,
    ProcessIssueCode,
    compose_histories,
)
from synkit.Rule.Apply import DPOError, DPOIssue, DPOIssueCode
from synkit.Rule.Compose import commute_independent

from .test_occurrence_history import (
    SCHEMA,
    _llg,
    _material,
    _single_event,
    _state_rule,
    _temporary_bond_process,
)


def _step(before, after, *, guard=None):
    rule = replace(_state_rule(before, after, "E"), condition=guard)
    initial = _material("m0", _llg({"input": before}))
    final = _material("m1", _llg({"output": after}))
    process = OccurrenceProcess(
        "step", (initial, final), (_single_event("E", rule, initial, final),)
    )
    return OccurrenceMTG.from_process(process)


def _port(output="m1", input_material="m0"):
    return PortGluing(output, input_material, {"output": "input"})


def _cross(left, right, pairs=None):
    pairs = (
        pairs
        if pairs is not None
        else [
            (a.occurrence_id, b.occurrence_id)
            for a in left.events
            for b in right.events
        ]
    )
    witnesses = []
    for a, b in pairs:
        first, second = left.event_by_id[a].rule, right.event_by_id[b].rule
        first_node, second_node = next(iter(first.left.node_ids)), next(
            iter(second.left.node_ids)
        )
        host = _llg(
            {
                "x": first.left.node_labels(first_node)["state"],
                "y": second.left.node_labels(second_node)["state"],
            }
        )
        certificate = commute_independent(
            first, {first_node: "x"}, second, {second_node: "y"}, host
        )
        witnesses.append(CrossIndependenceWitness(a, b, certificate))
    return tuple(witnesses)


def _root_maps(composition, side):
    embedding = getattr(composition, side + "_embedding")
    return dict(embedding.event_mapping), dict(embedding.carrier_mapping)


def _through(inner, outer):
    return {key: outer[value] for key, value in inner.items()}


def _semantic_signature(history, roots):
    events, carriers = {}, {}
    for name, (event_map, carrier_map) in roots.items():
        events.update({(name, key): value for key, value in event_map.items()})
        carriers.update({(name, key): value for key, value in carrier_map.items()})
    event_inverse = {value: key for key, value in events.items()}
    lineage_roots = {}
    for root, carrier in carriers.items():
        lineage_roots.setdefault(history.node_lineage_by_carrier[carrier], set()).add(
            root
        )
    partitions = frozenset(frozenset(values) for values in lineage_roots.values())
    order = frozenset(
        (event_inverse[a], event_inverse[b]) for a, b in history.causal_pairs
    )
    final = history.configuration_cut(history.event_by_id)
    states = frozenset(
        (
            frozenset(lineage_roots[node]),
            tuple(sorted(final.world.node_labels(node, semantic=True).items())),
        )
        for node in final.world.node_ids
    )
    return partitions, order, states


def test_sequential_gluing_retains_local_rules_conditions_and_atom_origins(monkeypatch):
    left, right = _step(0, 1), _step(1, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Port composition must not enumerate schedules or select an isomorphism."
        )

    monkeypatch.setattr(nx, "all_topological_sorts", forbidden)
    monkeypatch.setattr(
        nx.algorithms.isomorphism.GraphMatcher, "isomorphisms_iter", forbidden
    )
    composition = compose_histories(left, right, (_port(),))
    history = composition.history
    assert composition.replay()
    assert history.causal_pairs == (("left:E", "right:E"),)
    assert len(history.materials) == 3
    assert history.event_by_id["right:E"].rule is right.events[0].rule
    assert dict(history.event_by_id["right:E"].inputs[0].mapping) == {"E:L": "output"}
    assert dict(composition.right_embedding.carrier_mapping)["m0", "input"] == (
        "left:m1",
        "output",
    )
    assert len(history.node_lineages) == 1
    assert history.initial_material_ids == {"left:m0"}
    assert history.final_material_ids == {"right:m1"}
    assert len(composition.local_applications) == 2
    assert left.material_by_id["m1"].value.node_ids == {"output"}
    assert right.material_by_id["m0"].value.node_ids == {"input"}


def test_associativity_uses_transported_ports_and_external_origin_correspondence():
    a, b, c = _step(0, 1), _step(1, 2), _step(2, 3)
    ab = compose_histories(a, b, (_port(),))
    first = compose_histories(ab.history, c, (_port("right:m1"),))
    bc = compose_histories(b, c, (_port(),))
    second = compose_histories(a, bc.history, (_port(input_material="left:m0"),))
    first_left, first_right = _root_maps(first, "left"), _root_maps(first, "right")
    second_left, second_right = _root_maps(second, "left"), _root_maps(second, "right")
    first_roots = {
        "a": tuple(
            _through(inner, outer)
            for inner, outer in zip(_root_maps(ab, "left"), first_left)
        ),
        "b": tuple(
            _through(inner, outer)
            for inner, outer in zip(_root_maps(ab, "right"), first_left)
        ),
        "c": first_right,
    }
    second_roots = {
        "a": second_left,
        "b": tuple(
            _through(inner, outer)
            for inner, outer in zip(_root_maps(bc, "left"), second_right)
        ),
        "c": tuple(
            _through(inner, outer)
            for inner, outer in zip(_root_maps(bc, "right"), second_right)
        ),
    }
    assert first.replay() and second.replay()
    assert _semantic_signature(first.history, first_roots) == _semantic_signature(
        second.history, second_roots
    )


def test_parallel_interchange_keeps_the_two_chains_and_all_copy_origins():
    a, b, c, d = _step(0, 1), _step(1, 2), _step(0, 3), _step(3, 4)
    ab = compose_histories(a, b, (_port(),))
    cd = compose_histories(c, d, (_port(),))
    sequential_parallel = compose_histories(
        ab.history, cd.history, cross_independence=_cross(ab.history, cd.history)
    )
    ac = compose_histories(a, c, cross_independence=_cross(a, c))
    bd = compose_histories(b, d, cross_independence=_cross(b, d))
    parallel_sequential = compose_histories(
        ac.history,
        bd.history,
        (_port("left:m1", "left:m0"), _port("right:m1", "right:m0")),
        cross_independence=_cross(
            ac.history, bd.history, (("left:E", "right:E"), ("right:E", "left:E"))
        ),
    )
    first_roots, second_roots = {}, {}
    for name, inner, inner_side, outer_side in (
        ("a", ab, "left", "left"),
        ("b", ab, "right", "left"),
        ("c", cd, "left", "right"),
        ("d", cd, "right", "right"),
    ):
        first_roots[name] = tuple(
            _through(x, y)
            for x, y in zip(
                _root_maps(inner, inner_side),
                _root_maps(sequential_parallel, outer_side),
            )
        )
    for name, inner, inner_side, outer_side in (
        ("a", ac, "left", "left"),
        ("c", ac, "right", "left"),
        ("b", bd, "left", "right"),
        ("d", bd, "right", "right"),
    ):
        second_roots[name] = tuple(
            _through(x, y)
            for x, y in zip(
                _root_maps(inner, inner_side),
                _root_maps(parallel_sequential, outer_side),
            )
        )
    assert sequential_parallel.replay() and parallel_sequential.replay()
    assert _semantic_signature(
        sequential_parallel.history, first_roots
    ) == _semantic_signature(parallel_sequential.history, second_roots)
    assert len(parallel_sequential.history.node_lineages) == 2
    assert len(parallel_sequential.history.to_process().linear_extensions()) == 6


def test_equal_intermediate_copies_remain_distinct_and_selected_pairings_change_origin():
    a, b, consume = _step(0, 1), _step(0, 1), _step(1, 2)
    copies = compose_histories(a, b, cross_independence=_cross(a, b)).history
    alternatives = []
    for selected, spectator in (("left", "right"), ("right", "left")):
        result = compose_histories(
            copies,
            consume,
            (_port(f"{selected}:m1"),),
            cross_independence=_cross(copies, consume, ((f"{spectator}:E", "E"),)),
        )
        assert result.replay()
        assert len(result.history.node_lineages) == 2
        final = result.history.configuration_cut(result.history.event_by_id)
        carrier = dict(result.left_embedding.carrier_mapping)["left:m0", "input"]
        node = result.history.node_lineage_by_carrier[carrier]
        alternatives.append(final.world.node_labels(node)["state"])
    assert alternatives == [2, 1]


@pytest.mark.parametrize("side", ("left", "right", "both"))
def test_empty_typed_unit_preserves_the_nonempty_history(side):
    empty = OccurrenceMTG.empty(SCHEMA)
    step = _step(0, 1)
    left, right = (empty, step) if side == "left" else (step, empty)
    if side == "both":
        left, right = empty, empty
    result = compose_histories(left, right)
    assert result.replay()
    assert len(result.history.events) == (0 if side == "both" else 1)
    assert len(result.history.node_lineages) == (0 if side == "both" else 1)
    assert empty.configuration_cut(()).world.node_ids == frozenset()
    assert empty.outer_rule().left.node_ids == frozenset()
    assert empty.full_history(()).number_of_nodes() == 0


@pytest.mark.parametrize(
    "ports,code",
    (
        ((_port("absent"),), GluingIssueCode.UNKNOWN_PORT),
        ((_port("m0"),), GluingIssueCode.NOT_BOUNDARY),
        ((_port(input_material="m1"),), GluingIssueCode.NOT_BOUNDARY),
        ((_port(), _port()), GluingIssueCode.PORT_REUSED),
        ((PortGluing("m1", "m0", {}),), GluingIssueCode.PORT_ISOMORPHISM),
        (
            (PortGluing("m1", "m0", (("output", "input"), ("output", "input"))),),
            GluingIssueCode.PORT_ISOMORPHISM,
        ),
    ),
)
def test_bad_boundary_bindings_are_refused(ports, code):
    with pytest.raises(GluingError) as error:
        compose_histories(_step(0, 1), _step(1, 2), ports)
    assert error.value.code == code


def test_semantically_wrong_port_and_incompatible_schemas_are_refused():
    with pytest.raises(GluingError) as error:
        compose_histories(_step(0, 1), _step(0, 2), (_port(),))
    assert error.value.code == GluingIssueCode.PORT_ISOMORPHISM
    with pytest.raises(GluingError) as error:
        compose_histories(
            OccurrenceMTG.empty(SCHEMA),
            OccurrenceMTG.empty(LabelSchema(name="different")),
        )
    assert error.value.code == GluingIssueCode.SCHEMA_MISMATCH


def test_new_incomparable_pairs_require_correct_nonduplicated_evidence():
    left, right = _step(0, 1), _step(0, 2)
    with pytest.raises(ProcessError) as error:
        compose_histories(left, right)
    assert any(
        i.code == ProcessIssueCode.MISSING_INDEPENDENCE for i in error.value.issues
    )
    cross = _cross(left, right)
    with pytest.raises(GluingError) as error:
        compose_histories(left, right, cross_independence=cross + cross)
    assert error.value.code == GluingIssueCode.INVALID_CERTIFICATE
    with pytest.raises(GluingError):
        compose_histories(
            left, right, cross_independence=(replace(cross[0], left_event="missing"),)
        )
    unrelated = _cross(_step(0, 7), right)
    with pytest.raises(ProcessError) as error:
        compose_histories(left, right, cross_independence=unrelated)
    assert any(
        i.code == ProcessIssueCode.INVALID_INDEPENDENCE for i in error.value.issues
    )


def test_a_new_causal_pair_cannot_also_be_declared_independent():
    left, right = _step(0, 1), _step(1, 2)
    with pytest.raises(ProcessError) as error:
        compose_histories(
            left, right, (_port(),), cross_independence=_cross(left, right)
        )
    assert any(
        i.code == ProcessIssueCode.EXTRA_INDEPENDENCE for i in error.value.issues
    )


class _LocalGuard:
    def validate(self, host, match):
        if host.node_ids != {"E:L"}:
            raise DPOError(
                DPOIssue(DPOIssueCode.MATCH_LABEL, "Local coordinate changed.")
            )


class _RemoteGuard:
    def validate(self, host, match):
        if "remote" not in host.node_ids:
            raise DPOError(
                DPOIssue(DPOIssueCode.MATCH_LABEL, "Missing remote condition input.")
            )


def test_local_guard_objects_and_coordinates_survive_port_transport():
    guard = _LocalGuard()
    left, right = _step(0, 1), _step(1, 2, guard=guard)
    result = compose_histories(left, right, (_port(),))
    assert result.replay()
    assert result.history.event_by_id["right:E"].rule.condition is guard
    assert (
        dict(result.local_applications)["right:E"].certificate.rule.condition is guard
    )


def test_guard_not_admitted_on_bound_inputs_is_refused():
    left, right = _step(0, 1), _step(1, 2, guard=_RemoteGuard())
    assert right.replay().valid
    with pytest.raises(GluingError) as error:
        compose_histories(left, right, (_port(),))
    assert error.value.code == GluingIssueCode.LOCAL_APPLICATION
    assert isinstance(error.value.__cause__, DPOError)


@pytest.mark.parametrize(
    "corruption", ("origin", "lineage", "missing_application", "result", "rule")
)
def test_gluing_certificate_rejects_changed_origin_and_execution_records(corruption):
    result = compose_histories(_step(0, 1), _step(1, 2), (_port(),))
    if corruption == "origin":
        embedding = replace(result.right_embedding, carrier_mapping=())
        forged = replace(result, right_embedding=embedding)
    elif corruption == "lineage":
        embedding = replace(
            result.left_embedding, node_lineage_mapping=(("node:0", "missing"),)
        )
        forged = replace(result, left_embedding=embedding)
    elif corruption == "missing_application":
        forged = replace(result, local_applications=result.local_applications[:1])
    elif corruption == "result":
        key, application = result.local_applications[0]
        altered = replace(application, result=result.right.events[0].rule.right)
        forged = replace(
            result, local_applications=((key, altered),) + result.local_applications[1:]
        )
    else:
        event = result.history.events[0]
        changed = replace(event, rule=replace(event.rule))
        forged = replace(
            result,
            history=replace(
                result.history, events=(changed,) + result.history.events[1:]
            ),
        )
    assert not forged.replay()


def test_corrupted_source_history_cannot_be_composed():
    source = _step(0, 1)
    forged = replace(source, node_lineages=())
    with pytest.raises(HistoryError):
        compose_histories(forged, _step(1, 2), (_port(),))


def test_empty_history_requires_explicit_typing_and_cannot_hide_carriers():
    with pytest.raises(HistoryError):
        OccurrenceMTG.from_process(OccurrenceProcess("empty", (), ()))
    empty = OccurrenceMTG.from_process(
        OccurrenceProcess("empty", (), ()), empty_schema=SCHEMA
    )
    assert empty.replay().valid
    assert not replace(empty, node_lineages=_step(0, 1).node_lineages).replay().valid
    assert not replace(empty, empty_schema=None).replay().valid


def _isotope_identity(prefix):
    graph = nx.Graph()
    for isotope in (16, 18):
        graph.add_node(
            f"{prefix}{isotope}",
            element="O",
            isotope=isotope,
            aromatic=False,
            hcount=1,
            radical=0,
            lone_pairs=2,
            valence_electrons=6,
        )
    graph.add_edge(f"{prefix}16", f"{prefix}18", sigma_order=1, pi_order=0)
    value = LewisLabelledGraph.from_networkx(
        derive_electron_labeled_graph(graph), ISOTOPE_ELECTRON_LLG_SCHEMA
    )
    return OccurrenceMTG.from_process(
        OccurrenceProcess("isotope-identity", (MaterialOccurrence("m", value),), ())
    )


def test_isotope_resolved_port_maps_refuse_swapped_identity_and_transport_bonds():
    left, right = _isotope_identity("a"), _isotope_identity("b")
    result = compose_histories(
        left, right, (PortGluing("m", "m", {"a16": "b16", "a18": "b18"}),)
    )
    assert result.replay()
    assert len(result.history.materials) == 1
    assert len(result.history.edge_lineages) == 1
    assert dict(result.right_embedding.carrier_mapping)["m", "b18"] == ("left:m", "a18")
    cut = result.history.configuration_cut(())
    origin = result.history.node_lineage_by_carrier["left:m", "a18"]
    assert cut.world.node_labels(origin)["isotope"] == 18
    with pytest.raises(GluingError) as error:
        compose_histories(
            left, right, (PortGluing("m", "m", {"a16": "b18", "a18": "b16"}),)
        )
    assert error.value.code == GluingIssueCode.PORT_ISOMORPHISM


def test_gluing_after_a_deleted_bond_retains_a_distinct_recreated_bond_lineage():
    original = _temporary_bond_process()
    left = OccurrenceMTG.from_process(original)
    add = original.events[0]
    right = OccurrenceMTG.from_process(
        OccurrenceProcess("recreate", original.materials[:3], (add,))
    )
    result = compose_histories(
        left,
        right,
        (
            PortGluing("a2", "a0", {"a2": "a0"}),
            PortGluing("b2", "b0", {"b2": "b0"}),
        ),
    )
    assert result.replay()
    history = result.history
    assert len(history.edge_lineages) == 2
    old = dict(result.left_embedding.edge_lineage_mapping)[
        left.edge_lineages[0].lineage_id
    ]
    new = dict(result.right_embedding.edge_lineage_mapping)[
        right.edge_lineages[0].lineage_id
    ]
    assert old != new
    assert history.edge_lineages[0].endpoints == history.edge_lineages[1].endpoints
    timeline = history.full_history(("left:add", "left:remove", "right:add"))
    histories = {
        data["lineage_id"]: tuple(state.present for state in data["history"])
        for *_, data in timeline.edges(data=True)
    }
    assert histories[old] == (False, True, False, False)
    assert histories[new] == (False, False, False, True)


class _AttributeGuard:
    def validate(self, host, match):
        if not host.node_labels(match["E:L"]).get("annotation:allowed", False):
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.MATCH_LABEL, "Actual material does not meet the guard."
                )
            )


def test_local_application_reads_actual_bound_attributes_after_gluing():
    left, right = _step(0, 1), _step(1, 2, guard=_AttributeGuard())
    graph = right.materials[0].value.to_networkx()
    graph.nodes["input"]["allowed"] = True
    material = MaterialOccurrence("m0", LewisLabelledGraph.from_networkx(graph, SCHEMA))
    right = OccurrenceMTG.from_process(
        OccurrenceProcess("guarded", (material, right.materials[1]), right.events)
    )
    # This event is admitted on its actual original copy. The semantic port
    # isomorphism ignores annotations, but the resulting guard must still read
    # the attributes of the copy supplied by the left history.
    admitted = compose_histories(OccurrenceMTG.empty(SCHEMA), right)
    assert admitted.replay()
    assert dict(admitted.local_applications)["right:E"].certificate.host.node_labels(
        "E:L"
    )["annotation:allowed"]
    with pytest.raises(GluingError) as error:
        compose_histories(left, right, (_port(),))
    assert error.value.code == GluingIssueCode.LOCAL_APPLICATION


def test_port_noninjective_atom_map_is_refused():
    with pytest.raises(GluingError) as error:
        compose_histories(
            _isotope_identity("a"),
            _isotope_identity("b"),
            (PortGluing("m", "m", {"a16": "b16", "a18": "b16"}),),
        )
    assert error.value.code == GluingIssueCode.PORT_ISOMORPHISM


def test_annotation_sensitive_guards_prevent_a_general_associative_admission_claim():
    def identity(allowed):
        graph = _llg({"output": 1}).to_networkx()
        graph.nodes["output"]["allowed"] = allowed
        material = MaterialOccurrence(
            "m", LewisLabelledGraph.from_networkx(graph, SCHEMA)
        )
        return OccurrenceMTG.from_process(
            OccurrenceProcess("identity", (material,), ())
        )

    p, q = identity(True), identity(False)
    r = _step(1, 2, guard=_AttributeGuard())
    graph = r.materials[0].value.to_networkx()
    graph.nodes["input"]["allowed"] = True
    r = OccurrenceMTG.from_process(
        OccurrenceProcess(
            "guarded",
            (
                MaterialOccurrence(
                    "m0", LewisLabelledGraph.from_networkx(graph, SCHEMA)
                ),
                r.materials[1],
            ),
            r.events,
        )
    )
    # Individually admitted histories and semantic boundary isomorphisms do
    # not suffice: a guard may distinguish annotations ignored by that category.
    assert compose_histories(OccurrenceMTG.empty(SCHEMA), r).replay()
    pq = compose_histories(p, q, (PortGluing("m", "m", {"output": "output"}),))
    assert compose_histories(pq.history, r, (_port("left:m"),)).replay()
    with pytest.raises(GluingError) as error:
        compose_histories(q, r, (_port("m"),))
    assert error.value.code == GluingIssueCode.LOCAL_APPLICATION
