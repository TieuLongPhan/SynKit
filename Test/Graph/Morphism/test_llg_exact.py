"""Exact Lewis morphisms preserve definedness, resources and categorical laws."""

from copy import deepcopy
from itertools import product

import networkx as nx
import pytest

from synkit.Graph.Morphism import (
    ELECTRON_LLG_SCHEMA,
    ISOTOPE_ELECTRON_LLG_SCHEMA,
    LLGError,
    LLGIssueCode,
    LLGMorphism,
    LabelSchema,
    LewisLabelledGraph,
)
from synkit.Rule.Apply import DPOError, DPOIssueCode, RuleSpan, apply_dpo

SCHEMA = LabelSchema(node_identity=("kind",), node_state=("x",), edge_state=("b",))


def pattern(nodes, state=None, bond=None):
    graph = nx.Graph()
    graph.add_nodes_from((node, {"kind": "X"}) for node in nodes)
    if state is not None:
        nx.set_node_attributes(graph, state, "x")
    if bond is not None:
        graph.add_edges_from((u, v, {"b": bond}) for u, v in zip(nodes, nodes[1:]))
    return LewisLabelledGraph.from_networkx(graph, SCHEMA, partial=True)


def water():
    graph = nx.Graph()
    graph.add_node(1, element="O", aromatic=False, hcount=2, radical=0, lone_pairs=2, valence_electrons=6, atom_map=19)
    return graph


def aromatic(phase):
    graph = nx.cycle_graph(6)
    nx.set_node_attributes(
        graph,
        {
            n: {"element": "C", "aromatic": True, "hcount": 1, "radical": 0, "lone_pairs": 0, "valence_electrons": 4}
            for n in graph
        },
    )
    for n in graph:
        graph.edges[n, (n + 1) % 6].update(sigma_order=1, pi_order=(n + phase) % 2)
    return LewisLabelledGraph.from_electron_graph(graph)


@pytest.mark.parametrize("exact", [False, True])
def test_undefined_coordinates_specialize_but_do_not_form_isomorphisms(exact):
    source, target = pattern((1, 2)), pattern((10, 20), state=0)
    morphism = LLGMorphism(source, target, {1: 10, 2: 20}, exact=exact)
    assert not morphism.is_isomorphism
    with pytest.raises(LLGError, match="not preserved"):
        LLGMorphism(target, source, {10: 1, 20: 2}, exact=exact)


def test_none_is_a_defined_value_and_not_an_omitted_coordinate():
    graph = nx.Graph()
    graph.add_node(1, kind="X", x=None)
    defined_none = LewisLabelledGraph.from_networkx(graph, SCHEMA, partial=True)
    omitted = pattern((1,))
    assert not LLGMorphism(omitted, defined_none, {1: 1}, exact=True).is_isomorphism
    with pytest.raises(LLGError):
        LLGMorphism(defined_none, omitted, {1: 1}, exact=True)


def test_edge_undefinedness_is_preserved_and_reflected_by_isomorphisms():
    source = pattern((1, 2), bond=0).to_networkx()
    source.edges[1, 2].pop("b")
    source = LewisLabelledGraph.from_networkx(source, SCHEMA, partial=True)
    target = pattern((10, 20), bond=0)
    arrow = LLGMorphism(source, target, {1: 10, 2: 20}, exact=True)
    assert not arrow.is_isomorphism
    assert not source.is_isomorphic(target, exact=True)


def test_exact_partial_coordinate_chains_associate_and_preserve_identities():
    # Each coordinate is either undefined or one of two defined values. Test
    # every valid specialization chain, with proper carrier embeddings.
    checked = 0
    for values in product((None, 0, 1), repeat=4):
        if any(a is not None and a != b for a, b in zip(values, values[1:])):
            continue
        graphs = [pattern(tuple(range(size)), state=state) for size, state in zip((1, 2, 3, 4), values)]
        arrows = [LLGMorphism(a, b, {n: n for n in a.node_ids}, exact=True) for a, b in zip(graphs, graphs[1:])]
        f, g, h = arrows
        assert f.then(g).then(h) == f.then(g.then(h))
        assert LLGMorphism.identity(graphs[0], exact=True).then(f) == f
        assert f.then(LLGMorphism.identity(graphs[1], exact=True)) == f
        checked += 1
    assert checked == 9


def test_exact_policy_survives_relabeling_and_refuses_implicit_downgrade():
    source, target = pattern((1,)), pattern((10, 20))
    f = LLGMorphism(source, target, {1: 10}, exact=True)
    moved = f.relabel({1: "a"}, {10: "b", 20: "c"})
    assert moved.exact and moved.mapping == {"a": "b"}
    with pytest.raises(LLGError) as error:
        f.then(LLGMorphism.identity(target))
    assert error.value.issues[0].code is LLGIssueCode.MORPHISM_POLICY_MISMATCH


def test_kekule_matching_and_exact_morphisms_are_distinct():
    first, second = aromatic(0), aromatic(1)
    assert first.is_isomorphic(second)
    assert LLGMorphism(first, second, {n: n for n in first.node_ids}).is_isomorphism
    with pytest.raises(LLGError) as error:
        LLGMorphism(first, second, {n: n for n in first.node_ids}, exact=True)
    assert error.value.issues[0].code is LLGIssueCode.EDGE_LABEL_MISMATCH
    # The two phases do have exact isomorphisms using a ring rotation.
    assert LLGMorphism(first, second, {n: (n + 1) % 6 for n in first.node_ids}, exact=True).is_isomorphism


def test_aromatic_matching_is_not_claimed_closed_under_fragment_composition():
    first, second = aromatic(0), aromatic(1)
    fragment = LewisLabelledGraph.from_networkx(first.to_networkx().subgraph((0, 1)).copy(), ELECTRON_LLG_SCHEMA)
    f = LLGMorphism(fragment, first, {0: 0, 1: 1})
    g = LLGMorphism(first, second, {n: n for n in first.node_ids})
    with pytest.raises(LLGError):
        f.then(g)


def test_dpo_admission_uses_exact_coordinates_before_replay():
    first, second = aromatic(0), aromatic(1)
    rule = RuleSpan.from_mapping(first, first, {n: n for n in first.node_ids})
    application = apply_dpo(rule, first, {n: n for n in first.node_ids})
    assert application.certificate.replay().valid
    with pytest.raises(DPOError) as error:
        apply_dpo(rule, second, {n: n for n in first.node_ids})
    assert error.value.issues[0].code is DPOIssueCode.MATCH_LABEL


def test_checked_electron_factory_derives_state_without_mutation():
    graph = water()
    before = deepcopy(graph)
    value = LewisLabelledGraph.from_electron_graph(graph)
    assert value.node_labels(1)["charge"] == 0
    assert value.node_labels(1)["bond_order_sum"] == 0
    assert nx.utils.graphs_equal(graph, before)


def test_checked_factory_refuses_inconsistent_derived_charge():
    graph = water()
    graph.nodes[1]["charge"] = 9
    with pytest.raises(LLGError) as error:
        LewisLabelledGraph.from_electron_graph(graph)
    assert error.value.issues[0].code is LLGIssueCode.DERIVED_LABEL_MISMATCH


def test_integer_lewis_accounting_does_not_round_through_binary_float():
    graph = water()
    graph.nodes[1].update(valence_electrons=2**53 + 1, lone_pairs=0, hcount=0)
    value = LewisLabelledGraph.from_electron_graph(graph)
    assert value.node_labels(1)["charge"] == 2**53 + 1
    graph.nodes[1]["charge"] = 2**53
    with pytest.raises(LLGError) as error:
        LewisLabelledGraph.from_electron_graph(graph)
    assert error.value.issues[0].code is LLGIssueCode.DERIVED_LABEL_MISMATCH


@pytest.mark.parametrize(
    "key,value",
    list(
        product(
            ("hcount", "radical", "lone_pairs", "valence_electrons"),
            (-1, 0.5, float("nan"), float("inf"), True, "2"),
        )
    )
    + [("element", None), ("element", ""), ("aromatic", "False"), ("aromatic", 0)],
)
def test_checked_factory_rejects_invalid_atom_labels(key, value):
    graph = water()
    graph.nodes[1][key] = value
    with pytest.raises(LLGError) as error:
        LewisLabelledGraph.from_electron_graph(graph)
    assert error.value.issues[0].code is LLGIssueCode.INVALID_ELECTRON_STATE


@pytest.mark.parametrize(
    "sigma,pi", [(2, 0), (1, -1), (float("nan"), 0), (1, float("inf")), (0, 0), (True, 0), (0.5, 0.5), (1, 0.5)]
)
def test_checked_factory_rejects_invalid_bond_occupancies(sigma, pi):
    graph = water()
    graph.add_node(2, **graph.nodes[1])
    graph.add_edge(1, 2, sigma_order=sigma, pi_order=pi)
    with pytest.raises(LLGError) as error:
        LewisLabelledGraph.from_electron_graph(graph)
    assert error.value.issues[0].code is LLGIssueCode.INVALID_ELECTRON_STATE


@pytest.mark.parametrize("contract", ["bare", "role", "cached"])
def test_wildcard_contracts_cannot_be_erased_into_exact_lewis_objects(contract):
    graph = water()
    if contract == "bare":
        graph.nodes[1]["element"] = "*"
    elif contract == "role":
        graph.nodes[1]["wildcard_role"] = "query_atom"
    else:
        graph.nodes[1]["_wildcard_constraint"] = ("query_atom",)
    symbolic = LewisLabelledGraph.from_networkx(graph, ELECTRON_LLG_SCHEMA)
    for operation in (
        lambda: LewisLabelledGraph.from_electron_graph(graph),
        lambda: LLGMorphism.identity(symbolic, exact=True),
        lambda: symbolic.is_isomorphic(symbolic, exact=True),
    ):
        with pytest.raises(LLGError) as error:
            operation()
        assert error.value.issues[0].code is LLGIssueCode.SYMBOLIC_STATE_UNSUPPORTED


def test_isotope_schema_is_selected_or_requested_without_losing_identity():
    graph = water()
    graph.nodes[1]["isotope"] = 18
    for options in ({}, {"schema": ISOTOPE_ELECTRON_LLG_SCHEMA}):
        value = LewisLabelledGraph.from_electron_graph(graph, **options)
        assert value.schema == ISOTOPE_ELECTRON_LLG_SCHEMA
        assert value.node_labels(1, semantic=True)["isotope"] == 18
    with pytest.raises(LLGError) as error:
        LewisLabelledGraph.from_electron_graph(graph, schema=ELECTRON_LLG_SCHEMA)
    assert error.value.issues[0].code is LLGIssueCode.LOSSY_ELECTRON_ADAPTER


def test_nonstereo_checked_factory_does_not_silently_erase_stereo():
    graph = water()
    graph.graph["stereo_descriptors"] = {"atom:1": object()}
    with pytest.raises(LLGError) as error:
        LewisLabelledGraph.from_electron_graph(graph)
    assert error.value.issues[0].code is LLGIssueCode.STEREO_STATE_UNSUPPORTED


def test_disjoint_union_is_not_a_pushout_among_injective_arrows_only():
    # Over the empty graph the two singleton injections agree. Both also map
    # into the same singleton. The required union mediator would collapse
    # two vertices, so it is a morphism only in a larger ambient category.
    empty, left, right = pattern(()), pattern((1,)), pattern((2,))
    union, target = pattern((1, 2)), pattern((10,))
    LLGMorphism(empty, left, {}, exact=True)
    LLGMorphism(empty, right, {}, exact=True)
    LLGMorphism(left, union, {1: 1}, exact=True)
    LLGMorphism(right, union, {2: 2}, exact=True)
    LLGMorphism(left, target, {1: 10}, exact=True)
    LLGMorphism(right, target, {2: 10}, exact=True)
    with pytest.raises(LLGError) as error:
        LLGMorphism(union, target, {1: 10, 2: 10}, exact=True)
    assert error.value.issues[0].code is LLGIssueCode.NON_INJECTIVE
