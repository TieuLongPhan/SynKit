"""Category laws and adapter contracts for finite Lewis-labelled graphs."""

from __future__ import annotations

import networkx as nx
import pytest

from synkit.Graph.Morphism import (
    COMMON_CHEMICAL_SCHEMA,
    ELECTRON_LLG_SCHEMA,
    LLGError,
    LLGIssueCode,
    LLGMorphism,
    LabelSchema,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
    llg_from_its,
)
from synkit.IO import rsmi_to_its

SCHEMA = LabelSchema(
    node_identity=("kind",),
    node_state=("state",),
    edge_state=("weight",),
    node_annotations=("atom_map",),
    name="test/1",
)


def _path(name: str, nodes: tuple[object, ...]) -> LewisLabelledGraph:
    graph = nx.Graph()
    for index, node in enumerate(nodes):
        graph.add_node(node, kind="X", state=index, atom_map=100 + index)
    for left, right in zip(nodes, nodes[1:]):
        graph.add_edge(left, right, weight=1)
    return LewisLabelledGraph.from_networkx(graph, SCHEMA, name=name)


def _constant_path(name: str, nodes: tuple[object, ...]) -> LewisLabelledGraph:
    graph = nx.Graph()
    graph.add_nodes_from((node, {"kind": "X", "state": 0}) for node in nodes)
    graph.add_edges_from(
        (left, right, {"weight": 1}) for left, right in zip(nodes, nodes[1:])
    )
    return LewisLabelledGraph.from_networkx(graph, SCHEMA, name=name)


def test_schema_classes_are_disjoint() -> None:
    with pytest.raises(LLGError) as error:
        LabelSchema(node_identity=("element",), node_state=("element",))
    assert error.value.issues[0].code is LLGIssueCode.SCHEMA_OVERLAP


def test_objects_reject_non_simple_or_incompletely_labeled_graphs() -> None:
    directed = nx.DiGraph()
    directed.add_node(1, kind="X", state=0)
    with pytest.raises(LLGError) as error:
        LewisLabelledGraph.from_networkx(directed, SCHEMA)
    assert error.value.issues[0].code is LLGIssueCode.DIRECTED_GRAPH

    loop = nx.Graph()
    loop.add_node(1, kind="X", state=0)
    loop.add_edge(1, 1, weight=1)
    with pytest.raises(LLGError) as error:
        LewisLabelledGraph.from_networkx(loop, SCHEMA)
    assert error.value.issues[0].code is LLGIssueCode.SELF_LOOP

    missing = nx.Graph()
    missing.add_node(1, kind="X")
    with pytest.raises(LLGError) as error:
        LewisLabelledGraph.from_networkx(missing, SCHEMA)
    assert error.value.issues[0].code is LLGIssueCode.MISSING_NODE_LABEL


def test_annotation_and_carrier_names_are_not_graph_semantics() -> None:
    first = _path("first", (1, 2, 3))
    graph = nx.Graph()
    graph.add_node("a", kind="X", state=0, atom_map=999, color="red")
    graph.add_node("b", kind="X", state=1, atom_map=998, color="green")
    graph.add_node("c", kind="X", state=2, atom_map=997, color="blue")
    graph.add_edge("a", "b", weight=1, source_id="left")
    graph.add_edge("b", "c", weight=1, source_id="right")
    second = LewisLabelledGraph.from_networkx(graph, SCHEMA, name="second")

    assert first.is_isomorphic(second)
    isomorphism = LLGMorphism(first, second, {1: "a", 2: "b", 3: "c"})
    assert isomorphism.is_isomorphism


def test_structural_absence_is_not_a_present_zero_labeled_edge() -> None:
    absent = nx.Graph()
    absent.add_nodes_from(
        ((1, {"kind": "X", "state": 0}), (2, {"kind": "X", "state": 0}))
    )
    present = absent.copy()
    present.add_edge(1, 2, weight=0)

    left = LewisLabelledGraph.from_networkx(absent, SCHEMA)
    right = LewisLabelledGraph.from_networkx(present, SCHEMA)

    assert not left.is_isomorphic(right)
    assert frozenset((1, 2)) not in left.edge_keys
    assert right.edge_labels(frozenset((1, 2)), semantic=True) == {"weight": 0}


def test_morphisms_are_total_injective_incidence_and_label_preserving() -> None:
    source = _constant_path("source", (1, 2))
    target = _constant_path("target", (10, 20, 30))
    valid = LLGMorphism(source, target, {1: 10, 2: 20})
    assert valid.edge_mapping == {frozenset((1, 2)): frozenset((10, 20))}

    with pytest.raises(LLGError) as partial:
        LLGMorphism(source, target, {1: 10})
    assert partial.value.issues[0].code is LLGIssueCode.PARTIAL_MAPPING

    with pytest.raises(LLGError) as collapse:
        LLGMorphism(source, target, {1: 10, 2: 10})
    assert collapse.value.issues[0].code is LLGIssueCode.NON_INJECTIVE

    disconnected_graph = target.to_networkx()
    disconnected_graph.remove_edge(10, 20)
    disconnected = LewisLabelledGraph.from_networkx(disconnected_graph, SCHEMA)
    with pytest.raises(LLGError) as incidence:
        LLGMorphism(source, disconnected, {1: 10, 2: 20})
    assert incidence.value.issues[0].code is LLGIssueCode.MISSING_EDGE

    changed_graph = target.to_networkx()
    changed_graph.nodes[10]["state"] = 2
    changed = LewisLabelledGraph.from_networkx(changed_graph, SCHEMA)
    with pytest.raises(LLGError) as labels:
        LLGMorphism(source, changed, {1: 10, 2: 20})
    assert labels.value.issues[0].code is LLGIssueCode.NODE_LABEL_MISMATCH


def test_identity_composition_and_associativity_laws_hold_literally() -> None:
    a = _constant_path("A", (1, 2))
    b = _constant_path("B", (10, 20, 30))
    c = _constant_path("C", (100, 200, 300, 400))
    d = _constant_path("D", (1000, 2000, 3000, 4000, 5000))
    f = LLGMorphism(a, b, {1: 10, 2: 20})
    g = LLGMorphism(b, c, {10: 100, 20: 200, 30: 300})
    h = LLGMorphism(c, d, {100: 1000, 200: 2000, 300: 3000, 400: 4000})

    assert LLGMorphism.identity(a).then(f) == f
    assert f.then(LLGMorphism.identity(b)) == f
    assert f.then(g).then(h) == f.then(g.then(h))


def test_independent_relabeling_is_covariant_and_non_mutating() -> None:
    source = _constant_path("source", (1, 2))
    target = _constant_path("target", (10, 20, 30))
    morphism = LLGMorphism(source, target, {1: 10, 2: 20})
    before_source = source.to_networkx()
    before_target = target.to_networkx()

    relabeled = morphism.relabel(
        {1: "s1", 2: "s2"},
        {10: "t1", 20: "t2", 30: "t3"},
    )

    assert relabeled.mapping == {"s1": "t1", "s2": "t2"}
    assert source.is_isomorphic(relabeled.source)
    assert target.is_isomorphic(relabeled.target)
    assert nx.utils.graphs_equal(source.to_networkx(), before_source)
    assert nx.utils.graphs_equal(target.to_networkx(), before_target)


def _electron_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(
        1,
        element="C",
        aromatic=False,
        hcount=3,
        radical=0,
        lone_pairs=0,
        valence_electrons=4,
        atom_map=91,
    )
    graph.add_node(
        2,
        element="C",
        aromatic=False,
        hcount=3,
        radical=0,
        lone_pairs=0,
        valence_electrons=4,
        atom_map=92,
    )
    graph.add_edge(1, 2, sigma_order=1.0, pi_order=0.0)
    return graph


def test_electron_derived_charge_and_total_order_are_deterministic() -> None:
    source = _electron_graph()
    derived = derive_electron_labeled_graph(source)
    llg = LewisLabelledGraph.from_networkx(derived, ELECTRON_LLG_SCHEMA)

    assert derived.nodes[1]["charge"] == 0
    assert derived.nodes[1]["bond_order_sum"] == 1.0
    assert derived.edges[1, 2]["order"] == 1.0
    assert llg.edge_labels(frozenset((1, 2)))["kekule_order"] == 1.0
    assert "charge" not in source.nodes[1]

    inconsistent = _electron_graph()
    inconsistent.nodes[1]["charge"] = 1
    with pytest.raises(LLGError) as error:
        derive_electron_labeled_graph(inconsistent)
    assert error.value.issues[0].code is LLGIssueCode.DERIVED_LABEL_MISMATCH


def test_tuple_and_typesgh_adapt_to_the_same_common_contract() -> None:
    reaction = "[CH3:1][OH:2]>>[CH2:1]=[O:2]"
    tuple_its = rsmi_to_its(reaction, format="tuple")
    legacy_its = rsmi_to_its(reaction, format="typesGH")

    for side in ("reactant", "product"):
        tuple_llg = llg_from_its(tuple_its, side)
        legacy_llg = llg_from_its(legacy_its, side)
        assert tuple_llg.schema is COMMON_CHEMICAL_SCHEMA
        assert legacy_llg.schema is COMMON_CHEMICAL_SCHEMA
        assert tuple_llg.is_isomorphic(legacy_llg)


def test_electron_complete_adapter_is_explicitly_lossy_for_typesgh() -> None:
    reaction = "[CH3:1][OH:2]>>[CH2:1]=[O:2]"
    tuple_its = rsmi_to_its(reaction, format="tuple")
    complete = llg_from_its(tuple_its, "product", electron_complete=True)
    assert complete.schema is ELECTRON_LLG_SCHEMA
    assert complete.node_labels(1)["charge"] == 0

    legacy = rsmi_to_its(reaction, format="typesGH")
    with pytest.raises(LLGError) as error:
        llg_from_its(legacy, "product", electron_complete=True)
    assert error.value.issues[0].code is LLGIssueCode.LOSSY_ELECTRON_ADAPTER
