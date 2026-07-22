"""Canonical SynGraph stereo identity and exact-collision invariants."""

from __future__ import annotations

import networkx as nx
import pytest

from synkit.Graph.Stereo import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptor_id,
    stereo_isomorphic,
)
from synkit.Graph.canon_graph import GraphCanonicaliser
from synkit.Graph.syn_graph import SynGraph

ATOM_CENTERED = (
    TetrahedralStereo,
    SquarePlanarStereo,
    TrigonalBipyramidalStereo,
    OctahedralStereo,
)


def _descriptor_graph(descriptor, *, elements=None):
    graph = nx.Graph()
    atom_ids = sorted(descriptor.dependencies)
    elements = elements or {
        atom_id: f"E{index}" for index, atom_id in enumerate(atom_ids)
    }
    graph.add_nodes_from(
        (
            atom_id,
            {
                "atom_map": atom_id,
                "element": elements[atom_id],
                "charge": 0,
                "lone_pairs": 0,
                "radical": 0,
                "aromatic": False,
                "hcount": 0,
            },
        )
        for atom_id in atom_ids
    )
    if isinstance(descriptor, ATOM_CENTERED):
        graph.add_edges_from(
            (descriptor.atoms[0], reference, {"order": 1})
            for reference in descriptor.atoms[1:]
            if isinstance(reference, int)
        )
    else:
        left, right = descriptor.atoms[2:4]
        graph.add_edge(left, right, order=1)
        graph.add_edges_from(
            (left, reference, {"order": 1})
            for reference in descriptor.atoms[:2]
            if isinstance(reference, int)
        )
        graph.add_edges_from(
            (right, reference, {"order": 1})
            for reference in descriptor.atoms[4:]
            if isinstance(reference, int)
        )
    graph.graph["stereo_descriptors"] = {descriptor_id(descriptor): descriptor}
    return graph


def _relabel_descriptor_graph(graph, descriptor, mapping):
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    for old, new in mapping.items():
        relabeled.nodes[new]["atom_map"] = new
    mapped_descriptor = descriptor.relabel(mapping)
    relabeled.graph["stereo_descriptors"] = {
        descriptor_id(mapped_descriptor): mapped_descriptor
    }
    return relabeled, mapped_descriptor


def test_stereo_isomorphism_ignores_aromatic_kekule_phase():
    first = nx.Graph()
    second = nx.Graph()
    for graph in (first, second):
        graph.add_node(
            1,
            element="C",
            charge=0,
            lone_pairs=0,
            radical=0,
            aromatic=True,
            hcount=1,
        )
        graph.add_node(
            2,
            element="C",
            charge=0,
            lone_pairs=0,
            radical=0,
            aromatic=True,
            hcount=1,
        )
    first.add_edge(1, 2, order=1.5, sigma_order=1.0, pi_order=1.0)
    second.add_edge(1, 2, order=1.5, sigma_order=1.0, pi_order=0.0)

    assert stereo_isomorphic(first, second)

    second.edges[1, 2]["order"] = 1.0
    assert not stereo_isomorphic(first, second)


SIX_CLASS_DESCRIPTORS = (
    TetrahedralStereo((1, 2, 3, 4, 5), 1),
    SquarePlanarStereo((1, 2, 3, 4, 5), 0),
    TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), -1),
    OctahedralStereo((1, 2, 3, 4, 5, 6, 7), 1),
    PlanarBondStereo((1, 2, 3, 4, 5, 6), 0),
    AtropBondStereo((1, 2, 3, 4, 5, 6), -1),
)


@pytest.mark.parametrize("descriptor", SIX_CLASS_DESCRIPTORS)
def test_six_class_syn_graph_identity_is_map_invariant(descriptor):
    graph = _descriptor_graph(descriptor)
    mapping = {atom_id: atom_id + 100 for atom_id in descriptor.dependencies}
    relabeled, _mapped_descriptor = _relabel_descriptor_graph(
        graph,
        descriptor,
        mapping,
    )
    first = SynGraph(graph)
    second = SynGraph(relabeled)

    assert first.stereo_form == second.stereo_form
    assert first.stereo_signature == second.stereo_signature
    assert first.signature == second.signature
    assert first == second
    assert hash(first) == hash(second)
    assert stereo_isomorphic(graph, relabeled)


def _distinct_orientation(descriptor):
    if isinstance(descriptor, SquarePlanarStereo):
        atoms = (
            descriptor.atoms[0],
            descriptor.atoms[2],
            descriptor.atoms[1],
            *descriptor.atoms[3:],
        )
        return SquarePlanarStereo(atoms, 0)
    return descriptor.invert()


@pytest.mark.parametrize("descriptor", SIX_CLASS_DESCRIPTORS)
def test_six_class_non_equivalent_orientations_have_distinct_identity(descriptor):
    graph = _descriptor_graph(descriptor)
    distinct = graph.copy()
    orientation = _distinct_orientation(descriptor)
    distinct.graph["stereo_descriptors"] = {descriptor_id(orientation): orientation}

    first = SynGraph(graph)
    second = SynGraph(distinct)

    assert first.structural_signature == second.structural_signature
    assert first.stereo_signature != second.stereo_signature
    assert first.signature != second.signature
    assert first != second
    assert not stereo_isomorphic(graph, distinct)


def test_unknown_specified_absent_and_virtual_kinds_remain_distinct():
    specified = TetrahedralStereo((1, 2, 3, 4, "@H:1"), 1)
    unknown = TetrahedralStereo(specified.atoms, None)
    lone_pair = TetrahedralStereo((1, 2, 3, 4, "@LP:1"), 1)
    base = _descriptor_graph(specified)
    variants = []
    for descriptor in (specified, unknown, lone_pair):
        graph = base.copy()
        graph.graph["stereo_descriptors"] = {descriptor_id(descriptor): descriptor}
        variants.append(SynGraph(graph))
    absent = base.copy()
    absent.graph["stereo_descriptors"] = {}
    variants.append(SynGraph(absent))

    assert len({variant.signature for variant in variants}) == 4
    assert len(set(variants)) == 4
    assert stereo_isomorphic(
        variants[1].raw,
        variants[0].raw,
        unknown_policy="wildcard",
    )
    assert not stereo_isomorphic(variants[1].raw, variants[0].raw)


def test_symmetric_square_planar_cis_trans_pattern_is_not_merged():
    cis = SquarePlanarStereo((1, 2, 3, 4, 5), 0)
    trans = SquarePlanarStereo((1, 2, 4, 3, 5), 0)
    elements = {1: "M", 2: "A", 3: "A", 4: "B", 5: "B"}
    cis_graph = _descriptor_graph(cis, elements=elements)
    trans_graph = _descriptor_graph(trans, elements=elements)
    symmetric_mapping = {1: 101, 2: 103, 3: 102, 4: 105, 5: 104}
    relabeled_cis, _descriptor = _relabel_descriptor_graph(
        cis_graph,
        cis,
        symmetric_mapping,
    )

    assert SynGraph(cis_graph) == SynGraph(relabeled_cis)
    assert stereo_isomorphic(cis_graph, relabeled_cis)
    assert (
        SynGraph(cis_graph).stereo_signature != SynGraph(trans_graph).stereo_signature
    )
    assert SynGraph(cis_graph) != SynGraph(trans_graph)
    assert not stereo_isomorphic(cis_graph, trans_graph)


def test_named_its_side_and_transition_registries_relabel_together():
    graph = nx.Graph()
    for node in range(1, 6):
        graph.add_node(
            node,
            atom_map=(node, node + 10),
            element=f"E{node}",
        )
    graph.add_edges_from((1, node, {"order": (1, 1)}) for node in range(2, 6))
    reactant = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    product = TetrahedralStereo((11, 12, 13, 14, 15), -1)
    transition = SquarePlanarStereo((1, 2, 3, 4, 5), 0)
    graph.graph["stereo_descriptors"] = {
        "reactant": {"atom:1": reactant},
        "product": {"atom:11": product},
        "transition": {"atom:1": transition},
    }

    node_mapping = {node: node + 300 for node in graph}
    reactant_mapping = {node: node + 100 for node in range(1, 6)}
    product_mapping = {node + 10: node + 200 for node in range(1, 6)}
    relabeled = nx.relabel_nodes(graph, node_mapping, copy=True)
    for old, new in node_mapping.items():
        relabeled.nodes[new]["atom_map"] = (
            reactant_mapping[old],
            product_mapping[old + 10],
        )
    relabeled.graph["stereo_descriptors"] = {
        "reactant": {
            "atom:101": reactant.relabel(reactant_mapping),
        },
        "product": {
            "atom:201": product.relabel(product_mapping),
        },
        "transition": {
            "atom:301": transition.relabel(node_mapping),
        },
    }

    first = SynGraph(graph)
    second = SynGraph(relabeled)

    assert tuple(layer for layer, _registry in first.stereo_form) == (
        "product",
        "reactant",
        "transition",
    )
    assert first.signature == second.signature
    assert first == second
    assert hash(first) == hash(second)
    assert stereo_isomorphic(graph, relabeled)

    changed = relabeled.copy()
    changed.graph["stereo_descriptors"] = {
        **relabeled.graph["stereo_descriptors"],
        "transition": {
            "atom:301": SquarePlanarStereo((301, 303, 302, 304, 305), 0),
        },
    }
    assert SynGraph(graph) != SynGraph(changed)


def test_exact_isomorphism_resolves_a_wl_signature_collision():
    graph = nx.frucht_graph()
    nx.set_node_attributes(graph, {node: node + 1 for node in graph}, "atom_map")
    first_descriptor = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    second_descriptor = TetrahedralStereo((2, 1, 3, 4, 5), 1)
    first_graph = graph.copy()
    second_graph = graph.copy()
    first_graph.graph["stereo_descriptors"] = {"atom:1": first_descriptor}
    second_graph.graph["stereo_descriptors"] = {"atom:2": second_descriptor}
    first = SynGraph(first_graph)
    second = SynGraph(second_graph)

    # All nodes of this asymmetric regular graph have the same 1-WL colour,
    # so the fast signatures collide. Exact stereo isomorphism is authoritative.
    assert first.signature == second.signature
    assert hash(first) == hash(second)
    assert first != second
    assert not stereo_isomorphic(first_graph, second_graph)


def test_plain_graph_keeps_the_legacy_structural_signature():
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, "C", "element")
    canonicaliser = GraphCanonicaliser()
    wrapped = SynGraph(graph, canonicaliser=canonicaliser)

    assert wrapped.stereo_form == ()
    assert wrapped.stereo_signature is None
    assert wrapped.signature == canonicaliser.canonical_signature(graph)
