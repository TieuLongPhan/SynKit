"""Permutation determinism gates for configured tetrahedral stereographs."""

from __future__ import annotations

from itertools import permutations
import os
import random
import subprocess
import sys

import networkx as nx

from synkit.Graph.Canon.exact import ExactColoredGraphCanonicalizer
from synkit.Graph.Stereo import TetrahedralStereo
from synkit.Graph.Stereo.canonical import (
    StereoSlotVertex,
    StereoTupleVertex,
    canonicalize_tetrahedral_stereograph,
    expand_tetrahedral_stereograph,
)


def _star(colors: tuple[str, str, str, str]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, color="center")
    for reference, color in enumerate(colors, start=1):
        graph.add_node(reference, color=color)
        graph.add_edge(0, reference, color="single")
    return graph


def _canonical(
    graph: nx.Graph,
    descriptor: TetrahedralStereo,
):
    return canonicalize_tetrahedral_stereograph(
        graph,
        (descriptor,),
        atom_color="color",
        bond_color="color",
    )


def test_all_120_atom_relabelings_preserve_tetrahedral_code() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)
    expected = _canonical(graph, descriptor).canonical_code
    target_handles = (10, 20, 30, 40, 50)

    for images in permutations(target_handles):
        mapping = dict(zip(graph, images))
        relabeled = nx.relabel_nodes(graph, mapping, copy=True)
        transported = descriptor.relabel(mapping)
        assert (
            _canonical(
                relabeled,
                transported,
            ).canonical_code
            == expected
        )


def test_auxiliary_node_and_edge_insertion_order_is_irrelevant() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)
    auxiliary = expand_tetrahedral_stereograph(
        graph,
        (descriptor,),
        atom_color="color",
        bond_color="color",
    )
    expected = ExactColoredGraphCanonicalizer(
        auxiliary,
        node_color="color",
        edge_color=None,
    ).canonicalize()
    nodes = list(auxiliary)
    edges = list(auxiliary.edges())

    for seed in range(32):
        generator = random.Random(seed)
        generator.shuffle(nodes)
        generator.shuffle(edges)
        rebuilt = nx.Graph()
        for node in nodes:
            rebuilt.add_node(node, **auxiliary.nodes[node])
        for edge in edges:
            rebuilt.add_edge(*edge)
        observed = ExactColoredGraphCanonicalizer(
            rebuilt,
            node_color="color",
            edge_color=None,
        ).canonicalize()
        assert observed.canonical_code == expected.canonical_code


def test_orbit_tuple_enumeration_handles_are_not_semantic() -> None:
    graph = _star(("fluorine", "chlorine", "bromine", "hydrogen"))
    auxiliary = expand_tetrahedral_stereograph(
        graph,
        (TetrahedralStereo((0, 1, 2, 3, 4), 1),),
        atom_color="color",
        bond_color="color",
    )
    expected = ExactColoredGraphCanonicalizer(
        auxiliary,
        node_color="color",
        edge_color=None,
    ).canonicalize()

    for seed in range(32):
        images = list(range(12))
        random.Random(seed).shuffle(images)
        mapping = {}
        for source, target in enumerate(images):
            mapping[StereoTupleVertex(0, source)] = StereoTupleVertex(
                0,
                target,
            )
            for position in range(4):
                mapping[StereoSlotVertex(0, source, position)] = StereoSlotVertex(
                    0, target, position
                )
        relabeled = nx.relabel_nodes(auxiliary, mapping, copy=True)
        observed = ExactColoredGraphCanonicalizer(
            relabeled,
            node_color="color",
            edge_color=None,
        ).canonicalize()
        assert observed.canonical_code == expected.canonical_code


def test_every_three_locus_descriptor_order_has_identical_code() -> None:
    graph = nx.Graph()
    descriptors = []
    for locus, offset in enumerate((0, 10, 20)):
        center = offset
        graph.add_node(center, color=f"center:{locus}")
        references = []
        for position in range(1, 5):
            reference = offset + position
            references.append(reference)
            graph.add_node(
                reference,
                color=f"ligand:{locus}:{position}",
            )
            graph.add_edge(center, reference, color="single")
        descriptors.append(
            TetrahedralStereo(
                (center, *references),
                1 if locus != 1 else -1,
            )
        )

    observed = {
        canonicalize_tetrahedral_stereograph(
            graph,
            order,
            atom_color="color",
            bond_color="color",
        ).canonical_code
        for order in permutations(descriptors)
    }

    assert len(observed) == 1


def test_symmetric_atom_order_is_only_a_witness_modulo_automorphism() -> None:
    graph = _star(("same", "same", "same", "same"))
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)
    rebuilt = nx.Graph()
    for node in reversed(tuple(graph)):
        rebuilt.add_node(node, **graph.nodes[node])
    for edge in reversed(tuple(graph.edges())):
        rebuilt.add_edge(*edge, **graph.edges[edge])

    first = _canonical(graph, descriptor)
    second = _canonical(rebuilt, descriptor)
    induced = {
        first.atom_order[index]: second.atom_order[index]
        for index in range(len(first.atom_order))
    }

    assert first.canonical_code == second.canonical_code
    assert induced in tuple(witness.as_dict() for witness in first.atom_automorphisms)


def test_stereograph_code_is_deterministic_across_hash_seeds() -> None:
    script = """
import networkx as nx
from synkit.Graph.Stereo import TetrahedralStereo
from synkit.Graph.Stereo.canonical import canonicalize_tetrahedral_stereograph
ids = {"center": 0, "fluorine": 1, "chlorine": 2, "bromine": 3, "hydrogen": 4}
colors = {
    "center": "center",
    "fluorine": "fluorine",
    "chlorine": "chlorine",
    "bromine": "bromine",
    "hydrogen": "hydrogen",
}
names = list(set(ids))
graph = nx.Graph()
for name in names:
    graph.add_node(ids[name], color=colors[name])
for name in set(ids) - {"center"}:
    graph.add_edge(0, ids[name], color="single")
result = canonicalize_tetrahedral_stereograph(
    graph,
    (TetrahedralStereo((0, 1, 2, 3, 4), 1),),
    atom_color="color",
    bond_color="color",
)
print(",".join(names))
print(result.canonical_code)
"""
    construction_orders = []
    certificates = []
    for seed in ("1", "947"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        output = subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=environment,
            text=True,
        )
        construction_order, certificate = output.split("\n", maxsplit=1)
        construction_orders.append(construction_order)
        certificates.append(certificate)

    assert construction_orders[0] != construction_orders[1]
    assert certificates[0] == certificates[1]


def test_isotope_charge_and_bond_order_remain_semantic() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(
        (
            (0, {"element": "C", "charge": 0}),
            (1, {"element": "F", "charge": 0}),
            (2, {"element": "Cl", "charge": 0}),
            (3, {"element": "Br", "charge": 0}),
            (4, {"element": "H", "charge": 0}),
        )
    )
    for reference in range(1, 5):
        graph.add_edge(0, reference, order=1.0)
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)
    expected = canonicalize_tetrahedral_stereograph(
        graph,
        (descriptor,),
    ).canonical_code

    isotope = graph.copy()
    isotope.nodes[4]["isotope"] = 2
    charge = graph.copy()
    charge.nodes[1]["charge"] = -1
    bond = graph.copy()
    bond.edges[0, 1]["order"] = 2.0

    for changed in (isotope, charge, bond):
        assert (
            canonicalize_tetrahedral_stereograph(
                changed,
                (descriptor,),
            ).canonical_code
            != expected
        )
