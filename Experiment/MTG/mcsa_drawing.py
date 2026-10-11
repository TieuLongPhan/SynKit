"""Restricted, explicit translation of the selected M-CSA Marvin drawings.

This reads data, not converter source. Aliases remain annotations of depicted
fragments. Unexpanded H counts are inferred from drawn charge/electron fields;
stereo, noncovalent contacts and the rest of the protein are outside the model.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import networkx as nx

from synkit.Graph.Morphism import (
    ELECTRON_LLG_SCHEMA,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)

VALENCE = {"H": 1, "C": 4, "N": 5, "O": 6, "P": 5, "S": 6}


def read_drawing(path: Path):
    """Read the frozen unit pair-arrow domain, refusing unsupported primitives."""
    root = ET.parse(path).getroot()
    graph = nx.Graph()
    for atom in root.iter("atom"):
        data = atom.attrib
        node = int(data["id"][1:])
        if node in graph or data["elementType"] not in VALENCE:
            raise ValueError(
                "Unknown or duplicate atom in the selected drawing domain."
            )
        if data.get("radical", "0") != "0":
            raise ValueError("This selected converter does not infer radical states.")
        graph.add_node(
            node,
            element=data["elementType"],
            lone_pairs=int(data.get("lonePair", 0)),
            radical=0,
            valence_electrons=VALENCE[data["elementType"]],
            aromatic=False,
            charge=int(data.get("formalCharge", 0)),
            source_alias=data.get("mrvAlias"),
            source_extra=data.get("mrvExtraLabel"),
        )
    for bond in root.iter("bond"):
        data = bond.attrib
        a, b = [int(ref[1:]) for ref in data["atomRefs2"].split()]
        order = int(data["order"])
        if order not in (1, 2, 3) or graph.has_edge(a, b):
            raise ValueError("The selected domain requires simple integer-order bonds.")
        if a == b or a not in graph or b not in graph:
            raise ValueError("A bond must have distinct known endpoints.")
        graph.add_edge(a, b, sigma_order=1, pi_order=order - 1)
    for node, attrs in graph.nodes(data=True):
        order = sum(
            d["sigma_order"] + d["pi_order"] for _, _, d in graph.edges(node, data=True)
        )
        attrs["hcount"] = (
            attrs["valence_electrons"]
            - 2 * attrs["lone_pairs"]
            - order
            - attrs["charge"]
        )
        if attrs["hcount"] < 0 or (attrs["element"] == "H" and attrs["hcount"]):
            raise ValueError(
                "Inferred unexpanded H count is outside the selected domain."
            )
    return derive_electron_labeled_graph(graph), _read_arrows(root, graph)


def _read_arrows(root, graph):
    arrows = []
    for flow in root.iter("MEFlow"):
        if flow.attrib.get("headFlags") or flow.attrib.get("arrowType"):
            raise ValueError("Explicit alternate arrow styles need another translator.")
        positions = []
        for point in flow:
            refs = point.attrib.get("atomRefs", point.attrib.get("atomRef", "")).split()
            positions.append(tuple(int(ref.rsplit(".a", 1)[-1]) for ref in refs))
        if len(positions) != 2 or any(len(p) not in (1, 2) for p in positions):
            raise ValueError(
                "Each selected arrow needs an atom/bond source and target."
            )
        if any(node not in graph for position in positions for node in position):
            raise ValueError("An arrow references an unknown atom.")
        arrows.append(tuple(positions))
    return tuple(arrows)


def _locus(graph, refs, *, source):
    if len(refs) == 1:
        return "lone_pairs", refs[0]
    edge = frozenset(refs)
    order = (
        graph.edges[tuple(refs)]["sigma_order"] + graph.edges[tuple(refs)]["pi_order"]
        if graph.has_edge(*refs)
        else 0
    )
    if source and not order:
        raise ValueError("An arrow cannot consume an absent bond.")
    kind = (
        ("pi_order" if order > 1 else "sigma_order")
        if source
        else ("pi_order" if order else "sigma_order")
    )
    return kind, edge


def push_group(graph, arrows):
    """Check common-prestate availability and commit the declared pair arrows."""
    consumed, delta = Counter(), Counter()
    for source, target in arrows:
        start, end = _locus(graph, source, source=True), _locus(
            graph, target, source=False
        )
        consumed[start] += 1
        delta[start] -= 1
        delta[end] += 1
    for (kind, carrier), amount in consumed.items():
        available = (
            graph.nodes[carrier][kind]
            if kind == "lone_pairs"
            else graph.edges[tuple(carrier)][kind]
        )
        if available < amount:
            raise ValueError(
                "A drawn group consumes unavailable common-prestate electrons."
            )
    result = graph.copy()
    for (kind, carrier), amount in delta.items():
        if kind == "lone_pairs":
            result.nodes[carrier][kind] += amount
        else:
            a, b = tuple(carrier)
            if not result.has_edge(a, b):
                result.add_edge(a, b, sigma_order=0, pi_order=0)
            result.edges[a, b][kind] += amount
    for a, b, attrs in list(result.edges(data=True)):
        if attrs["sigma_order"] < 0 or attrs["pi_order"] < 0:
            raise ValueError("A drawn group creates a negative bond resource.")
        if attrs["sigma_order"] + attrs["pi_order"] == 0:
            result.remove_edge(a, b)
        else:
            attrs.pop("bond_order", None)
            attrs.pop("order", None)
            attrs.pop("kekule_order", None)
    for _, attrs in result.nodes(data=True):
        attrs.pop("charge", None)
        attrs.pop("bond_order_sum", None)
    return derive_electron_labeled_graph(result)


def value(graph):
    return LewisLabelledGraph.from_networkx(graph, ELECTRON_LLG_SCHEMA)


def observed_atoms(graph):
    """Identify anatomical query roles in the depicted active-site fragment."""
    sulfur = [n for n, d in graph.nodes(data=True) if d["element"] == "S"]
    carbon = [
        n
        for n, d in graph.nodes(data=True)
        if d["element"] == "C"
        and d.get("source_extra") == "res:Asp169A"
        and sum(graph.nodes[v]["element"] == "O" for v in graph.neighbors(n)) >= 1
    ]
    if len(sulfur) != 1 or len(carbon) != 1:
        raise ValueError("The selected observed atom roles must be unambiguous.")
    return carbon[0], sulfur[0]


def align_drawing(host, drawing, limit=32):
    """Retain exact IDs when valid, otherwise enumerate bounded strict bindings."""
    keys = (*ELECTRON_LLG_SCHEMA.declared_node_keys, "source_alias", "source_extra")

    def node_match(a, b):
        return all(a.get(k) == b.get(k) for k in keys)

    def edge_match(a, b):
        return all(a[k] == b[k] for k in ("sigma_order", "pi_order"))

    nodes = set(drawing)
    if nodes <= set(host) and nx.utils.graphs_equal(host.subgraph(nodes), drawing):
        mappings = [{n: n for n in drawing}]
    else:
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            host, drawing, node_match=node_match, edge_match=edge_match
        )
        mappings = []
        for mapping in matcher.subgraph_isomorphisms_iter():
            inverse = {v: k for k, v in mapping.items()}
            if any(
                (a in inverse.values()) != (b in inverse.values())
                for a, b in host.edges
            ):
                continue  # Omitted carriers must be complete untouched components.
            mappings.append(inverse)
            if len(mappings) > limit:
                raise ValueError(
                    "Correspondence enumeration exceeded the declared bound."
                )
    if not mappings:
        raise ValueError(
            "The next drawing does not strictly bind into the retained host."
        )
    target_roles, host_roles = observed_atoms(drawing), observed_atoms(host)
    if any(
        tuple(mapping[n] for n in target_roles) != host_roles for mapping in mappings
    ):
        raise ValueError("Correspondence alternatives disagree on an observed anchor.")
    mappings.sort(
        key=lambda mapping: (
            -sum(k == v for k, v in mapping.items()),
            tuple(sorted(mapping.items())),
        )
    )
    return mappings
