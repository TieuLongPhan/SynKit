from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, List
import logging

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CRNGraphInfo:
    species_nodes: List[Hashable]
    rule_nodes: List[Hashable]
    is_dag: bool


def node_sort_key(graph: nx.DiGraph, node: Hashable):
    data = graph.nodes[node]
    kind = data.get("kind")
    if kind == "rule":
        return (
            1,
            data.get("step", 10**9),
            data.get("rule_index", 10**9),
            data.get("app_index", 10**9),
            str(data.get("label", "")),
            str(node),
        )
    return (
        0,
        data.get("step", -1),
        str(data.get("label", "")),
        str(data.get("smiles", "")),
        str(node),
    )


def validate_crn_graph(graph: nx.DiGraph, *, strict: bool = True) -> CRNGraphInfo:
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("CRNVis expects a networkx.DiGraph.")

    valid_kinds = {"species", "rule"}
    invalid_kind_nodes = [
        n for n, d in graph.nodes(data=True) if d.get("kind") not in valid_kinds
    ]
    if invalid_kind_nodes:
        msg = (
            "Found nodes with invalid or missing 'kind': "
            f"{invalid_kind_nodes!r}. Expected only 'species' or 'rule'."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    species_nodes = sorted(
        [n for n, d in graph.nodes(data=True) if d.get("kind") == "species"],
        key=lambda n: node_sort_key(graph, n),
    )
    rule_nodes = sorted(
        [n for n, d in graph.nodes(data=True) if d.get("kind") == "rule"],
        key=lambda n: node_sort_key(graph, n),
    )

    invalid_edges = []
    unknown_roles = []
    for u, v, d in graph.edges(data=True):
        role = d.get("role")
        u_kind = graph.nodes[u].get("kind")
        v_kind = graph.nodes[v].get("kind")
        if role == "reactant":
            if not (u_kind == "species" and v_kind == "rule"):
                invalid_edges.append((u, v, role))
        elif role == "product":
            if not (u_kind == "rule" and v_kind == "species"):
                invalid_edges.append((u, v, role))
        elif role is not None:
            unknown_roles.append((u, v, role))

    if invalid_edges:
        msg = (
            "Found edges whose direction is inconsistent with their role: "
            f"{invalid_edges!r}."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    if unknown_roles:
        logger.warning(
            "Found edges with unknown roles; they will be drawn with fallback style: %r",
            unknown_roles,
        )

    return CRNGraphInfo(
        species_nodes=species_nodes,
        rule_nodes=rule_nodes,
        is_dag=nx.is_directed_acyclic_graph(graph),
    )
