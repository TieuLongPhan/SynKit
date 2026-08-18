from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, List
import logging

import networkx as nx

from ..kinds import ALL_KINDS, is_reaction_node, is_species_node

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CRNGraphInfo:
    species_nodes: List[Hashable]
    rule_nodes: List[Hashable]
    is_dag: bool


def node_sort_key(graph: nx.DiGraph, node: Hashable):
    """Order reaction nodes after species nodes, deterministically.

    Reaction nodes are recognised through :func:`~synkit.CRN.kinds.is_reaction_node`,
    so ``kind="reaction"`` and the legacy ``kind="rule"`` sort identically. A
    direct ``kind == "rule"`` comparison here used to file ``kind="reaction"``
    nodes under the species branch, giving two different drawings of the same
    network.

    :param graph:
        Bipartite CRN graph.
    :type graph: networkx.DiGraph

    :param node:
        Node to build a sort key for.
    :type node: Hashable

    :return:
        Comparable sort key.
    :rtype: tuple
    """
    data = graph.nodes[node]
    if is_reaction_node(data):
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

    invalid_kind_nodes = [
        n
        for n, d in graph.nodes(data=True)
        if not (is_species_node(d) or is_reaction_node(d))
    ]
    if invalid_kind_nodes:
        msg = (
            "Found nodes with invalid or missing 'kind': "
            f"{invalid_kind_nodes!r}. Expected one of {sorted(ALL_KINDS)!r}."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    species_nodes = sorted(
        [n for n, d in graph.nodes(data=True) if is_species_node(d)],
        key=lambda n: node_sort_key(graph, n),
    )
    rule_nodes = sorted(
        [n for n, d in graph.nodes(data=True) if is_reaction_node(d)],
        key=lambda n: node_sort_key(graph, n),
    )

    invalid_edges = []
    unknown_roles = []
    for u, v, d in graph.edges(data=True):
        role = d.get("role")
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        if role == "reactant":
            if not (is_species_node(u_data) and is_reaction_node(v_data)):
                invalid_edges.append((u, v, role))
        elif role == "product":
            if not (is_reaction_node(u_data) and is_species_node(v_data)):
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
