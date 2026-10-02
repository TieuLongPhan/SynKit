from __future__ import annotations

"""Atom-map helpers and edge lookup utilities."""

from typing import Dict, Optional, Tuple

import networkx as nx

from .utils import normalize_scalar_attr


def node_to_amap(graph: nx.Graph, atom_map_key: str = "atom_map") -> Dict[int, int]:
    """Return node -> atom-map dictionary."""
    return {
        n: int(normalize_scalar_attr(graph.nodes[n].get(atom_map_key, n), n))
        for n in graph.nodes()
    }


def amap_to_node(graph: nx.Graph, atom_map_key: str = "atom_map") -> Dict[int, int]:
    """Return atom-map -> node dictionary."""
    return {
        int(normalize_scalar_attr(graph.nodes[n].get(atom_map_key, n), n)): n
        for n in graph.nodes()
    }


def edge_amap_key_from_nodes(
    graph: nx.Graph,
    edge: Tuple[int, int],
    atom_map_key: str = "atom_map",
) -> Tuple[int, int]:
    """Return a sorted atom-map edge key from a node edge."""
    n2a = node_to_amap(graph, atom_map_key=atom_map_key)
    u, v = edge
    return tuple(sorted((n2a[u], n2a[v])))


def edge_nodes_from_amap_key(
    graph: Optional[nx.Graph],
    amap_edge: Tuple[int, int],
    atom_map_key: str = "atom_map",
) -> Optional[Tuple[int, int]]:
    """Resolve an atom-map edge back to node ids."""
    if graph is None:
        return None
    a2n = amap_to_node(graph, atom_map_key=atom_map_key)
    a1, a2 = amap_edge
    if a1 not in a2n or a2 not in a2n:
        return None
    return tuple(sorted((a2n[a1], a2n[a2])))
