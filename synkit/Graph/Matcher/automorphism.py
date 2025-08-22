"""automorphism.py
~~~~~~~~~~~~~~~~~~~
Utility for computing graph automorphisms and pruning redundant sub-graph
mappings equivalent under those symmetries.

This module provides the :class:`Automorphism` helper, which computes the
node-orbits of a graph and uses them to deduplicate subgraph-match mappings.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import networkx as nx
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)

# ---------------------------------------------------------------------------
# Typing aliases
# ---------------------------------------------------------------------------
NodeId = Union[int, str, Tuple, object]
MappingDict = Mapping[NodeId, NodeId]

__all__ = ["Automorphism", "NodeId", "MappingDict"]


class Automorphism:
    """
    Analyze the automorphism group of a graph and prune sub-graph mappings
    that are equivalent under those symmetries.

    Two nodes are in the same orbit if there exists an automorphism σ
    such that σ(u) = v.

    :param graph: The host graph for which to compute automorphisms.
    :type graph: nx.Graph
    :param node_attr_keys: Sequence of node attribute keys to preserve in matches.
        Defaults to ('element', 'charge').
    :type node_attr_keys: Sequence[str] | None
    :param edge_attr_keys: Sequence of edge attribute keys to preserve in matches.
        Defaults to ('order',).
    :type edge_attr_keys: Sequence[str] | None

    Usage::

        auto = Automorphism(G)
        orbits = auto.orbits  # list of frozenset of NodeId
        unique_maps = auto.deduplicate(mappings)
    """

    _DEF_NODE_ATTRS: Tuple[str, ...] = ("element", "charge")
    _DEF_EDGE_ATTRS: Tuple[str, ...] = ("order",)

    def __init__(
        self,
        graph: nx.Graph,
        node_attr_keys: Sequence[str] | None = None,
        edge_attr_keys: Sequence[str] | None = None,
    ) -> None:
        self._graph: nx.Graph = graph
        self._nkeys: Tuple[str, ...] = (
            tuple(node_attr_keys) if node_attr_keys else self._DEF_NODE_ATTRS
        )
        self._ekeys: Tuple[str, ...] = (
            tuple(edge_attr_keys) if edge_attr_keys else self._DEF_EDGE_ATTRS
        )
        self._orbits: List[frozenset[NodeId]] | None = None

    @property
    def orbits(self) -> List[frozenset[NodeId]]:
        """
        Compute and return node-orbits of the graph.

        :returns: List of frozensets, each containing nodes mutually mapped by some automorphism.
        :rtype: List[frozenset[NodeId]]
        """
        if self._orbits is None:
            self._orbits = self._compute_orbits()
        return self._orbits

    def deduplicate(self, mappings: List[MappingDict]) -> List[MappingDict]:
        """
        Remove mappings that are equivalent under graph automorphisms.

        :param mappings: List of mapping dicts from pattern-node to host-node.
        :type mappings: List[MappingDict]
        :returns: Pruned list retaining one representative per equivalence class.
        :rtype: List[MappingDict]
        """
        if not mappings:
            return []

        # Map each node to its orbit index
        orbit_index: Dict[NodeId, int] = {
            node: idx for idx, orb in enumerate(self.orbits) for node in orb
        }

        def signature(m: MappingDict) -> Tuple[int, ...]:
            # Sorted tuple of orbit indices hit by mapping values
            return tuple(sorted(orbit_index[n] for n in m.values()))

        mappings.sort(key=signature)
        unique = [next(group) for _, group in groupby(mappings, key=signature)]
        return unique

    def _compute_orbits(self) -> List[frozenset[NodeId]]:
        """
        Internal: enumerate all automorphisms and group nodes into orbits.

        :returns: List of frozensets of NodeId
        :rtype: List[frozenset[NodeId]]
        """
        gm = GraphMatcher(
            self._graph,
            self._graph,
            node_match=categorical_node_match(
                self._nkeys, ["*", 0][: len(self._nkeys)]
            ),
            edge_match=categorical_edge_match(self._ekeys, [1.0][: len(self._ekeys)]),
        )
        orbit_sets: Dict[NodeId, set[NodeId]] = defaultdict(set)
        for auto in gm.isomorphisms_iter():
            for u, v in auto.items():
                orbit_sets[u].add(v)
                orbit_sets[v].add(u)

        if not orbit_sets:
            # No symmetries: each node is its own orbit
            return [frozenset({n}) for n in self._graph.nodes]
        return list({frozenset(s) for s in orbit_sets.values()})

    def __len__(self) -> int:
        """
        :returns: Number of distinct node orbits
        :rtype: int
        """
        return len(self.orbits)

    def __repr__(self) -> str:
        """
        :returns: Summary of Automorphism helper state
        :rtype: str
        """
        return (
            f"<Automorphism | orbits={len(self)} nodes={self._graph.number_of_nodes()}>"
        )
