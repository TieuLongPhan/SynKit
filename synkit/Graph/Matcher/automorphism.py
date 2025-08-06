"""automorphism.py
~~~~~~~~~~~~~~~~~~~
Utility for computing graph automorphisms and pruning redundant sub-graph
mappings that are equivalent up to those symmetries.

The main entry-point is the :class:`Automorphism` helper which exposes two
public attributes/methods:

* :pyattr:`Automorphism.orbits` – the list of *node orbits* of the graph.
* :pymeth:`Automorphism.deduplicate` – collapse a list of sub-graph
  ``mappings`` (``Dict[sub_node -> pattern_node]``) to unique
  representatives under the automorphism group.

Only **standard-library** modules and *networkx* are required.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------
NodeId = int | str | Tuple | object  # NetworkX is permissive – keep it generic
MappingDict = Mapping[NodeId, NodeId]

__all__ = ["Automorphism", "NodeId", "MappingDict"]


class Automorphism:
    """Compute automorphism *orbits* of a graph and use them to prune duplicate
    sub-graph mappings that are equivalent up to those automorphisms.

    Parameters
    ----------
    graph : nx.Graph
        The host graph *G* whose automorphism group will be analysed.
    node_attr_keys : Sequence[str], optional
        Node-attribute keys that must be preserved by an automorphism.
        Defaults to ``("element", "charge")``.
    edge_attr_keys : Sequence[str], optional
        Edge-attribute keys that must be preserved.  Defaults to ``("order",)``.

    Notes
    -----
    The implementation follows the standard *naïve backtracking* approach via
    :class:`networkx.algorithms.isomorphism.GraphMatcher`.  For molecules of
    typical size (<= 100 atoms) this is more than fast enough (<1 ms).
    """

    #: default node and edge attributes used for matching
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
        # Lazily computed cache
        self._orbits: List[frozenset[NodeId]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def orbits(self) -> List[frozenset[NodeId]]:
        """Return the list of node-orbits of *G*.

        Two nodes *u, v* are in the same orbit iff ∃ automorphism σ s.t.
        σ(u)=v.
        """
        if self._orbits is None:
            self._orbits = self._compute_orbits()
        return self._orbits

    # ------------------------------------------------------------------
    def deduplicate(self, mappings: List[MappingDict]) -> List[MappingDict]:
        """Collapse *mappings* to a single representative per automorphism-class.

        Parameters
        ----------
        mappings
            List of *pattern → host* node-mapping dictionaries as returned by a
            sub-graph matcher.

        Returns
        -------
        List[MappingDict]
            The pruned list with duplicates removed.
        """
        if not mappings:
            return []

        # Build invariant signature: which orbit-IDs are hit by each mapping
        orbit_id: Dict[NodeId, int] = {
            node: idx for idx, orb in enumerate(self.orbits) for node in orb
        }

        def _signature(m: MappingDict) -> Tuple[int, ...]:
            # multiset of orbit-IDs, sorted for determinism
            return tuple(sorted(orbit_id[n] for n in m.values()))

        mappings.sort(key=_signature)  # required for itertools.groupby
        unique: List[MappingDict] = [next(g) for _, g in groupby(mappings, _signature)]
        return unique

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_orbits(self) -> List[frozenset[NodeId]]:
        """Enumerate *all* automorphisms of the graph and derive node orbits."""
        gm = GraphMatcher(
            self._graph,
            self._graph,
            node_match=categorical_node_match(
                self._nkeys, ["*", 0][: len(self._nkeys)]
            ),
            edge_match=categorical_edge_match(self._ekeys, [1.0][: len(self._ekeys)]),
        )

        orbit_sets: Dict[NodeId, set] = defaultdict(set)
        for auto in gm.isomorphisms_iter():  # each is a Dict[u -> σ(u)]
            for u, v in auto.items():
                orbit_sets[u].add(v)
                orbit_sets[v].add(u)

        # Freeze & deduplicate
        if not orbit_sets:  # totally asymmetric graph
            return [frozenset({n}) for n in self._graph.nodes]
        return list({frozenset(s) for s in orbit_sets.values()})

    # ------------------------------------------------------------------
    # Magic methods
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover – tiny utility
        """Number of distinct automorphism orbits."""
        return len(self.orbits)

    # String representations --------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Automorphism | orbits={len(self)} nodes={self._graph.number_of_nodes()}>"
        )
