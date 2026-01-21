"""automorphism.py
~~~~~~~~~~~~~~~~~~
Utility for computing graph automorphisms and pruning redundant sub-graph
mappings equivalent under those symmetries.

This module provides the :class:`Automorphism` helper, which computes the
node-orbits of a graph and uses them to deduplicate subgraph-match mappings.

Key idea
--------
Group host nodes into **orbits** under the automorphism group of the host
graph: two nodes are in the same orbit if there exists an automorphism
:math:`\\sigma` such that :math:`\\sigma(u) = v`.

Mappings are considered equivalent if they hit the same **multiset of orbits**
on the host side, and a single representative is kept.

Disconnected graphs
-------------------
If the host graph is disconnected, we choose one connected component as an
**anchor** (by default the *largest* component) to suppress automorphisms that
swap isomorphic components. We then compute automorphisms **within each
component** independently and combine component orbits. The total number of
automorphisms is the product of component automorphism counts (excluding
component-permutation symmetries by design due to anchoring).
"""

from __future__ import annotations

from collections import defaultdict
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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

    Two nodes are in the same orbit if there exists an automorphism
    :math:`\\sigma` such that :math:`\\sigma(u) = v`.

    Parameters
    ----------
    graph : nx.Graph
        The host graph for which to compute automorphisms.
    node_attr_keys : Sequence[str] | None, optional
        Sequence of node attribute keys to respect in the automorphism
        computation (i.e., nodes must match on these attributes). Defaults to
        ``("element", "charge")``.
    edge_attr_keys : Sequence[str] | None, optional
        Sequence of edge attribute keys to respect in the automorphism
        computation. Defaults to ``("order",)``.
    anchor_largest_component : bool, optional
        If ``True`` and the graph is disconnected, chooses the largest connected
        component as an "anchor" to suppress automorphisms that swap isomorphic
        components. Defaults to ``True``.
    """

    _DEF_NODE_ATTRS: Tuple[str, ...] = ("element", "charge")
    _DEF_EDGE_ATTRS: Tuple[str, ...] = ("order",)

    def __init__(
        self,
        graph: nx.Graph,
        node_attr_keys: Optional[Sequence[str]] = None,
        edge_attr_keys: Optional[Sequence[str]] = None,
        *,
        anchor_largest_component: bool = True,
    ) -> None:
        self._graph: nx.Graph = graph
        self._nkeys: Tuple[str, ...] = (
            tuple(node_attr_keys) if node_attr_keys else self._DEF_NODE_ATTRS
        )
        self._ekeys: Tuple[str, ...] = (
            tuple(edge_attr_keys) if edge_attr_keys else self._DEF_EDGE_ATTRS
        )
        self._anchor_largest: bool = bool(anchor_largest_component)

        self._orbits: Optional[List[frozenset[NodeId]]] = None
        self._n_automorphisms: Optional[int] = None
        self._orbit_index: Optional[Dict[NodeId, int]] = None
        self._components: Optional[List[frozenset[NodeId]]] = None
        self._anchor_component: Optional[frozenset[NodeId]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def orbits(self) -> List[frozenset[NodeId]]:
        """
        Node-orbits of the graph under its automorphism group.

        Returns
        -------
        list[frozenset[NodeId]]
            Orbits, computed lazily and cached.
        """
        if self._orbits is None:
            self._analyze()
        return self._orbits  # type: ignore[return-value]

    @property
    def n_automorphisms(self) -> int:
        """
        Number of automorphisms of the host graph.

        Returns
        -------
        int
            Group order (at least 1).
        """
        if self._n_automorphisms is None:
            self._analyze()
        return int(self._n_automorphisms)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Connected / disconnected handling
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        """
        Whether the host graph is connected (weakly for directed graphs).

        Returns
        -------
        bool
            ``True`` if connected or has 0/1 nodes, else ``False``.
        """
        n = self._graph.number_of_nodes()
        if n <= 1:
            return True
        return len(self.components) == 1

    @property
    def components(self) -> List[frozenset[NodeId]]:
        """
        Connected components (weakly for directed graphs).

        Returns
        -------
        list[frozenset[NodeId]]
            Components as frozensets of node IDs.
        """
        if self._components is None:
            self._components = self._compute_components()
        return self._components

    @property
    def anchor_component(self) -> Optional[frozenset[NodeId]]:
        """
        Anchor component used for disconnected graphs.

        Returns
        -------
        frozenset[NodeId] | None
            Anchor component node-set, or ``None`` if graph is connected.
        """
        if self._orbits is None:
            self._analyze()
        return self._anchor_component

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_components(self) -> List[frozenset[NodeId]]:
        if self._graph.number_of_nodes() == 0:
            return []
        if self._graph.is_directed():
            comps: Iterable[Iterable[NodeId]] = nx.weakly_connected_components(self._graph)  # type: ignore[arg-type]
        else:
            comps = nx.connected_components(self._graph)  # type: ignore[arg-type]
        return [frozenset(c) for c in comps]

    def _node_defaults(self) -> List[object]:
        defaults: List[object] = []
        for k in self._nkeys:
            defaults.append(0 if k == "charge" else "*")
        return defaults

    def _edge_defaults(self) -> List[object]:
        return [1.0 for _ in self._ekeys]

    def _make_matcher(self, g: nx.Graph) -> GraphMatcher:
        return GraphMatcher(
            g,
            g,
            node_match=categorical_node_match(self._nkeys, self._node_defaults()),
            edge_match=categorical_edge_match(self._ekeys, self._edge_defaults()),
        )

    def _analyze_component(self, g: nx.Graph) -> Tuple[List[frozenset[NodeId]], int]:
        if g.number_of_nodes() == 0:
            return ([], 1)
        if g.number_of_nodes() == 1:
            node = next(iter(g.nodes))
            return ([frozenset({node})], 1)

        gm = self._make_matcher(g)

        orbit_sets: Dict[NodeId, set[NodeId]] = defaultdict(set)
        n_aut = 0

        for auto in gm.isomorphisms_iter():
            n_aut += 1
            for u, v in auto.items():
                orbit_sets[u].add(v)
                orbit_sets[v].add(u)

        if not orbit_sets:
            return ([frozenset({n}) for n in g.nodes], 1)

        unique_orbits = {frozenset(nodes) for nodes in orbit_sets.values()}
        return (list(unique_orbits), n_aut if n_aut > 0 else 1)

    def _choose_anchor(
        self, comps: List[frozenset[NodeId]]
    ) -> Optional[frozenset[NodeId]]:
        if len(comps) <= 1 or not self._anchor_largest:
            return None
        return max(comps, key=len)

    def _sorted_orbits(
        self, orbits: List[frozenset[NodeId]]
    ) -> List[frozenset[NodeId]]:
        def _key(o: frozenset[NodeId]) -> Tuple[int, str]:
            return (len(o), "|".join(sorted((repr(x) for x in o))))

        return sorted(orbits, key=_key)

    def _analyze(self) -> None:
        comps = self.components

        if self._graph.number_of_nodes() == 0:
            self._orbits = []
            self._n_automorphisms = 1
            self._anchor_component = None
            return

        if len(comps) <= 1:
            orbits, n_aut = self._analyze_component(self._graph)
            self._orbits = self._sorted_orbits(orbits)
            self._n_automorphisms = n_aut
            self._anchor_component = None
            return

        # Disconnected: anchor largest component; compute per-component autos
        self._anchor_component = self._choose_anchor(comps)

        all_orbits: List[frozenset[NodeId]] = []
        total_aut = 1

        for comp in comps:
            sub = self._graph.subgraph(comp).copy()
            orbits, n_aut = self._analyze_component(sub)
            all_orbits.extend(orbits)
            total_aut *= int(n_aut)

        self._orbits = self._sorted_orbits(list({frozenset(o) for o in all_orbits}))
        self._n_automorphisms = int(total_aut) if total_aut > 0 else 1

    def _get_orbit_index(self) -> Dict[NodeId, int]:
        if self._orbit_index is None:
            self._orbit_index = {
                node: idx for idx, orb in enumerate(self.orbits) for node in orb
            }
        return self._orbit_index

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.orbits)

    def __repr__(self) -> str:
        n_nodes = self._graph.number_of_nodes()
        conn = "connected" if self.is_connected else "disconnected"
        n_comp = len(self.components)
        n_orb = len(self.orbits) if self._orbits is not None else "?"
        n_aut = self._n_automorphisms if self._n_automorphisms is not None else "?"
        return (
            f"<Automorphism | {conn} nodes={n_nodes} comps={n_comp} "
            f"orbits={n_orb} automorphisms={n_aut}>"
        )
