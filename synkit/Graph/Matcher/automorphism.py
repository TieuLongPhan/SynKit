"""automorphism.py
~~~~~~~~~~~~~~~~~~~
Utility for computing graph automorphisms and pruning redundant sub-graph
mappings equivalent under those symmetries.

This module provides the :class:`Automorphism` helper, which computes the
node-orbits of a graph and uses them to deduplicate subgraph-match mappings.

The key idea is to group host nodes into **orbits** under the automorphism
group of the host graph: two nodes are in the same orbit if there exists an
automorphism :math:`\\sigma` such that :math:`\\sigma(u) = v`.

Mappings are then considered equivalent if they hit the **same multiset of
orbits** on the host side, and a single representative is kept from each
equivalence class.
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

    Notes
    -----
    The automorphism computation is performed via
    :class:`networkx.algorithms.isomorphism.GraphMatcher` on ``graph`` against
    itself. For chemical graphs this is usually feasible, but for very large
    or highly symmetric graphs, the number of automorphisms can grow quickly.

    Examples
    --------
    Basic usage:

    .. code-block:: python

        import networkx as nx
        from synkit.Graph.automorphism import Automorphism

        # A 4-cycle where all nodes are symmetric
        G = nx.cycle_graph(4)

        auto = Automorphism(G)
        orbits = auto.orbits
        # orbits == [frozenset({0, 1, 2, 3})]
        n_sym = auto.n_automorphisms
        # n_sym == 8 for a 4-cycle

        # Imagine three subgraph mappings that only differ by rotation
        mappings = [
            {"a": 0, "b": 1},
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]
        unique = auto.deduplicate(mappings)
        # len(unique) == 1
    """

    _DEF_NODE_ATTRS: Tuple[str, ...] = ("element", "charge")
    _DEF_EDGE_ATTRS: Tuple[str, ...] = ("order",)

    def __init__(
        self,
        graph: nx.Graph,
        node_attr_keys: Sequence[str] | None = None,
        edge_attr_keys: Sequence[str] | None = None,
    ) -> None:
        """
        Initialize an :class:`Automorphism` helper.

        Parameters
        ----------
        graph : nx.Graph
            Host graph whose automorphisms and node orbits will be analyzed.
        node_attr_keys : Sequence[str] | None, optional
            Node attribute keys to be matched in the automorphism search.
            If ``None``, defaults to :attr:`_DEF_NODE_ATTRS`.
        edge_attr_keys : Sequence[str] | None, optional
            Edge attribute keys to be matched in the automorphism search.
            If ``None``, defaults to :attr:`_DEF_EDGE_ATTRS`.
        """
        self._graph: nx.Graph = graph
        self._nkeys: Tuple[str, ...] = (
            tuple(node_attr_keys) if node_attr_keys else self._DEF_NODE_ATTRS
        )
        self._ekeys: Tuple[str, ...] = (
            tuple(edge_attr_keys) if edge_attr_keys else self._DEF_EDGE_ATTRS
        )

        self._orbits: List[frozenset[NodeId]] | None = None
        self._n_automorphisms: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def orbits(self) -> List[frozenset[NodeId]]:
        """
        Node-orbits of the graph under its automorphism group.

        Returns
        -------
        List[frozenset[NodeId]]
            List of frozensets, each containing nodes mutually mapped by some
            automorphism.

        Notes
        -----
        The orbits are computed lazily on first access and cached afterwards.
        """
        if self._orbits is None:
            self._analyze()
        return self._orbits  # type: ignore[return-value]

    @property
    def n_automorphisms(self) -> int:
        """
        Number of distinct automorphisms of the host graph.

        Returns
        -------
        int
            The order (size) of the automorphism group. This is at least 1
            (the identity automorphism), even for asymmetric graphs.

        Notes
        -----
        The value is computed together with :attr:`orbits` and cached.
        """
        if self._n_automorphisms is None:
            self._analyze()
        return int(self._n_automorphisms)  # type: ignore[return-value]

    def deduplicate(self, mappings: List[MappingDict]) -> List[MappingDict]:
        """
        Remove mappings that are equivalent under graph automorphisms.

        Two mappings are considered equivalent if the multiset of **orbit
        indices** hit by their host-side node assignments is the same. In
        other words, we ignore which specific nodes in an orbit are hit and
        keep a single representative mapping.

        Parameters
        ----------
        mappings : list of MappingDict
            List of mapping dicts from pattern-node to host-node.

        Returns
        -------
        list of MappingDict
            Pruned list retaining one representative per equivalence class.

        Examples
        --------
        .. code-block:: python

            auto = Automorphism(G)
            unique_mappings = auto.deduplicate(all_mappings)
        """
        if not mappings:
            return []

        orbit_index: Dict[NodeId, int] = {
            node: idx for idx, orb in enumerate(self.orbits) for node in orb
        }

        def signature(mapping: MappingDict) -> Tuple[int, ...]:
            """Sorted tuple of orbit indices hit by mapping values."""
            return tuple(sorted(orbit_index[n] for n in mapping.values()))

        mappings_sorted = list(mappings)
        mappings_sorted.sort(key=signature)

        unique: List[MappingDict] = [
            next(group) for _, group in groupby(mappings_sorted, key=signature)
        ]
        return unique

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _analyze(self) -> None:
        """
        Compute node orbits and number of automorphisms in a single pass.

        This method is idempotent and safe to call multiple times.
        """
        gm = GraphMatcher(
            self._graph,
            self._graph,
            node_match=categorical_node_match(
                self._nkeys,
                ["*", 0][: len(self._nkeys)],  # defaults if attr missing
            ),
            edge_match=categorical_edge_match(
                self._ekeys,
                [1.0][: len(self._ekeys)],  # defaults if attr missing
            ),
        )

        orbit_sets: Dict[NodeId, set[NodeId]] = defaultdict(set)
        n_aut: int = 0

        for auto in gm.isomorphisms_iter():
            n_aut += 1
            for u, v in auto.items():
                orbit_sets[u].add(v)
                orbit_sets[v].add(u)

        if not orbit_sets:
            # Empty graph: no nodes, but still one (trivial) automorphism
            if self._graph.number_of_nodes() == 0:
                self._orbits = []
                self._n_automorphisms = 1
                return

            # Asymmetric graph: only the identity automorphism
            self._orbits = [frozenset({n}) for n in self._graph.nodes]
            self._n_automorphisms = 1
            return

        # Deduplicate orbit sets (each node's set should already be its orbit)
        unique_orbits = {frozenset(nodes) for nodes in orbit_sets.values()}

        self._orbits = list(unique_orbits)
        self._n_automorphisms = n_aut if n_aut > 0 else 1

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """
        Return the number of distinct node orbits.

        Returns
        -------
        int
            Number of orbits in the host graph.
        """
        return len(self.orbits)

    def __repr__(self) -> str:
        """
        Return a short textual summary of the helper state.

        Returns
        -------
        str
            Summary representation of the :class:`Automorphism` instance.
        """
        n_nodes = self._graph.number_of_nodes()
        n_orb = len(self)
        n_aut = self.n_automorphisms
        return (
            f"<Automorphism | nodes={n_nodes} " f"orbits={n_orb} automorphisms={n_aut}>"
        )
