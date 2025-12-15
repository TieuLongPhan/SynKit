from __future__ import annotations

from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple, Mapping

import networkx as nx
from itertools import groupby


class AutoEst:
    """
    Approximate node automorphism groups (orbits) via 1-WL color refinement.

    This class performs a Weisfeiler–Lehman (WL-1) style color refinement
    on the input graph to approximate the partition of nodes into
    automorphism orbits. In many practical graphs (especially with
    chemically meaningful node/edge labels), the WL partition coincides
    with, or closely approximates, the true orbit partition and is much
    cheaper than enumerating all automorphisms.

    :param graph: Input NetworkX graph. It is not modified in-place.
    :type graph: nx.Graph
    :param node_attrs: Node attribute keys whose values will be included in
        the initial coloring. If ``None`` or empty, only structural
        information (degree) is used initially.
    :type node_attrs: list[str] or None
    :param edge_attrs: Edge attribute keys whose values will be incorporated
        into the neighborhood signatures. If ``None`` or empty, only
        neighbor colors are used.
    :type edge_attrs: list[str] or None
    :param max_iter: Maximum number of WL refinement iterations.
    :type max_iter: int

    .. note::
       This is an **approximate** estimator of automorphism orbits:
       two nodes with different WL colors cannot be in the same orbit,
       but nodes with the same WL color might still be distinguishable
       by higher-order invariants. For many molecular graphs where
       node/edge labels are informative, this partition is typically
       very close to the true automorphism partition and is often
       sufficient for automorphism pruning.

    .. seealso::
       For a high-level discussion of how Weisfeiler–Lehman refinements
       relate to automorphism indistinguishability and orbit structure,
       see:

       * A. Dawar and G. Vagnozzi, *Generalizations of k-dimensional
         Weisfeiler–Leman stabilization*, arXiv preprint (2019/2020).

    Example
    -------
    .. code-block:: python

        import networkx as nx
        from synkit.Graph.automorphism import AutoEst

        # Simple 4-cycle where all nodes are symmetric under rotation/reflection
        G = nx.cycle_graph(4)

        est = AutoEst(G, node_attrs=[], edge_attrs=[])
        est = est.fit()

        print(est.orbits)
        # [frozenset({0, 1, 2, 3})]

        print(est.n_orbits)
        # 1
    """

    _DEF_NODE_ATTRS: Tuple[str, ...] = ("element", "charge")
    _DEF_EDGE_ATTRS: Tuple[str, ...] = ("order",)

    def __init__(
        self,
        graph: nx.Graph,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        max_iter: int = 10,
    ) -> None:
        self._graph: nx.Graph = graph
        self._node_attrs: Tuple[str, ...] = (
            tuple(node_attrs) if node_attrs is not None else self._DEF_NODE_ATTRS
        )
        self._edge_attrs: Tuple[str, ...] = (
            tuple(edge_attrs) if edge_attrs is not None else self._DEF_EDGE_ATTRS
        )
        self._max_iter: int = int(max_iter)

        self._colors: Dict[Hashable, int] = {}
        self._orbits: List[frozenset[Hashable]] = []
        self._orbit_index: Dict[Hashable, int] | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def graph(self) -> nx.Graph:
        """
        Underlying host graph.

        :returns: Graph passed to the constructor.
        :rtype: nx.Graph
        """
        return self._graph

    @property
    def node_attrs(self) -> Tuple[str, ...]:
        """
        Node attribute keys used in WL initialization.

        :returns: Node-attribute keys considered by the estimator.
        :rtype: tuple[str, ...]
        """
        return self._node_attrs

    @property
    def edge_attrs(self) -> Tuple[str, ...]:
        """
        Edge attribute keys used in WL refinement.

        :returns: Edge-attribute keys considered by the estimator.
        :rtype: tuple[str, ...]
        """
        return self._edge_attrs

    def fit(self) -> AutoEst:
        """
        Run WL-1 refinement and compute approximate automorphism orbits.

        :returns: The fitted estimator (``self``), allowing chained use.
        :rtype: AutoEst
        """
        self._initialize_colors()
        self._refine_colors()
        self._build_orbits()
        self._orbit_index = None
        self._fitted = True
        return self

    @property
    def node_colors(self) -> Dict[Hashable, int]:
        """
        Node-to-color mapping after WL refinement.

        :returns: Mapping from node identifiers to integer color IDs.
        :rtype: dict[hashable, int]

        :raises RuntimeError: If :meth:`fit` has not been called.
        """
        self._ensure_fitted()
        return self._colors

    @property
    def orbits(self) -> List[frozenset[Hashable]]:
        """
        WL-equivalence classes (approximate automorphism orbits).

        :returns: List of frozensets; each frozenset is a set of nodes
            that share the same final WL color.
        :rtype: list[frozenset[hashable]]

        :raises RuntimeError: If :meth:`fit` has not been called.
        """
        self._ensure_fitted()
        return self._orbits

    @property
    def groups(self) -> List[List[Hashable]]:
        """
        Orbits represented as sorted lists instead of frozensets.

        This is a convenience view over :attr:`orbits` to match APIs that
        expect a list-of-lists representation.

        :returns: List of sorted node-lists corresponding to orbits.
        :rtype: list[list[hashable]]
        """
        groups: List[List[Hashable]] = []
        for orb in self.orbits:
            group = sorted(orb, key=lambda x: x)
            groups.append(group)
        groups.sort(key=lambda g: (len(g), g[0]))
        return groups

    @property
    def orbit_index(self) -> Dict[Hashable, int]:
        """
        Map each node to its WL-orbit index.

        :returns: Mapping ``node -> orbit_id`` where ``orbit_id`` indexes
            :attr:`orbits`.
        :rtype: dict[hashable, int]

        :raises RuntimeError: If :meth:`fit` has not been called.
        """
        self._ensure_fitted()
        if self._orbit_index is None:
            index: Dict[Hashable, int] = {}
            for i, orb in enumerate(self._orbits):
                for node in orb:
                    index[node] = i
            self._orbit_index = index
        return self._orbit_index

    @property
    def n_orbits(self) -> int:
        """
        Number of WL-equivalence classes (approximate automorphism orbits).

        :returns: Number of orbits in :attr:`orbits`.
        :rtype: int

        :raises RuntimeError: If :meth:`fit` has not been called.
        """
        self._ensure_fitted()
        return len(self._orbits)

    @property
    def n_groups(self) -> int:
        """
        Alias for :attr:`n_orbits`, for symmetry with exact Automorphism code.

        :returns: Number of approximate orbits.
        :rtype: int
        """
        return self.n_orbits

    def __len__(self) -> int:
        """
        Number of approximate automorphism orbits.

        :returns: Equivalent to :attr:`n_orbits` after fitting, or ``0``
            if the estimator has not been fitted yet.
        :rtype: int
        """
        return self.n_orbits if self._fitted else 0

    def __repr__(self) -> str:
        """
        Return a short textual summary of the estimator state.

        :returns: Summary representation of the
            :class:`AutoEst` instance.
        :rtype: str
        """
        n_nodes = self._graph.number_of_nodes()
        n_orb: object = len(self) if self._fitted else "?"
        return (
            f"<AutoEst | nodes={n_nodes} "
            f"orbits={n_orb} approx='WL-1' max_iter={self._max_iter}>"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_fitted(self) -> None:
        """
        Raise if estimator has not been fitted yet.

        :raises RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Call 'fit()' before accessing results.")

    def _initialize_colors(self) -> None:
        """
        Initialize node colors using degree and selected node attributes.
        """
        colors: Dict[Hashable, int] = {}
        palette: Dict[Tuple[Any, ...], int] = {}
        next_color = 0

        for node in self._graph.nodes():
            label = self._initial_label(node)
            if label not in palette:
                palette[label] = next_color
                next_color += 1
            colors[node] = palette[label]

        self._colors = colors

    def _initial_label(self, node: Hashable) -> Tuple[Any, ...]:
        """
        Build the initial label for a node.

        :param node: Node identifier.
        :type node: hashable

        :returns: Initial structural/attribute label for the node.
        :rtype: tuple
        """
        degree = self._graph.degree(node)
        attrs = self._graph.nodes[node]
        values = [attrs.get(key) for key in self._node_attrs]
        return (degree, *values)

    def _refine_colors(self) -> None:
        """
        Perform WL-1 color refinement up to ``max_iter`` or convergence.
        """
        for _ in range(self._max_iter):
            new_colors, changed = self._refine_once()
            self._colors = new_colors
            if not changed:
                break

    def _refine_once(self) -> Tuple[Dict[Hashable, int], bool]:
        """
        Perform a single WL refinement sweep.

        :returns: A pair ``(new_colors, changed)`` where ``new_colors`` is
            the updated node-to-color mapping, and ``changed`` indicates
            whether any node color changed in this iteration.
        :rtype: tuple[dict[hashable, int], bool]
        """
        new_colors: Dict[Hashable, int] = {}
        palette: Dict[Tuple[Any, ...], int] = {}
        next_color = 0
        changed = False

        for node in self._graph.nodes():
            label = self._refined_label(node)
            if label not in palette:
                palette[label] = next_color
                next_color += 1
            color = palette[label]
            new_colors[node] = color
            if color != self._colors.get(node):
                changed = True

        return new_colors, changed

    def _refined_label(self, node: Hashable) -> Tuple[Any, ...]:
        """
        Build refined label combining current node color and neighbor signatures.

        :param node: Node identifier.
        :type node: hashable

        :returns: Refined label used by the WL procedure.
        :rtype: tuple
        """
        base_color = self._colors[node]
        neigh_sigs: List[Tuple[Any, ...]] = []

        for nbr in self._graph.neighbors(node):
            sig = self._neighbor_signature(node, nbr)
            neigh_sigs.append(sig)

        neigh_sigs.sort()
        return (base_color, tuple(neigh_sigs))

    def _neighbor_signature(
        self,
        node: Hashable,
        neighbor: Hashable,
    ) -> Tuple[Any, ...]:
        """
        Build a neighborhood signature for a neighbor edge.

        :param node: Central node identifier (currently unused but kept
            for extensibility).
        :type node: hashable
        :param neighbor: Neighbor node identifier.
        :type neighbor: hashable

        :returns: Signature including neighbor color and selected edge
            attributes.
        :rtype: tuple
        """
        edge_data = self._graph.get_edge_data(node, neighbor, default={})
        edge_vals = [edge_data.get(key) for key in self._edge_attrs]
        return (self._colors[neighbor], *edge_vals)

    def _build_orbits(self) -> None:
        """
        Group nodes by final WL color into approximate orbits.
        """
        color_to_nodes: Dict[int, List[Hashable]] = {}
        for node, color in self._colors.items():
            color_to_nodes.setdefault(color, []).append(node)

        orbits: List[frozenset[Hashable]] = []
        for nodes in color_to_nodes.values():
            orbit = frozenset(nodes)
            orbits.append(orbit)

        # sort by (size, min node) for deterministic order
        self._orbits = sorted(
            orbits,
            key=lambda orb: (len(orb), min(orb)),
        )

    def deduplicate(
        self, mappings: List[Mapping[Hashable, Hashable]]
    ) -> List[Mapping[Hashable, Hashable]]:
        """
        Remove mappings that are equivalent under the approximate WL-orbit partition.

        Two mappings are considered equivalent if the multiset of orbit indices
        hit by their *host-side* node assignments is identical. In other words,
        we ignore which specific nodes inside an orbit are chosen and keep a
        single representative mapping per equivalence class.

        Parameters
        ----------
        mappings : list[Mapping[hashable, hashable]]
            List of mapping dictionaries mapping *pattern node* -> *host node*.

        Returns
        -------
        list[Mapping[hashable, hashable]]
            Pruned list retaining one representative per equivalence class.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted (call ``fit()`` first).
        ValueError
            If any host node referenced in `mappings` is not present in the
            fitted orbit index (indicative of inconsistent host nodes).

        Notes
        -----
        - Because AutoEst uses WL-1 approximation, its orbits may be coarser
          (or equal) to the true automorphism orbits; deduplication therefore
          is *approximate* and may retain more mappings than an exact automorphism
          based deduplication would.
        """
        # Ensure estimator is fitted and orbits are available
        self._ensure_fitted()

        if not mappings:
            return []

        # Build node -> orbit index mapping from fitted orbits
        orbit_index: Dict[Hashable, int] = {
            node: idx for idx, orb in enumerate(self._orbits) for node in orb
        }

        # Validate that all host nodes referenced by mappings are known
        missing_nodes = set()
        for mapping in mappings:
            for host_node in mapping.values():
                if host_node not in orbit_index:
                    missing_nodes.add(host_node)
        if missing_nodes:
            raise ValueError(
                "Found host nodes in `mappings` not present in fitted orbits: "
                f"{sorted(missing_nodes)}. Did you fit() on the correct host graph?"
            )

        # Signature function: sorted tuple of orbit indices hit by mapping values
        def signature(mapping: Mapping[Hashable, Hashable]) -> Tuple[int, ...]:
            return tuple(sorted(orbit_index[n] for n in mapping.values()))

        # Sort and group by signature, then pick the first mapping from each group
        mappings_sorted = sorted(mappings, key=signature)
        unique: List[Mapping[Hashable, Hashable]] = [
            next(group) for _, group in groupby(mappings_sorted, key=signature)
        ]
        return unique


def estimate_automorphism_groups(
    graph: nx.Graph,
    node_attrs: Optional[Iterable[str]] = None,
    edge_attrs: Optional[Iterable[str]] = None,
    max_iter: int = 10,
) -> AutoEst:
    """
    Convenience function to fit :class:`AutoEst`.

    :param graph: Input NetworkX graph.
    :type graph: nx.Graph
    :param node_attrs: Node attribute keys to include in the WL labels.
    :type node_attrs: iterable[str] or None
    :param edge_attrs: Edge attribute keys to include in neighborhood
        signatures.
    :type edge_attrs: iterable[str] or None
    :param max_iter: Maximum number of WL refinement iterations.
    :type max_iter: int

    :returns: Fitted estimator. Access results via :attr:`orbits`,
        :attr:`groups` or :attr:`n_orbits`.
    :rtype: AutoEst
    """
    node_list = list(node_attrs) if node_attrs is not None else None
    edge_list = list(edge_attrs) if edge_attrs is not None else None
    return AutoEst(
        graph=graph,
        node_attrs=node_list,
        edge_attrs=edge_list,
        max_iter=max_iter,
    ).fit()
