"""
auto_est.py
~~~~~~~~~~~

Approximate node automorphism groups (orbits) via 1-WL color refinement,
plus orbit- and component-aware deduplication utilities.

Design goals (SynKit style)
---------------------------
- OOP with a scikit-like ``fit() -> self``.
- Deterministic output ordering.
- Sphinx-style docstrings.
- Helper methods and useful properties.
- Optional "components style" grouping, analogous to exact Automorphism:
  you can obtain *orbit-components* induced by a subset of nodes (anchors).

Important note
--------------
WL-1 provides an *approximate* orbit partition: distinct WL colors imply
distinct orbits, but equal WL colors do not guarantee true symmetry.

This module offers:
- ``AutoEst.orbits``: WL-equivalence classes on the given graph
- ``AutoEst.components(nodes)``: connected components on an induced subgraph
- ``AutoEst.orbit_components(nodes)``: components of the orbit-quotient graph
- ``AutoEst.deduplicate_host_orbits(mappings)``: host-orbit based pruning
- ``AutoEst.deduplicate_pattern_orbits(mappings, pattern_orbits, ...)``:
  pattern-orbit based pruning (anchor-aware)

The anchor-aware deduplication follows your recent constraint:
anchor components must not be pruned by orbit-independence.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import (
    Any,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

import networkx as nx


# --------------------------------------------------------------------------- #
# Small, typed config container (keeps complexity low)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _WLConfig:
    """
    Internal WL configuration.

    :param node_attrs: Node attribute keys for initial labels.
    :type node_attrs: tuple[str, ...]
    :param edge_attrs: Edge attribute keys for neighbor signatures.
    :type edge_attrs: tuple[str, ...]
    :param max_iter: Maximum refinement iterations.
    :type max_iter: int
    """

    node_attrs: Tuple[str, ...]
    edge_attrs: Tuple[str, ...]
    max_iter: int


# --------------------------------------------------------------------------- #
# Core estimator
# --------------------------------------------------------------------------- #
class AutoEst:
    """
    Approximate node automorphism groups (orbits) via 1-WL color refinement.

    This class performs a Weisfeiler–Lehman (WL-1) style color refinement
    on the input graph to approximate a partition of nodes into
    *automorphism-indistinguishability* classes (often called “WL-orbits”).
    In many practical graphs (especially with chemically meaningful
    node/edge labels), the WL partition coincides with, or closely
    approximates, the true orbit partition and is much cheaper than
    enumerating automorphisms.

    Besides the basic orbit partition, :class:`AutoEst` provides a
    “components-style” interface analogous to the exact automorphism helper:

    - :meth:`components` returns connected components of an induced subgraph
      (useful for “anchor components”).
    - :meth:`orbit_components` returns connected components in the
      **orbit-quotient graph** restricted to a node subset, capturing which
      orbits are *coupled* by edges inside the subset.
    - :meth:`deduplicate_host_orbits` prunes mappings by **host-side** WL-orbits.
    - :meth:`deduplicate_pattern_orbits` prunes mappings by **pattern** orbits,
      with optional anchor-aware behavior (no pruning inside anchored nodes).

    :param graph: Input NetworkX graph. It is not modified in-place.
    :type graph: nx.Graph
    :param node_attrs: Node attribute keys whose values will be included in
        the initial coloring. If ``None``, defaults are used.
    :type node_attrs: list[str] or None
    :param edge_attrs: Edge attribute keys whose values will be incorporated
        into the neighborhood signatures. If ``None``, defaults are used.
    :type edge_attrs: list[str] or None
    :param max_iter: Maximum number of WL refinement iterations.
    :type max_iter: int

    .. note::
       This is an **approximate** estimator of automorphism orbits:
       two nodes with different final WL colors cannot be in the same orbit,
       but nodes with the same WL color might still be distinguishable by
       higher-order invariants (e.g. higher-dimensional WL, spectral
       invariants, or full automorphism search). In many molecular graphs
       where node/edge labels are informative, this partition is typically
       very close to the true automorphism partition and is often sufficient
       for symmetry-aware pruning.

    .. note::
       “Anchor components” can invalidate orbit-wise independence:
       when a subset of nodes is treated as anchored, orbits that are
       connected through the anchored subgraph should be considered *coupled*.
       Use :meth:`orbit_components` and anchor-aware
       :meth:`deduplicate_pattern_orbits` to avoid incorrect pruning.

    .. seealso::
       For discussions relating Weisfeiler–Lehman refinement to
       automorphism indistinguishability and orbit structure, see:

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

        # "components style": connected components of an induced subgraph
        comps = est.components(nodes=[0, 1])
        print(comps)
        # [frozenset({0, 1})]

        # orbit-quotient components of an induced subgraph
        oc = est.orbit_components(nodes=[0, 1, 2, 3])
        print(oc)
        # [frozenset({0})]   # one orbit-id component in this symmetric case
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

        cfg = _WLConfig(
            node_attrs=(
                tuple(node_attrs) if node_attrs is not None else self._DEF_NODE_ATTRS
            ),
            edge_attrs=(
                tuple(edge_attrs) if edge_attrs is not None else self._DEF_EDGE_ATTRS
            ),
            max_iter=int(max_iter),
        )
        self._cfg: _WLConfig = cfg

        self._colors: Dict[Hashable, int] = {}
        self._orbits: List[FrozenSet[Hashable]] = []
        self._orbit_index: Optional[Dict[Hashable, int]] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    # Public API: properties
    # ------------------------------------------------------------------ #
    @property
    def graph(self) -> nx.Graph:
        """
        Underlying graph.

        :returns: Graph passed to the constructor.
        :rtype: nx.Graph
        """
        return self._graph

    @property
    def node_attrs(self) -> Tuple[str, ...]:
        """
        Node attribute keys used in WL initialization.

        :returns: Node-attribute keys.
        :rtype: tuple[str, ...]
        """
        return self._cfg.node_attrs

    @property
    def edge_attrs(self) -> Tuple[str, ...]:
        """
        Edge attribute keys used in WL refinement.

        :returns: Edge-attribute keys.
        :rtype: tuple[str, ...]
        """
        return self._cfg.edge_attrs

    @property
    def max_iter(self) -> int:
        """
        Maximum number of WL refinement iterations.

        :returns: Maximum refinement iterations.
        :rtype: int
        """
        return self._cfg.max_iter

    @property
    def anchor_component(self) -> FrozenSet[Hashable]:
        """
        Largest connected component of the fitted graph.

        This is a convenience “components-style” accessor. It is commonly used as
        an anchor set for match pruning and symmetry breaking.

        :returns: The node-set of the largest connected component. If multiple
            components share the maximum size, the one with the smallest
            (sorted) node is returned for determinism.
        :rtype: frozenset[hashable]

        :raises RuntimeError: If :meth:`fit` has not been called.
        """
        self._ensure_fitted()
        comps = self._components_on_induced(nodes=None)
        if not comps:
            return frozenset()

        # Deterministic tie-break: size desc, then smallest node
        comps_sorted = sorted(
            comps,
            key=lambda c: (-len(c), min(c) if c else 0),
        )
        return comps_sorted[0]

    # ------------------------------------------------------------------ #
    # Fitting and results
    # ------------------------------------------------------------------ #
    def fit(self) -> AutoEst:
        """
        Run WL-1 refinement and compute approximate orbits.

        :returns: The fitted estimator (``self``).
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
        Node-to-color mapping after refinement.

        :returns: Mapping node -> WL color id.
        :rtype: dict[hashable, int]
        """
        self._ensure_fitted()
        return dict(self._colors)

    @property
    def orbits(self) -> List[FrozenSet[Hashable]]:
        """
        WL-equivalence classes (approximate automorphism orbits).

        :returns: List of frozensets, each representing an orbit.
        :rtype: list[frozenset[hashable]]
        """
        self._ensure_fitted()
        return list(self._orbits)

    @property
    def groups(self) -> List[List[Hashable]]:
        """
        Orbits represented as sorted lists.

        :returns: List of sorted node lists.
        :rtype: list[list[hashable]]
        """
        self._ensure_fitted()
        out: List[List[Hashable]] = []
        for orb in self._orbits:
            out.append(sorted(orb, key=lambda x: x))
        out.sort(key=lambda g: (len(g), g[0] if g else 0))
        return out

    @property
    def orbit_index(self) -> Dict[Hashable, int]:
        """
        Map each node to its orbit id.

        :returns: Mapping node -> orbit_id.
        :rtype: dict[hashable, int]
        """
        self._ensure_fitted()
        if self._orbit_index is None:
            self._orbit_index = self._build_orbit_index()
        return dict(self._orbit_index)

    @property
    def n_orbits(self) -> int:
        """
        Number of approximate orbits.

        :returns: Number of orbits.
        :rtype: int
        """
        self._ensure_fitted()
        return len(self._orbits)

    @property
    def n_groups(self) -> int:
        """
        Alias for :attr:`n_orbits`.

        :returns: Number of orbits.
        :rtype: int
        """
        return self.n_orbits

    def __len__(self) -> int:
        """
        Number of orbits (0 if not fitted).

        :returns: Orbit count or 0.
        :rtype: int
        """
        return self.n_orbits if self._fitted else 0

    def __repr__(self) -> str:
        """
        Summary representation.

        :returns: Debug-friendly repr string.
        :rtype: str
        """
        n_nodes = self._graph.number_of_nodes()
        n_orb: object = len(self) if self._fitted else "?"
        return (
            f"<AutoEst | nodes={n_nodes} "
            f"orbits={n_orb} approx='WL-1' max_iter={self._cfg.max_iter}>"
        )

    # ------------------------------------------------------------------ #
    # "Components style" (similar spirit to Automorphism)
    # ------------------------------------------------------------------ #
    def components(
        self, nodes: Optional[Iterable[Hashable]] = None
    ) -> List[FrozenSet[Hashable]]:
        """
        Compute connected components on an induced subgraph.

        This mirrors the "components" utilities you used around Automorphism.

        :param nodes: Subset of nodes to induce. If None, uses all nodes.
        :type nodes: iterable[hashable] or None

        :returns: Connected components as frozensets (deterministic order).
        :rtype: list[frozenset[hashable]]

        :raises RuntimeError: If not fitted.
        """
        self._ensure_fitted()
        return self._components_on_induced(nodes)

    def orbit_components(
        self, nodes: Optional[Iterable[Hashable]] = None
    ) -> List[FrozenSet[int]]:
        """
        Components of the orbit-quotient graph restricted to an induced subgraph.

        - First restrict to `nodes` (or all nodes).
        - Collapse nodes to their orbit ids.
        - Build orbit-quotient adjacency based on edges between orbits.
        - Return connected components of orbit ids.

        This is useful when you have an "anchor component" defined as a set of
        pattern nodes and want to treat coupled orbits as a single unit.

        :param nodes: Subset of nodes. If None, uses all nodes.
        :type nodes: iterable[hashable] or None

        :returns: List of connected components in orbit-id space.
        :rtype: list[frozenset[int]]

        :raises RuntimeError: If not fitted.
        """
        self._ensure_fitted()
        keep = self._normalize_nodes(nodes)
        orbit_idx = self.orbit_index
        q = self._orbit_quotient_graph(keep, orbit_idx)
        comps = [frozenset(c) for c in nx.connected_components(q)]
        return sorted(comps, key=lambda c: (len(c), min(c)))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_fitted(self) -> None:
        """
        Ensure estimator is fitted.

        :raises RuntimeError: If not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Call 'fit()' before accessing results.")

    def _initialize_colors(self) -> None:
        """
        Initialize node colors using degree and selected node attributes.
        """
        palette: Dict[Tuple[Any, ...], int] = {}
        colors: Dict[Hashable, int] = {}
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
        Build initial label.

        :param node: Node id.
        :type node: hashable

        :returns: Tuple label (degree + attr values).
        :rtype: tuple
        """
        degree = self._graph.degree(node)
        attrs = self._graph.nodes[node]
        vals = [attrs.get(k) for k in self._cfg.node_attrs]
        return (degree, *vals)

    def _refine_colors(self) -> None:
        """
        Iterate WL refinement until convergence or max_iter.
        """
        for _ in range(self._cfg.max_iter):
            new_colors, changed = self._refine_once()
            self._colors = new_colors
            if not changed:
                break

    def _refine_once(self) -> Tuple[Dict[Hashable, int], bool]:
        """
        Single WL sweep.

        :returns: (new_colors, changed)
        :rtype: tuple[dict[hashable, int], bool]
        """
        palette: Dict[Tuple[Any, ...], int] = {}
        new_colors: Dict[Hashable, int] = {}
        next_color = 0
        changed = False

        for node in self._graph.nodes():
            label = self._refined_label(node)
            if label not in palette:
                palette[label] = next_color
                next_color += 1
            c = palette[label]
            new_colors[node] = c
            if c != self._colors.get(node):
                changed = True

        return new_colors, changed

    def _refined_label(self, node: Hashable) -> Tuple[Any, ...]:
        """
        Combine current color with sorted neighbor signatures.

        :param node: Node id.
        :type node: hashable

        :returns: Refined label.
        :rtype: tuple
        """
        base = self._colors[node]
        sigs: List[Tuple[Any, ...]] = []

        for nbr in self._graph.neighbors(node):
            sigs.append(self._neighbor_signature(node, nbr))

        sigs.sort()
        return (base, tuple(sigs))

    def _neighbor_signature(
        self, node: Hashable, neighbor: Hashable
    ) -> Tuple[Any, ...]:
        """
        Neighbor signature: (neighbor_color, edge_attr_1, ...).

        :param node: Central node.
        :type node: hashable
        :param neighbor: Neighbor node.
        :type neighbor: hashable

        :returns: Neighbor signature.
        :rtype: tuple
        """
        edge_data = self._graph.get_edge_data(node, neighbor, default={})
        edge_vals = [edge_data.get(k) for k in self._cfg.edge_attrs]
        return (self._colors[neighbor], *edge_vals)

    def _build_orbits(self) -> None:
        """
        Group nodes by final colors.
        """
        color_to_nodes: Dict[int, List[Hashable]] = {}
        for node, color in self._colors.items():
            color_to_nodes.setdefault(color, []).append(node)

        orbits = [frozenset(v) for v in color_to_nodes.values()]
        self._orbits = sorted(orbits, key=lambda o: (len(o), min(o)))

    def _build_orbit_index(self) -> Dict[Hashable, int]:
        """
        Build node -> orbit id mapping.

        :returns: Orbit index.
        :rtype: dict[hashable, int]
        """
        idx: Dict[Hashable, int] = {}
        for i, orb in enumerate(self._orbits):
            for n in orb:
                idx[n] = i
        return idx

    def _normalize_nodes(
        self, nodes: Optional[Iterable[Hashable]]
    ) -> FrozenSet[Hashable]:
        """
        Normalize subset nodes.

        :param nodes: Subset or None.
        :type nodes: iterable[hashable] or None

        :returns: Frozenset of nodes (validated).
        :rtype: frozenset[hashable]
        """
        if nodes is None:
            return frozenset(self._graph.nodes())
        keep = frozenset(nodes)
        unknown = [n for n in keep if n not in self._graph]
        if unknown:
            raise ValueError(f"Unknown nodes in subset: {unknown}")
        return keep

    def _components_on_induced(
        self, nodes: Optional[Iterable[Hashable]]
    ) -> List[FrozenSet[Hashable]]:
        """
        Connected components on induced subgraph.

        :param nodes: Subset or None.
        :type nodes: iterable[hashable] or None

        :returns: Components in deterministic order.
        :rtype: list[frozenset[hashable]]
        """
        keep = self._normalize_nodes(nodes)
        sub = self._graph.subgraph(keep)
        comps = [frozenset(c) for c in nx.connected_components(sub)]
        return sorted(comps, key=lambda c: (len(c), min(c)))

    def _orbit_quotient_graph(
        self,
        keep_nodes: FrozenSet[Hashable],
        orbit_idx: Dict[Hashable, int],
    ) -> nx.Graph:
        """
        Build orbit quotient graph restricted to keep_nodes.

        :param keep_nodes: Nodes to keep.
        :type keep_nodes: frozenset[hashable]
        :param orbit_idx: Node->orbit id.
        :type orbit_idx: dict[hashable, int]

        :returns: Quotient graph in orbit-id space.
        :rtype: nx.Graph
        """
        q = nx.Graph()
        for n in keep_nodes:
            q.add_node(orbit_idx[n])

        for u, v in self._graph.edges():
            if u not in keep_nodes or v not in keep_nodes:
                continue
            ou = orbit_idx[u]
            ov = orbit_idx[v]
            if ou != ov:
                q.add_edge(ou, ov)

        return q

    def _validate_host_nodes_in_mappings(
        self,
        mappings: List[Mapping[Hashable, Hashable]],
        orbit_idx: Dict[Hashable, int],
    ) -> None:
        """
        Validate all host nodes are known.

        :param mappings: Mappings list.
        :type mappings: list[Mapping[hashable, hashable]]
        :param orbit_idx: Node->orbit id.
        :type orbit_idx: dict[hashable, int]

        :raises ValueError: If unknown nodes appear.
        """
        missing: List[Hashable] = []
        for mp in mappings:
            for h in mp.values():
                if h not in orbit_idx:
                    missing.append(h)
        if missing:
            raise ValueError(
                "Host nodes in mappings not present in fitted graph: "
                f"{sorted(set(missing))}"
            )

    def _dedup_by_signature(
        self,
        mappings: List[Mapping[Hashable, Hashable]],
        sig_fn: Any,
    ) -> List[Mapping[Hashable, Hashable]]:
        """
        Deduplicate by computed signature (stable representative selection).

        :param mappings: List of mappings.
        :type mappings: list[Mapping[hashable, hashable]]
        :param sig_fn: Function mapping mapping->signature.
        :type sig_fn: callable

        :returns: Deduplicated list.
        :rtype: list[Mapping[hashable, hashable]]
        """
        mappings_sorted = sorted(mappings, key=sig_fn)
        out: List[Mapping[Hashable, Hashable]] = []
        for _, grp in groupby(mappings_sorted, key=sig_fn):
            out.append(next(grp))
        return out

    def _sort_orbits(
        self, pattern_orbits: Iterable[FrozenSet[Hashable]]
    ) -> Tuple[Tuple[Hashable, ...], ...]:
        """
        Deterministically sort orbits.

        :param pattern_orbits: Orbits.
        :type pattern_orbits: iterable[frozenset[hashable]]

        :returns: Tuple of sorted orbit tuples.
        :rtype: tuple[tuple[hashable, ...], ...]
        """
        orbs = [tuple(sorted(o, key=lambda x: x)) for o in pattern_orbits]
        orbs.sort(key=lambda o: (len(o), o[0] if o else 0))
        return tuple(orbs)

    def _dedup_pattern_orbits_no_anchor(
        self,
        mappings: List[Mapping[Hashable, Hashable]],
        orbits: Tuple[Tuple[Hashable, ...], ...],
    ) -> List[Mapping[Hashable, Hashable]]:
        """
        Pattern-orbit dedup without anchor.

        :param mappings: Mappings.
        :type mappings: list[Mapping[hashable, hashable]]
        :param orbits: Sorted orbit tuples.
        :type orbits: tuple[tuple[hashable, ...], ...]

        :returns: Deduplicated mappings.
        :rtype: list[Mapping[hashable, hashable]]
        """

        def _sig(m: Mapping[Hashable, Hashable]) -> Tuple[Tuple[Hashable, ...], ...]:
            return tuple(tuple(sorted(m[p] for p in orb)) for orb in orbits)

        return self._dedup_by_signature(mappings, _sig)

    def _dedup_pattern_orbits_with_anchor(
        self,
        mappings: List[Mapping[Hashable, Hashable]],
        orbits: Tuple[Tuple[Hashable, ...], ...],
        anchor: FrozenSet[Hashable],
    ) -> List[Mapping[Hashable, Hashable]]:
        """
        Pattern-orbit dedup with anchor (no pruning inside anchor nodes).

        :param mappings: Mappings.
        :type mappings: list[Mapping[hashable, hashable]]
        :param orbits: Sorted orbit tuples.
        :type orbits: tuple[tuple[hashable, ...], ...]
        :param anchor: Anchor nodes.
        :type anchor: frozenset[hashable]

        :returns: Deduplicated mappings.
        :rtype: list[Mapping[hashable, hashable]]
        """
        anchor_nodes = tuple(sorted(anchor, key=lambda x: x))
        free_orbits = tuple(orb for orb in orbits if not (set(orb) & anchor))

        def _sig(m: Mapping[Hashable, Hashable]) -> Tuple[Any, ...]:
            free_part = tuple(tuple(sorted(m[p] for p in orb)) for orb in free_orbits)
            anchor_part = tuple(m[p] for p in anchor_nodes)
            return (free_part, anchor_part)

        return self._dedup_by_signature(mappings, _sig)


# --------------------------------------------------------------------------- #
# Convenience function (kept minimal)
# --------------------------------------------------------------------------- #
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
    :param node_attrs: Node attribute keys to include in WL initialization.
    :type node_attrs: iterable[str] or None
    :param edge_attrs: Edge attribute keys to include in neighbor signatures.
    :type edge_attrs: iterable[str] or None
    :param max_iter: Maximum WL iterations.
    :type max_iter: int

    :returns: Fitted estimator.
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
