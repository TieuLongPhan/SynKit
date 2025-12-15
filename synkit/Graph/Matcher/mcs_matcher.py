"""mcs_matcher.py — Maximum/Common Subgraph Matcher
===================================================

A convenience wrapper around :class:`networkx.algorithms.isomorphism.GraphMatcher`
that finds *all* common-subgraph (or maximum-common-subgraph) node mappings
between two molecular graphs.

Highlights
----------
* **Flexible node matching** via :func:`generic_node_match`.
* **Multi-attribute edge matching** via a list of ``edge_attrs``.
* Optional **wildcard pruning**: if ``prune_wc=True``, nodes with
  ``attrs[element_key] == wildcard_element`` are removed (non-inplace)
  from both graphs before searching.
* Optional **automorphism pruning**: if ``prune_automorphisms=True``,
  mappings that cover the same host-node set are collapsed, greatly
  reducing equivalent mappings from symmetric subgraphs.
* Results are **cached** – call :py:meth:`get_mappings` (or use the
  :pyattr:`mappings` property) to retrieve them.
* Can report mappings as pattern→host, G1→G2 or G2→G1 via the
  :py:meth:`get_mappings` helper.
* Helpful :pyattr:`help` and :py:meth:`__repr__` utilities, in the same
  OOP style as :class:`PartialMatcher`.

Public API
~~~~~~~~~~
``MCSMatcher(node_attrs=None,
            node_defaults=None,
            allow_shift=True,
            edge_attrs=None,
            prune_wc=False,
            prune_automorphisms=False,
            wildcard_element='*',
            element_key='element')``
    Construct a matcher instance.

``matcher.find_common_subgraph(G1, G2, mcs=False, mcs_mol=False)``
    Run the search (stores and returns ``self``). If ``mcs_mol=True``,
    match by entire connected components (molecule-level matching).

``matcher.get_mappings(direction='pattern_to_host')``
    Retrieve the stored mapping list. ``direction`` can be one of
    ``"pattern_to_host"``, ``"G1_to_G2"``, ``"G2_to_G1"``.

``matcher.mappings``
    Shorthand for ``get_mappings(direction='pattern_to_host')``.

``matcher.mapping_direction``
    String describing internal orientation: ``"G1_to_G2"``, ``"G2_to_G1"``,
    or ``"unknown"`` if no search has been run yet.

``matcher.find_rc_mapping(rc1, rc2,
                         side='op',
                         mcs=True,
                         mcs_mol=False,
                         component=True)``
    Convenience wrapper for ITS reaction-centre or ITS-like graph
    objects (via :func:`synkit.Graph.ITS.its_decompose` when applicable).
    ``side`` chooses which ITS sides to compare:

    * ``'r'``:   compare right sides (``r1`` vs ``r2``)
    * ``'l'``:   compare left  sides (``l1`` vs ``l2``)
    * ``'op'``:  compare opposite  (``r1`` vs ``l2``)
    * ``'its'``: treat ``rc1`` and ``rc2`` directly as graphs (no
      decomposition).
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher, generic_node_match

try:
    from synkit.Graph.ITS import its_decompose  # optional
except ImportError:  # pragma: no cover – allow standalone use
    its_decompose = None  # type: ignore[assignment]

__all__ = ["MCSMatcher"]

MappingDict = Dict[int, int]


class MCSMatcher:
    """
    Common / maximum-common subgraph matcher.

    This class wraps :class:`networkx.algorithms.isomorphism.GraphMatcher`
    to provide higher-level utilities for computing sets of common subgraphs
    between two graphs, with a focus on molecular graphs (atoms/bonds).

    Node matching is controlled via :func:`generic_node_match` using one or
    more attribute names and default values. Edge matching compares one or
    more scalar edge attributes (e.g. bond order) specified in
    :paramref:`edge_attrs`.

    Mappings and orientation
    ------------------------
    Internally, mappings are always stored as **pattern → host**, where
    the *pattern* is the smaller of the two (after optional wildcard
    pruning). The helper :py:meth:`get_mappings` can convert these to
    ``G1→G2`` or ``G2→G1`` orientation as needed, and the property
    :pyattr:`mapping_direction` exposes which graph acted as the pattern.

    Optional wildcard pruning
    -------------------------
    If :paramref:`prune_wc` is ``True``, nodes with
    ``attrs[element_key] == wildcard_element`` are removed from both input
    graphs **non-inplace** before the MCS search. Node labels are preserved,
    so the resulting mappings still reference the original node ids.

    Optional automorphism pruning
    -----------------------------
    If :paramref:`prune_automorphisms` is ``True``, mappings that induce
    the same **host node set** (i.e. same image in the host graph) are
    collapsed. This is especially useful for highly symmetric hosts (rings,
    repeated subunits, etc.) where many mappings are equivalent at the level
    of “which part of the host is covered”.

    :param node_attrs: Node attribute keys to compare. If ``None``,
        defaults to ``["element"]``.
    :type node_attrs: list[str] | None
    :param node_defaults: Fallback values for each node attribute
        when missing. If ``None``, defaults to a list of ``"*"``
        of the same length as :paramref:`node_attrs`.
    :type node_defaults: list[Any] | None
    :param allow_shift: Placeholder for future asymmetric rules. Currently
        unused but kept for API compatibility.
    :type allow_shift: bool
    :param edge_attrs: Edge attribute keys to use for scalar comparison
        (e.g. ``["order"]`` or ``["order", "standard_order"]``). If
        ``None``, defaults to ``["order"]``.
    :type edge_attrs: list[str] | None
    :param prune_wc: If ``True``, strip wildcard nodes (see
        :paramref:`wildcard_element`, :paramref:`element_key`) from both
        graphs before searching.
    :type prune_wc: bool
    :param prune_automorphisms: If ``True``, collapse mappings that have
        the same host node set (automorphism pruning).
    :type prune_automorphisms: bool
    :param wildcard_element: Attribute value denoting wildcard nodes
        (typically ``"*"``, used together with :paramref:`element_key`).
    :type wildcard_element: Any
    :param element_key: Node attribute key used to detect wildcard nodes
        when :paramref:`prune_wc` is ``True``.
    :type element_key: str
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        node_attrs: Optional[List[str]] = None,
        node_defaults: Optional[List[Any]] = None,
        allow_shift: bool = True,
        *,
        edge_attrs: Optional[List[str]] = None,
        prune_wc: bool = False,
        prune_automorphisms: bool = False,
        wildcard_element: Any = "*",
        element_key: str = "element",
    ) -> None:
        if node_attrs is None:
            node_attrs = ["element"]
        if node_defaults is None:
            node_defaults = ["*"] * len(node_attrs)
        if len(node_defaults) != len(node_attrs):
            raise ValueError(
                "MCSMatcher: node_defaults must have the same length as node_attrs."
            )

        self._node_attrs: List[str] = node_attrs
        self._node_defaults: List[Any] = node_defaults
        self._edge_attrs: List[str] = edge_attrs or ["order"]
        self.allow_shift: bool = allow_shift

        self.prune_wc: bool = prune_wc
        self.prune_automorphisms: bool = prune_automorphisms
        self.wildcard_element: Any = wildcard_element
        self.element_key: str = element_key

        comparators: List[Callable[[Any, Any], bool]] = [
            (lambda x, y: x == y) for _ in node_attrs
        ]
        self.node_match: Callable[[Dict[str, Any], Dict[str, Any]], bool] = (
            generic_node_match(
                node_attrs,
                node_defaults,
                comparators,
            )
        )

        # Internal cache
        self._mappings: List[MappingDict] = []  # pattern -> host
        self._last_size: int = 0
        self._last_pattern_is_G1: Optional[bool] = None  # None until a search runs

    # ------------------------------------------------------------------
    # Internal edge / mapping helpers
    # ------------------------------------------------------------------
    def _edge_match(
        self,
        host_attrs: Dict[str, Any],
        pat_attrs: Dict[str, Any],
    ) -> bool:
        """
        Compare edge attributes listed in :pyattr:`_edge_attrs`.

        For each name in :pyattr:`_edge_attrs`:

        * If both values can be cast to ``float``, use numeric equality.
        * Otherwise, fall back to direct ``==`` comparison.
        * Missing values on both sides are ignored.

        :param host_attrs: Edge attribute dictionary from host graph.
        :type host_attrs: dict[str, Any]
        :param pat_attrs: Edge attribute dictionary from pattern graph.
        :type pat_attrs: dict[str, Any]
        :returns: ``True`` if all configured edge attributes match, otherwise
            ``False``.
        :rtype: bool
        """
        for name in self._edge_attrs:
            hv = host_attrs.get(name, None)
            pv = pat_attrs.get(name, None)
            if hv is None and pv is None:
                continue
            try:
                if float(hv) != float(pv):
                    return False
            except (TypeError, ValueError):
                if hv != pv:
                    return False
        return True

    @staticmethod
    def _invert_mapping(gm_mapping: MappingDict) -> MappingDict:
        """
        Convert *host→pattern* dict to *pattern→host*.

        :param gm_mapping: Mapping from host nodes to pattern nodes.
        :type gm_mapping: dict[int, int]
        :returns: Mapping from pattern nodes to host nodes.
        :rtype: dict[int, int]
        """
        return {pat: host for host, pat in gm_mapping.items()}

    def _prune_graph(self, G: nx.Graph) -> nx.Graph:
        """
        Return a version of ``G`` with wildcard nodes removed if needed.

        This method does **not** mutate the input graph. When
        :pyattr:`prune_wc` is ``False``, it simply returns ``G`` as-is.

        :param G: Input graph.
        :type G: nx.Graph
        :returns: Possibly pruned graph (wildcard nodes removed).
        :rtype: nx.Graph
        """
        if not self.prune_wc:
            return G

        key = self.element_key
        wc = self.wildcard_element
        keep_nodes = [n for n, data in G.nodes(data=True) if data.get(key) != wc]
        return G.subgraph(keep_nodes).copy()

    def _componentwise_mcs(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        *,
        mcs: bool,
    ) -> MappingDict:
        """
        Perform size-sorted, component-wise MCS between ``G1`` and ``G2``.

        The graphs are decomposed into connected components, which are
        sorted by size (descending). Components are then matched in
        order (largest with largest, etc.) using the internal subgraph-
        search machinery.

        The combined mapping is always reported as **G1 → G2** in terms
        of the original node ids; components that fail to yield any
        mapping are simply skipped.

        :param G1: First input graph.
        :type G1: nx.Graph
        :param G2: Second input graph.
        :type G2: nx.Graph
        :param mcs: If ``True``, restrict each component pair to
            maximum-common-subgraph mappings.
        :type mcs: bool
        :returns: Combined mapping from nodes of ``G1`` to nodes of
            ``G2``.
        :rtype: dict[int, int]
        """
        comps1 = [G1.subgraph(c).copy() for c in nx.connected_components(G1)]
        comps2 = [G2.subgraph(c).copy() for c in nx.connected_components(G2)]

        comps1.sort(key=lambda g: g.number_of_nodes(), reverse=True)
        comps2.sort(key=lambda g: g.number_of_nodes(), reverse=True)

        limit = min(len(comps1), len(comps2))
        combined: MappingDict = {}

        for idx in range(limit):
            sub1 = comps1[idx]
            sub2 = comps2[idx]

            pattern, host, pattern_is_G1 = self._prepare_orientation(sub1, sub2)
            local_maps = self._search_subgraphs(pattern, host, mcs=mcs)
            if not local_maps:
                continue

            best = local_maps[0]
            if pattern_is_G1:
                # pattern is subgraph of G1 → host is subgraph of G2
                combined.update(best)
            else:
                # pattern is subgraph of G2, host is subgraph of G1 → invert
                combined.update(self._invert_mapping(best))

        return combined

    # ------------------------------------------------------------------
    # Connected-component (molecule) level matching
    # ------------------------------------------------------------------
    def _find_mcs_mol(self, G1: nx.Graph, G2: nx.Graph) -> MappingDict:
        """
        Match connected components of ``G1`` to ``G2`` of the same size.

        Components are sorted by size (descending) and matched greedily.
        For each component in ``G1``, the method looks for a component in
        ``G2`` with the same size and an isomorphic mapping, combining
        component-mappings into a single dictionary.

        Direction: the returned mapping is **G1 → G2** in this mode.

        :param G1: First graph (treated as source of components).
        :type G1: nx.Graph
        :param G2: Second graph (target for component mapping).
        :type G2: nx.Graph
        :returns: Combined mapping from nodes of ``G1`` to nodes of ``G2``.
        :rtype: dict[int, int]
        """
        comps1 = sorted(nx.connected_components(G1), key=len, reverse=True)
        comps2 = sorted(nx.connected_components(G2), key=len, reverse=True)

        used2: Set[frozenset[int]] = set()
        combined: MappingDict = {}

        for comp1 in comps1:
            size = len(comp1)
            sub1 = G1.subgraph(comp1)

            for comp2 in comps2:
                if len(comp2) != size:
                    continue
                key2 = frozenset(comp2)
                if key2 in used2:
                    continue

                sub2 = G2.subgraph(comp2)
                gm = GraphMatcher(
                    sub1,
                    sub2,
                    node_match=self.node_match,
                    edge_match=self._edge_match,
                )
                if gm.is_isomorphic():
                    combined.update(gm.mapping)
                    used2.add(key2)
                    break

        return combined

    # ------------------------------------------------------------------
    # Core subgraph search
    # ------------------------------------------------------------------
    def _prepare_orientation(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
    ) -> Tuple[nx.Graph, nx.Graph, bool]:
        """
        Ensure the smaller graph is used as pattern for efficiency.

        :param G1: Original first graph.
        :type G1: nx.Graph
        :param G2: Original second graph.
        :type G2: nx.Graph
        :returns: Tuple ``(pattern, host, pattern_is_G1)`` where
            ``pattern_is_G1=True`` indicates that the pattern is ``G1``.
        :rtype: tuple[nx.Graph, nx.Graph, bool]
        """
        if G1.number_of_nodes() <= G2.number_of_nodes():
            return G1, G2, True
        return G2, G1, False

    def _search_subgraphs(
        self,
        pattern: nx.Graph,
        host: nx.Graph,
        *,
        mcs: bool,
    ) -> List[MappingDict]:
        """
        Enumerate common subgraphs between ``pattern`` and ``host``.

        The returned mappings always map **pattern nodes → host nodes**.

        If :pyattr:`prune_automorphisms` is ``True``, mappings with the
        same host-node set are collapsed.

        :param pattern: Graph treated as pattern (smaller or equal).
        :type pattern: nx.Graph
        :param host: Graph treated as host (larger or equal).
        :type host: nx.Graph
        :param mcs: If ``True``, retain only maximum-size mappings.
        :type mcs: bool
        :returns: List of mappings from pattern nodes to host nodes.
        :rtype: list[MappingDict]
        """
        max_k = min(pattern.number_of_nodes(), host.number_of_nodes())
        seen: Set[Tuple[Tuple[int, int], ...]] = set()
        mappings: List[MappingDict] = []
        best_size = 0

        # For automorphism pruning: track host node sets we have already seen
        host_sets_seen: Set[frozenset[Any]] = set()

        for k in range(max_k, 0, -1):
            if mcs and best_size and k < best_size:
                break

            level_found = False
            for nodes in itertools.combinations(pattern.nodes(), k):
                sub_pat = pattern.subgraph(nodes).copy()
                gm = GraphMatcher(
                    host,
                    sub_pat,
                    node_match=self.node_match,
                    edge_match=self._edge_match,
                )
                for iso in gm.subgraph_isomorphisms_iter():
                    inv = self._invert_mapping(iso)  # pattern -> host
                    key = tuple(sorted(inv.items()))
                    if key in seen:
                        continue
                    seen.add(key)

                    if self.prune_automorphisms:
                        host_set = frozenset(inv.values())
                        if host_set in host_sets_seen:
                            # Same image in host as an earlier mapping -> skip
                            continue
                        host_sets_seen.add(host_set)

                    mappings.append(inv)
                    level_found = True

            if level_found:
                best_size = k
                if mcs:
                    break

        if mcs and best_size:
            mappings = [m for m in mappings if len(m) == best_size]

        mappings.sort(key=lambda d: (-len(d), tuple(sorted(d.items()))))
        self._last_size = (
            best_size if best_size else (len(mappings[0]) if mappings else 0)
        )
        return mappings

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------
    def find_common_subgraph(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        *,
        mcs: bool = False,
        mcs_mol: bool = False,
    ) -> "MCSMatcher":
        """
        Search for common subgraphs between two graphs.

        The results are cached in :pyattr:`mappings` and
        :pyattr:`last_size`. The method returns ``self`` to enable a
        fluent style.

        If :pyattr:`prune_wc` is ``True``, wildcard nodes are stripped
        (non-inplace) from both graphs before the search.

        :param G1: First input graph.
        :type G1: nx.Graph
        :param G2: Second input graph.
        :type G2: nx.Graph
        :param mcs: If ``True``, restrict to maximum-common-subgraph
            mappings (largest possible node count).
        :type mcs: bool
        :param mcs_mol: If ``True``, perform connected-component
            (molecule-level) matching using :py:meth:`_find_mcs_mol`.
            In this mode, ``mcs`` is ignored.
        :type mcs_mol: bool
        :returns: The matcher instance (with internal cache updated).
        :rtype: MCSMatcher
        """
        self._mappings = []
        self._last_size = 0
        self._last_pattern_is_G1 = None

        G1_use = self._prune_graph(G1)
        G2_use = self._prune_graph(G2)

        if mcs_mol:
            combined = self._find_mcs_mol(G1_use, G2_use)
            self._mappings = [combined]  # G1 -> G2
            self._last_size = len(combined)
            self._last_pattern_is_G1 = True
            return self

        pattern, host, pattern_is_G1 = self._prepare_orientation(G1_use, G2_use)
        self._last_pattern_is_G1 = pattern_is_G1
        self._mappings = self._search_subgraphs(pattern, host, mcs=mcs)

        return self

    def find_rc_mapping(
        self,
        rc1: Any,
        rc2: Any,
        *,
        side: str = "op",
        mcs: bool = True,
        mcs_mol: bool = False,
        component: bool = True,
    ) -> "MCSMatcher":
        """
        Convenience wrapper for ITS reaction-centre or ITS-like graph
        objects.

        Depending on :paramref:`side`, this either uses
        :func:`synkit.Graph.ITS.its_decompose` to obtain left/right
        graphs or treats the inputs directly as graphs.

        Side selection
        --------------
        * ``'r'``   → compare right sides:  ``r1`` vs ``r2``.
        * ``'l'``   → compare left sides:   ``l1`` vs ``l2``.
        * ``'op'``  → compare opposite:     ``r1`` vs ``l2``.
        * ``'its'`` → treat ``rc1`` and ``rc2`` directly as graphs
          (no decomposition), useful when the inputs are already ITS
          (or ITS-like) :class:`networkx.Graph` objects.

        Component-wise mode
        -------------------
        If :paramref:`component` is ``True``, the selected graphs are
        decomposed into connected components, sorted by size
        (descending), and matched pairwise (largest with largest, etc.)
        using a common-/maximum-common-subgraph search for each pair.
        The resulting mappings are combined into a single **G1 → G2**
        mapping in terms of the original node ids. In this mode,
        :paramref:`mcs_mol` is ignored.

        :param rc1: First reaction-centre or ITS-like graph object.
        :type rc1: Any
        :param rc2: Second reaction-centre or ITS-like graph object.
        :type rc2: Any
        :param side: Which ITS sides to compare (``'r'``, ``'l'``,
            ``'op'``, or ``'its'``).
        :type side: str
        :param mcs: If ``True``, restrict to maximum-common-subgraph
            mappings (for the whole graph or per-component in
            component-wise mode).
        :type mcs: bool
        :param mcs_mol: If ``True``, use connected-component matching
            via :py:meth:`_find_mcs_mol`. Ignored if
            :paramref:`component` is ``True``.
        :type mcs_mol: bool
        :param component: If ``True``, perform size-sorted,
            component-wise MCS between the selected sides and combine
            the per-component mappings into a single mapping.
        :type component: bool
        :returns: The matcher instance (with internal cache updated).
        :rtype: MCSMatcher
        :raises ImportError: If :mod:`synkit` ITS utilities are not
            available for ``side`` in ``{'r', 'l', 'op'}``.
        :raises ValueError: If ``side`` is not one of
            ``'r'``, ``'l'``, ``'op'``, ``'its'``.
        """
        # reset cache
        self._mappings = []
        self._last_size = 0
        self._last_pattern_is_G1 = None

        side_norm = side.lower()

        if side_norm == "its":
            # Treat rc1 and rc2 directly as graphs
            G1, G2 = rc1, rc2
        else:
            if its_decompose is None:
                raise ImportError(
                    "synkit is not available; cannot decompose reaction centres "
                    "for side values 'r', 'l' or 'op'."
                )

            l1, r1 = its_decompose(rc1)
            l2, r2 = its_decompose(rc2)

            if side_norm == "r":
                G1, G2 = r1, r2
            elif side_norm == "l":
                G1, G2 = l1, l2
            elif side_norm == "op":
                G1, G2 = r1, l2
            else:
                raise ValueError(
                    "MCSMatcher.find_rc_mapping: side must be one of "
                    "'r', 'l', 'op', 'its', got "
                    f"{side!r}."
                )

        if component:
            G1_use = self._prune_graph(G1)
            G2_use = self._prune_graph(G2)
            combined = self._componentwise_mcs(G1_use, G2_use, mcs=mcs)
            self._mappings = [combined]  # G1 -> G2
            self._last_size = len(combined)
            self._last_pattern_is_G1 = True
            return self

        return self.find_common_subgraph(G1, G2, mcs=mcs, mcs_mol=mcs_mol)

    # ------------------------------------------------------------------
    # Accessors / properties
    # ------------------------------------------------------------------
    def get_mappings(self, direction: str = "pattern_to_host") -> List[MappingDict]:
        """
        Return a copy of the cached mapping list in the requested orientation.

        Internal cache is **pattern → host** (where pattern is the smaller
        graph after pruning). This method can convert to original
        ``G1→G2`` or ``G2→G1`` orientation based on the last call to
        :py:meth:`find_common_subgraph`.

        :param direction: Orientation of the returned mappings. One of:
            - ``"pattern_to_host"`` (default): internal orientation.
            - ``"G1_to_G2"``: mapping from first input graph to second.
            - ``"G2_to_G1"``: mapping from second input graph to first.
        :type direction: str
        :returns: List of node-mapping dictionaries.
        :rtype: list[dict[int, int]]
        :raises ValueError: If ``direction`` is not supported.
        """
        if direction == "pattern_to_host" or self._last_pattern_is_G1 is None:
            return [dict(m) for m in self._mappings]

        if direction not in {"G1_to_G2", "G2_to_G1"}:
            raise ValueError(
                "MCSMatcher.get_mappings: direction must be one of "
                "'pattern_to_host', 'G1_to_G2', 'G2_to_G1'."
            )

        pattern_is_G1 = self._last_pattern_is_G1
        result: List[MappingDict] = []

        for m in self._mappings:
            if direction == "G1_to_G2":
                if pattern_is_G1:
                    result.append(dict(m))
                else:
                    result.append(self._invert_mapping(m))
            else:  # direction == "G2_to_G1"
                if pattern_is_G1:
                    result.append(self._invert_mapping(m))
                else:
                    result.append(dict(m))

        return result

    @property
    def mappings(self) -> List[MappingDict]:
        """
        Cached node mappings from the most recent search (pattern→host).

        To obtain G1→G2 or G2→G1 orientation, use
        :py:meth:`get_mappings` with ``direction='G1_to_G2'`` or
        ``direction='G2_to_G1'``.

        :returns: List of node-mapping dictionaries (pattern→host).
        :rtype: list[dict[int, int]]
        """
        return self.get_mappings(direction="pattern_to_host")

    @property
    def last_size(self) -> int:
        """
        Number of nodes in the most recent maximum mapping set.

        This is the size of the largest mapping found in the last call
        to :py:meth:`find_common_subgraph` (or zero if no mappings
        exist).

        :returns: Size of the largest mapping.
        :rtype: int
        """
        return self._last_size

    @property
    def num_mappings(self) -> int:
        """
        Number of mappings stored from the most recent search.

        :returns: Count of cached mappings.
        :rtype: int
        """
        return len(self._mappings)

    @property
    def mapping_direction(self) -> str:
        """
        Human-readable description of internal mapping orientation.

        :returns: ``"G1_to_G2"``, ``"G2_to_G1"``, or ``"unknown"`` if
            no search has been run yet.
        :rtype: str
        """
        if self._last_pattern_is_G1 is None:
            return "unknown"
        return "G1_to_G2" if self._last_pattern_is_G1 else "G2_to_G1"

    # ------------------------------------------------------------------
    # Iteration & niceties
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterable[MappingDict]:
        """
        Iterate over cached mappings in pattern→host orientation.

        :returns: Iterator over mapping dictionaries.
        :rtype: Iterable[dict[int, int]]
        """
        return iter(self._mappings)

    def __repr__(self) -> str:
        """
        Short textual representation for debugging.

        :returns: Summary string with key attributes.
        :rtype: str
        """
        return (
            f"<MCSMatcher mappings={self.num_mappings} "
            f"last_size={self.last_size} "
            f"prune_wc={self.prune_wc} "
            f"prune_automorphisms={self.prune_automorphisms} "
            f"direction={self.mapping_direction}>"
        )

    __str__ = __repr__

    @property
    def help(self) -> str:
        """
        Return the module-level documentation string.

        :returns: The full module docstring, if available.
        :rtype: str
        """
        return __doc__ or ""
