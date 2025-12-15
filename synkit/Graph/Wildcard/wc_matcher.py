from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Set, Tuple, Union
from collections import Counter
import logging
import numbers

import networkx as nx
from networkx.algorithms import isomorphism

GraphType = Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]

__all__ = ["WCMatcher", "WildcardPatternMatcher"]


class WCMatcher:
    """
    Wildcard-aware pattern→host matcher with subgraph regions for wildcard nodes.

    Semantics
    ---------
    • The graph containing nodes with ``attrs[element_key] == wildcard_element``
      is treated as the **pattern**.
    • Wildcard nodes (``element == "*"`` by default) are **removed** from the
      pattern when computing the *core isomorphism*.
    • Only the **core** pattern nodes (non-wildcards) are matched against the
      host using :class:`networkx.algorithms.isomorphism.GraphMatcher`.
    • Wildcard nodes are treated as “don’t care” substituents attached to core
      atoms. Their **exact number and structure are ignored** – we only ensure
      that the core scaffold is present in the host.
    • After a core mapping is found, each wildcard node is associated with a
      **host subgraph region**, obtained as neighbours of its mapped anchor
      nodes in the host.

    Attribute matching
    ------------------
    Node attributes (for keys listed in :pydata:`node_attrs`):

    • ``str`` and ``bool``: must match **exactly**.
    • ``int`` and ``float``: pattern value ≤ host value (lower-bound semantics).
    • Special handling for a ``"neighbors"`` attribute if present:
        - Pattern neighbour list may contain wildcard entries (e.g. ``"*"``
          for any element).
        - Concrete neighbour labels in the pattern act as **lower bounds** on
          host counts (multiset inclusion).
        - Wildcard neighbour labels impose no constraint.

    Edge attributes (for keys listed in :pydata:`edge_attrs`):

    • Same typed semantics as node attributes (str/bool exact, numeric ≤).

    Typical usage
    -------------
    .. code-block:: python

        import networkx as nx
        from synkit.Graph.Matcher.wc_matcher import WCMatcher

        # Host: C-C-C chain
        host = nx.path_graph(3)
        nx.set_node_attributes(host, "C", "element")

        # Pattern: C-C-* (wildcard substituent on the second carbon)
        pattern = nx.path_graph(3)
        nx.set_node_attributes(pattern, "C", "element")
        pattern.nodes[2]["element"] = "*"  # wildcard node

        matcher = WCMatcher(
            pattern,
            host,
            wildcard_element="*",
            element_key="element",
            node_attrs=["element"],
            edge_attrs=[],
        ).fit()

        if matcher.is_match:
            print("Core mapping:", matcher.core_mapping_without_wildcard_regions)
            print("Wildcard regions:", matcher.wildcard_subgraph_mapping)

    :param G1: First input graph.
    :type G1: GraphType
    :param G2: Second input graph.
    :type G2: GraphType
    :param wildcard_element: Node attribute value that denotes a wildcard
        in the pattern. Defaults to ``"*"``
    :type wildcard_element: str
    :param element_key: Name of the node attribute storing the element /
        atom type. Defaults to ``"element"``.
    :type element_key: str
    :param node_attrs: Node attribute keys to be checked in addition to
        :pydata:`element_key`. If ``"neighbors"`` is included, neighbour
        lists are compared with lower-bound / wildcard semantics.
    :type node_attrs: Sequence[str] | None
    :param edge_attrs: Edge attribute keys to be checked with typed
        semantics.
    :type edge_attrs: Sequence[str] | None
    :param node_match: Optional additional node predicate, combined with
        the default matcher. Signature:
        ``(host_attr: dict, pattern_attr: dict) -> bool``.
    :type node_match: Callable[[Dict[str, Any], Dict[str, Any]], bool] | None
    :param edge_match: Optional additional edge predicate, combined with
        the default matcher. Signature:
        ``(host_attr: dict, pattern_attr: dict) -> bool``.
    :type edge_match: Callable[[Dict[str, Any], Dict[str, Any]], bool] | None
    :param logger: Optional logger for diagnostics.
    :type logger: logging.Logger | None
    """

    # ------------------------------------------------------------------ init / basic props

    def __init__(
        self,
        G1: GraphType,
        G2: GraphType,
        *,
        wildcard_element: str = "*",
        element_key: str = "element",
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
        node_match: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
        edge_match: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._element_key = element_key
        self._wildcard_element = wildcard_element
        self._user_node_match = node_match
        self._user_edge_match = edge_match
        self._node_attrs = tuple(node_attrs or [])
        self._edge_attrs = tuple(edge_attrs or [])

        self._host: GraphType
        self._pattern: GraphType
        self._pattern_is_G1: bool

        self._host, self._pattern, self._pattern_is_G1 = self._choose_host_pattern(
            G1, G2
        )

        # Core mapping: pattern(non-wc node) -> host node
        self._core_mapping: Dict[Any, Any] = {}
        self._is_match: bool = False

    def __repr__(self) -> str:
        """
        String representation for debugging.

        :returns: Short summary including match status and node counts.
        :rtype: str
        """
        status = "matched" if self._is_match else "unmatched"
        return (
            f"<WCMatcher status={status} "
            f"n_pattern={self._pattern.number_of_nodes()} "
            f"n_host={self._host.number_of_nodes()}>"
        )

    # ------------------------------------------------------------------ public API

    def fit(self) -> "WCMatcher":
        """
        Run the wildcard-aware core isomorphism search.

        The method computes the **core** subgraph isomorphism (ignoring
        wildcard nodes) and stores the mapping internally. Use
        :pyattr:`is_match` and :pyattr:`core_mapping_without_wildcard_regions`
        to inspect the result.

        :returns: Self, to allow fluent chaining.
        :rtype: WCMatcher
        """
        pattern_core = self._build_pattern_core()

        # Quick size check – if core is larger than host, no match is possible.
        if (
            pattern_core.number_of_nodes() > self._host.number_of_nodes()
            or pattern_core.number_of_edges() > self._host.number_of_edges()
        ):
            self._logger.debug(
                "WCMatcher: pattern core larger than host (no match possible)."
            )
            self._core_mapping = {}
            self._is_match = False
            return self

        gm = self._make_graph_matcher(pattern_core)

        for host_to_pattern_core in gm.subgraph_isomorphisms_iter():
            # networkx gives host→pattern mapping; we want pattern→host
            core_to_host = {p: h for h, p in host_to_pattern_core.items()}
            self._core_mapping = core_to_host
            self._is_match = True
            self._logger.debug("WCMatcher: core match found: %s", core_to_host)
            return self

        self._logger.debug("WCMatcher: no core mapping found.")
        self._core_mapping = {}
        self._is_match = False
        return self

    @property
    def is_match(self) -> bool:
        """
        Whether a wildcard-compatible **core** mapping was found.

        :returns: ``True`` if the core pattern matches a subgraph of the host.
        :rtype: bool
        """
        return self._is_match

    @property
    def pattern_graph(self) -> GraphType:
        """
        Graph that was treated as the **pattern** (may contain wildcards).

        :returns: Pattern graph.
        :rtype: GraphType
        """
        return self._pattern

    @property
    def host_graph(self) -> GraphType:
        """
        Graph that was treated as the **host**.

        :returns: Host graph.
        :rtype: GraphType
        """
        return self._host

    @property
    def core_mapping_without_wildcard_regions(self) -> Dict[Any, Any]:
        """
        Mapping from **non-wildcard pattern nodes → host nodes**.

        Wildcard nodes in the pattern are ignored in this mapping. This is the
        clean "core" mapping without any enlargement due to wildcard regions.

        :returns: Mapping from pattern-core nodes to host nodes.
        :rtype: dict[Any, Any]
        """
        return dict(self._core_mapping)

    @property
    def wildcard_subgraph_mapping(self) -> Dict[Any, Set[Any]]:
        """
        Mapping from each **wildcard pattern node** to a set of host nodes
        forming its wildcard subgraph region.

        Construction heuristic
        ----------------------
        For each wildcard node ``w`` in the pattern:

        1. Collect its **non-wildcard neighbour(s)** in the pattern.
        2. Map those neighbours to host anchors using the core mapping.
        3. For each anchor ``h`` in the host, add all neighbours of ``h``
           that are **not already used** by any core mapping.

        Notes
        -----
        • If there is no core mapping (``is_match == False``), the result is
          an empty dict.
        • If a wildcard's anchors cannot be mapped (e.g. missing in core),
          its region is an empty set.

        :returns: Mapping ``wildcard_pattern_node → set(host_nodes_in_region)``.
        :rtype: dict[Any, set[Any]]
        """
        if not self._is_match:
            return {}

        wc_nodes = [
            n
            for n, data in self._pattern.nodes(data=True)
            if data.get(self._element_key) == self._wildcard_element
        ]

        if not wc_nodes:
            return {}

        core_hosts: Set[Any] = set(self._core_mapping.values())
        regions: Dict[Any, Set[Any]] = {}

        for w in wc_nodes:
            neigh = list(self._pattern.neighbors(w))
            anchors = [
                p
                for p in neigh
                if self._pattern.nodes[p].get(self._element_key)
                != self._wildcard_element
                and p in self._core_mapping
            ]

            region: Set[Any] = set()
            for p in anchors:
                h_anchor = self._core_mapping[p]
                for h_nbr in self._host.neighbors(h_anchor):
                    if h_nbr not in core_hosts:
                        region.add(h_nbr)

            regions[w] = region

        return regions

    @property
    def help(self) -> str:
        """
        Human-readable summary of the matcher behaviour.

        :returns: Description string summarising semantics and attributes.
        :rtype: str
        """
        return (
            "WCMatcher (WildcardPatternMatcher):\n"
            "  • pattern = graph with wildcard nodes (element == wildcard_element)\n"
            "  • host    = other graph (or larger if both/none have wildcards)\n"
            "  • semantics:\n"
            "      - core match ignores wildcard nodes\n"
            "      - node attrs: str/bool exact, numeric pattern ≤ host\n"
            "      - neighbours: concrete labels are lower bounds; '*' is wildcard\n"
            "      - wildcard_subgraph_mapping gives host regions per wildcard node"
        )

    # ------------------------------------------------------------------ internals: host/pattern selection

    def _choose_host_pattern(
        self, G1: GraphType, G2: GraphType
    ) -> Tuple[GraphType, GraphType, bool]:
        """
        Decide which graph is host and which is pattern.

        Preference
        ----------
        1. Graph containing wildcard nodes is the pattern.
        2. If both (or neither) contain wildcards, the **smaller** graph is
           treated as the pattern.

        :param G1: First input graph.
        :type G1: GraphType
        :param G2: Second input graph.
        :type G2: GraphType
        :returns: Tuple ``(host, pattern, pattern_is_G1)``.
        :rtype: tuple[GraphType, GraphType, bool]
        """
        has_wc1 = self._graph_has_wildcard(G1)
        has_wc2 = self._graph_has_wildcard(G2)

        if has_wc1 and not has_wc2:
            return G2, G1, True
        if has_wc2 and not has_wc1:
            return G1, G2, False

        if G1.number_of_nodes() <= G2.number_of_nodes():
            return G2, G1, True
        return G1, G2, False

    def _graph_has_wildcard(self, G: GraphType) -> bool:
        """
        Check whether a graph contains at least one wildcard node.

        :param G: Graph to inspect.
        :type G: GraphType
        :returns: ``True`` if a wildcard node is present.
        :rtype: bool
        """
        key = self._element_key
        wc = self._wildcard_element
        for _, data in G.nodes(data=True):
            if data.get(key) == wc:
                return True
        return False

    # ------------------------------------------------------------------ internals: pattern core construction

    def _build_pattern_core(self) -> GraphType:
        """
        Build the core pattern graph (non-wildcard nodes only).

        Wildcard nodes (with ``element == wildcard_element``) are removed,
        and the induced subgraph is returned.

        :returns: Core pattern graph without wildcard nodes.
        :rtype: GraphType
        """
        key = self._element_key
        wc = self._wildcard_element

        core_nodes: Set[Any] = {
            n for n, data in self._pattern.nodes(data=True) if data.get(key) != wc
        }

        return self._pattern.subgraph(core_nodes).copy()

    # ------------------------------------------------------------------ internals: typed attribute comparison

    def _typed_leq(self, host_val: Any, pat_val: Any) -> bool:
        """
        Typed comparison of attribute values.

        Rules
        -----
        • If ``pat_val`` is ``None``: no constraint → ``True``.
        • If both values are numeric (int/float): require ``host_val >= pat_val``.
        • If both are ``str`` or both are ``bool``: require equality.
        • Otherwise: fall back to equality.

        :param host_val: Attribute value from the host graph.
        :type host_val: Any
        :param pat_val: Attribute value from the pattern graph.
        :type pat_val: Any
        :returns: ``True`` if ``host_val`` satisfies the pattern constraint.
        :rtype: bool
        """
        if pat_val is None:
            return True

        if isinstance(host_val, numbers.Number) and isinstance(pat_val, numbers.Number):
            return host_val >= pat_val

        if isinstance(host_val, str) and isinstance(pat_val, str):
            return host_val == pat_val
        if isinstance(host_val, bool) and isinstance(pat_val, bool):
            return host_val == pat_val

        return host_val == pat_val

    def _neighbors_compatible(
        self,
        host_neigh: Optional[Iterable[Any]],
        pat_neigh: Optional[Iterable[Any]],
    ) -> bool:
        """
        Compare neighbour \"element\" lists with wildcard & lower-bound semantics.

        • Pattern neighbour list may contain wildcard entries
          (``self._wildcard_element``).
        • Concrete labels in the pattern act as **lower bounds** on host counts:
          ``count_host[label] >= count_pattern[label]``.
        • Wildcards in the pattern impose no constraint on the host.

        :param host_neigh: Neighbour labels in the host (e.g. list of element
            symbols).
        :type host_neigh: Iterable[Any] | None
        :param pat_neigh: Neighbour labels in the pattern.
        :type pat_neigh: Iterable[Any] | None
        :returns: ``True`` if host neighbours satisfy the pattern constraints.
        :rtype: bool
        """
        if pat_neigh is None:
            return True

        wc = self._wildcard_element
        host_neigh = list(host_neigh or [])
        pat_neigh = list(pat_neigh)

        pat_counts = Counter(lbl for lbl in pat_neigh if lbl != wc)
        host_counts = Counter(host_neigh)

        for lbl, p_cnt in pat_counts.items():
            if host_counts[lbl] < p_cnt:
                return False

        return True

    # ------------------------------------------------------------------ internals: attribute-based node / edge checks

    def _node_attrs_ok(
        self,
        host_attr: Dict[str, Any],
        pat_attr: Dict[str, Any],
        *,
        key: str,
        wc: str,
    ) -> bool:
        """
        Check node-level attributes including element & neighbours.

        :param host_attr: Host node attribute dictionary.
        :type host_attr: dict[str, Any]
        :param pat_attr: Pattern node attribute dictionary.
        :type pat_attr: dict[str, Any]
        :param key: Element attribute key (e.g. ``"element"``).
        :type key: str
        :param wc: Wildcard element value.
        :type wc: str
        :returns: ``True`` if host node satisfies pattern node constraints.
        :rtype: bool
        """
        host_el = host_attr.get(key)
        pat_el = pat_attr.get(key)
        if pat_el is not None and pat_el != wc:
            if host_el != pat_el:
                return False

        for name in self._node_attrs:
            if name == "neighbors":
                if not self._neighbors_compatible(
                    host_attr.get(name), pat_attr.get(name)
                ):
                    return False
            else:
                if not self._typed_leq(host_attr.get(name), pat_attr.get(name)):
                    return False

        return True

    def _edge_attrs_ok(
        self,
        host_attr: Dict[str, Any],
        pat_attr: Dict[str, Any],
    ) -> bool:
        """
        Check edge-level attributes using typed semantics.

        :param host_attr: Host edge attribute dictionary.
        :type host_attr: dict[str, Any]
        :param pat_attr: Pattern edge attribute dictionary.
        :type pat_attr: dict[str, Any]
        :returns: ``True`` if host edge satisfies pattern edge constraints.
        :rtype: bool
        """
        for name in self._edge_attrs:
            if not self._typed_leq(host_attr.get(name), pat_attr.get(name)):
                return False
        return True

    # ------------------------------------------------------------------ internals: GraphMatcher construction

    def _make_graph_matcher(self, pattern_core: GraphType) -> Any:
        """
        Build a NetworkX GraphMatcher for host vs. pattern_core.

        Default behaviour
        -----------------
        • Node match:
            - :pydata:`element_key`: exact, unless pattern element is
              :pydata:`wildcard_element`.
            - For keys in :pyattr:`_node_attrs`:
                * If key == ``"neighbors"``: use neighbour lower-bound semantics.
                * Else: typed comparison via :meth:`_typed_leq`.
            - Then apply optional :pydata:`node_match` predicate if given.
        • Edge match:
            - For keys in :pyattr:`_edge_attrs`: typed comparison via
              :meth:`_typed_leq`.
            - Then optional :pydata:`edge_match` predicate if given.

        :param pattern_core: Induced subgraph of the pattern on non-wildcard
            nodes.
        :type pattern_core: GraphType
        :returns: Instance of :class:`GraphMatcher` or
            :class:`MultiGraphMatcher` depending on the host type.
        :rtype: Any
        """
        key = self._element_key
        wc = self._wildcard_element
        user_node_match = self._user_node_match
        user_edge_match = self._user_edge_match

        def default_node_match(
            host_attr: Dict[str, Any], pat_attr: Dict[str, Any]
        ) -> bool:
            if not self._node_attrs_ok(host_attr, pat_attr, key=key, wc=wc):
                return False
            if user_node_match is not None and not user_node_match(host_attr, pat_attr):
                return False
            return True

        def default_edge_match(
            host_attr: Dict[str, Any], pat_attr: Dict[str, Any]
        ) -> bool:
            if not self._edge_attrs_ok(host_attr, pat_attr):
                return False
            if user_edge_match is not None and not user_edge_match(host_attr, pat_attr):
                return False
            return True

        if isinstance(self._host, (nx.MultiGraph, nx.MultiDiGraph)):
            matcher_cls = isomorphism.MultiGraphMatcher
        else:
            matcher_cls = isomorphism.GraphMatcher

        return matcher_cls(
            self._host,
            pattern_core,
            node_match=default_node_match,
            edge_match=default_edge_match,
        )


# Backwards-compatible alias
WildcardPatternMatcher = WCMatcher
