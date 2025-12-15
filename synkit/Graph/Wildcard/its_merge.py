from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union, Set
import logging

import networkx as nx

GraphType = Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]

__all__ = ["ITSMerge", "fuse_its_graphs"]


class ITSMerge:
    """
    Merge two ITS graphs given a node mapping between them.

    This class encapsulates the logic of fusing two ITS graphs (e.g. from
    wildcard pattern matching) in an object-oriented way. The result is a
    **fused** graph that:

    * starts as a copy of the **host** graph,
    * merges ITS typing (``typesGH``) on mapped node pairs,
    * adds leftover (non-mapped, non-wildcard) pattern nodes and edges, and
    * optionally removes all wildcard nodes in the final fused graph.

    Orientation
    -----------
    The graph whose nodes appear as **values** of the mapping is treated as
    the **host**; the other graph is the **pattern**. If the mapping is
    given in the opposite direction (host → pattern), the class detects this
    and automatically inverts the mapping.

    ITS merging semantics
    ---------------------
    * For mapped node pairs (p → h), the ``typesGH`` attribute is merged:
      the hydrogen count entries (index 2 in each inner tuple) are set to the
      **maximum** of host vs pattern.
    * Leftover pattern nodes:
        - If ``element == wildcard_element``, they are **ignored**.
        - Otherwise they are added as new nodes with new IDs and edges are
          created according to the pattern topology.
    * If :paramref:`remove_wildcards` is ``True`` (default), **all wildcard
      nodes** (``element == wildcard_element``) are removed from the fused
      graph; their incident edges disappear. If ``False``, wildcard nodes are
      kept.

    Examples
    --------
    Simple usage with integer-labeled graphs:

    .. code-block:: python

        import networkx as nx
        from synkit.Graph.ITS.its_merge import ITSMerge

        G1 = nx.Graph()
        G2 = nx.Graph()

        # Example: two ITS graphs with 'typesGH' and 'element' attributes
        G1.add_node(1, element="C", typesGH=(("C", False, 2, 0, ["O"]),
                                             ("C", False, 2, 0, ["O"])))
        G2.add_node(10, element="C", typesGH=(("C", False, 1, 0, ["O"]),
                                              ("C", False, 1, 0, ["O"])))

        mapping = {1: 10}  # pattern node → host node

        merger = ITSMerge(G1, G2, mapping, remove_wildcards=True).merge()
        F = merger.fused_graph
        print("Fused nodes:", F.nodes(data=True))

    :param G1: First input ITS graph.
    :type G1: GraphType
    :param G2: Second input ITS graph.
    :type G2: GraphType
    :param mapping: Node mapping between the graphs. Must be a bijection
        either from pattern → host or host → pattern; the class detects
        orientation automatically.
    :type mapping: dict[Any, Any]
    :param types_key: Node attribute key holding the ITS typing tuple,
        e.g. ``(('C', False, 3, 0, ['O']), ('C', False, 3, 0, ['O']))``.
    :type types_key: str
    :param element_key: Node attribute key for element / atom type.
    :type element_key: str
    :param wildcard_element: Value of :paramref:`element_key` that denotes
        wildcard nodes.
    :type wildcard_element: str
    :param remove_wildcards: If ``True``, remove wildcard nodes in the final
        fused graph. If ``False``, wildcard nodes are kept.
    :type remove_wildcards: bool
    :param logger: Optional logger for debug output.
    :type logger: logging.Logger | None
    """

    def __init__(
        self,
        G1: GraphType,
        G2: GraphType,
        mapping: Dict[Any, Any],
        *,
        types_key: str = "typesGH",
        element_key: str = "element",
        wildcard_element: str = "*",
        remove_wildcards: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._G1 = G1
        self._G2 = G2
        self._mapping_input: Dict[Any, Any] = dict(mapping)
        self._types_key = types_key
        self._element_key = element_key
        self._wildcard_element = wildcard_element
        self._remove_wildcards_flag = remove_wildcards
        self._logger = logger or logging.getLogger(__name__)

        self._host: GraphType
        self._pattern: GraphType
        self._pat_is_G1: bool
        self._pat_to_host: Dict[Any, Any]

        self._fused: Optional[GraphType] = None

        self._orient_graphs_and_mapping()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def merge(self) -> "ITSMerge":
        """
        Execute the ITS fusion process.

        The method:

        1. Starts from a copy of the host graph.
        2. Merges ``typesGH`` attributes on mapped node pairs.
        3. Adds leftover non-wildcard pattern nodes.
        4. Adds pattern edges between mapped/added nodes.
        5. Optionally removes wildcard nodes from the fused graph.

        :returns: Self, with :pyattr:`fused_graph` updated.
        :rtype: ITSMerge
        """
        fused = self._host.copy()
        self._merge_anchor_nodes(fused)
        leftover_map = self._add_leftover_pattern_nodes(fused)
        self._add_pattern_edges(fused, leftover_map)
        if self._remove_wildcards_flag:
            self._remove_wildcards(fused)
        self._fused = fused
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def fused_graph(self) -> GraphType:
        """
        Fused ITS graph.

        The graph is in the **host's node ID space**, plus any extra IDs
        for leftover pattern nodes. Wildcard nodes may have been removed,
        depending on :paramref:`remove_wildcards`.

        :returns: Fused ITS graph.
        :rtype: GraphType
        :raises RuntimeError: If :meth:`merge` has not been called yet.
        """
        if self._fused is None:
            raise RuntimeError(
                "ITSMerge.merge() must be called before accessing fused_graph."
            )
        return self._fused

    @property
    def host_graph(self) -> GraphType:
        """
        Graph that was treated as the **host** for merging.

        :returns: Host graph.
        :rtype: GraphType
        """
        return self._host

    @property
    def pattern_graph(self) -> GraphType:
        """
        Graph that was treated as the **pattern** for merging.

        :returns: Pattern graph.
        :rtype: GraphType
        """
        return self._pattern

    @property
    def pattern_to_host(self) -> Dict[Any, Any]:
        """
        Mapping from pattern node IDs to host node IDs.

        Orientation is resolved automatically during construction.

        :returns: Pattern → host node mapping.
        :rtype: dict[Any, Any]
        """
        return dict(self._pat_to_host)

    def __repr__(self) -> str:
        """
        Short textual representation for debugging.

        :returns: Summary string with node/edge counts when available.
        :rtype: str
        """
        fused_info = "unmerged"
        if self._fused is not None:
            fused_info = (
                f"fused_nodes={self._fused.number_of_nodes()}, "
                f"fused_edges={self._fused.number_of_edges()}"
            )
        return (
            f"<ITSMerge host_nodes={self._host.number_of_nodes()} "
            f"pattern_nodes={self._pattern.number_of_nodes()} "
            f"remove_wildcards={self._remove_wildcards_flag} "
            f"{fused_info}>"
        )

    # ------------------------------------------------------------------
    # Internal: orientation & mapping
    # ------------------------------------------------------------------
    def _orient_graphs_and_mapping(self) -> None:
        """
        Decide which graph is host and which is pattern, and orient mapping.

        Orientation rules
        -----------------
        1. If mapping keys belong to G1 and values to G2, then
           pattern = G1, host = G2 and pat_to_host = mapping.
        2. If mapping keys belong to G2 and values to G1, then the provided
           mapping appears to be host→pattern. Invert it so that
           pattern = G1, host = G2 and pat_to_host = inverted mapping.
        3. Otherwise, try the inverse mapping similarly.
        4. If none of these combinations works, raise :class:`ValueError`.
        """
        mapping = self._mapping_input

        def all_in(G: GraphType, nodes: Set[Any]) -> bool:
            return all(n in G for n in nodes)

        keys = set(mapping.keys())
        vals = set(mapping.values())

        # Case A: mapping is already pattern(G1) -> host(G2)
        if all_in(self._G1, keys) and all_in(self._G2, vals):
            self._pattern, self._host = self._G1, self._G2
            self._pat_is_G1 = True
            self._pat_to_host = dict(mapping)
            return

        # Case B: mapping keys in G2 and values in G1 -> mapping likely host->pattern
        # invert it to get pattern(G1)->host(G2)
        if all_in(self._G2, keys) and all_in(self._G1, vals):
            inv = {v: k for k, v in mapping.items()}
            if all_in(self._G1, set(inv.keys())) and all_in(
                self._G2, set(inv.values())
            ):
                self._pattern, self._host = self._G1, self._G2
                self._pat_is_G1 = True
                self._pat_to_host = inv
                return

        # Case C: try inverse mapping explicitly (in case mapping was provided in
        # unexpected orientation)
        inv = {v: k for k, v in mapping.items()}
        inv_keys = set(inv.keys())
        inv_vals = set(inv.values())

        if all_in(self._G1, inv_keys) and all_in(self._G2, inv_vals):
            # inv is pattern(G1) -> host(G2)
            self._pattern, self._host = self._G1, self._G2
            self._pat_is_G1 = True
            self._pat_to_host = dict(inv)
            return

        if all_in(self._G2, inv_keys) and all_in(self._G1, inv_vals):
            # inv is pattern(G2) -> host(G1) -> invert it to pattern(G1)->host(G2)
            inv2 = {v: k for k, v in inv.items()}
            if all_in(self._G1, set(inv2.keys())) and all_in(
                self._G2, set(inv2.values())
            ):
                self._pattern, self._host = self._G1, self._G2
                self._pat_is_G1 = True
                self._pat_to_host = dict(inv2)
                return

        # Fallback: if we reach here, we cannot reliably orient the mapping
        if all_in(self._G2, keys) and all_in(self._G1, vals):
            # keep original mapping as pattern=G2, host=G1 (fallback)
            self._pattern, self._host = self._G2, self._G1
            self._pat_is_G1 = False
            self._pat_to_host = dict(mapping)
            return

        raise ValueError(
            "ITSMerge: cannot orient mapping; keys/values do not consistently "
            "belong to G1/G2."
        )

    # ------------------------------------------------------------------
    # Internal: ITS types merging
    # ------------------------------------------------------------------
    def _merge_types_gh(
        self,
        host_data: Dict[str, Any],
        pat_data: Dict[str, Any],
    ) -> Optional[Tuple[Tuple[Any, ...], Tuple[Any, ...]]]:
        """
        Merge host vs pattern ``typesGH``, keeping max hydrogen counts.

        Only index 2 (H count) of each side is altered; remaining entries are
        taken from the host when present.

        :param host_data: Host node attribute dictionary.
        :type host_data: dict[str, Any]
        :param pat_data: Pattern node attribute dictionary.
        :type pat_data: dict[str, Any]
        :returns: Merged ``typesGH`` tuple (left, right) or ``None`` if both
            are missing.
        :rtype: tuple[tuple[Any, ...], tuple[Any, ...]] | None
        """
        t_host = host_data.get(self._types_key)
        t_pat = pat_data.get(self._types_key)

        if t_host is None and t_pat is None:
            return None
        if t_host is None:
            t_host = t_pat
        if t_host is None:
            return None

        try:
            left_h = list(t_host[0])
            right_h = list(t_host[1])
        except Exception:  # pragma: no cover - defensive
            self._logger.debug(
                "ITSMerge: host typesGH not a 2-tuple, leaving as-is: %r", t_host
            )
            return t_host

        if t_pat is not None:
            try:
                left_p = t_pat[0]
                right_p = t_pat[1]
                if len(left_h) > 2 and len(left_p) > 2:
                    left_h[2] = max(left_h[2], left_p[2])
                if len(right_h) > 2 and len(right_p) > 2:
                    right_h[2] = max(right_h[2], right_p[2])
            except Exception:  # pragma: no cover - defensive
                self._logger.debug(
                    "ITSMerge: pattern typesGH not mergeable, keeping host: %r",
                    t_pat,
                )

        return (tuple(left_h), tuple(right_h))

    def _merge_anchor_nodes(self, fused: GraphType) -> None:
        """
        Merge ITS typing for mapped pattern→host node pairs.

        :param fused: Graph being constructed (copy of host).
        :type fused: GraphType
        """
        for p_node, h_node in self._pat_to_host.items():
            if h_node not in fused or p_node not in self._pattern:
                continue

            host_attrs = fused.nodes[h_node]
            pat_attrs = self._pattern.nodes[p_node]
            merged_t = self._merge_types_gh(host_attrs, pat_attrs)
            if merged_t is not None:
                host_attrs[self._types_key] = merged_t

    # ------------------------------------------------------------------
    # Internal: leftover nodes & edges
    # ------------------------------------------------------------------
    def _next_int_id(self, fused: GraphType) -> Optional[int]:
        """
        Determine starting integer ID for new nodes, if applicable.

        :param fused: Graph being constructed.
        :type fused: GraphType
        :returns: Next integer ID or ``None`` if node IDs are not all ints.
        :rtype: int | None
        """
        if not fused.nodes:
            return 0
        if all(isinstance(n, int) for n in fused.nodes):
            return max(fused.nodes) + 1
        return None

    def _add_leftover_pattern_nodes(self, fused: GraphType) -> Dict[Any, Any]:
        """
        Add leftover non-wildcard pattern nodes to the fused graph.

        :param fused: Graph being constructed.
        :type fused: GraphType
        :returns: Mapping from pattern node IDs to new fused node IDs.
        :rtype: dict[Any, Any]
        """
        mapped_p_nodes = set(self._pat_to_host.keys())
        leftover_map: Dict[Any, Any] = {}

        next_int_id = self._next_int_id(fused)
        tag = "p1" if self._pat_is_G1 else "p2"

        for p_node, p_data in self._pattern.nodes(data=True):
            if p_node in mapped_p_nodes:
                continue
            if p_data.get(self._element_key) == self._wildcard_element:
                continue

            if next_int_id is not None:
                f_node = next_int_id
                next_int_id += 1
            else:
                f_node = (tag, p_node)

            leftover_map[p_node] = f_node
            fused.add_node(f_node, **p_data)

        return leftover_map

    def _add_pattern_edges(
        self,
        fused: GraphType,
        leftover_map: Dict[Any, Any],
    ) -> None:
        """
        Add edges from the pattern graph into the fused graph.

        Edges are added between:

        * mapped pattern nodes, via their host IDs, and
        * leftover pattern nodes that were newly added.

        :param fused: Graph being constructed.
        :type fused: GraphType
        :param leftover_map: Mapping from pattern node IDs to new fused node IDs.
        :type leftover_map: dict[Any, Any]
        """

        def map_p_to_f(n: Any) -> Optional[Any]:
            if n in self._pat_to_host:
                return self._pat_to_host[n]
            return leftover_map.get(n)

        is_multi = isinstance(fused, (nx.MultiGraph, nx.MultiDiGraph))

        for u, v, data in self._pattern.edges(data=True):
            fu = map_p_to_f(u)
            fv = map_p_to_f(v)
            if fu is None or fv is None:
                continue

            if is_multi:
                fused.add_edge(fu, fv, **data)
            else:
                if fused.has_edge(fu, fv):
                    continue
                fused.add_edge(fu, fv, **data)

    # ------------------------------------------------------------------
    # Internal: wildcard removal
    # ------------------------------------------------------------------
    def _remove_wildcards(self, fused: GraphType) -> None:
        """
        Remove wildcard nodes (and incident edges) from the fused graph.

        :param fused: Graph being constructed.
        :type fused: GraphType
        """
        wc_nodes = [
            n
            for n, d in fused.nodes(data=True)
            if d.get(self._element_key) == self._wildcard_element
        ]
        if wc_nodes:
            self._logger.debug(
                "ITSMerge: removing %d wildcard nodes from fused graph",
                len(wc_nodes),
            )
            fused.remove_nodes_from(wc_nodes)


# ----------------------------------------------------------------------
# Functional wrapper for backwards compatibility
# ----------------------------------------------------------------------
def fuse_its_graphs(
    G1: GraphType,
    G2: GraphType,
    mapping: Dict[Any, Any],
    *,
    types_key: str = "typesGH",
    element_key: str = "element",
    wildcard_element: str = "*",
    remove_wildcards: bool = True,
    logger: Optional[logging.Logger] = None,
) -> GraphType:
    """
    Functional wrapper around :class:`ITSMerge`.

    :param G1: First input ITS graph.
    :type G1: GraphType
    :param G2: Second input ITS graph.
    :type G2: GraphType
    :param mapping: Node mapping between the graphs.
    :type mapping: dict[Any, Any]
    :param types_key: Node attribute key holding ITS typing information.
    :type types_key: str
    :param element_key: Node attribute key for element / atom type.
    :type element_key: str
    :param wildcard_element: Value of :paramref:`element_key` that denotes
        wildcard nodes.
    :type wildcard_element: str
    :param remove_wildcards: If ``True``, remove wildcard nodes from the
        fused graph. If ``False``, keep them.
    :type remove_wildcards: bool
    :param logger: Optional logger for debug output.
    :type logger: logging.Logger | None
    :returns: Fused ITS graph.
    :rtype: GraphType
    """
    merger = ITSMerge(
        G1,
        G2,
        mapping,
        types_key=types_key,
        element_key=element_key,
        wildcard_element=wildcard_element,
        remove_wildcards=remove_wildcards,
        logger=logger,
    ).merge()
    return merger.fused_graph
