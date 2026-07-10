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
        self._normalize_atom_maps(fused)
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

        Uses score-based voting: for each of the four candidate orientations
        (direct/inverted mapping × G1-as-pattern/G2-as-pattern) count how many
        (key, value) pairs in the candidate mapping are consistent with
        key∈pattern and value∈host. The highest-scoring orientation wins; ties
        are broken by preferring G1-as-pattern with the direct mapping.
        """
        mapping = self._mapping_input
        inv = {v: k for k, v in mapping.items()}

        def _score(m: Dict[Any, Any], pattern: GraphType, host: GraphType) -> int:
            return sum(1 for k, v in m.items() if k in pattern and v in host)

        candidates = [
            (mapping, self._G1, self._G2, True),
            (inv, self._G1, self._G2, True),
            (mapping, self._G2, self._G1, False),
            (inv, self._G2, self._G1, False),
        ]

        best_score = -1
        best = None
        for m, pat, host, pat_is_G1 in candidates:
            score = _score(m, pat, host)
            if score > best_score:
                best_score = score
                best = (m, pat, host, pat_is_G1)

        if best is None or best_score == 0:
            raise ValueError(
                "ITSMerge: cannot orient mapping; keys/values do not consistently "
                "belong to G1/G2."
            )

        mapping, pattern, host, pat_is_G1 = best
        self._pattern, self._host = pattern, host
        self._pat_is_G1 = pat_is_G1
        self._pat_to_host = dict(mapping)

    # ------------------------------------------------------------------
    # Internal: wildcard detection helper
    # ------------------------------------------------------------------
    def _is_wc_node(self, attrs: Dict[str, Any]) -> bool:
        """Return True if a node's element is the wildcard element."""
        el = attrs.get(self._element_key)
        wc = self._wildcard_element
        wc0 = wc[0] if isinstance(wc, (tuple, list)) else wc
        if isinstance(el, (tuple, list)):
            return el[0] == wc0
        return el == wc0

    def _host_has_wc_neighbor(self, h_node: Any) -> bool:
        """Return True if a host node has at least one wildcard neighbor."""
        if h_node not in self._host:
            return False
        return any(
            self._is_wc_node(self._host.nodes[nb])
            for nb in self._host.neighbors(h_node)
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

    def _merge_tuple_format(
        self,
        host_data: Dict[str, Any],
        pat_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge tuple-format ITS node attributes (``hcount``, ``sigma_order``,
        ``pi_order``), each stored as a 2-tuple ``(reactant_side, product_side)``.

        ``hcount`` keeps the host value; bond orders keep the host value.
        The host (bw ITS) has the correct hcounts from the actual substrate
        molecule, so we never let the pattern override them.  Taking max()
        was wrong when MCS maps a pattern atom with more Hs onto a host atom
        with fewer Hs (e.g. wrong cross-molecule match due to incompatible
        fw/bw atom_map numbering).

        :param host_data: Host node attribute dictionary.
        :param pat_data: Pattern node attribute dictionary.
        :returns: Dict of updated attributes (empty if no tuple-format attrs found).
        """
        updates: Dict[str, Any] = {}
        for attr in ("hcount", "sigma_order", "pi_order"):
            h_val = host_data.get(attr)
            p_val = pat_data.get(attr)
            if h_val is None or p_val is None:
                continue
            if not (
                isinstance(h_val, (tuple, list)) and isinstance(p_val, (tuple, list))
            ):
                continue
            if len(h_val) != 2 or len(p_val) != 2:
                continue
            updates[attr] = tuple(h_val)
        return updates

    def _merge_anchor_nodes(self, fused: GraphType) -> None:
        """
        Merge ITS typing for mapped pattern→host node pairs.

        When the host node is a wildcard but the pattern node is a real atom,
        the wildcard placeholder is replaced entirely by the pattern's data so
        that the fused graph carries the correct atom identity and hydrogen
        counts from the forward ITS instead of the wildcard stub.

        :param fused: Graph being constructed (copy of host).
        :type fused: GraphType
        """
        for p_node, h_node in self._pat_to_host.items():
            if h_node not in fused or p_node not in self._pattern:
                continue

            host_attrs = fused.nodes[h_node]
            pat_attrs = self._pattern.nodes[p_node]

            # If the host node is a wildcard placeholder and the pattern node is
            # a real atom, replace the wildcard's attributes with the real atom's
            # data so the fused ITS carries the actual element/hcount/charge.
            # Preserve the host node's original atom_map so the fused-graph node
            # ID space stays consistent (the pattern's atom_map would collide with
            # other host nodes that legitimately share that atom_map number).
            if self._is_wc_node(host_attrs) and not self._is_wc_node(pat_attrs):
                old_amap = host_attrs.get("atom_map")
                host_attrs.clear()
                host_attrs.update(pat_attrs)
                if old_amap is not None:
                    host_attrs["atom_map"] = old_amap
                continue

            merged_t = self._merge_types_gh(host_attrs, pat_attrs)
            if merged_t is not None:
                host_attrs[self._types_key] = merged_t
            tuple_updates = self._merge_tuple_format(host_attrs, pat_attrs)
            host_attrs.update(tuple_updates)

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

        Pattern nodes whose atom_map already appears in the host are
        treated as implicitly mapped (they are not added as new nodes
        and their host counterpart is recorded in :attr:`_pat_to_host`
        so that edges between them are handled correctly).

        :param fused: Graph being constructed.
        :type fused: GraphType
        :returns: Mapping from pattern node IDs to new fused node IDs.
        :rtype: dict[Any, Any]
        """
        mapped_p_nodes = set(self._pat_to_host.keys())
        leftover_map: Dict[Any, Any] = {}

        # Build reverse lookup: host atom_map → host node id
        host_amap_to_node: Dict[Any, Any] = {}
        for h_node in self._host.nodes():
            amap = self._host.nodes[h_node].get("atom_map")
            if amap is not None:
                host_amap_to_node[amap] = h_node

        # Host nodes already occupied by the explicit MCS mapping must not be
        # re-used for implicit atom_map-based matching.  fw and bw ITS use
        # independent atom_map numbering (reactant vs product SMILES), so the
        # same atom_map number can refer to chemically different atoms — the
        # element-equality check below is not sufficient to distinguish them
        # when the MCS has already assigned the host node to a different pattern
        # atom.
        used_host_nodes: Set[Any] = set(self._pat_to_host.values())

        next_int_id = self._next_int_id(fused)
        tag = "p1" if self._pat_is_G1 else "p2"

        for p_node, p_data in self._pattern.nodes(data=True):
            if p_node in mapped_p_nodes:
                continue
            if p_data.get(self._element_key) == self._wildcard_element:
                continue

            # If this pattern atom's atom_map already exists in the host
            # AND the elements match AND the host node is not already occupied
            # by the explicit MCS mapping, record the host counterpart instead
            # of creating a duplicate node.  The element check and the
            # used_host_nodes guard are both required because fw and bw ITS use
            # independent atom_map numbering (reactant vs product SMILES
            # ordering), so the same atom_map number can refer to chemically
            # different atoms.
            p_amap = p_data.get("atom_map")
            if p_amap is not None and p_amap in host_amap_to_node:
                h_node = host_amap_to_node[p_amap]
                h_elem = self._host.nodes[h_node].get(self._element_key)
                p_elem = p_data.get(self._element_key)
                if h_elem == p_elem and h_node not in used_host_nodes:
                    self._pat_to_host[p_node] = h_node
                    mapped_p_nodes.add(p_node)
                    used_host_nodes.add(h_node)
                    continue
                # Element mismatch or host node already used: atom_maps are from
                # incompatible numbering schemes — fall through to genuine
                # leftover handling.

            if next_int_id is not None:
                f_node = next_int_id
                next_int_id += 1
            else:
                f_node = (tag, p_node)

            leftover_map[p_node] = f_node
            fused.add_node(f_node, **p_data)
            # Reassign atom_map to the new unique fused node ID so that
            # tuple-format SMILES generation does not collide with existing
            # host atom_maps (fw and bw ITS use independent atom_map
            # numbering, so the same number may already be taken by a host
            # atom with a different identity).
            nd = fused.nodes[f_node]
            am_val = nd.get("atom_map")
            if am_val is not None:
                if isinstance(am_val, (tuple, list)):
                    nd["atom_map"] = type(am_val)([f_node] * len(am_val))
                else:
                    nd["atom_map"] = f_node
            tgh = nd.get("typesGH")
            if tgh and len(tgh) == 2:

                def _upd_amap(side: tuple, new_am: int) -> tuple:
                    lst = list(side)
                    if lst:
                        lst[-1] = new_am
                    return tuple(lst)

                nd["typesGH"] = (_upd_amap(tgh[0], f_node), _upd_amap(tgh[1], f_node))
            # Propagate aromatic/charge from the nearest mapped anchor neighbour
            # when those attributes are absent from the pattern node's own data.
            for p_nb in self._pattern.neighbors(p_node):
                h_nb = self._pat_to_host.get(p_nb)
                if h_nb is None or h_nb not in fused:
                    continue
                h_nb_data = fused.nodes[h_nb]
                for attr in ("aromatic", "charge"):
                    if attr in h_nb_data and attr not in p_data:
                        fused.nodes[f_node][attr] = h_nb_data[attr]
                break

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

            # Fix 2b: only add edges that involve at least one genuinely new
            # leftover node.  Edges between two host-already-present atoms are
            # skipped: the host ITS already encodes the correct bonds between
            # its own atoms, and adding pattern bonds would create duplicate or
            # invalid connectivity (valence errors).
            if u not in leftover_map and v not in leftover_map:
                continue

            # Fix 2d: for edges between exactly one leftover node and one
            # host-mapped node, skip if the bond is static (no order change
            # between reactant and product sides, i.e. standard_order == 0).
            # Static bonds from leftover nodes are phantom bonds produced by
            # the fw-ITS double-representation artifact — the host ITS already
            # encodes any real bond to the mapped atom, so adding these would
            # create wrong connectivity.  Only reaction bonds (forming/breaking,
            # standard_order != 0) should be added.
            #
            # Exception: if the mapped host node IS a wildcard or HAS a wildcard
            # neighbor, the leftover pattern atom was expected to bond there (the
            # wildcard was a placeholder for the missing group), so even static
            # bonds must be preserved.
            if (u in leftover_map) != (v in leftover_map):
                std_ord = data.get("standard_order")
                is_static: bool
                if std_ord is not None:
                    is_static = abs(float(std_ord)) < 1e-9
                else:
                    ord_val = data.get("order")
                    if isinstance(ord_val, (tuple, list)) and len(ord_val) == 2:
                        is_static = abs(float(ord_val[0]) - float(ord_val[1])) < 1e-9
                    else:
                        is_static = False
                if is_static:
                    # Check whether the mapped host node is/has a wildcard:
                    # if so, the static bond is genuine and must be kept.
                    mapped_end_node = None
                    for lo, me in ((u, v), (v, u)):
                        if lo in leftover_map:
                            mapped_end_node = self._pat_to_host.get(me)
                            break
                    if mapped_end_node is not None and mapped_end_node in self._host:
                        wc_placeholder = self._is_wc_node(
                            self._host.nodes[mapped_end_node]
                        ) or self._host_has_wc_neighbor(mapped_end_node)
                    else:
                        wc_placeholder = False
                    if not wc_placeholder:
                        continue

            # Fix 2c: for edges between a leftover atom and a non-leftover
            # (host-mapped) atom, skip if the corresponding HOST atom has
            # no wildcard neighbor and no neighbor of the leftover's element.
            # This prevents phantom bonds when the MCS maps a pattern atom to
            # a host atom that is structurally similar but chemically different
            # (e.g. mapping a ylide-C adjacent to P onto an aldehyde-C that
            # should never bond to P).
            # Exception: if the mapped host node itself is a wildcard, any
            # leftover neighbor is chemically valid by construction.
            for leftover_end, mapped_end in ((u, v), (v, u)):
                if leftover_end not in leftover_map:
                    continue
                h_mapped = self._pat_to_host.get(mapped_end)
                if h_mapped is None or h_mapped not in self._host:
                    break
                # If the mapped host node is itself a wildcard placeholder,
                # any leftover pattern neighbor is chemically valid.
                if self._is_wc_node(self._host.nodes[h_mapped]):
                    break
                left_elem = self._pattern.nodes[leftover_end].get(self._element_key)
                wc = self._wildcard_element
                host_neighbors = list(self._host.neighbors(h_mapped))
                has_wc = any(
                    self._host.nodes[nb].get(self._element_key) == wc
                    for nb in host_neighbors
                )
                has_elem = any(
                    self._host.nodes[nb].get(self._element_key) == left_elem
                    for nb in host_neighbors
                )
                # Only block phantom bonds when the host node already has real
                # (non-wildcard) neighbors: if it is isolated, the leftover edge
                # is legitimate by construction (nothing to protect against).
                has_real_neighbors = any(
                    not self._is_wc_node(self._host.nodes[nb]) for nb in host_neighbors
                )
                if has_real_neighbors and not has_wc and not has_elem:
                    fu = None  # signal to skip
                break

            if fu is None or fv is None:
                continue

            if is_multi:
                fused.add_edge(fu, fv, **data)
            else:
                if fused.has_edge(fu, fv):
                    continue
                fused.add_edge(fu, fv, **data)

    # ------------------------------------------------------------------
    # Internal: atom_map normalization
    # ------------------------------------------------------------------
    def _normalize_atom_maps(self, fused: GraphType) -> None:
        """
        Reassign every node's ``atom_map`` to its node index.

        After merging fw and bw ITS graphs, the two graphs' independent
        atom_map numbering schemes can produce duplicate atom_map values in
        the fused graph (e.g. fw node 29 and bw node 29 both end up in the
        fused graph with ``atom_map=(29,29)``).  Using the node index as the
        canonical atom_map guarantees uniqueness.

        :param fused: Fused graph to normalize in-place.
        :type fused: GraphType
        """
        for n, d in fused.nodes(data=True):
            amap = d.get("atom_map")
            if amap is None:
                continue
            if isinstance(amap, (tuple, list)):
                d["atom_map"] = type(amap)([n] * len(amap))
            else:
                d["atom_map"] = n

    # ------------------------------------------------------------------
    # Internal: wildcard removal
    # ------------------------------------------------------------------
    def _remove_wildcards(self, fused: GraphType) -> None:
        """
        Remove wildcard nodes (and incident edges) from the fused graph.

        Before removal, aromatic atoms that will lose their last ring bond
        to a wildcard are dearomatized so that RDKit sanitization does not
        fail with "non-ring atom marked aromatic".

        :param fused: Graph being constructed.
        :type fused: GraphType
        """
        wc_nodes = {
            n
            for n, d in fused.nodes(data=True)
            if d.get(self._element_key) == self._wildcard_element
        }

        if wc_nodes:
            self._logger.debug(
                "ITSMerge: removing %d wildcard nodes from fused graph",
                len(wc_nodes),
            )

        # Dearomatize aromatic atoms that lack enough aromatic-bond neighbors
        # to form a valid ring.  This check runs unconditionally (even when
        # there are no wildcards) because leftover pattern atoms can be
        # marked aromatic but have no ring partners in the fused graph.
        # An aromatic atom needs at least 2 aromatic-bond neighbors to form
        # a valid ring.  If fewer than 2 aromatic-bond neighbors remain after
        # wildcard removal, the atom can no longer be in a ring and must be
        # dearomatized to avoid RDKit "non-ring atom marked aromatic" errors.
        AROMATIC_ORDER = 1.5

        def _edge_order_r(node: Any, nb: Any) -> float:
            val = fused.edges[node, nb].get("order", 0)
            if isinstance(val, (tuple, list)):
                return float(val[0])
            return float(val or 0)

        def _aromatic_degree(node: Any) -> int:
            return sum(
                1
                for nb in fused.neighbors(node)
                if nb not in wc_nodes
                and abs(_edge_order_r(node, nb) - AROMATIC_ORDER) < 1e-9
            )

        def _is_aromatic(attr: Any) -> bool:
            if isinstance(attr, (tuple, list)):
                return any(attr)
            return bool(attr)

        SINGLE_ORDER = 1.0

        def _dearom_side(side: tuple) -> tuple:
            lst = list(side)
            if len(lst) > 1:
                lst[1] = False
            return tuple(lst)

        def _fix_bonds(n: Any) -> None:
            for nb in fused.neighbors(n):
                if nb in wc_nodes:
                    continue
                edata = fused.edges[n, nb]
                for key in ("order", "kekule_order", "sigma_order"):
                    val = edata.get(key)
                    if val is None:
                        continue
                    if isinstance(val, (tuple, list)):
                        edata[key] = tuple(
                            SINGLE_ORDER if abs(v - AROMATIC_ORDER) < 1e-9 else v
                            for v in val
                        )
                    elif abs(val - AROMATIC_ORDER) < 1e-9:
                        edata[key] = SINGLE_ORDER

        dearom_nodes: set = set()

        # Iterative passes: dearomatizing a node changes its bond orders,
        # which can reduce the aromatic degree of its neighbors and trigger
        # further dearomatization (cascade effect).  Repeat until convergence.
        while True:
            newly_dearom: set = set()
            for n, d in fused.nodes(data=True):
                if n in wc_nodes or n in dearom_nodes:
                    continue
                aromatic = d.get("aromatic")
                if not _is_aromatic(aromatic):
                    continue
                if _aromatic_degree(n) < 2:
                    if isinstance(aromatic, (tuple, list)):
                        d["aromatic"] = tuple(False for _ in aromatic)
                    else:
                        d["aromatic"] = False
                    tgh = d.get("typesGH")
                    if tgh and len(tgh) == 2:
                        d["typesGH"] = (_dearom_side(tgh[0]), _dearom_side(tgh[1]))
                    newly_dearom.add(n)

            if not newly_dearom:
                break

            # Fix bond orders immediately so the next iteration sees updated
            # aromatic degrees for nodes that lost an aromatic neighbour.
            for n in newly_dearom:
                _fix_bonds(n)

            dearom_nodes.update(newly_dearom)

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
