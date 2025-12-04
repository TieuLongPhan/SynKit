"""mcs_matcher.py — Maximum/Common Subgraph Matcher
===================================================

A convenience wrapper around :class:`networkx.algorithms.isomorphism.GraphMatcher`
that finds *all* common-subgraph (or maximum-common-subgraph) node mappings
between two molecular graphs.

Highlights
----------
* **Flexible node matching** via :func:`generic_node_match`.
* **Scalar edge attribute** comparison (e.g. ``order``).
* Results are **cached** – call :py:meth:`get_mappings` (or use the
  :pyattr:`mappings` property) to retrieve them.
* Helpful :pyattr:`help` and :py:meth:`__repr__` utilities, in the same
  OOP style as :class:`PartialMatcher`.

Public API
~~~~~~~~~~
``MCSMatcher(node_label_names, node_label_defaults, edge_attribute='order', allow_shift=True)``
    Construct a matcher instance.

``matcher.find_common_subgraph(G1, G2, mcs=False, mcs_mol=False)``
    Run the search (stores and returns ``self``). If ``mcs_mol=True``,
    match by entire connected components (molecule-level matching).

``matcher.get_mappings()`` / ``matcher.mappings``
    Retrieve the stored mapping list.

``matcher.find_rc_mapping(rc1, rc2, mcs=False, mcs_mol=False)``
    Convenience wrapper for ITS reaction-centre objects (via
    :func:`synkit.Graph.ITS.its_decompose`).

Dependencies
~~~~~~~~~~~~
* Python 3.9+
* NetworkX ≥ 3.0
* :mod:`synkit.Graph.ITS.its_decompose` (optional helper for
  :py:meth:`find_rc_mapping`).

Examples
--------
Basic usage with two isomorphic graphs:

.. code-block:: python

    import networkx as nx
    from synkit.Graph.Matcher.mcs_matcher import MCSMatcher

    G1 = nx.cycle_graph(4)
    G2 = nx.cycle_graph(4)

    matcher = MCSMatcher()
    matcher.find_common_subgraph(G1, G2, mcs=True)
    print("Mappings:", matcher.mappings)
    print("Last size:", matcher.last_size)

Subgraph case where G1 is smaller than G2:

.. code-block:: python

    G1 = nx.path_graph(3)
    G2 = nx.path_graph(5)

    matcher = MCSMatcher()
    matcher.find_common_subgraph(G1, G2, mcs=True)
    # Mappings are dicts mapping nodes of G1 to nodes of G2
    for m in matcher:
        print(m)
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
    more attribute names and default values. Edge matching compares a single
    scalar edge attribute (e.g. bond order).

    The matcher stores discovered mappings internally so that downstream
    code can retrieve them via :py:meth:`get_mappings` or the
    :pyattr:`mappings` property.

    :param node_label_names: Node attribute keys to compare. If ``None``,
        defaults to ``["element"]``.
    :type node_label_names: list[str] | None
    :param node_label_defaults: Fallback values for each node attribute
        when missing. If ``None``, defaults to a list of ``"*"``
        of the same length as :paramref:`node_label_names`.
    :type node_label_defaults: list[Any] | None
    :param edge_attribute: Edge attribute storing the scalar "order"
        (e.g. bond order). Defaults to ``"order"``.
    :type edge_attribute: str
    :param allow_shift: Placeholder for future asymmetric rules. Currently
        unused but kept for API compatibility.
    :type allow_shift: bool

    Examples
    --------
    Find all common subgraphs (not just maximum size):

    .. code-block:: python

        import networkx as nx
        from synkit.Graph.Matcher.mcs_matcher import MCSMatcher

        G1 = nx.path_graph(3)
        G2 = nx.path_graph(4)

        matcher = MCSMatcher()
        matcher.find_common_subgraph(G1, G2, mcs=False)
        for mapping in matcher.mappings:
            print(mapping)

    Restrict to maximum common subgraph (MCS) only:

    .. code-block:: python

        matcher = MCSMatcher()
        matcher.find_common_subgraph(G1, G2, mcs=True)
        print("Maximum size:", matcher.last_size)
        print("Number of MCS mappings:", matcher.num_mappings)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        node_label_names: Optional[List[str]] = None,
        node_label_defaults: Optional[List[Any]] = None,
        edge_attribute: str = "order",
        allow_shift: bool = True,
    ) -> None:
        if node_label_names is None:
            node_label_names = ["element"]
        if node_label_defaults is None:
            node_label_defaults = ["*"] * len(node_label_names)

        self._node_label_names: List[str] = node_label_names
        self._node_label_defaults: List[Any] = node_label_defaults
        self.edge_attr: str = edge_attribute
        self.allow_shift: bool = allow_shift

        comparators: List[Callable[[Any, Any], bool]] = [
            lambda x, y: x == y for _ in node_label_names
        ]
        self.node_match: Callable[[Dict[str, Any], Dict[str, Any]], bool] = (
            generic_node_match(
                node_label_names,
                node_label_defaults,
                comparators,
            )
        )

        # Internal cache
        self._mappings: List[MappingDict] = []
        self._last_size: int = 0

    # ------------------------------------------------------------------
    # Internal edge / mapping helpers
    # ------------------------------------------------------------------
    def _edge_match(
        self, host_attrs: Dict[str, Any], pat_attrs: Dict[str, Any]
    ) -> bool:
        """
        Compare scalar edge attributes (exact equality on ``edge_attr``).

        :param host_attrs: Edge attribute dictionary from host graph.
        :type host_attrs: dict[str, Any]
        :param pat_attrs: Edge attribute dictionary from pattern graph.
        :type pat_attrs: dict[str, Any]
        :returns: ``True`` if the scalar attributes match, otherwise ``False``.
        :rtype: bool
        """
        hv = host_attrs.get(self.edge_attr, None)
        pv = pat_attrs.get(self.edge_attr, None)
        try:
            return float(hv) == float(pv)
        except (TypeError, ValueError):
            # If coercion fails, fall back to direct equality
            return hv == pv

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
        :returns: Tuple ``(pattern, host, swapped)`` where
            ``swapped=True`` indicates that ``G1`` and ``G2`` were
            swapped (pattern is now ``G2``).
        :rtype: tuple[nx.Graph, nx.Graph, bool]
        """
        if G1.number_of_nodes() <= G2.number_of_nodes():
            return G1, G2, False
        return G2, G1, True

    def _search_subgraphs(
        self,
        pattern: nx.Graph,
        host: nx.Graph,
        *,
        mcs: bool,
    ) -> List[MappingDict]:
        """
        Enumerate common subgraphs between ``pattern`` and ``host``.

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
                    inv = self._invert_mapping(iso)
                    key = tuple(sorted(inv.items()))
                    if key in seen:
                        continue
                    seen.add(key)
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
            best_size
            if best_size
            else (mappings[0] and len(mappings[0]) if mappings else 0)
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

        :param G1: First input graph (conceptually the "pattern").
        :type G1: nx.Graph
        :param G2: Second input graph (conceptually the "host").
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

        if mcs_mol:
            combined = self._find_mcs_mol(G1, G2)
            self._mappings = [combined]
            self._last_size = len(combined)
            return self

        pattern, host, swapped = self._prepare_orientation(G1, G2)
        mappings = self._search_subgraphs(pattern, host, mcs=mcs)

        if swapped:
            # mappings are pattern→host; if pattern is G2, invert to
            # get mappings from original G1 to G2.
            self._mappings = [self._invert_mapping(m) for m in mappings]
        else:
            self._mappings = mappings

        return self

    def find_rc_mapping(
        self,
        rc1: Any,
        rc2: Any,
        *,
        mcs: bool = False,
        mcs_mol: bool = False,
    ) -> "MCSMatcher":
        """
        Convenience wrapper for ITS reaction-centre objects.

        This uses :func:`synkit.Graph.ITS.its_decompose` to obtain the
        underlying graphs and then delegates to
        :py:meth:`find_common_subgraph`.

        :param rc1: First reaction-centre object.
        :type rc1: Any
        :param rc2: Second reaction-centre object.
        :type rc2: Any
        :param mcs: If ``True``, restrict to maximum-common-subgraph
            mappings.
        :type mcs: bool
        :param mcs_mol: If ``True``, use connected-component matching.
        :type mcs_mol: bool
        :returns: The matcher instance (with internal cache updated).
        :rtype: MCSMatcher
        :raises ImportError: If :mod:`synkit` ITS utilities are not
            available.
        """
        if its_decompose is None:
            raise ImportError(
                "synkit is not available; cannot decompose reaction centres."
            )

        _, r1 = its_decompose(rc1)
        l2, _ = its_decompose(rc2)
        return self.find_common_subgraph(r1, l2, mcs=mcs, mcs_mol=mcs_mol)

    # ------------------------------------------------------------------
    # Accessors / properties
    # ------------------------------------------------------------------
    def get_mappings(self) -> List[MappingDict]:
        """
        Return a copy of the cached mapping list.

        Each mapping dictionary maps nodes from the *first* input graph
        ``G1`` to nodes of the *second* input graph ``G2``, regardless
        of any internal swapping used for efficiency.

        :returns: List of node-mapping dictionaries.
        :rtype: list[dict[int, int]]
        """
        return list(self._mappings)

    @property
    def mappings(self) -> List[MappingDict]:
        """
        Cached node mappings from the most recent search.

        :returns: List of node-mapping dictionaries.
        :rtype: list[dict[int, int]]
        """
        return self.get_mappings()

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

    # ------------------------------------------------------------------
    # Iteration & niceties
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterable[MappingDict]:
        """
        Iterate over cached mappings.

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
            f"<MCSMatcher mappings={self.num_mappings} " f"last_size={self.last_size}>"
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
