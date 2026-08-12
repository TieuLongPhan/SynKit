"""NetworkX-backed graph isomorphism and subgraph matching."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional
import networkx as nx
from networkx.algorithms.isomorphism import (
    DiGraphMatcher as _NXDiGraphMatcher,
    GraphMatcher as _NXGraphMatcher,
    MultiDiGraphMatcher as _NXMultiDiGraphMatcher,
    MultiGraphMatcher as _NXMultiGraphMatcher,
    categorical_multiedge_match,
)

MappingDict = Dict[int, int]

__all__ = ["GraphMatcherEngine", "MappingDict"]

################################################################################
# Utility helpers
################################################################################


def _wl1_hash(g: nx.Graph, node_attrs: tuple[str, ...]) -> Counter:
    """Single‑pass Weisfeiler–Lehman (k=1) colour‑refinement of *g*.

    The result is a multiset (Counter) of the form::
        Counter({(base_label, neigh_multiset): multiplicity})

    where *base_label* is the tuple of ``node_attrs`` extracted from the node
    and *neigh_multiset* is the sorted tuple of base labels of its neighbours.
    """

    base = {
        n: tuple(g.nodes[n].get(a) for a in node_attrs)
        for n in g.nodes  # local alias to avoid global look‑up inside loop
    }
    refined: Counter = Counter()
    for n, b in base.items():
        neigh = tuple(sorted(base[v] for v in g.neighbors(n)))
        refined[(b, neigh)] += 1
    return refined


################################################################################
# Main engine
################################################################################


class GraphMatcherEngine:
    """Reusable engine for (sub‑)graph isomorphism checks & embeddings.

    :param backend: ``"nx"`` – the native NetworkX implementation.
    :param node_attrs: Lists of attribute keys used for matching. ``hcount`` and
                       ``lone_pairs`` are treated specially: the host must be **≥** the
                       pattern. Other requested attributes, including ``radical``, match
                       exactly.
    :param edge_attrs: Lists of attribute keys used for matching. ``hcount`` and
                       ``lone_pairs`` are treated specially: the host must be **≥** the
                       pattern. Other requested attributes, including ``radical``, match
                       exactly.
    :param wl1_filter: If *True*, a fast WL‑based colour refinement pre‑filter discards host
                       graphs that cannot possibly contain the pattern.
    :param max_mappings: Upper bound on the number of mappings to enumerate in
                         :py:meth:`get_mappings`.  *None* means "no limit".
    """

    # Construction

    def __init__(
        self,
        *,
        backend: str = "nx",
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        wl1_filter: bool = False,
        max_mappings: Optional[int] = 1,
    ) -> None:
        self.backend = backend.lower()
        if self.backend not in self.available_backends():
            raise ValueError(f"Unsupported backend: {backend!r}")

        # Store attributes as *immutable* tuples so they can be hashed & used in
        # lru_cache‑decorated helpers.
        self.node_attrs: tuple[str, ...] = tuple(node_attrs or ())
        self.edge_attrs: tuple[str, ...] = tuple(edge_attrs or ())

        self.wl1_filter = bool(wl1_filter)
        if max_mappings is not None and max_mappings < 0:
            raise ValueError("max_mappings must be non-negative or None")
        self.max_mappings = max_mappings

        # Reuse the match functions across calls on this engine.
        self._nm = self._compile_node_matcher()
        self._em = self._compile_edge_matcher()

    # Public helpers

    @staticmethod
    def available_backends() -> List[str]:
        return ["nx"]

    # ------------------------------------------------------------------
    # Node / edge matchers – compiled only once per engine instance.
    # ------------------------------------------------------------------

    def _compile_node_matcher(self):
        attrs = self.node_attrs  # local copy for closure

        if not attrs:  # Only the special *hcount* semantics apply.

            def nm(nh, np):  # noqa: ANN001 – external signature
                return nh.get("hcount", 0) >= np.get("hcount", 0)

            return nm

        def nm(nh, np, _attrs=attrs):  # noqa: ANN001 – external signature
            for k in _attrs:
                host_value = nh.get(k, 0 if k in {"hcount", "lone_pairs"} else None)
                pattern_value = np.get(k, 0 if k in {"hcount", "lone_pairs"} else None)
                if k in {"hcount", "lone_pairs"}:
                    if host_value < pattern_value:
                        return False
                    continue
                if host_value != pattern_value:
                    return False
            return nh.get("hcount", 0) >= np.get("hcount", 0)

        return nm

    def _compile_edge_matcher(self):
        attrs = self.edge_attrs  # local copy for closure
        if not attrs:
            return lambda *_: True  # noqa: E731 – intentionally anonymous

        def em(eh, ep, _attrs=attrs):  # noqa: ANN001 – external signature
            for k in _attrs:
                if eh.get(k) != ep.get(k):
                    return False
            return True

        return em

    def _matcher(self, host: nx.Graph, pattern: nx.Graph):
        """Build the NetworkX matcher appropriate for the graph kind."""
        if host.is_directed() != pattern.is_directed():
            return None
        if host.is_multigraph() != pattern.is_multigraph():
            return None

        if host.is_multigraph():
            matcher_cls = (
                _NXMultiDiGraphMatcher if host.is_directed() else _NXMultiGraphMatcher
            )
            edge_match = (
                categorical_multiedge_match(
                    self.edge_attrs,
                    [None] * len(self.edge_attrs),
                )
                if self.edge_attrs
                else (lambda *_: True)
            )
        else:
            matcher_cls = _NXDiGraphMatcher if host.is_directed() else _NXGraphMatcher
            edge_match = self._em
        return matcher_cls(
            host,
            pattern,
            node_match=self._nm,
            edge_match=edge_match,
        )

    # Public API

    def isomorphic(self, obj1: Any, obj2: Any) -> bool:
        return self._isomorphic_nx(obj1, obj2)

    def get_mappings(self, host: Any, pattern: Any) -> List[MappingDict]:
        return self._get_mappings_nx(host, pattern)

    # NetworkX backend

    def _pre_check(self, host: nx.Graph, pattern: nx.Graph) -> bool:
        """Return *True* if the inexpensive sanity checks pass."""

        # Basic size test – also rejects host < pattern immediately.
        if (
            host.number_of_nodes() < pattern.number_of_nodes()
            or host.number_of_edges() < pattern.number_of_edges()
        ):
            return False

        if not self.wl1_filter:
            return True

        # Whole-graph 1-WL colours are not a necessary invariant for a
        # subgraph embedding: neighbours outside the selected image change the
        # host colours. Likewise, hcount/lone_pairs use a directional >=
        # comparator rather than exact label equality. In either situation a
        # WL mismatch cannot soundly reject the candidate.
        if (
            host.number_of_nodes() != pattern.number_of_nodes()
            or host.number_of_edges() != pattern.number_of_edges()
            or {"hcount", "lone_pairs"}.intersection(self.node_attrs)
        ):
            return True

        try:
            # Recompute from current graph state. NetworkX graphs are mutable,
            # so caching solely by object identity can retain stale colours.
            return _wl1_hash(host, self.node_attrs) == _wl1_hash(
                pattern,
                self.node_attrs,
            )
        except (TypeError, ValueError):
            # Heterogeneous or unhashable user attributes are still supported
            # by the authoritative matcher. A pre-filter must fail open.
            return True

    def _isomorphic_nx(
        self, g1: nx.Graph, g2: nx.Graph
    ) -> bool:  # noqa: C901 – complexity OK here
        if not isinstance(g1, nx.Graph) or not isinstance(g2, nx.Graph):
            raise TypeError("NX backend expects `networkx.Graph` objects.")

        # Treat the larger graph as host so comparator semantics remain
        # host-first for subgraph checks.
        host, pattern = (g1, g2)
        if g1.number_of_nodes() < g2.number_of_nodes():
            host, pattern = g2, g1

        if not self._pre_check(host, pattern):
            return False

        gm = self._matcher(host, pattern)
        if gm is None:
            return False
        return (
            gm.is_isomorphic()
            if host.number_of_nodes() == pattern.number_of_nodes()
            else gm.subgraph_is_isomorphic()
        )

    def _get_mappings_nx(
        self, host: nx.Graph, pattern: nx.Graph
    ) -> List[MappingDict]:  # noqa: C901 – complexity OK here
        if not isinstance(host, nx.Graph) or not isinstance(pattern, nx.Graph):
            raise TypeError("NX backend expects `networkx.Graph` objects.")

        if not self._pre_check(host, pattern):
            return []

        if self.max_mappings == 0:
            return []

        gm = self._matcher(host, pattern)
        if gm is None:
            return []

        # Full blow isomorphism (same #nodes / #edges)? Then a single call tells
        # us everything and is much faster than iterating via *isomorphisms_iter*.
        if (
            self.max_mappings == 1
            and pattern.number_of_nodes() == host.number_of_nodes()
            and pattern.number_of_edges() == host.number_of_edges()
        ):
            return (
                [
                    {
                        pattern_node: host_node
                        for host_node, pattern_node in gm.mapping.items()
                    }
                ]
                if gm.is_isomorphic()
                else []
            )

        # Sub‑isomorphisms.
        iso_iter = gm.subgraph_isomorphisms_iter()
        if self.max_mappings is not None:
            from itertools import (
                islice,
            )  # local import – cheap and avoids polluting global namespace

            iso_iter = islice(iso_iter, self.max_mappings)
        return [
            {pattern_node: host_node for host_node, pattern_node in mapping.items()}
            for mapping in iso_iter
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover – debug only
        cls = self.__class__.__name__
        return (
            f"<{cls} backend={self.backend!r} node_attrs={list(self.node_attrs)!r} "
            f"edge_attrs={list(self.edge_attrs)!r} wl1_filter={self.wl1_filter} "
            f"max_mappings={self.max_mappings}>"
        )

    __str__ = __repr__

    # Keep the help() method for API compatibility (slightly condensed).
    def help(self) -> str:  # noqa: D401 – imperative mood
        return (
            "GraphMatcherEngine(backend='nx', node_attrs=[...], edge_attrs=[...], "
            "wl1_filter=True|False, max_mappings=N)\n"
            "Methods: isomorphic(obj1, obj2)  get_mappings(host, pattern)"
        )
