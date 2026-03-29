from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from ._common import (
    SymmetryConfig,
    approx_automorphism_count_from_cells,
    build_fast_signature,
    edge_token,
    graph_key_from_order,
    hash_text,
    node_token,
    prepare_graph,
)


@dataclass(frozen=True)
class _WLState:
    """
    Internal cached WL state.

    :param colors:
        Final node colors after WL refinement.
    :type colors: Dict[Any, str]

    :param cells:
        Final WL color cells as sorted node lists.
    :type cells: List[List[Any]]

    :param orbits:
        Approximate node orbits induced by final WL colors.
    :type orbits: List[Set[Any]]

    :param color_hist:
        Histogram of final colors.
    :type color_hist: Dict[str, int]

    :param iters_run:
        Number of WL iterations actually performed.
    :type iters_run: int

    :param stabilized:
        Whether refinement stabilized before the iteration limit.
    :type stabilized: bool

    :param canonical_order:
        Deterministic node order induced by final WL colors.
    :type canonical_order: List[Any]

    :param approx_automorphism_count:
        Approximate automorphism count derived from WL cells.
    :type approx_automorphism_count: Optional[int]
    """

    colors: Dict[Any, str]
    cells: List[List[Any]]
    orbits: List[Set[Any]]
    color_hist: Dict[str, int]
    iters_run: int
    stabilized: bool
    canonical_order: List[Any]
    approx_automorphism_count: Optional[int]


@dataclass(frozen=True)
class WLCanonicalResult:
    """
    Approximate canonicalization result from WL refinement.

    This mirrors the exact canonicalizer style, but remains approximate.

    :param canon_graph:
        Graph canonically relabeled according to the WL order.
    :type canon_graph: nx.DiGraph

    :param graph_type:
        Graph representation type, e.g. ``"bipartite"`` or ``"species"``.
    :type graph_type: str

    :param canonical_order:
        Deterministic node order induced by the final WL partition.
    :type canonical_order: List[Any]

    :param canonical_key:
        Canonical graph key derived from the WL order.
    :type canonical_key: Any

    :param automorphism_count:
        Approximate automorphism count from WL cells.
    :type automorphism_count: Optional[int]

    :param orbits:
        Approximate node orbits from the final WL cells.
    :type orbits: List[Set[Any]]

    :param colors:
        Final WL color mapping.
    :type colors: Dict[Any, str]

    :param color_hist:
        Histogram of final WL colors.
    :type color_hist: Dict[str, int]

    :param iters_run:
        Number of WL iterations actually performed.
    :type iters_run: int

    :param stabilized:
        Whether WL refinement stabilized before the maximum iteration count.
    :type stabilized: bool

    :param exact:
        Always ``False`` for WL refinement.
    :type exact: bool

    :param elapsed_seconds:
        Runtime in seconds for building the result object.
    :type elapsed_seconds: float
    """

    canon_graph: nx.DiGraph
    graph_type: str
    canonical_order: List[Any]
    canonical_key: Any
    automorphism_count: Optional[int]
    orbits: List[Set[Any]]
    colors: Dict[Any, str]
    color_hist: Dict[str, int]
    iters_run: int
    stabilized: bool
    exact: bool
    elapsed_seconds: float


class WLCanonicalizer:
    """
    Fast approximate canonicalizer for SynKit CRN graphs using direction-aware
    1-WL refinement.

    This class is designed as a lightweight companion to the exact CRN
    canonicalizer. It gives:

    - deterministic WL-based canonical relabeling
    - approximate orbit partitions
    - approximate automorphism counts from WL cells
    - fast signatures for cheap prefiltering

    Compared with the exact canonicalizer, this class is much faster but not
    guaranteed to distinguish all non-isomorphic graphs or recover exact
    automorphism groups.

    :param source:
        Input CRN representation. This may be a prepared
        :class:`networkx.DiGraph`, an object exposing ``to_digraph()``, or any
        object accepted by :func:`prepare_graph`.
    :type source: Any

    :param include_rule:
        Whether rule nodes should be included in the prepared graph.
    :type include_rule: bool

    :param integer_ids:
        Whether integer node identifiers should be used during graph
        preparation.
    :type integer_ids: bool

    :param include_stoich:
        Whether stoichiometric edge attributes should be preserved during graph
        preparation.
    :type include_stoich: bool

    :param n_iter:
        Maximum number of WL refinement iterations.
    :type n_iter: int

    :param digest_size:
        Digest size used when hashing node and edge signatures.
    :type digest_size: int

    :param include_in_neighbors:
        Whether incoming neighbors should contribute to refinement.
    :type include_in_neighbors: bool

    :param include_out_neighbors:
        Whether outgoing neighbors should contribute to refinement.
    :type include_out_neighbors: bool

    :param estimate_automorphisms:
        Whether to compute an approximate automorphism count from the final WL
        cells.
    :type estimate_automorphisms: bool

    :param automorphism_cap:
        Cap applied to approximate automorphism counts.
    :type automorphism_cap: int

    :param config:
        Symmetry semantics configuration controlling which node and edge
        attributes participate in WL coloring.
    :type config: Optional[SymmetryConfig]

    Example
    -------
    .. code-block:: python

        from synkit.CRN.Sym import WLCanonicalizer, SymmetryConfig

        wl = WLCanonicalizer(
            syn.to_digraph(),
            include_rule=True,
            config=SymmetryConfig.topological(),
        )

        print(wl.has_nontrivial_automorphism())
        print(wl.orbits())
        print(wl.canonical_order())
        print(wl.summary()["automorphism_count"])
    """

    def __init__(
        self,
        source: Any,
        *,
        include_rule: bool = True,
        integer_ids: bool = False,
        include_stoich: bool = True,
        n_iter: int = 20,
        digest_size: int = 16,
        include_in_neighbors: bool = True,
        include_out_neighbors: bool = True,
        estimate_automorphisms: bool = True,
        automorphism_cap: int = 10**18,
        config: Optional[SymmetryConfig] = None,
    ) -> None:
        """
        Initialize the WL canonicalizer.

        :param source:
            Input CRN representation.
        :type source: Any

        :param include_rule:
            Whether rule nodes should be included in the prepared graph.
        :type include_rule: bool

        :param integer_ids:
            Whether integer node identifiers should be used during graph
            preparation.
        :type integer_ids: bool

        :param include_stoich:
            Whether stoichiometric edge attributes should be preserved.
        :type include_stoich: bool

        :param n_iter:
            Maximum number of WL refinement rounds.
        :type n_iter: int

        :param digest_size:
            Digest size used when hashing WL signatures.
        :type digest_size: int

        :param include_in_neighbors:
            Whether incoming neighborhoods should be used.
        :type include_in_neighbors: bool

        :param include_out_neighbors:
            Whether outgoing neighborhoods should be used.
        :type include_out_neighbors: bool

        :param estimate_automorphisms:
            Whether to estimate automorphism count from final WL cells.
        :type estimate_automorphisms: bool

        :param automorphism_cap:
            Maximum cap for approximate automorphism counting.
        :type automorphism_cap: int

        :param config:
            Symmetry semantics configuration.
        :type config: Optional[SymmetryConfig]

        :returns:
            None.
        :rtype: None

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(
                syn.to_digraph(),
                include_rule=True,
                n_iter=20,
                digest_size=16,
            )
        """
        self.source = source
        self.include_rule = bool(include_rule)
        self.integer_ids = bool(integer_ids)
        self.include_stoich = bool(include_stoich)
        self.n_iter = int(n_iter)
        self.digest_size = int(digest_size)
        self.include_in_neighbors = bool(include_in_neighbors)
        self.include_out_neighbors = bool(include_out_neighbors)
        self.estimate_automorphisms = bool(estimate_automorphisms)
        self.automorphism_cap = int(automorphism_cap)
        self.config = config or SymmetryConfig.semantic()

        self._G, self._graph_type = prepare_graph(
            source,
            include_rule=self.include_rule,
            integer_ids=self.integer_ids,
            include_stoich=self.include_stoich,
        )

        self._state_cache: Optional[_WLState] = None
        self._summary_cache: Optional[WLCanonicalResult] = None
        self._fast_signature: Optional[Tuple[Any, ...]] = None
        self._cache_key_last: Optional[Tuple[Any, ...]] = None

    def __repr__(self) -> str:
        """
        Return a concise representation.

        :returns:
            String representation.
        :rtype: str

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            print(wl)
        """
        return (
            f"WLCanonicalizer(include_rule={self.include_rule}, "
            f"graph_type={self.graph_type}, n_iter={self.n_iter}, "
            f"digest_size={self.digest_size})"
        )

    @property
    def G(self) -> nx.DiGraph:
        """
        Return the prepared graph.

        :returns:
            Prepared directed graph.
        :rtype: nx.DiGraph

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            print(wl.G.number_of_nodes(), wl.G.number_of_edges())
        """
        return self._G

    @property
    def graph_type(self) -> str:
        """
        Return the graph representation type.

        :returns:
            Graph representation type.
        :rtype: str

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn, include_rule=True)
            print(wl.graph_type)
        """
        return self._graph_type

    def _cache_key(self) -> Tuple[Any, ...]:
        """
        Build a conservative cache key for the current graph and parameters.

        :returns:
            Cache key tuple.
        :rtype: Tuple[Any, ...]

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            print(wl._cache_key())
        """
        return (
            id(self.G),
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
            self.include_rule,
            self.integer_ids,
            self.include_stoich,
            self.n_iter,
            self.digest_size,
            self.include_in_neighbors,
            self.include_out_neighbors,
            self.estimate_automorphisms,
            self.automorphism_cap,
            self.config,
        )

    def _node_seed(self, v: Any) -> str:
        """
        Compute the initial WL color for one node.

        The seed combines the semantic node token and the directed degree.

        :param v:
            Node identifier.
        :type v: Any

        :returns:
            Initial hashed node color.
        :rtype: str
        """
        tok = node_token(self.G.nodes[v], self.config)
        deg = (self.G.in_degree(v), self.G.out_degree(v))
        return hash_text(f"N|{tok}|{deg}", digest_size=self.digest_size)

    def _edge_sig(self, attrs: Dict[str, Any]) -> str:
        """
        Compute a hashed signature for one edge attribute dictionary.

        :param attrs:
            Edge attributes.
        :type attrs: Dict[str, Any]

        :returns:
            Hashed edge signature.
        :rtype: str
        """
        return hash_text(
            f"E|{edge_token(attrs, self.config)}",
            digest_size=self.digest_size,
        )

    def _edge_sig_between(self, u: Any, v: Any) -> str:
        """
        Return a stable signature for the edge between two nodes.

        For multigraphs, the minimum signature over parallel edges is used to
        keep the behavior deterministic.

        :param u:
            Source node.
        :type u: Any

        :param v:
            Target node.
        :type v: Any

        :returns:
            Stable edge signature.
        :rtype: str
        """
        data = self.G.get_edge_data(u, v, default=None)
        if data is None:
            return self._edge_sig({})

        if self.G.is_multigraph():
            sigs: List[str] = []
            if isinstance(data, dict):
                for _, attrs in data.items():
                    if isinstance(attrs, dict):
                        sigs.append(self._edge_sig(attrs))
            return min(sigs) if sigs else self._edge_sig({})

        if isinstance(data, dict):
            return self._edge_sig(data)

        return self._edge_sig({})

    def _neighbors_items(
        self,
        colors: Dict[Any, str],
        v: Any,
        *,
        direction: str,
    ) -> List[str]:
        """
        Collect colored neighbor-edge descriptors for one node.

        :param colors:
            Current node colors.
        :type colors: Dict[Any, str]

        :param v:
            Node identifier.
        :type v: Any

        :param direction:
            Neighborhood direction, one of ``"in"``, ``"out"``, or ``"undir"``.
        :type direction: str

        :returns:
            Sorted color-edge descriptors.
        :rtype: List[str]
        """
        items: List[str] = []

        if direction == "in":
            if not self.G.is_directed():
                direction = "undir"
            else:
                for u in self.G.predecessors(v):
                    items.append(f"{colors[u]}#{self._edge_sig_between(u, v)}")

        if direction == "out":
            if not self.G.is_directed():
                direction = "undir"
            else:
                for u in self.G.successors(v):
                    items.append(f"{colors[u]}#{self._edge_sig_between(v, u)}")

        if direction == "undir":
            for u in self.G.neighbors(v):
                es = self._edge_sig_between(v, u) if self.G.has_edge(v, u) else ""
                if not es and self.G.has_edge(u, v):
                    es = self._edge_sig_between(u, v)
                items.append(f"{colors[u]}#{es}")

        items.sort()
        return items

    def _refine_once(self, colors: Dict[Any, str]) -> Dict[Any, str]:
        """
        Perform one WL refinement round.

        :param colors:
            Current node colors.
        :type colors: Dict[Any, str]

        :returns:
            Refined node colors.
        :rtype: Dict[Any, str]
        """
        new_colors: Dict[Any, str] = {}

        for v in self.G.nodes():
            parts: List[str] = [colors[v]]

            if self.include_in_neighbors:
                parts.append(
                    "IN["
                    + "|".join(self._neighbors_items(colors, v, direction="in"))
                    + "]"
                )

            if self.include_out_neighbors:
                parts.append(
                    "OUT["
                    + "|".join(self._neighbors_items(colors, v, direction="out"))
                    + "]"
                )

            new_colors[v] = hash_text("||".join(parts), digest_size=self.digest_size)

        return new_colors

    @staticmethod
    def _colors_equal(a: Dict[Any, str], b: Dict[Any, str]) -> bool:
        """
        Compare two color mappings exactly.

        :param a:
            First color mapping.
        :type a: Dict[Any, str]

        :param b:
            Second color mapping.
        :type b: Dict[Any, str]

        :returns:
            ``True`` if both mappings are identical.
        :rtype: bool
        """
        if a.keys() != b.keys():
            return False
        return all(a[k] == b[k] for k in a)

    @staticmethod
    def _buckets_from_colors(colors: Dict[Any, str]) -> Dict[str, List[Any]]:
        """
        Group nodes by final color.

        :param colors:
            Node-to-color mapping.
        :type colors: Dict[Any, str]

        :returns:
            Color buckets.
        :rtype: Dict[str, List[Any]]
        """
        buckets: Dict[str, List[Any]] = {}
        for v, c in colors.items():
            buckets.setdefault(c, []).append(v)
        return buckets

    @staticmethod
    def _orbits_from_buckets(buckets: Dict[str, List[Any]]) -> List[Set[Any]]:
        """
        Build approximate orbit sets from color buckets.

        :param buckets:
            Color buckets.
        :type buckets: Dict[str, List[Any]]

        :returns:
            Approximate orbit sets.
        :rtype: List[Set[Any]]
        """
        items = sorted(buckets.items(), key=lambda kv: (kv[0], len(kv[1])))
        out: List[Set[Any]] = []
        for _, nodes in items:
            out.append(set(sorted(nodes, key=str)))
        return out

    @staticmethod
    def _canonical_order_from_colors(
        G: nx.DiGraph,
        colors: Dict[Any, str],
    ) -> List[Any]:
        """
        Build a deterministic node order from final colors.

        :param G:
            Input graph.
        :type G: nx.DiGraph

        :param colors:
            Final color mapping.
        :type colors: Dict[Any, str]

        :returns:
            Deterministic canonical order.
        :rtype: List[Any]
        """
        return sorted(G.nodes(), key=lambda v: (colors[v], str(v)))

    def _run(self) -> _WLState:
        """
        Run WL refinement once and cache the result.

        :returns:
            Internal cached WL state.
        :rtype: _WLState

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            state = wl._run()
            print(state.iters_run, state.stabilized)
        """
        key = self._cache_key()
        if self._state_cache is not None and self._cache_key_last == key:
            return self._state_cache

        colors: Dict[Any, str] = {v: self._node_seed(v) for v in self.G.nodes()}
        stabilized = False
        iters_run = 0

        for it in range(self.n_iter):
            iters_run = it + 1
            new_colors = self._refine_once(colors)
            if self._colors_equal(new_colors, colors):
                stabilized = True
                colors = new_colors
                break
            colors = new_colors

        buckets = self._buckets_from_colors(colors)
        cells = [
            sorted(nodes, key=str)
            for _, nodes in sorted(
                buckets.items(),
                key=lambda kv: (kv[0], tuple(map(str, kv[1]))),
            )
        ]
        orbits = self._orbits_from_buckets(buckets)
        color_hist = {c: len(nodes) for c, nodes in buckets.items()}
        canonical_order = self._canonical_order_from_colors(self.G, colors)

        approx_count: Optional[int] = None
        if self.estimate_automorphisms:
            approx_count = approx_automorphism_count_from_cells(
                cells,
                cap=self.automorphism_cap,
            )

        self._state_cache = _WLState(
            colors=colors,
            cells=cells,
            orbits=orbits,
            color_hist=color_hist,
            iters_run=iters_run,
            stabilized=stabilized,
            canonical_order=canonical_order,
            approx_automorphism_count=approx_count,
        )
        self._summary_cache = None
        self._fast_signature = None
        self._cache_key_last = key
        return self._state_cache

    def colors(self) -> Dict[Any, str]:
        """
        Return final WL colors.

        :returns:
            Mapping from node to final color.
        :rtype: Dict[Any, str]

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            print(wl.colors())
        """
        return dict(self._run().colors)

    def color_of(self, v: Any) -> str:
        """
        Return the final WL color of one node.

        :param v:
            Node identifier.
        :type v: Any

        :returns:
            Final WL color.
        :rtype: str

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            print(wl.color_of(1))
        """
        return self._run().colors[v]

    def orbits(self) -> List[Set[Any]]:
        """
        Return approximate WL orbit sets.

        :returns:
            Approximate orbits induced by final WL colors.
        :rtype: List[Set[Any]]

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            print(wl.orbits())
        """
        return [set(x) for x in self._run().orbits]

    def wl_orbits(self) -> List[Set[Any]]:
        """
        Alias for :meth:`orbits`.

        :returns:
            Approximate WL orbit sets.
        :rtype: List[Set[Any]]
        """
        return self.orbits()

    def has_nontrivial_automorphism(self) -> bool:
        """
        Heuristically detect whether symmetry may be present.

        This is approximate and simply checks whether any WL color cell has size
        greater than one.

        :returns:
            ``True`` if WL detects a non-singleton cell, else ``False``.
        :rtype: bool

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            print(wl.has_nontrivial_automorphism())
        """
        return any(len(cell) > 1 for cell in self._run().cells)

    def canonical_order(self) -> List[Any]:
        """
        Return the deterministic WL node order.

        :returns:
            WL-based canonical node order.
        :rtype: List[Any]

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            print(wl.canonical_order())
        """
        return list(self._run().canonical_order)

    def canonical_key(self) -> Any:
        """
        Return the canonical key induced by the WL order.

        :returns:
            WL canonical key.
        :rtype: Any

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            print(wl.canonical_key())
        """
        return graph_key_from_order(self.G, self.canonical_order(), self.config)

    def canonical_graph(self) -> nx.DiGraph:
        """
        Return the canonically relabeled graph using the WL order.

        :returns:
            WL-canonically relabeled graph.
        :rtype: nx.DiGraph

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            G_can = wl.canonical_graph()
            print(sorted(G_can.nodes()))
        """
        order = self.canonical_order()
        mapping = {v: i + 1 for i, v in enumerate(order)}
        return nx.relabel_nodes(self.G, mapping, copy=True)

    def graph(self) -> nx.DiGraph:
        """
        Alias for :meth:`canonical_graph`, matching the older canon style.

        :returns:
            WL-canonically relabeled graph.
        :rtype: nx.DiGraph

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            G_can = wl.graph()
        """
        return self.canonical_graph()

    def canonical_result(self) -> WLCanonicalResult:
        """
        Build an approximate canonicalization result in a CRN-canon-like format.

        :returns:
            Approximate canonicalization result.
        :rtype: WLCanonicalResult

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            result = wl.canonical_result()
            print(result.canonical_order)
            print(result.automorphism_count)
        """
        if self._summary_cache is not None:
            return self._summary_cache

        start = perf_counter()
        state = self._run()
        can_graph = self.canonical_graph()
        can_key = graph_key_from_order(self.G, state.canonical_order, self.config)

        self._summary_cache = WLCanonicalResult(
            canon_graph=can_graph,
            graph_type=self.graph_type,
            canonical_order=list(state.canonical_order),
            canonical_key=can_key,
            automorphism_count=state.approx_automorphism_count,
            orbits=[set(x) for x in state.orbits],
            colors=dict(state.colors),
            color_hist=dict(state.color_hist),
            iters_run=state.iters_run,
            stabilized=state.stabilized,
            exact=False,
            elapsed_seconds=perf_counter() - start,
        )
        return self._summary_cache

    def summary(self) -> Dict[str, Any]:
        """
        Return a dictionary summary in a format close to the exact canonicalizer.

        The reported automorphism count and orbit sets are WL-based
        approximations.

        :returns:
            Summary dictionary.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            info = wl.summary()
            print(info["automorphism_count"])
            print(info["orbits"])
        """
        res = self.canonical_result()
        return {
            "canon_graph": res.canon_graph,
            "graph_type": res.graph_type,
            "automorphism_count": res.automorphism_count,
            "orbits": res.orbits,
            "canonical_perm": res.canonical_order,
            "canonical_key": res.canonical_key,
            "colors": res.colors,
            "color_hist": res.color_hist,
            "iters_run": res.iters_run,
            "stabilized": res.stabilized,
            "exact": res.exact,
            "elapsed_seconds": res.elapsed_seconds,
        }

    def fast_signature(self) -> Tuple[Any, ...]:
        """
        Return a fast graph signature using graph statistics and WL color
        histogram.

        This is useful as a cheap prefilter before exact graph isomorphism or
        exact canonicalization.

        :returns:
            Fast graph signature.
        :rtype: Tuple[Any, ...]

        Example
        -------
        .. code-block:: python

            wl = WLCanonicalizer(syn)
            print(wl.fast_signature())
        """
        if self._fast_signature is None:
            state = self._run()
            self._fast_signature = build_fast_signature(
                self.G,
                self.graph_type,
                self.config,
                wl_color_hist=state.color_hist,
            )
        return self._fast_signature


def wl_canonical(source: Any, **kwargs: Any) -> nx.DiGraph:
    """
    Convenience function returning the WL-canonically relabeled graph.

    :param source:
        Input CRN representation.
    :type source: Any

    :param kwargs:
        Additional keyword arguments forwarded to :class:`WLCanonicalizer`.
    :type kwargs: Any

    :returns:
        WL-canonically relabeled graph.
    :rtype: nx.DiGraph

    Example
    -------
    .. code-block:: python

        from synkit.CRN.Sym import SymmetryConfig, wl_canonical

        G_can = wl_canonical(
            syn.to_digraph(),
            include_rule=True,
            config=SymmetryConfig.topological(),
        )
    """
    return WLCanonicalizer(source, **kwargs).canonical_graph()
