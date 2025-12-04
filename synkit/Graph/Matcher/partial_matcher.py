from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterator, List, Optional, Sequence, Set, Union

import networkx as nx

from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine
from synkit.Synthesis.Reactor.strategy import Strategy

MappingDict = Dict[int, int]

__all__ = ["PartialMatcher"]


class PartialMatcher:
    """
    Component-subset helper for pattern→host subgraph matching.

    This matcher treats each connected component of the pattern as an
    independent "micro-pattern" and searches for consistent embeddings
    of subsets of these components into one or more host graphs.

    Semantics of ``partial``
    ------------------------
    * ``partial=False`` (full mode):

      - Auto-mode (``k=None``) only tries **full-pattern** matching,
        i.e. embeddings that use **all** connected components.

    * ``partial=True`` (strict partial mode):

      - Auto-mode (``k=None``) tries **only partial subsets**:
        ``k = n_components - 1, n_components - 2, ..., 1``.
      - Full-pattern matches (using all components) are excluded in
        this default mode.

      Explicit ``k`` still does what you ask for; i.e. you can request
      ``k = n_components`` even if ``partial=True``. The ``partial``
      flag only affects the behaviour when ``k`` is ``None``.

    Efficiency
    ----------
    For efficiency, all embeddings for each pair (host, pattern
    component) are pre-computed once and then re-used for all component
    combinations. This avoids repeated calls to
    :class:`SubgraphSearchEngine` when exploring many subsets.

    :param host: Single host graph or sequence of host graphs.
    :type host: nx.Graph | Sequence[nx.Graph]
    :param pattern: Pattern graph whose connected components act as
        building blocks.
    :type pattern: nx.Graph
    :param node_attrs: Node attribute keys enforced equal during
        matching.
    :type node_attrs: list[str]
    :param edge_attrs: Edge attribute keys enforced equal during
        matching.
    :type edge_attrs: list[str]
    :param strategy: Matching strategy forwarded to
        :class:`SubgraphSearchEngine`.
    :type strategy: Strategy
    :param max_results: Global cap on number of embeddings to store.
        If ``None``, no explicit cap is applied.
    :type max_results: int | None
    :param partial: If ``False``, auto-mode (``k=None``) behaves as
        full-pattern matcher. If ``True``, auto-mode only returns
        strict partial matches (no full-pattern embeddings).
    :type partial: bool

    Examples
    --------
    Basic usage with a pattern consisting of two disconnected edges
    and a single host graph:

    .. code-block:: python

        import networkx as nx
        from synkit.Graph.Matcher.partial_matcher import PartialMatcher
        from synkit.Synthesis.Reactor.strategy import Strategy

        # Host: path on 5 nodes
        host = nx.path_graph(5)

        # Pattern: two disconnected edges (0-1) and (2-3)
        pattern = nx.Graph()
        pattern.add_edges_from([(0, 1), (2, 3)])

        # Strict partial mode: only partial embeddings (no full-pattern)
        matcher = PartialMatcher(
            host=host,
            pattern=pattern,
            node_attrs=[],
            edge_attrs=[],
            strategy=Strategy.COMPONENT,
            max_results=None,
            partial=True,
        )
        mappings = matcher.get_mappings()
        print("Partial mappings:", mappings)

    One-liner via the functional wrapper:

    .. code-block:: python

        mappings = PartialMatcher.find_partial_mappings(
            host=host,
            pattern=pattern,
            node_attrs=[],
            edge_attrs=[],
            k=None,  # auto-mode
            strategy=Strategy.COMPONENT,
            max_results=100,
            partial=True,
        )
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        host: Union[nx.Graph, Sequence[nx.Graph]],
        pattern: nx.Graph,
        node_attrs: List[str],
        edge_attrs: List[str],
        *,
        strategy: Strategy = Strategy.COMPONENT,
        max_results: Optional[int] = None,
        partial: bool = True,
    ) -> None:
        if isinstance(host, nx.Graph):
            self.hosts: List[nx.Graph] = [host]
        elif isinstance(host, Sequence):
            self.hosts = list(host)
        else:
            raise TypeError(
                "host must be a networkx.Graph or a sequence of such graphs"
            )

        self.pattern: nx.Graph = pattern
        self.node_attrs: List[str] = node_attrs
        self.edge_attrs: List[str] = edge_attrs
        self.strategy: Strategy = strategy
        self.max_results: Optional[int] = max_results
        self.partial: bool = partial

        self._pattern_ccs: List[nx.Graph] = self._split_pattern_components()
        # _host_embeddings[host_index][component_index] -> list[MappingDict]
        self._host_embeddings: List[List[List[MappingDict]]] = []
        self._precompute_embeddings()

        # Auto-run matching on construction with k=None semantics
        self._mappings: List[MappingDict] = self._match_components(k=None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _split_pattern_components(self) -> List[nx.Graph]:
        """
        Split the pattern into connected components.

        :returns: List of connected component subgraphs.
        :rtype: list[nx.Graph]
        :raises ValueError: If the pattern has no components.
        """
        components = [
            self.pattern.subgraph(c).copy()
            for c in nx.connected_components(self.pattern)
        ]
        if not components:
            raise ValueError("Pattern graph has no components.")
        return components

    def _precompute_embeddings(self) -> None:
        """
        Pre-compute embeddings for each (host, component) pair.

        The results are stored in :attr:`_host_embeddings` as a nested
        list indexed as ``[host_index][component_index]``.
        """
        host_embeddings: List[List[List[MappingDict]]] = []
        for host in self.hosts:
            comp_embeddings: List[List[MappingDict]] = []
            for pat_cc in self._pattern_ccs:
                embeddings = SubgraphSearchEngine.find_subgraph_mappings(
                    host,
                    pat_cc,
                    node_attrs=self.node_attrs,
                    edge_attrs=self.edge_attrs,
                    strategy=self.strategy,
                    max_results=self.max_results,
                    strict_cc_count=False,
                )
                comp_embeddings.append(embeddings)
            host_embeddings.append(comp_embeddings)
        self._host_embeddings = host_embeddings

    # ------------------------------------------------------------------
    # Core matching logic
    # ------------------------------------------------------------------
    def _match_components(self, k: Optional[int] = None) -> List[MappingDict]:
        """
        Internal search – returns a *flat* list of embeddings.

        :param k: Number of connected components of the pattern to use.

            * If an integer, the search is restricted to subsets of
              exactly ``k`` pattern components.
            * If ``None``, behaviour depends on :attr:`partial`:

              - ``partial=False`` → only full pattern (``k = n_components``).
              - ``partial=True`` → only strict partials:
                ``k = n_components - 1, ..., 1``.

        :type k: int | None
        :returns: Flat list of pattern→host node mappings.
        :rtype: list[MappingDict]
        """
        n_cc = len(self._pattern_ccs)

        if k is not None:
            return self._match_fixed_k(k, n_cc)

        if not self.partial:
            # Full mode: only full-pattern embeddings
            return self._match_fixed_k(n_cc, n_cc)

        # Strict partial mode: no full pattern; start from n_cc - 1
        if n_cc <= 1:
            # Cannot have strict partial when there is only one component
            return []

        start_k = n_cc - 1
        return self._match_k_range(start_k=start_k, stop_k=1, n_cc=n_cc)

    def _match_k_range(
        self,
        *,
        start_k: int,
        stop_k: int,
        n_cc: int,
    ) -> List[MappingDict]:
        """
        Aggregate embeddings over k in [stop_k, start_k] (descending).

        :param start_k: Maximum number of components to use.
        :type start_k: int
        :param stop_k: Minimum number of components to use.
        :type stop_k: int
        :param n_cc: Total number of connected components.
        :type n_cc: int
        :returns: Flat list of pattern→host node mappings.
        :rtype: list[MappingDict]
        """
        start_k = min(start_k, n_cc)
        stop_k = max(stop_k, 1)
        if start_k < stop_k:
            return []

        all_mappings: List[MappingDict] = []
        for k_try in range(start_k, stop_k - 1, -1):
            mappings = self._match_fixed_k(k_try, n_cc)
            if not mappings:
                continue
            for emb in mappings:
                all_mappings.append(emb)
                if self.max_results and len(all_mappings) >= self.max_results:
                    return all_mappings
        return all_mappings

    def _match_fixed_k(self, k: int, n_cc: int) -> List[MappingDict]:
        """
        Match using exactly ``k`` connected components of the pattern.

        :param k: Number of connected components to select.
        :type k: int
        :param n_cc: Total number of connected components.
        :type n_cc: int
        :returns: Flat list of pattern→host node mappings.
        :rtype: list[MappingDict]
        :raises ValueError: If ``k`` is outside ``[1, n_cc]``.
        """
        if k <= 0 or k > n_cc:
            raise ValueError(f"k must be between 1 and {n_cc}")

        all_mappings: List[MappingDict] = []
        cc_indices = range(n_cc)

        for combo in combinations(cc_indices, k):
            for host_index, _host in enumerate(self.hosts):
                self._backtrack_components(
                    combo=combo,
                    host_index=host_index,
                    level=0,
                    used_nodes=set(),
                    accum={},
                    out=all_mappings,
                )
                if self.max_results and len(all_mappings) >= self.max_results:
                    return all_mappings

        return all_mappings

    def _backtrack_components(
        self,
        combo: Sequence[int],
        host_index: int,
        level: int,
        used_nodes: Set[int],
        accum: MappingDict,
        out: List[MappingDict],
    ) -> None:
        """
        Backtracking across selected components within a single host.

        :param combo: Sequence of component indices to match in order.
        :type combo: Sequence[int]
        :param host_index: Index of the current host in :attr:`hosts`.
        :type host_index: int
        :param level: Current recursion depth (index in ``combo``).
        :type level: int
        :param used_nodes: Set of host node ids already used.
        :type used_nodes: set[int]
        :param accum: Accumulated pattern→host mapping.
        :type accum: MappingDict
        :param out: List where completed mappings are appended.
        :type out: list[MappingDict]
        """
        if self.max_results and len(out) >= self.max_results:
            return

        if level == len(combo):
            out.append(accum.copy())
            return

        cc_idx = combo[level]
        embeddings = self._host_embeddings[host_index][cc_idx]
        if not embeddings:
            return

        for emb in embeddings:
            mapped = set(emb.values())
            if mapped & used_nodes:
                continue

            new_used = used_nodes | mapped
            new_accum = {**accum, **emb}
            self._backtrack_components(
                combo=combo,
                host_index=host_index,
                level=level + 1,
                used_nodes=new_used,
                accum=new_accum,
                out=out,
            )

    # ------------------------------------------------------------------
    # Public instance helpers
    # ------------------------------------------------------------------
    def get_mappings(self) -> List[MappingDict]:
        """
        Return the list of discovered embeddings (auto-computed).

        :returns: List of pattern→host node mappings.
        :rtype: list[MappingDict]
        """
        return self._mappings

    @property
    def num_mappings(self) -> int:
        """
        Number of embeddings found.

        :returns: Count of discovered embeddings.
        :rtype: int
        """
        return len(self._mappings)

    @property
    def num_pattern_components(self) -> int:
        """
        Number of connected components in the pattern graph.

        :returns: Number of pattern connected components.
        :rtype: int
        """
        return len(self._pattern_ccs)

    # Iteration support -------------------------------------------------
    def __iter__(self) -> Iterator[MappingDict]:
        """
        Iterate over discovered embeddings.

        :returns: Iterator over mapping dictionaries.
        :rtype: Iterator[MappingDict]
        """
        return iter(self._mappings)

    # Niceties ----------------------------------------------------------
    def __repr__(self) -> str:
        """
        Representation string for debugging.

        :returns: Short summary of matcher state.
        :rtype: str
        """
        return (
            f"<PartialMatcher pattern_ccs={self.num_pattern_components} "
            f"hosts={len(self.hosts)} mappings={self.num_mappings} "
            f"partial={self.partial}>"
        )

    __str__ = __repr__

    @property
    def help(self) -> str:
        """
        Return the full module docstring.

        :returns: Module-level documentation string.
        :rtype: str
        """
        return __doc__ or ""

    # ------------------------------------------------------------------
    # Functional/staticmethod wrapper
    # ------------------------------------------------------------------
    @staticmethod
    def find_partial_mappings(
        host: Union[nx.Graph, Sequence[nx.Graph]],
        pattern: nx.Graph,
        *,
        node_attrs: List[str],
        edge_attrs: List[str],
        k: Optional[int] = None,
        strategy: Strategy = Strategy.COMPONENT,
        max_results: Optional[int] = None,
        partial: bool = True,
    ) -> List[MappingDict]:
        """
        Stateless convenience wrapper – one-liner for users in a hurry.

        The :param:`partial` flag here behaves exactly like on the class:

        * ``partial=False`` and ``k=None`` → full-pattern only.
        * ``partial=True`` and ``k=None`` → strict partials only.

        :param host: A single host graph or a sequence of host graphs.
        :type host: nx.Graph | Sequence[nx.Graph]
        :param pattern: Pattern graph whose connected components are used
            as building blocks.
        :type pattern: nx.Graph
        :param node_attrs: Node attribute keys to enforce equality on
            during matching.
        :type node_attrs: list[str]
        :param edge_attrs: Edge attribute keys to enforce equality on
            during matching.
        :type edge_attrs: list[str]
        :param k: If an integer, restricts the search to subsets of
            exactly ``k`` pattern connected components. If ``None``,
            behaviour follows the ``partial`` flag.
        :type k: int | None
        :param strategy: Matching strategy forwarded to
            :class:`SubgraphSearchEngine`.
        :type strategy: Strategy
        :param max_results: Optional global cap on the number of
            embeddings to return.
        :type max_results: int | None
        :param partial: If ``True``, all k in a strict-partial range are
            tried in auto-mode; if ``False``, only the full pattern is
            used in auto-mode.
        :type partial: bool
        :returns: Flat list of pattern→host node mappings.
        :rtype: list[MappingDict]
        """
        matcher = PartialMatcher(
            host=host,
            pattern=pattern,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            strategy=strategy,
            max_results=max_results,
            partial=partial,
        )
        if k is not None:
            return matcher._match_components(k)
        return matcher.get_mappings()
