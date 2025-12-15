from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx

from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine
from synkit.Synthesis.Reactor.strategy import Strategy
from synkit.Graph.Matcher.auto_est import AutoEst  # WL-1 orbit estimator

MappingDict = Dict[int, int]

__all__ = ["PartialMatcher"]


class PartialMatcher:
    """
    Component-subset helper for pattern→host subgraph matching.

    This matcher treats each connected component of the pattern as an
    independent "micro-pattern" and searches for consistent embeddings
    of subsets of these components into one or more host graphs. It can
    behave like a classic "partial matcher" (searching all component
    counts) or like a strict full-pattern matcher, depending on the
    ``partial`` flag.

    Internally, all embeddings for each pair (host, pattern component)
    are pre-computed once and then re-used for all combinations. This
    significantly reduces redundant calls to
    :class:`SubgraphSearchEngine` when exploring many subsets.

    Optionally, approximate WL-1 automorphism orbits can be used to
    prune embeddings that are equivalent under host symmetries via
    :paramref:`prune_auto`.

    Parameters
    ----------
    host : nx.Graph | Sequence[nx.Graph]
        Single host graph or sequence of host graphs.
    pattern : nx.Graph
        Pattern graph whose connected components act as building blocks.
    node_attrs : list[str]
        Node attribute keys enforced equal during matching.
    edge_attrs : list[str]
        Edge attribute keys enforced equal during matching.
    strategy : Strategy, optional
        Matching strategy forwarded to :class:`SubgraphSearchEngine`.
    max_results : int | None, optional
        Global cap on number of embeddings to store. If ``None``, no
        explicit cap is applied.
    partial : bool, optional
        If ``True``, auto-mode (``k=None``) searches all component counts
        from full pattern down to 1. If ``False``, auto-mode only tries
        ``k = n_components`` (i.e. full-pattern matching only).
    threshold : int | None, optional
        Optional cap on embeddings *per (host, component)* pairing. If
        exceeded, that pairing is treated as "no valid embeddings" and
        skipped. Defaults to :data:`SubgraphSearchEngine.DEFAULT_THRESHOLD`.
    pre_filter : bool, optional
        If ``True``, enable the cheap Cartesian-product pre-filter in
        :class:`SubgraphSearchEngine` for each (host, component) pair.
    prune_auto : bool, optional
        If ``True``, apply approximate automorphism-based pruning on the
        final list of embeddings using WL-1 orbits computed by
        :class:`AutoEst`. For safety, pruning is only applied when there
        is a single host graph. Defaults to ``False``.
    wl_max_iter : int, optional
        Maximum number of WL refinement iterations in :class:`AutoEst`
        when :paramref:`prune_auto` is enabled. Defaults to 10.
    """

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
        threshold: Optional[int] = None,
        pre_filter: bool = False,
        prune_auto: bool = False,
        wl_max_iter: int = 10,
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
        self._threshold: int = (
            threshold
            if threshold is not None
            else SubgraphSearchEngine.DEFAULT_THRESHOLD
        )
        self._pre_filter: bool = pre_filter

        # WL-1 approximate automorphism pruning settings
        self._prune_auto: bool = bool(prune_auto)
        self._wl_max_iter: int = int(wl_max_iter)

        # WL-style approximate embedding count
        self._approx_embedding_count: Optional[int] = None

        self._pattern_ccs: List[nx.Graph] = self._split_pattern_components()
        self._host_embeddings: List[List[List[MappingDict]]] = []
        self._precompute_embeddings()

        mappings = self._match_components(k=None)
        if self._prune_auto:
            mappings = self._prune_automorphic_mappings(mappings)
        self._mappings: List[MappingDict] = mappings

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
                    threshold=self._threshold,
                    pre_filter=self._pre_filter,
                )
                comp_embeddings.append(embeddings)
            host_embeddings.append(comp_embeddings)
        self._host_embeddings = host_embeddings

    def _prune_automorphic_mappings(
        self,
        mappings: List[MappingDict],
    ) -> List[MappingDict]:
        """
        Approximate automorphism-based pruning using WL-1 orbits.

        This uses :class:`AutoEst` on the (single) host graph to collapse
        mappings that hit the same multiset of WL-orbit indices. For
        multiple hosts, the input list is returned unchanged to avoid
        node-id collisions across different hosts.

        :param mappings: Flat list of pattern→host mappings.
        :type mappings: list[MappingDict]
        :returns: Possibly pruned list of mappings.
        :rtype: list[MappingDict]
        """
        if not mappings:
            return []

        if len(self.hosts) != 1:
            return mappings

        host = self.hosts[0]
        est = AutoEst(
            graph=host,
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            max_iter=self._wl_max_iter,
        ).fit()

        # AutoEst.deduplicate expects Mapping[hashable, hashable]
        return est.deduplicate(mappings)

    @staticmethod
    def _build_label_hist(
        graph: nx.Graph,
        node_attrs: Sequence[str],
    ) -> Dict[Tuple[Any, ...], int]:
        """
        Build WL-style initial label histogram for a graph.

        The label is ``(degree, attrs[node_attr_0], attrs[node_attr_1], ...)``.

        :param graph: Input NetworkX graph.
        :type graph: nx.Graph
        :param node_attrs: Node attribute keys to include in the label.
        :type node_attrs: Sequence[str]
        :returns: Mapping from label tuple to count.
        :rtype: dict[tuple, int]
        """
        hist: Dict[Tuple[Any, ...], int] = {}
        for node in graph.nodes():
            degree = graph.degree(node)
            attrs = graph.nodes[node]
            values = [attrs.get(key) for key in node_attrs]
            label = (degree, *values)
            hist[label] = hist.get(label, 0) + 1
        return hist

    @staticmethod
    def _approx_pair_count(
        host_hist: Dict[Tuple[Any, ...], int],
        comp_hist: Dict[Tuple[Any, ...], int],
    ) -> int:
        """
        Approximate number of label-consistent injective mappings for a
        (host, component) pair, ignoring adjacency.

        :param host_hist: Label histogram for the host graph.
        :type host_hist: dict
        :param comp_hist: Label histogram for the pattern component.
        :type comp_hist: dict
        :returns: Upper-bound estimate of injective mappings.
        :rtype: int
        """
        total = 1
        for label, p_count in comp_hist.items():
            h_count = host_hist.get(label, 0)
            if h_count < p_count:
                return 0
            for i in range(p_count):
                total *= h_count - i
        return total

    # ------------------------------------------------------------------
    # WL-style approximate helpers (factorised to reduce complexity)
    # ------------------------------------------------------------------
    def _normalise_k_values(self, k: Optional[int]) -> List[int]:
        """
        Normalise ``k`` to a list of component counts to consider.

        :param k: Desired number of components or ``None`` for auto-mode.
        :type k: int | None
        :returns: List of component counts to iterate over.
        :rtype: list[int]
        :raises ValueError: If ``k`` is outside ``[1, n_components]``.
        """
        n_cc = len(self._pattern_ccs)
        if k is not None:
            if k <= 0 or k > n_cc:
                raise ValueError(f"k must be between 1 and {n_cc}")
            return [k]
        if not self.partial:
            return [n_cc]
        return list(range(n_cc, 0, -1))

    def _build_host_hists(self) -> List[Dict[Tuple[Any, ...], int]]:
        """
        Build WL-style label histograms for all host graphs.

        :returns: List of label histograms, one per host.
        :rtype: list[dict[tuple, int]]
        """
        host_hists: List[Dict[Tuple[Any, ...], int]] = []
        for host in self.hosts:
            hist = self._build_label_hist(host, self.node_attrs)
            host_hists.append(hist)
        return host_hists

    def _build_comp_hists(self) -> List[Dict[Tuple[Any, ...], int]]:
        """
        Build WL-style label histograms for all pattern components.

        :returns: List of label histograms, one per component.
        :rtype: list[dict[tuple, int]]
        """
        comp_hists: List[Dict[Tuple[Any, ...], int]] = []
        for pat_cc in self._pattern_ccs:
            hist = self._build_label_hist(pat_cc, self.node_attrs)
            comp_hists.append(hist)
        return comp_hists

    def _compute_pair_counts(
        self,
        host_hists: List[Dict[Tuple[Any, ...], int]],
        comp_hists: List[Dict[Tuple[Any, ...], int]],
    ) -> List[List[int]]:
        """
        Pre-compute approximate counts for all (host, component) pairs.

        :param host_hists: WL label histograms for hosts.
        :type host_hists: list[dict[tuple, int]]
        :param comp_hists: WL label histograms for components.
        :type comp_hists: list[dict[tuple, int]]
        :returns: Matrix of approximate counts indexed as [host][component].
        :rtype: list[list[int]]
        """
        pair_counts: List[List[int]] = []
        for host_hist in host_hists:
            row: List[int] = []
            for comp_hist in comp_hists:
                count = self._approx_pair_count(host_hist, comp_hist)
                row.append(count)
            pair_counts.append(row)
        return pair_counts

    def _product_for_combo(
        self,
        row: List[int],
        combo: Sequence[int],
    ) -> int:
        """
        Compute product of pair counts for a single host/combination.

        :param row: List of pair counts for one host over all components.
        :type row: list[int]
        :param combo: Selected component indices.
        :type combo: Sequence[int]
        :returns: Product of counts, or 0 if any factor is 0.
        :rtype: int
        """
        prod_val = 1
        for cc_idx in combo:
            pair_count = row[cc_idx]
            if pair_count == 0:
                return 0
            prod_val *= pair_count
        return prod_val

    def _aggregate_pair_counts(
        self,
        pair_counts: List[List[int]],
        k_values: List[int],
    ) -> int:
        """
        Aggregate per-pair estimates over component subsets and hosts.

        :param pair_counts: Matrix of approximate counts [host][component].
        :type pair_counts: list[list[int]]
        :param k_values: Component counts to consider.
        :type k_values: list[int]
        :returns: Aggregated estimate (may be truncated by ``max_results``).
        :rtype: int
        """
        total_est = 0
        n_cc = len(self._pattern_ccs)
        cc_indices = range(n_cc)

        for k_try in k_values:
            for combo in combinations(cc_indices, k_try):
                for host_idx, row in enumerate(pair_counts):
                    prod_val = self._product_for_combo(row, combo)
                    if prod_val == 0:
                        continue
                    total_est += prod_val
                    if self.max_results and total_est >= self.max_results:
                        return int(self.max_results)
        return int(total_est)

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

              - ``partial=False`` → only ``k = n_components`` is used.
              - ``partial=True`` → searches all feasible ``k`` from
                ``n_components`` down to 1.

        :type k: int | None
        :returns: Flat list of pattern→host node mappings.
        :rtype: list[MappingDict]
        """
        if k is not None:
            return self._match_fixed_k(k)

        if not self.partial:
            return self._match_fixed_k(len(self._pattern_ccs))

        return self._match_all_k()

    def _match_all_k(self) -> List[MappingDict]:
        """
        Aggregate embeddings over all feasible component counts.

        This tries ``k = n_cc, n_cc-1, ..., 1`` and stops once
        :attr:`max_results` is reached (if set).

        :returns: Flat list of pattern→host node mappings.
        :rtype: list[MappingDict]
        """
        all_mappings: List[MappingDict] = []
        n_cc = len(self._pattern_ccs)

        for k_try in range(n_cc, 0, -1):
            mappings = self._match_fixed_k(k_try)
            if not mappings:
                continue

            for emb in mappings:
                all_mappings.append(emb)
                if self.max_results and len(all_mappings) >= self.max_results:
                    return all_mappings

        return all_mappings

    def _match_fixed_k(self, k: int) -> List[MappingDict]:
        """
        Match using exactly ``k`` connected components of the pattern.

        :param k: Number of connected components to select.
        :type k: int
        :returns: Flat list of pattern→host node mappings.
        :rtype: list[MappingDict]
        :raises ValueError: If ``k`` is outside ``[1, n_components]``.
        """
        n_cc = len(self._pattern_ccs)
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
    # WL-style approximate embedding count
    # ------------------------------------------------------------------
    def estimate_embeddings_wl(self, k: Optional[int] = None) -> "PartialMatcher":
        """
        Estimate the number of embeddings using WL-style initial labels.

        This is a **cheap, approximate upper bound** that:

        * Builds WL-style labels ``(degree, node_attrs...)`` on the host
          and pattern components.
        * For each (host, component) pair, estimates the number of
          label-consistent injective mappings ignoring adjacency, via
          a product of falling factorials per label class.
        * Aggregates these per-pair estimates over subsets of pattern
          components using the same semantics as :meth:`_match_components`.

        No calls to :class:`SubgraphSearchEngine` or backtracking are
        performed. The result is stored in
        :attr:`approx_embedding_count`.

        :param k: Number of pattern components to use. If ``None``,
            behaviour mirrors :meth:`_match_components`:

            * ``partial=False`` → use only full pattern (``k=n_cc``).
            * ``partial=True`` → aggregate over all k from ``n_cc`` down to 1.

        :type k: int | None
        :returns: The estimator itself (for chained use).
        :rtype: PartialMatcher
        """
        k_values = self._normalise_k_values(k)
        host_hists = self._build_host_hists()
        comp_hists = self._build_comp_hists()
        pair_counts = self._compute_pair_counts(host_hists, comp_hists)
        total_est = self._aggregate_pair_counts(pair_counts, k_values)

        if self.max_results and total_est >= self.max_results:
            self._approx_embedding_count = int(self.max_results)
        else:
            self._approx_embedding_count = int(total_est)

        return self

    @property
    def approx_embedding_count(self) -> int:
        """
        WL-style approximate embedding count.

        :returns: Last estimated embedding count.
        :rtype: int
        :raises RuntimeError: If :meth:`estimate_embeddings_wl` has not
            been called.
        """
        if self._approx_embedding_count is None:
            raise RuntimeError(
                "Call 'estimate_embeddings_wl()' before accessing "
                "'approx_embedding_count'."
            )
        return self._approx_embedding_count

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

    @property
    def threshold(self) -> int:
        """
        Per-(host, component) embedding threshold.

        :returns: Threshold passed to :class:`SubgraphSearchEngine`.
        :rtype: int
        """
        return self._threshold

    @property
    def pre_filter(self) -> bool:
        """
        Whether the cheap pre-filter is enabled.

        :returns: Current value of the pre-filter flag.
        :rtype: bool
        """
        return self._pre_filter

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
            f"partial={self.partial} threshold={self._threshold} "
            f"pre_filter={self._pre_filter} prune_auto={self._prune_auto}>"
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
        threshold: Optional[int] = None,
        pre_filter: bool = False,
        prune_auto: bool = False,
        wl_max_iter: int = 10,
    ) -> List[MappingDict]:
        """
        Stateless convenience wrapper – one-liner for users in a hurry.

        This mirrors the OO API but avoids explicitly instantiating the
        matcher in user code.

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
        :param partial: If ``True``, all component counts are tried in
            auto-mode. If ``False``, only the full pattern is used.
        :type partial: bool
        :param threshold: Optional per-(host, component) embedding cap
            forwarded to :class:`SubgraphSearchEngine`.
        :type threshold: int | None
        :param pre_filter: Whether to enable the cheap pre-filter in
            :class:`SubgraphSearchEngine`.
        :type pre_filter: bool
        :param prune_auto: If ``True``, apply WL-1-based approximate
            automorphism pruning on the final mappings.
        :type prune_auto: bool
        :param wl_max_iter: Maximum number of WL iterations for the
            internal :class:`AutoEst` if :paramref:`prune_auto` is
            enabled.
        :type wl_max_iter: int
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
            threshold=threshold,
            pre_filter=pre_filter,
            prune_auto=prune_auto,
            wl_max_iter=wl_max_iter,
        )
        if k is not None:
            return matcher._match_components(k)
        return matcher.get_mappings()
