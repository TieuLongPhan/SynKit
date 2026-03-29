from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from ._common import (
    AutomorphismResult,
    CanonicalResult,
    SymmetryConfig,
    edge_token,
    graph_key_from_order,
    node_token,
    orbits_from_mappings,
    should_stop,
)
from .wl_canon import WLCanonicalizer


@dataclass(frozen=True)
class IRInternalResult:
    """
    Internal exact-search result produced by :class:`IRCanonicalEngine`.

    :param canonical_order:
        Canonical node ordering selected by the exact individualize-refine search.
    :type canonical_order: List[Any]

    :param canonical_key:
        Canonical graph key associated with ``canonical_order``.
    :type canonical_key: Tuple[Any, ...]

    :param automorphism_count:
        Number of automorphisms discovered for the best canonical form. When the
        search is truncated, this may be a lower bound or capped count depending
        on the caller configuration.
    :type automorphism_count: int

    :param sample_permutations:
        Sample node orders equivalent to the canonical form.
    :type sample_permutations: List[List[Any]]

    :param sample_mappings:
        Sample automorphism mappings represented as ``source_node -> image_node``.
    :type sample_mappings: List[Dict[Any, Any]]

    :param orbits:
        Orbit partition induced by the sampled automorphism mappings.
    :type orbits: List[set[Any]]

    :param elapsed_seconds:
        Wall-clock runtime in seconds for the executed search.
    :type elapsed_seconds: float

    :param stopped_early:
        Whether the search stopped early because of timeout, count limit, or
        ``stop_after_two``.
    :type stopped_early: bool
    """

    canonical_order: List[Any]
    canonical_key: Tuple[Any, ...]
    automorphism_count: int
    sample_permutations: List[List[Any]]
    sample_mappings: List[Dict[Any, Any]]
    orbits: List[set[Any]]
    elapsed_seconds: float
    stopped_early: bool


class IRCanonicalEngine:
    """
    Exact individualize-refine engine for canonical labeling and automorphism search.

    This engine starts from a Weisfeiler-Lehman (WL) coloring, then applies an
    exact individualize-refine search on ambiguous cells. It is shared by both
    canonicalization and automorphism queries so that an exact completed result
    can be cached and reused.

    :param source:
        Input CRN-like object or graph accepted by :class:`WLCanonicalizer`.
    :type source: Any

    :param include_rule:
        Whether rule/reaction node semantics should be encoded into node tokens.
    :type include_rule: bool

    :param integer_ids:
        Whether integer-style IDs should be preferred when supported by the
        upstream graph builder.
    :type integer_ids: bool

    :param include_stoich:
        Whether stoichiometric edge/node information should be included in the
        canonicalization tokens.
    :type include_stoich: bool

    :param wl_iters:
        Maximum number of WL refinement iterations used to initialize the exact
        search.
    :type wl_iters: int

    :param wl_digest_size:
        Digest size used by the WL canonicalizer for hashed color refinement.
    :type wl_digest_size: int

    :param config:
        Symmetry configuration controlling which semantic attributes are included
        in node and edge tokens.
    :type config: Optional[SymmetryConfig]

    Notes
    -----
    Exact automorphism counting is generally harder than obtaining a single
    canonical order. Therefore, this class caches completed exact runs so later
    canonicalization and automorphism queries can reuse the same result instead
    of rerunning the search.

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Sym._ir import IRCanonicalEngine

        engine = IRCanonicalEngine(
            source=crn,
            include_rule=True,
            include_stoich=True,
        )

        canon = engine.canonical_result()
        print(canon.canonical_order)
        print(canon.canonical_key)

        auto = engine.automorphism_result(max_count=10)
        print(auto.automorphism_count)
        print(auto.orbits)
    """

    def __init__(
        self,
        source: Any,
        *,
        include_rule: bool = True,
        integer_ids: bool = False,
        include_stoich: bool = True,
        wl_iters: int = 20,
        wl_digest_size: int = 16,
        config: Optional[SymmetryConfig] = None,
    ) -> None:
        """
        Initialize the exact IR engine.

        :param source:
            Input CRN-like object or graph.
        :type source: Any

        :param include_rule:
            Whether rule/reaction semantics are included.
        :type include_rule: bool

        :param integer_ids:
            Whether integer IDs should be preserved or preferred when possible.
        :type integer_ids: bool

        :param include_stoich:
            Whether stoichiometric information is encoded into the graph tokens.
        :type include_stoich: bool

        :param wl_iters:
            Number of WL refinement iterations.
        :type wl_iters: int

        :param wl_digest_size:
            Digest size used internally by WL hashing.
        :type wl_digest_size: int

        :param config:
            Optional semantic symmetry configuration. If ``None``, semantic mode
            is used.
        :type config: Optional[SymmetryConfig]
        """
        self.config = config or SymmetryConfig.semantic()
        self.wl = WLCanonicalizer(
            source,
            include_rule=include_rule,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
            n_iter=wl_iters,
            digest_size=wl_digest_size,
            config=self.config,
        )
        self._refine_cache: Dict[Tuple[Tuple[Any, ...], ...], List[List[Any]]] = {}
        self._sig_cache: Dict[
            Tuple[Tuple[Tuple[Any, ...], ...], Any], Tuple[Any, ...]
        ] = {}
        self._node_tok_cache: Dict[Any, Tuple[Any, ...]] = {}
        self._deg_cache: Dict[Any, Tuple[int, int]] = {}
        self._pred_cache: Dict[Any, List[Tuple[Any, Tuple[Any, ...]]]] = {}
        self._succ_cache: Dict[Any, List[Tuple[Any, Tuple[Any, ...]]]] = {}
        self._act_cache: Dict[Any, int] = {}
        self._exact_full_cache: Optional[IRInternalResult] = None
        self._prepare_static_caches()

    @property
    def G(self) -> nx.DiGraph:
        """
        Return the internal directed graph used by the engine.

        :returns:
            Directed graph produced by :class:`WLCanonicalizer`.
        :rtype: nx.DiGraph
        """
        return self.wl.G

    @property
    def graph_type(self) -> str:
        """
        Return the graph type label reported by the WL canonicalizer.

        :returns:
            Graph type identifier.
        :rtype: str
        """
        return self.wl.graph_type

    def _prepare_static_caches(self) -> None:
        """
        Precompute static per-node caches used during refinement.

        This stores node tokens, degree pairs, predecessor signatures, successor
        signatures, and a simple activity score for branch ordering.

        :returns:
            ``None``.
        :rtype: None
        """
        G = self.G
        for v in G.nodes():
            self._node_tok_cache[v] = node_token(G.nodes[v], self.config)
            self._deg_cache[v] = (G.in_degree(v), G.out_degree(v))
            self._pred_cache[v] = [
                (u, edge_token(G[u][v], self.config)) for u in G.predecessors(v)
            ]
            self._succ_cache[v] = [
                (u, edge_token(G[v][u], self.config)) for u in G.successors(v)
            ]
            self._act_cache[v] = len(self._pred_cache[v]) + len(self._succ_cache[v])

    @staticmethod
    def _part_key(part: Sequence[Sequence[Any]]) -> Tuple[Tuple[Any, ...], ...]:
        """
        Convert a partition into a hashable cache key.

        :param part:
            Partition represented as a sequence of cells.
        :type part: Sequence[Sequence[Any]]

        :returns:
            Immutable tuple-of-tuples representation of the partition.
        :rtype: Tuple[Tuple[Any, ...], ...]
        """
        return tuple(tuple(cell) for cell in part)

    def _cell_signature(self, v: Any, part: Sequence[Sequence[Any]]) -> Tuple[Any, ...]:
        """
        Compute the refinement signature of a node with respect to a partition.

        The signature combines node token, degree, and per-cell incident edge
        token multisets. This is the core feature used to split cells during
        refinement.

        :param v:
            Node whose signature is requested.
        :type v: Any

        :param part:
            Current partition.
        :type part: Sequence[Sequence[Any]]

        :returns:
            Signature tuple used for exact refinement.
        :rtype: Tuple[Any, ...]
        """
        pkey = self._part_key(part)
        ck = (pkey, v)
        if ck in self._sig_cache:
            return self._sig_cache[ck]

        tok = self._node_tok_cache[v]
        deg = self._deg_cache[v]

        pred_map = defaultdict(list)
        succ_map = defaultdict(list)
        for u, etok in self._pred_cache[v]:
            pred_map[u].append(etok)
        for u, etok in self._succ_cache[v]:
            succ_map[u].append(etok)

        per_cell: List[Any] = []
        for cell in part:
            cell_set = set(cell)
            in_items: List[Tuple[Any, ...]] = []
            out_items: List[Tuple[Any, ...]] = []
            for u in cell_set:
                if u in pred_map:
                    in_items.extend(pred_map[u])
                if u in succ_map:
                    out_items.extend(succ_map[u])
            in_items.sort(key=str)
            out_items.sort(key=str)
            if self.config.use_edge_direction:
                per_cell.append((tuple(in_items), tuple(out_items)))
            else:
                mixed = in_items + out_items
                mixed.sort(key=str)
                per_cell.append(tuple(mixed))

        sig = (tok, deg, tuple(per_cell))
        self._sig_cache[ck] = sig
        return sig

    def _refine(self, part: Sequence[Sequence[Any]]) -> List[List[Any]]:
        """
        Refine a partition until it becomes stable.

        Cells are repeatedly split using :meth:`_cell_signature` until no further
        refinement is possible.

        :param part:
            Input partition.
        :type part: Sequence[Sequence[Any]]

        :returns:
            Stable refined partition.
        :rtype: List[List[Any]]
        """
        pkey = self._part_key(part)
        if pkey in self._refine_cache:
            return [list(cell) for cell in self._refine_cache[pkey]]

        cur = [sorted(cell, key=str) for cell in part]
        while True:
            changed = False
            new_part: List[List[Any]] = []
            cur_key = self._part_key(cur)
            for cell in cur:
                if len(cell) <= 1:
                    new_part.append(list(cell))
                    continue
                buckets: Dict[Tuple[Any, ...], List[Any]] = defaultdict(list)
                for v in cell:
                    sig = self._cell_signature(v, cur)
                    buckets[sig].append(v)
                if len(buckets) == 1:
                    new_part.append(list(cell))
                else:
                    changed = True
                    for _, block in sorted(
                        buckets.items(),
                        key=lambda kv: (str(kv[0]), tuple(map(str, kv[1]))),
                    ):
                        new_part.append(sorted(block, key=str))
            cur = new_part
            if not changed:
                result = [list(cell) for cell in cur]
                self._refine_cache[cur_key] = [list(cell) for cell in result]
                self._refine_cache[pkey] = [list(cell) for cell in result]
                return result

    def _initial_partition(self) -> List[List[Any]]:
        """
        Build the initial partition from WL colors and node tokens.

        :returns:
            Refined initial partition.
        :rtype: List[List[Any]]
        """
        buckets: Dict[Tuple[Any, ...], List[Any]] = defaultdict(list)
        for v in self.G.nodes():
            buckets[(self.wl.color_of(v), self._node_tok_cache[v])].append(v)
        part = [
            sorted(vs, key=str)
            for _, vs in sorted(
                buckets.items(), key=lambda kv: (str(kv[0]), tuple(map(str, kv[1])))
            )
        ]
        return self._refine(part)

    def _target_cell(self, part: Sequence[Sequence[Any]]) -> List[Any]:
        """
        Choose the next ambiguous cell to individualize.

        Preference is given to smaller cells, then cells with higher total
        activity, then token and string-based tie breaking.

        :param part:
            Current partition.
        :type part: Sequence[Sequence[Any]]

        :returns:
            Selected ambiguous cell.
        :rtype: List[Any]
        """
        ambiguous = [list(cell) for cell in part if len(cell) > 1]
        return min(
            ambiguous,
            key=lambda cell: (
                len(cell),
                -sum(self._act_cache[v] for v in cell),
                tuple(self._node_tok_cache[v] for v in cell),
                tuple(map(str, cell)),
            ),
        )

    def _candidate_order_key(
        self, v: Any, part: Sequence[Sequence[Any]]
    ) -> Tuple[Any, ...]:
        """
        Build a sorting key for candidates within the chosen target cell.

        :param v:
            Candidate node.
        :type v: Any

        :param part:
            Current partition.
        :type part: Sequence[Sequence[Any]]

        :returns:
            Candidate ordering key.
        :rtype: Tuple[Any, ...]
        """
        return (
            self._cell_signature(v, part),
            -self._act_cache[v],
            str(v),
        )

    @staticmethod
    def _slice_result(
        res: IRInternalResult, max_count: Optional[int], *, force_stopped: bool = False
    ) -> IRInternalResult:
        """
        Return a truncated copy of an exact result.

        :param res:
            Full internal result.
        :type res: IRInternalResult

        :param max_count:
            Maximum number of permutations/mappings to retain. If ``None``, the
            original result is returned unchanged.
        :type max_count: Optional[int]

        :param force_stopped:
            Whether the sliced result should be marked as stopped early even if
            the original full result completed.
        :type force_stopped: bool

        :returns:
            Possibly truncated internal result.
        :rtype: IRInternalResult
        """
        if max_count is None:
            return res
        perms = [list(p) for p in res.sample_permutations[:max_count]]
        maps = [dict(m) for m in res.sample_mappings[:max_count]]
        stopped = (
            force_stopped or (res.automorphism_count > max_count) or res.stopped_early
        )
        orbits = (
            res.orbits
            if not stopped
            else orbits_from_mappings(
                res.canonical_order,
                maps if maps else [{v: v for v in res.canonical_order}],
            )
        )
        return IRInternalResult(
            canonical_order=list(res.canonical_order),
            canonical_key=res.canonical_key,
            automorphism_count=min(res.automorphism_count, max_count),
            sample_permutations=perms,
            sample_mappings=maps,
            orbits=orbits,
            elapsed_seconds=res.elapsed_seconds,
            stopped_early=stopped,
        )

    def _reuse_cached_run(
        self,
        *,
        max_count: Optional[int],
        timeout_sec: Optional[float],
        stop_after_two: bool,
    ) -> Optional[IRInternalResult]:
        """
        Reuse a previously completed exact result when valid.

        :param max_count:
            Requested maximum number of stored automorphisms.
        :type max_count: Optional[int]

        :param timeout_sec:
            Optional timeout for the current request.
        :type timeout_sec: Optional[float]

        :param stop_after_two:
            Whether the caller only needs to distinguish uniqueness from
            non-uniqueness.
        :type stop_after_two: bool

        :returns:
            Cached result if reusable, otherwise ``None``.
        :rtype: Optional[IRInternalResult]
        """
        if timeout_sec is not None or self._exact_full_cache is None:
            return None
        if stop_after_two:
            return self._slice_result(self._exact_full_cache, 2)
        return self._slice_result(self._exact_full_cache, max_count)

    def _initialize_run_state(
        self,
    ) -> Tuple[float, List[List[Any]], List[Any], Tuple[Any, ...]]:
        """
        Prepare the initial state for an exact run.

        :returns:
            Tuple containing start time, initial refined partition, WL order, and
            the canonical key of the WL order.
        :rtype: Tuple[float, List[List[Any]], List[Any], Tuple[Any, ...]]
        """
        start = perf_counter()
        init = self._initial_partition()
        wl_order = self.wl.canonical_order()
        wl_key = graph_key_from_order(self.G, wl_order, self.config)
        return start, init, list(wl_order), wl_key

    def _build_child_partition(
        self, part: Sequence[Sequence[Any]], target: Sequence[Any], chosen: Any
    ) -> List[List[Any]]:
        """
        Create a child partition by individualizing one node from a target cell.

        :param part:
            Current partition.
        :type part: Sequence[Sequence[Any]]

        :param target:
            Target ambiguous cell to split.
        :type target: Sequence[Any]

        :param chosen:
            Node to isolate into its own singleton cell.
        :type chosen: Any

        :returns:
            Child partition after individualization.
        :rtype: List[List[Any]]
        """
        target_set = set(target)
        rem = [u for u in target if u != chosen]
        child: List[List[Any]] = []
        for cell in part:
            if set(cell) == target_set and len(cell) == len(target):
                child.append([chosen])
                if rem:
                    child.append(rem)
            else:
                child.append(list(cell))
        return child

    def _finalize_leaf(
        self,
        part: Sequence[Sequence[Any]],
        *,
        best_key: Optional[Tuple[Any, ...]],
        best_order: Optional[List[Any]],
        sample_perms: List[List[Any]],
        count: int,
        max_count: Optional[int],
        stop_after_two: bool,
    ) -> Tuple[
        Optional[Tuple[Any, ...]], Optional[List[Any]], List[List[Any]], int, bool
    ]:
        """
        Process a fully individualized partition.

        :param part:
            Fully discrete partition.
        :type part: Sequence[Sequence[Any]]

        :param best_key:
            Current best canonical key.
        :type best_key: Optional[Tuple[Any, ...]]

        :param best_order:
            Current best canonical order.
        :type best_order: Optional[List[Any]]

        :param sample_perms:
            Collected sample permutations for the current best key.
        :type sample_perms: List[List[Any]]

        :param count:
            Current automorphism count for the best key.
        :type count: int

        :param max_count:
            Maximum number of stored permutations.
        :type max_count: Optional[int]

        :param stop_after_two:
            Whether to stop once two equivalent canonical orders are found.
        :type stop_after_two: bool

        :returns:
            Updated ``(best_key, best_order, sample_perms, count, should_stop_now)``.
        :rtype: Tuple[Optional[Tuple[Any, ...]], Optional[List[Any]], List[List[Any]], int, bool]
        """
        order = [cell[0] for cell in part]
        key = graph_key_from_order(self.G, order, self.config)

        if best_key is None or key < best_key:
            return key, order, [list(order)], 1, False

        if key == best_key:
            count += 1
            if max_count is None or len(sample_perms) < max_count:
                sample_perms.append(list(order))
            return (
                best_key,
                best_order,
                sample_perms,
                count,
                (stop_after_two and count >= 2),
            )

        return best_key, best_order, sample_perms, count, False

    def _assemble_result(
        self,
        *,
        start: float,
        best_key: Tuple[Any, ...],
        best_order: List[Any],
        sample_perms: List[List[Any]],
        count: int,
        stopped: bool,
    ) -> IRInternalResult:
        """
        Assemble the final :class:`IRInternalResult`.

        :param start:
            Start time returned by :func:`perf_counter`.
        :type start: float

        :param best_key:
            Best canonical key found.
        :type best_key: Tuple[Any, ...]

        :param best_order:
            Best canonical order found.
        :type best_order: List[Any]

        :param sample_perms:
            Sample canonical-equivalent permutations.
        :type sample_perms: List[List[Any]]

        :param count:
            Automorphism count accumulated for the best key.
        :type count: int

        :param stopped:
            Whether the search stopped early.
        :type stopped: bool

        :returns:
            Internal result object.
        :rtype: IRInternalResult
        """
        sample_mappings = [
            {best_order[i]: p[i] for i in range(len(best_order))} for p in sample_perms
        ]
        orbits = orbits_from_mappings(
            best_order,
            sample_mappings if sample_mappings else [{v: v for v in best_order}],
        )
        return IRInternalResult(
            canonical_order=best_order,
            canonical_key=best_key,
            automorphism_count=count,
            sample_permutations=sample_perms,
            sample_mappings=sample_mappings,
            orbits=orbits,
            elapsed_seconds=perf_counter() - start,
            stopped_early=stopped,
        )

    def run(
        self,
        *,
        max_count: Optional[int] = None,
        timeout_sec: Optional[float] = None,
        stop_after_two: bool = False,
    ) -> IRInternalResult:
        """
        Run exact individualize-refine search.

        This method computes a canonical order exactly and, when requested,
        counts automorphisms of the best canonical form.

        :param max_count:
            Optional cap on the number of stored sample permutations and mappings.
            This does not necessarily limit the search unless enforced by
            :func:`should_stop`.
        :type max_count: Optional[int]

        :param timeout_sec:
            Optional wall-clock timeout in seconds.
        :type timeout_sec: Optional[float]

        :param stop_after_two:
            If ``True``, stop as soon as two canonical-equivalent leaves are found.
            This is useful for fast uniqueness testing.
        :type stop_after_two: bool

        :returns:
            Exact-search internal result.
        :rtype: IRInternalResult
        """
        cached = self._reuse_cached_run(
            max_count=max_count,
            timeout_sec=timeout_sec,
            stop_after_two=stop_after_two,
        )
        if cached is not None:
            return cached

        start, init, wl_order, wl_key = self._initialize_run_state()
        best_key: Optional[Tuple[Any, ...]] = wl_key
        best_order: Optional[List[Any]] = list(wl_order)
        sample_perms: List[List[Any]] = []
        count = 0
        stopped = False

        def dfs(part: List[List[Any]]) -> None:
            nonlocal best_key, best_order, sample_perms, count, stopped
            if stopped:
                return
            if should_stop(start, timeout_sec, count=count, max_count=max_count):
                stopped = True
                return

            refined = self._refine(part)
            ambiguous = [cell for cell in refined if len(cell) > 1]
            if not ambiguous:
                (
                    best_key,
                    best_order,
                    sample_perms,
                    count,
                    stop_now,
                ) = self._finalize_leaf(
                    refined,
                    best_key=best_key,
                    best_order=best_order,
                    sample_perms=sample_perms,
                    count=count,
                    max_count=max_count,
                    stop_after_two=stop_after_two,
                )
                if stop_now:
                    stopped = True
                return

            target = self._target_cell(refined)
            ordered_candidates = sorted(
                target, key=lambda v: self._candidate_order_key(v, refined)
            )
            for v in ordered_candidates:
                child = self._build_child_partition(refined, target, v)
                dfs(child)
                if stopped:
                    return

        dfs(init)
        assert best_key is not None and best_order is not None

        result = self._assemble_result(
            start=start,
            best_key=best_key,
            best_order=best_order,
            sample_perms=sample_perms,
            count=count,
            stopped=stopped,
        )
        if timeout_sec is None and not result.stopped_early and not stop_after_two:
            self._exact_full_cache = result
        return result

    def canonical_result(
        self, *, timeout_sec: Optional[float] = None
    ) -> CanonicalResult:
        """
        Compute the exact canonicalization result.

        :param timeout_sec:
            Optional timeout in seconds.
        :type timeout_sec: Optional[float]

        :returns:
            Public canonicalization result.
        :rtype: CanonicalResult
        """
        res = self.run(timeout_sec=timeout_sec)
        return CanonicalResult(
            canonical_order=res.canonical_order,
            canonical_key=res.canonical_key,
            exact=True,
            elapsed_seconds=res.elapsed_seconds,
        )

    def automorphism_result(
        self,
        *,
        max_count: Optional[int] = None,
        timeout_sec: Optional[float] = None,
        stop_after_two: bool = False,
    ) -> AutomorphismResult:
        """
        Compute the automorphism result.

        :param max_count:
            Optional cap on the number of stored sample mappings.
        :type max_count: Optional[int]

        :param timeout_sec:
            Optional timeout in seconds.
        :type timeout_sec: Optional[float]

        :param stop_after_two:
            Whether to stop once two automorphisms are found.
        :type stop_after_two: bool

        :returns:
            Public automorphism result.
        :rtype: AutomorphismResult
        """
        res = self.run(
            max_count=max_count, timeout_sec=timeout_sec, stop_after_two=stop_after_two
        )
        return AutomorphismResult(
            graph_type=self.graph_type,
            automorphism_count=res.automorphism_count,
            sample_mappings=res.sample_mappings,
            orbits=res.orbits,
            elapsed_seconds=res.elapsed_seconds,
            stopped_early=res.stopped_early,
        )
