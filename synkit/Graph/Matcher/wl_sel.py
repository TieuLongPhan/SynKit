from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Iterator

import networkx as nx


NodeSig = Counter  # Counter[str, int]
EdgeSig = Counter  # Counter[str, int]
DegSig = Counter  # Counter[int, int]
TieTuple = Tuple[Any, ...]


class WLSel:
    """
    WL-based selector for pairing two lists of graphs.

    Parameters
    ----------
    fw : Sequence[nx.Graph]
        Forward graphs (indices form first element of pairs).
    bw : Sequence[nx.Graph]
        Backward graphs (indices form second element of pairs).
    element_key : str or None
        Node attribute name used to detect wildcard nodes. Nodes with
        ``data[element_key] == "*"`` are removed from the core. If None,
        no wildcard filtering is applied.
    node_attrs : sequence of str or None
        Node attributes used to build base labels. If provided, the base
        label for a node is ``str(tuple(data[k] for k in node_attrs))``.
        If empty and element_key is provided, the element value is used.
        If both are empty/None, node degree is used as base label.
    edge_attrs : sequence of str or None
        Edge attributes used inside WL neighbor signatures. If multiple keys
        are provided, temporary edge tuples are formed internally.
    wl_iters : int
        WL refinement iterations (0 disables WL, uses base labels).
    min_score : float
        Minimum score (0..1) for pairs to be kept by default in scoring.
    node_weight : float
        Weight for node-overlap in final score (size-sim gets 1-node_weight).

    Notes
    -----
    - Use :meth:`build_signatures` then :meth:`score_pairs`.
    - Results available via :attr:`pair_scores` and :attr:`pair_indices`.
    """

    def __init__(
        self,
        fw: Sequence[nx.Graph],
        bw: Sequence[nx.Graph],
        element_key: Optional[str] = "element",
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
        wl_iters: int = 1,
        min_score: float = 0.8,
        node_weight: float = 0.85,
    ) -> None:
        self._fw = list(fw)
        self._bw = list(bw)

        self.element_key = element_key
        self.node_attrs = list(node_attrs) if node_attrs else []
        self.edge_attrs = list(edge_attrs) if edge_attrs else []

        self.wl_iters = max(0, int(wl_iters))
        self.min_score = float(min_score)
        self.node_weight = float(node_weight)

        # cached per-graph signatures
        self._fw_node_sigs: Dict[int, NodeSig] = {}
        self._bw_node_sigs: Dict[int, NodeSig] = {}
        self._fw_edge_sigs: Dict[int, EdgeSig] = {}
        self._bw_edge_sigs: Dict[int, EdgeSig] = {}
        self._fw_deg_sigs: Dict[int, DegSig] = {}
        self._bw_deg_sigs: Dict[int, DegSig] = {}
        self._fw_sizes: Dict[int, int] = {}
        self._bw_sizes: Dict[int, int] = {}

        self._signatures_built = False

        # scored pair storage: list of (i, j, primary_score, tie_tuple)
        self._pair_scores: List[Tuple[int, int, float, TieTuple]] = []
        self._pairs: List[Tuple[int, int]] = []

    # ---------------- fluent API ----------------
    def build_signatures(self) -> "WLSel":
        """
        Build WL-based node label multisets, edge multisets and degree multisets.
        Returns self for fluent usage.
        """
        self._fw_node_sigs.clear()
        self._bw_node_sigs.clear()
        self._fw_edge_sigs.clear()
        self._bw_edge_sigs.clear()
        self._fw_deg_sigs.clear()
        self._bw_deg_sigs.clear()
        self._fw_sizes.clear()
        self._bw_sizes.clear()

        for idx, g in enumerate(self._fw):
            core = self._core_subgraph(g)
            labels = self._wl_node_labels(core)
            self._fw_node_sigs[idx] = Counter(labels)
            self._fw_edge_sigs[idx] = self._edge_signature(core)
            self._fw_deg_sigs[idx] = Counter(d for _, d in core.degree())
            self._fw_sizes[idx] = core.number_of_nodes()

        for idx, g in enumerate(self._bw):
            core = self._core_subgraph(g)
            labels = self._wl_node_labels(core)
            self._bw_node_sigs[idx] = Counter(labels)
            self._bw_edge_sigs[idx] = self._edge_signature(core)
            self._bw_deg_sigs[idx] = Counter(d for _, d in core.degree())
            self._bw_sizes[idx] = core.number_of_nodes()

        self._signatures_built = True
        return self

    def score_pairs(
        self,
        top_k: Optional[int] = None,
        require_label_exact: bool = False,
    ) -> "WLSel":
        """
        Score all fw–bw pairs using WL-overlap + size similarity.

        Parameters
        ----------
        top_k : int or None
            If provided, keep only top_k pairs after sorting.
        require_label_exact : bool
            If True, keep only pairs whose WL label multisets are identical.

        Returns
        -------
        WLSel
            self (pairs stored in .pair_scores and .pair_indices).
        """
        if not self._signatures_built:
            self.build_signatures()

        scored: List[Tuple[int, int, float, TieTuple]] = []
        w_node = self.node_weight
        min_sc = self.min_score

        for i, sig1 in self._fw_node_sigs.items():
            n1 = self._fw_sizes.get(i, 0)
            e1 = self._fw_edge_sigs.get(i, Counter())
            deg1 = self._fw_deg_sigs.get(i, Counter())
            for j, sig2 in self._bw_node_sigs.items():
                n2 = self._bw_sizes.get(j, 0)
                e2 = self._bw_edge_sigs.get(j, Counter())
                deg2 = self._bw_deg_sigs.get(j, Counter())

                node_overlap = self._overlap_counters(sig1, sig2)
                size_sim = self._size_similarity(n1, n2)
                primary = w_node * node_overlap + (1.0 - w_node) * size_sim

                if primary < min_sc:
                    continue

                if require_label_exact and sig1 != sig2:
                    continue

                # tie breakers (higher is better)
                label_exact = 1 if sig1 == sig2 else 0
                edge_overlap = (
                    self._overlap_counters(e1, e2) if self.edge_attrs else 0.0
                )
                degree_overlap = self._overlap_counters(deg1, deg2)
                unique_label_overlap = self._unique_label_overlap(sig1, sig2)

                tie_tuple: TieTuple = (
                    label_exact,
                    edge_overlap,
                    degree_overlap,
                    unique_label_overlap,
                )

                scored.append((i, j, primary, tie_tuple))

        # sort by primary then tie_tuple (descending)
        scored.sort(key=lambda t: (t[2], t[3]), reverse=True)

        if top_k is not None:
            scored = scored[: int(top_k)]

        self._pair_scores = scored
        self._pairs = [(i, j) for (i, j, _, _) in scored]
        return self

    # ---------------- accessors ----------------
    @property
    def pair_scores(self) -> List[Tuple[int, int, float, TieTuple]]:
        """Return scored pairs as (i, j, primary_score, tie_tuple)."""
        return list(self._pair_scores)

    @property
    def pair_indices(self) -> List[Tuple[int, int]]:
        """Return list of pair indices (i, j) in sorted order."""
        return list(self._pairs)

    def candidate_pairs(
        self, max_pairs: Optional[int] = None
    ) -> Generator[Tuple[int, int], None, None]:
        """
        Yield candidate index pairs (i, j). If scoring hasn't been run, it will
        be invoked with default settings.
        """
        if not self._pairs:
            self.score_pairs()

        if max_pairs is None:
            for i, j in self._pairs:
                yield (i, j)
        else:
            for i, j in self._pairs[: int(max_pairs)]:
                yield (i, j)

    def filter_best_pairs(
        self, top_k: int = 1, min_score: Optional[float] = None
    ) -> "WLSel":
        """
        Keep only the best `top_k` pairs (by current ordering) and optionally
        enforce a minimum primary score. Returns self.
        """
        if not self._pair_scores:
            self.score_pairs()

        threshold = float(min_score) if min_score is not None else -1.0
        filtered: List[Tuple[int, int, float, TieTuple]] = []
        for i, j, sc, tie in self._pair_scores:
            if sc >= threshold:
                filtered.append((i, j, sc, tie))
        if top_k > 0:
            filtered = filtered[:top_k]

        self._pair_scores = filtered
        self._pairs = [(i, j) for (i, j, _, _) in filtered]
        return self

    # ---------------- internal helpers ----------------
    def _core_subgraph(self, g: nx.Graph) -> nx.Graph:
        """Return induced subgraph with wildcard nodes removed."""
        if self.element_key is None:
            return g
        key = self.element_key
        keep = [n for n, d in g.nodes(data=True) if d.get(key) != "*"]
        return g.subgraph(keep)

    def _base_labels(self, g: nx.Graph) -> List[str]:
        """Build base labels for nodes in graph g (in g.nodes() order)."""
        labels: List[str] = []

        if self.node_attrs:
            keys = self.node_attrs
            for _, data in g.nodes(data=True):
                vals = tuple(data.get(k) for k in keys)
                labels.append(str(vals))
            return labels

        if self.element_key is not None:
            key = self.element_key
            for _, data in g.nodes(data=True):
                labels.append(str(data.get(key, "X")))
            return labels

        for n in g.nodes():
            labels.append(str(g.degree[n]))
        return labels

    # -------- WL label plumbing (refactored) --------
    def _wl_node_labels(self, g: nx.Graph) -> List[str]:
        """
        Return WL-refined labels.

        Strategy:
        - If empty graph or wl_iters <= 0 -> base labels.
        - Prefer networkx WL node hashes when available.
        - Otherwise fall back to local deterministic WL.
        """
        if g.number_of_nodes() == 0 or self.wl_iters <= 0:
            return self._base_labels(g)

        nx_wl = self._nx_wl_hashes()
        if nx_wl is None:
            return self._local_wl_node_labels(g, iters=self.wl_iters)

        node_attr_arg, edge_attr_arg = self._resolve_wl_attr_args()

        with self._inject_temp_attrs(
            g,
            node_attr_arg=node_attr_arg,
            edge_attr_arg=edge_attr_arg,
        ) as (node_attr_final, edge_attr_final):
            node_hash_dict = nx_wl(
                g,
                node_attr=node_attr_final,
                edge_attr=edge_attr_final,
                iterations=self.wl_iters,
                include_initial_labels=False,
            )

        return self._labels_from_nx_hash_dict(g, node_hash_dict)

    def _nx_wl_hashes(self):
        """Return networkx WL-hash function if available, else None."""
        try:
            from networkx.algorithms.graph_hashing import (
                weisfeiler_lehman_subgraph_hashes,
            )
        except Exception:
            return None
        return weisfeiler_lehman_subgraph_hashes

    def _resolve_wl_attr_args(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Decide which node_attr and edge_attr keys to use for networkx WL.

        Returns
        -------
        (node_attr_arg, edge_attr_arg)
            These may be:
            - a real attribute key,
            - the special sentinel "__TEMP__" meaning "needs temp injection",
            - or None.
        """
        node_attr_arg: Optional[str]
        edge_attr_arg: Optional[str]

        # Node attributes resolution
        if self.node_attrs and len(self.node_attrs) > 1:
            node_attr_arg = "__TEMP_NODE__"
        elif self.node_attrs:
            node_attr_arg = self.node_attrs[0]
        elif self.element_key is not None:
            node_attr_arg = self.element_key
        else:
            node_attr_arg = None

        # Edge attributes resolution
        if self.edge_attrs and len(self.edge_attrs) > 1:
            edge_attr_arg = "__TEMP_EDGE__"
        elif self.edge_attrs:
            edge_attr_arg = self.edge_attrs[0]
        else:
            edge_attr_arg = None

        return node_attr_arg, edge_attr_arg

    @contextmanager
    def _inject_temp_attrs(
        self,
        g: nx.Graph,
        *,
        node_attr_arg: Optional[str],
        edge_attr_arg: Optional[str],
    ) -> Iterator[Tuple[Optional[str], Optional[str]]]:
        """
        Context manager that injects temporary combined attrs if needed.

        If node_attr_arg is "__TEMP_NODE__", we create "__wl_node_temp__".
        If edge_attr_arg is "__TEMP_EDGE__", we create "__wl_edge_temp__".

        Yields
        ------
        (node_attr_final, edge_attr_final)
            The actual attribute names to pass into networkx WL.
        """
        node_temp_key: Optional[str] = None
        edge_temp_key: Optional[str] = None

        node_attr_final = node_attr_arg
        edge_attr_final = edge_attr_arg

        try:
            # Inject combined node attribute
            if node_attr_arg == "__TEMP_NODE__":
                node_temp_key = "__wl_node_temp__"
                keys = self.node_attrs
                for n, data in g.nodes(data=True):
                    data[node_temp_key] = str(tuple(data.get(k) for k in keys))
                node_attr_final = node_temp_key

            # Inject combined edge attribute
            if edge_attr_arg == "__TEMP_EDGE__":
                edge_temp_key = "__wl_edge_temp__"
                keys = self.edge_attrs
                for u, v, data in g.edges(data=True):
                    data[edge_temp_key] = str(tuple(data.get(k) for k in keys))
                edge_attr_final = edge_temp_key

            yield (node_attr_final, edge_attr_final)

        finally:
            # Cleanup injected attrs
            if node_temp_key is not None:
                for _, data in g.nodes(data=True):
                    data.pop(node_temp_key, None)
            if edge_temp_key is not None:
                for _, _, data in g.edges(data=True):
                    data.pop(edge_temp_key, None)

    def _labels_from_nx_hash_dict(
        self,
        g: nx.Graph,
        node_hash_dict: Dict[Any, List[str]],
    ) -> List[str]:
        """
        Convert networkx WL hash dict to labels in g.nodes() order.
        """
        last_idx = max(0, self.wl_iters - 1)
        labels: List[str] = []

        for n in g.nodes():
            hashes = node_hash_dict.get(n, [])
            if not hashes:
                labels.append(str(g.degree[n]))
                continue

            # Prefer the requested iteration index when available
            if last_idx < len(hashes):
                labels.append(hashes[last_idx])
            else:
                labels.append(hashes[-1])

        return labels

    # -------- Local WL remains unchanged --------
    def _local_wl_node_labels(self, g: nx.Graph, iters: int = 1) -> List[str]:
        """
        Canonical local WL refinement (deterministic token assignment).
        Returns labels in g.nodes() order.
        """
        base = self._base_labels(g)
        curr: Dict[Any, str] = {n: base[idx] for idx, n in enumerate(g.nodes())}
        num_iter = max(0, int(iters))
        edge_keys = self.edge_attrs

        for _ in range(num_iter):
            struct: Dict[Any, str] = {}
            for n in g.nodes():
                neigh = []
                for m in g.neighbors(n):
                    token = curr[m]
                    if edge_keys:
                        e_data = g.get_edge_data(n, m, default={})
                        e_vals = tuple(e_data.get(k) for k in edge_keys)
                        token = f"{token}|{e_vals}"
                    neigh.append(token)
                neigh.sort()
                struct[n] = f"{curr[n]}|{'-'.join(neigh)}"

            unique = sorted(set(struct.values()))
            token_map = {s: f"t{idx}" for idx, s in enumerate(unique)}
            curr = {n: token_map[struct[n]] for n in g.nodes()}

        return [curr[n] for n in g.nodes()]

    # -------- Other helpers unchanged --------
    def _edge_signature(self, g: nx.Graph) -> EdgeSig:
        """Build multiset of edge labels for graph g (stringified tuples)."""
        if not self.edge_attrs:
            return Counter()
        keys = self.edge_attrs
        labels = []
        for _, _, data in g.edges(data=True):
            vals = tuple(data.get(k) for k in keys)
            labels.append(str(vals))
        return Counter(labels)

    @staticmethod
    def _overlap_counters(c1: Counter, c2: Counter) -> float:
        """Return min-sum overlap normalized by min(total1, total2)."""
        t1 = sum(c1.values())
        t2 = sum(c2.values())
        if t1 == 0 or t2 == 0:
            return 0.0
        inter = sum(min(c1.get(k, 0), c2.get(k, 0)) for k in set(c1) | set(c2))
        return inter / min(t1, t2)

    @staticmethod
    def _size_similarity(n1: int, n2: int) -> float:
        """Return size similarity in [0,1]."""
        if n1 == 0 and n2 == 0:
            return 1.0
        if n1 == 0 or n2 == 0:
            return 0.0
        max_n = max(n1, n2)
        return max(0.0, 1.0 - abs(n1 - n2) / max_n)

    @staticmethod
    def _unique_label_overlap(c1: Counter, c2: Counter) -> float:
        u1 = set(c1.keys())
        u2 = set(c2.keys())
        if not u1 and not u2:
            return 1.0
        if not u1 or not u2:
            return 0.0
        inter = len(u1 & u2)
        denom = min(len(u1), len(u2))
        return inter / denom

    def __repr__(self) -> str:
        return (
            f"<WLSel fw={len(self._fw)} bw={len(self._bw)} "
            f"wl_iters={self.wl_iters} node_attrs={self.node_attrs} "
            f"edge_attrs={self.edge_attrs} min_score={self.min_score:.2f}>"
        )
