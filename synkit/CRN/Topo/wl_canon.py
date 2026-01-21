from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import hashlib

import networkx as nx

from ..Hypergraph.hypergraph import CRNHyperGraph
from ..Hypergraph.backend import _CRNGraphBackend


# -------------------------------------------------------------------------
# Approx canon: Weisfeiler–Lehman (WL) color refinement (fast, approximate)
# -------------------------------------------------------------------------


def _blake_str(x: str, *, digest_size: int = 16) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(x.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _freeze(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(_freeze(v) for v in x)
    if isinstance(x, dict):
        return tuple((k, _freeze(v)) for k, v in sorted(x.items()))
    if isinstance(x, set):
        return tuple(sorted(_freeze(v) for v in x))
    return x


@dataclass(frozen=True)
class WLResult:
    G_can: nx.DiGraph
    colors: Dict[Any, str]  # node -> final WL color
    orbits: List[Set[Any]]  # approximate orbits (same color)
    iters_run: int
    stabilized: bool
    color_hist: Dict[str, int]  # color -> count
    automorphism_count: Optional[int]  # approximate (optional)


class WLCanonicalizer(_CRNGraphBackend):
    """
    Fast *approximate* canonicalization + orbit-like partition using
    Weisfeiler–Lehman (1-WL) refinement on the CRN graph view.

    What you get:
      - `orbits()` returns sets of nodes that are *WL-indistinguishable*
        (same final color). This is an *approximation* of true automorphism orbits.
      - `graph()` returns a deterministic relabeling 1..N based on (color, tie-break).

    `automorphism_count` in `summary()` is **approximate**:
      - If `estimate_automorphisms=True`, we return ∏_c (|cell_c|!) over WL color cells.
        This is an upper bound in many cases and can be a very loose proxy.
      - Otherwise it is None.

    This will NOT enumerate automorphisms / exact orbits, but it scales to graphs
    where exact IR search explodes.
    """

    def __init__(
        self,
        hg: CRNHyperGraph,
        *,
        include_rule: bool = False,
        node_attr_keys: Iterable[str] = ("kind",),
        edge_attr_keys: Iterable[str] = ("role", "stoich"),
        integer_ids: bool = False,
        include_stoich: bool = True,
        n_iter: int = 20,
        digest_size: int = 16,
        include_in_neighbors: bool = True,
        include_out_neighbors: bool = True,
        estimate_automorphisms: bool = True,
        automorphism_cap: int = 10**18,
    ) -> None:
        super().__init__(
            hg,
            include_rule=include_rule,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
        )
        self.node_attr_keys: Tuple[str, ...] = tuple(node_attr_keys)
        self.edge_attr_keys: Tuple[str, ...] = tuple(edge_attr_keys)
        self.n_iter = int(n_iter)
        self.digest_size = int(digest_size)
        self.include_in_neighbors = bool(include_in_neighbors)
        self.include_out_neighbors = bool(include_out_neighbors)
        self.estimate_automorphisms = bool(estimate_automorphisms)
        self.automorphism_cap = int(automorphism_cap)

        self._last: Optional[WLResult] = None
        self._last_key: Optional[Tuple[int, int, bool, bool, bool, int]] = None

    def __repr__(self) -> str:
        return (
            f"WLCanonicalizer(include_rule={self.include_rule}, "
            f"node_attr_keys={self.node_attr_keys}, edge_attr_keys={self.edge_attr_keys}, "
            f"n_iter={self.n_iter}, graph_type={getattr(self, '_graph_type', None)})"
        )

    # ------------------------- WL core -------------------------

    def _node_seed(self, G: nx.DiGraph, v: Any) -> str:
        attrs = tuple(_freeze(G.nodes[v].get(k, None)) for k in self.node_attr_keys)
        if G.is_directed():
            deg = (G.in_degree(v), G.out_degree(v))
        else:
            d = G.degree(v)
            deg = (d, d)
        return _blake_str(f"N|{attrs}|{deg}", digest_size=self.digest_size)

    def _edge_sig(self, attrs: Dict[str, Any]) -> str:
        t = tuple(_freeze(attrs.get(k, None)) for k in self.edge_attr_keys)
        return _blake_str(f"E|{t}", digest_size=self.digest_size)

    @staticmethod
    def _fact_cap(n: int, cap: int) -> int:
        """Compute n! but stop growing beyond cap (to avoid huge ints)."""
        out = 1
        for k in range(2, n + 1):
            out *= k
            if out >= cap:
                return cap
        return out

    def _estimate_aut_count(self, cell_sizes: List[int]) -> int:
        """
        Very rough proxy: product of factorials of WL color cell sizes.
        Often an over-estimate; capped at `automorphism_cap`.
        """
        out = 1
        for s in cell_sizes:
            out *= self._fact_cap(s, self.automorphism_cap)
            if out >= self.automorphism_cap:
                return self.automorphism_cap
        return out

    def _wl_refine(self, G: nx.DiGraph) -> WLResult:
        key = (
            self.n_iter,
            self.digest_size,
            self.include_in_neighbors,
            self.include_out_neighbors,
            self.estimate_automorphisms,
            self.automorphism_cap,
        )
        if self._last is not None and self._last_key == key:
            return self._last

        colors: Dict[Any, str] = {v: self._node_seed(G, v) for v in G.nodes()}

        stabilized = False
        iters_run = 0

        for it in range(self.n_iter):
            iters_run = it + 1
            new_colors: Dict[Any, str] = {}

            for v in G.nodes():
                parts: List[str] = [colors[v]]

                if self.include_in_neighbors:
                    in_items: List[str] = []
                    if G.is_directed():
                        for u in G.predecessors(v):
                            es = self._edge_sig(G[u][v])
                            in_items.append(f"{colors[u]}#{es}")
                    else:
                        for u in G.neighbors(v):
                            es = self._edge_sig(G[u][v]) if G.has_edge(u, v) else ""
                            in_items.append(f"{colors[u]}#{es}")
                    in_items.sort()
                    parts.append("IN[" + "|".join(in_items) + "]")

                if self.include_out_neighbors:
                    out_items: List[str] = []
                    if G.is_directed():
                        for u in G.successors(v):
                            es = self._edge_sig(G[v][u])
                            out_items.append(f"{colors[u]}#{es}")
                    else:
                        for u in G.neighbors(v):
                            es = self._edge_sig(G[v][u]) if G.has_edge(v, u) else ""
                            out_items.append(f"{colors[u]}#{es}")
                    out_items.sort()
                    parts.append("OUT[" + "|".join(out_items) + "]")

                new_colors[v] = _blake_str(
                    "||".join(parts), digest_size=self.digest_size
                )

            if all(new_colors[v] == colors[v] for v in G.nodes()):
                stabilized = True
                colors = new_colors
                break

            colors = new_colors

        # buckets = WL color classes
        buckets: Dict[str, List[Any]] = {}
        for v, c in colors.items():
            buckets.setdefault(c, []).append(v)

        orbits: List[Set[Any]] = [
            set(sorted(nodes))
            for _, nodes in sorted(buckets.items(), key=lambda kv: (kv[0], len(kv[1])))
        ]

        # deterministic canonical relabeling
        order = sorted(G.nodes(), key=lambda v: (colors[v], str(v)))
        mapping = {v: i + 1 for i, v in enumerate(order)}
        G_can = nx.relabel_nodes(G, mapping, copy=True)

        hist: Dict[str, int] = {c: len(nodes) for c, nodes in buckets.items()}

        aut_count: Optional[int] = None
        if self.estimate_automorphisms:
            cell_sizes = [len(nodes) for nodes in buckets.values()]
            aut_count = self._estimate_aut_count(cell_sizes)

        out = WLResult(
            G_can=G_can,
            colors=colors,
            orbits=orbits,
            iters_run=iters_run,
            stabilized=stabilized,
            color_hist=hist,
            automorphism_count=aut_count,
        )
        self._last = out
        self._last_key = key
        return out

    # ------------------------- public API -------------------------

    def orbits(self) -> List[Set[Any]]:
        r = self._wl_refine(self.G)
        return r.orbits

    def graph(self) -> nx.DiGraph:
        r = self._wl_refine(self.G)
        return r.G_can

    def summary(self) -> Dict[str, Any]:
        r = self._wl_refine(self.G)
        return {
            "canon_graph": r.G_can,
            "graph_type": self.graph_type,
            "node_attr_keys": self.node_attr_keys,
            "edge_attr_keys": self.edge_attr_keys,
            "n_iter": self.n_iter,
            "iters_run": r.iters_run,
            "stabilized": r.stabilized,
            "orbits": r.orbits,
            "colors": r.colors,
            "color_hist": r.color_hist,
            # requested key (approximate; see class docstring)
            "automorphism_count": r.automorphism_count,
        }


# -------------------------------------------------------------------------
# Functional convenience API
# -------------------------------------------------------------------------


def wl_canonical(
    hg: CRNHyperGraph,
    *,
    include_rule: bool = False,
    node_attr_keys: Iterable[str] = ("kind",),
    edge_attr_keys: Iterable[str] = ("role", "stoich"),
    integer_ids: bool = False,
    include_stoich: bool = True,
    n_iter: int = 20,
    digest_size: int = 16,
    include_in_neighbors: bool = True,
    include_out_neighbors: bool = True,
    estimate_automorphisms: bool = True,
    automorphism_cap: int = 10**18,
) -> WLCanonicalizer:
    canon = WLCanonicalizer(
        hg,
        include_rule=include_rule,
        node_attr_keys=node_attr_keys,
        edge_attr_keys=edge_attr_keys,
        integer_ids=integer_ids,
        include_stoich=include_stoich,
        n_iter=n_iter,
        digest_size=digest_size,
        include_in_neighbors=include_in_neighbors,
        include_out_neighbors=include_out_neighbors,
        estimate_automorphisms=estimate_automorphisms,
        automorphism_cap=automorphism_cap,
    )
    canon.summary()
    return canon
