from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import time

import networkx as nx

from ..Hypergraph.hypergraph import CRNHyperGraph
from ..Hypergraph.backend import _CRNGraphBackend


# -------------------------------------------------------------------------
# Canon: Exact Nauty-style individualization + refinement (no heuristic pruning)
# -------------------------------------------------------------------------


class CRNCanonicalizer(_CRNGraphBackend):
    """
    Exact canonicalization and automorphism enumeration for a CRN via
    individualization-refinement (IR) search.

    This implementation is **exact**:
      - No branch-shape skipping
      - No prefix/label pruning that could drop valid automorphisms
      - Enumerates all permutations achieving the minimal canonical label
        (these correspond to automorphisms w.r.t. the labeling function)
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
    ) -> None:
        super().__init__(
            hg,
            include_rule=include_rule,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
        )
        self.node_attr_keys: Tuple[str, ...] = tuple(node_attr_keys)
        self.edge_attr_keys: Tuple[str, ...] = tuple(edge_attr_keys)

    def __repr__(self) -> str:
        return (
            f"CRNCanonicalizer(include_rule={self.include_rule}, "
            f"node_attr_keys={self.node_attr_keys}, "
            f"edge_attr_keys={self.edge_attr_keys}, "
            f"graph_type={getattr(self, '_graph_type', None)})"
        )

    # --- core helpers -------------------------------------------------------

    @staticmethod
    def _freeze(x: Any) -> Any:
        """Convert nested containers into hashable equivalents."""
        if isinstance(x, list):
            return tuple(CRNCanonicalizer._freeze(v) for v in x)
        if isinstance(x, dict):
            return frozenset(
                (k, CRNCanonicalizer._freeze(v)) for k, v in sorted(x.items())
            )
        if isinstance(x, set):
            return tuple(sorted(CRNCanonicalizer._freeze(v) for v in x))
        return x

    def _init_part(self, G: nx.Graph) -> List[List[Any]]:
        """Initial partition by node attributes."""
        if not self.node_attr_keys:
            return [sorted(G.nodes())]
        buckets: Dict[Tuple[Any, ...], List[Any]] = {}
        for v in G.nodes():
            key = tuple(
                self._freeze(G.nodes[v].get(a, None)) for a in self.node_attr_keys
            )
            buckets.setdefault(key, []).append(v)
        return [
            sorted(nodes) for _, nodes in sorted(buckets.items(), key=lambda kv: kv[0])
        ]

    def _sig(
        self,
        G: nx.DiGraph,
        v: Any,
        part: List[List[Any]],
    ) -> Tuple[Any, ...]:
        """
        Refinement signature for node v under current partition.

        Uses:
          - node attributes
          - (in_degree, out_degree)
          - neighbor counts per cell (undirected union for directed graphs)
          - multiset of outgoing edge attributes
        """
        node_attrs = tuple(
            self._freeze(G.nodes[v].get(a, None)) for a in self.node_attr_keys
        )

        if G.is_directed():
            degree = (G.in_degree(v), G.out_degree(v))
            nbrs = set(G.predecessors(v)) | set(G.successors(v))
        else:
            d = G.degree[v]
            degree = (d, d)
            nbrs = set(G.neighbors(v))

        counts: List[int] = []
        for cell in part:
            s = set(cell)
            counts.append(sum(1 for n in nbrs if n in s))
        counts_t = tuple(counts)

        edge_mult: List[Tuple[Any, ...]] = []
        for nbr in G.successors(v) if G.is_directed() else G.neighbors(v):
            attrs = G[v][nbr]
            vals: List[Any] = []
            for a in self.edge_attr_keys:
                val = attrs.get(a, None)
                if a == "order" and isinstance(val, tuple):
                    val = tuple(sorted(round(float(x), 3) for x in val))
                vals.append(self._freeze(val))
            edge_mult.append(tuple(vals))
        edge_mult_t = tuple(sorted(edge_mult))

        return (node_attrs, degree, counts_t, edge_mult_t)

    def _refine(self, G: nx.DiGraph, part: List[List[Any]]) -> List[List[Any]]:
        """Refine partition until stable (exact, deterministic)."""
        changed = True
        cache: Dict[Tuple[Any, int], Tuple[Any, ...]] = {}
        while changed:
            changed = False
            new_part: List[List[Any]] = []
            # note: cache depends on part, but we keep it simple & exact:
            # cache key includes an epoch id via id(part) to avoid reusing across parts.
            epoch = id(tuple(tuple(c) for c in part))
            for cell in part:
                if len(cell) <= 1:
                    new_part.append(cell)
                    continue
                sigs: Dict[Tuple[Any, ...], List[Any]] = {}
                for v in cell:
                    ck = (v, epoch)
                    if ck not in cache:
                        cache[ck] = self._sig(G, v, part)
                    s = cache[ck]
                    sigs.setdefault(s, []).append(v)

                if len(sigs) > 1:
                    changed = True
                    for s in sorted(sigs.keys()):
                        new_part.append(sorted(sigs[s]))
                else:
                    new_part.append(sorted(cell))
            part = new_part
        return part

    def _label(self, G: nx.DiGraph, perm: List[Any]) -> str:
        """
        Canonical label string for a full node permutation.
        Deterministic and exact (no approximations).
        """
        node_seg = "|".join(
            ":".join(
                str(self._freeze(G.nodes[v].get(a, ""))) for a in self.node_attr_keys
            )
            for v in perm
        )
        n = len(perm)
        edge_bits: List[str] = []
        for i in range(n):
            vi = perm[i]
            for j in range(n):
                if i == j:
                    continue
                vj = perm[j]
                if G.has_edge(vi, vj):
                    attrs = G[vi][vj]
                    frozen = tuple(
                        self._freeze(attrs.get(a, "")) for a in self.edge_attr_keys
                    )
                    edge_bits.append("1:" + ":".join(str(x) for x in frozen))
                else:
                    edge_bits.append("0:" + ":".join("" for _ in self.edge_attr_keys))
        edge_seg = "|".join(edge_bits)
        return node_seg + "||" + edge_seg

    def _search(
        self,
        G: nx.DiGraph,
        part: List[List[Any]],
        prefix: List[Any],
        best: Dict[str, Optional[str]],
        perms: List[List[Any]],
        *,
        depth: int,
        max_depth: Optional[int],
        start: float,
        timeout_sec: Optional[float],
    ) -> bool:
        """
        Exact recursive IR search for minimal canonical label.

        Returns True only if stopped early due to max_depth/timeout.
        """
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            return True
        if max_depth is not None and depth > max_depth:
            return True

        part = self._refine(G, part)

        # Fully discrete partition: finalize permutation and score it.
        if all(len(c) == 1 for c in part):
            perm = prefix + [v for c in part for v in c]
            lab = self._label(G, perm)
            if best["label"] is None or lab < best["label"]:
                best["label"], best["perm"] = lab, perm  # type: ignore[assignment]
                perms.clear()
                perms.append(perm)
            elif lab == best["label"]:
                perms.append(perm)
            return False

        # Choose a non-singleton cell (deterministic choice keeps exactness).
        idx = next(i for i, c in enumerate(part) if len(c) > 1)
        cell = sorted(part[idx])

        for v in cell:
            rest = [w for w in cell if w != v]
            new_part = (
                part[:idx] + [[v]] + ([sorted(rest)] if rest else []) + part[idx + 1 :]
            )
            pref = prefix + [v]

            if self._search(
                G,
                new_part,
                pref,
                best,
                perms,
                depth=depth + 1,
                max_depth=max_depth,
                start=start,
                timeout_sec=timeout_sec,
            ):
                return True
        return False

    @staticmethod
    def _orbits_from_perms(perms: List[List[Any]]) -> List[Set[Any]]:
        """Derive node orbits from a list of automorphism permutations."""
        if not perms:
            return []
        orbit_map: Dict[Any, int] = {}
        orbits: List[Set[Any]] = []

        def merge(i: int, j: int) -> None:
            if i == j:
                return
            o1 = orbits[i]
            o2 = orbits[j]
            if len(o1) < len(o2):
                i, j = j, i
                o1, o2 = o2, o1
            o1.update(o2)
            orbits[j] = set()
            for v in o2:
                orbit_map[v] = i

        first = perms[0]
        for idx, v in enumerate(first):
            orbit_map[v] = idx
            orbits.append({v})

        for p in perms[1:]:
            for idx, v in enumerate(p):
                merge(idx, orbit_map[v])

        return [o for o in orbits if o]

    @staticmethod
    def _maps_from_perms(
        ref: List[Any], perms: List[List[Any]]
    ) -> List[Dict[Any, Any]]:
        """Convert permutations into mapping dicts relative to a reference order."""
        maps: List[Dict[Any, Any]] = []
        n = len(ref)
        for p in perms:
            if len(p) != n:
                continue
            maps.append({ref[i]: p[i] for i in range(n)})
        return maps

    def _canon(
        self,
        *,
        max_depth: Optional[int],
        timeout_sec: Optional[float],
    ) -> Tuple[
        nx.DiGraph,
        List[Any],
        List[List[Any]],
        List[Set[Any]],
        List[Dict[Any, Any]],
        bool,
    ]:
        """Compute canonical graph, minimal permutations, orbits and mappings (exact)."""
        G = self.G
        best: Dict[str, Optional[str]] = {"label": None, "perm": None}
        perms: List[List[Any]] = []

        part = self._init_part(G)
        start = time.time()
        early = self._search(
            G,
            part,
            [],
            best,
            perms,
            depth=0,
            max_depth=max_depth,
            start=start,
            timeout_sec=timeout_sec,
        )

        perm = best.get("perm")  # type: ignore[assignment]
        if perm is None:
            raise RuntimeError(
                f"Canonical form not found; early stop (max_depth={max_depth}, timeout_sec={timeout_sec})"
            )

        mapping = {v: i + 1 for i, v in enumerate(perm)}
        G_can = nx.relabel_nodes(G, mapping, copy=True)

        orbits = self._orbits_from_perms(perms)
        maps = self._maps_from_perms(perm, perms)
        return G_can, perm, perms, orbits, maps, early

    # --- public API ---------------------------------------------------------

    def graph(
        self,
        *,
        max_depth: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> nx.DiGraph:
        """Return the canonical relabeled graph."""
        G_can, _, _, _, _, _ = self._canon(max_depth=max_depth, timeout_sec=timeout_sec)
        return G_can

    def has_nontrivial_automorphism(
        self,
        *,
        max_depth: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> bool:
        """True iff >1 minimal-label permutation exists."""
        _, _, perms, _, _, _ = self._canon(max_depth=max_depth, timeout_sec=timeout_sec)
        return len(perms) > 1

    def summary(
        self,
        *,
        max_depth: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run canonicalization and return a summary dictionary."""
        G_can, perm, perms, orbits, maps, early = self._canon(
            max_depth=max_depth,
            timeout_sec=timeout_sec,
        )
        return {
            "canon_graph": G_can,
            "graph_type": self.graph_type,
            "node_attr_keys": self.node_attr_keys,
            "edge_attr_keys": self.edge_attr_keys,
            "automorphism_count": len(perms),
            "sample_permutations": perms,
            "mappings": maps,
            "orbits": orbits,
            "early_stop": early,
            "canonical_perm": perm,
        }

    def orbits(
        self,
        *,
        max_depth: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> List[Set[Any]]:
        """Compute node orbits for the canonical form."""
        _, _, _, orbits, _, _ = self._canon(
            max_depth=max_depth,
            timeout_sec=timeout_sec,
        )
        return orbits


# -------------------------------------------------------------------------
# Functional convenience API
# -------------------------------------------------------------------------


def canonical(
    hg: CRNHyperGraph,
    *,
    include_rule: bool = False,
    node_attr_keys: Iterable[str] = ("kind",),
    edge_attr_keys: Iterable[str] = ("role", "stoich"),
    integer_ids: bool = False,
    include_stoich: bool = True,
    max_depth: Optional[int] = None,
    timeout_sec: Optional[float] = None,
) -> CRNCanonicalizer:
    """
    Run canonicalization and return a CRNCanonicalizer instance (exact).

    NOTE: returns the canonicalizer object (not the summary dict).
    """
    canon = CRNCanonicalizer(
        hg,
        include_rule=include_rule,
        node_attr_keys=node_attr_keys,
        edge_attr_keys=edge_attr_keys,
        integer_ids=integer_ids,
        include_stoich=include_stoich,
    )
    # compute once to validate / warm results (optional)
    canon.summary(max_depth=max_depth, timeout_sec=timeout_sec)
    return canon
