from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import time

import networkx as nx

from ..Hypergraph.hypergraph import CRNHyperGraph
from ..Hypergraph.backend import _CRNGraphBackend


# -------------------------------------------------------------------------
# Canon: Nauty-style canonicalization
# -------------------------------------------------------------------------


class CRNCanonicalizer(_CRNGraphBackend):
    """
    Nauty-style canonicalization and automorphism enumeration for a CRN.

    The underlying :class:`CRNHyperGraph` is first converted to either a
    bipartite species→reaction→species DiGraph (include_rule=True) or a
    collapsed species→species DiGraph (include_rule=False). A
    partition-refinement search then computes a canonical labeling and
    associated automorphisms.
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
        """
        Initialize a canonicalizer for a CRN.

        :param hg: Hypergraph to canonicalize.
        :type hg: CRNHyperGraph
        :param include_rule: If True, work on bipartite species→reaction→species
                             graph; if False, use collapsed species graph.
        :type include_rule: bool
        :param node_attr_keys: Node attribute keys used in partition refinement
                               and canonical label construction.
        :type node_attr_keys: Iterable[str]
        :param edge_attr_keys: Edge attribute keys included in canonical labels.
        :type edge_attr_keys: Iterable[str]
        :param integer_ids: If True, use integer node ids in the bipartite view.
        :type integer_ids: bool
        :param include_stoich: If True, include stoichiometry on bipartite edges.
        :type include_stoich: bool
        """
        super().__init__(
            hg,
            include_rule=include_rule,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
        )
        self.node_attr_keys: Tuple[str, ...] = tuple(node_attr_keys)
        self.edge_attr_keys: Tuple[str, ...] = tuple(edge_attr_keys)

    def __repr__(self) -> str:
        """
        :returns: String representation of the canonicalizer.
        :rtype: str
        """
        return (
            f"CRNCanonicalizer(include_rule={self.include_rule}, "
            f"node_attr_keys={self.node_attr_keys}, "
            f"edge_attr_keys={self.edge_attr_keys}, "
            f"graph_type={getattr(self, '_graph_type', None)})"
        )

    # --- core helpers -------------------------------------------------------

    @staticmethod
    def _freeze(x: Any) -> Any:
        """
        Convert nested containers into hashable equivalents.

        Lists become tuples; dicts become frozensets of sorted (key, value)
        pairs; other types are returned unchanged.

        :param x: Object to convert.
        :type x: Any
        :returns: Hashable representation of x.
        :rtype: Any
        """
        if isinstance(x, list):
            return tuple(CRNCanonicalizer._freeze(v) for v in x)
        if isinstance(x, dict):
            return frozenset(
                (k, CRNCanonicalizer._freeze(v)) for k, v in sorted(x.items())
            )
        return x

    def _init_part(self, G: nx.Graph) -> List[List[Any]]:
        """
        Build the initial node partition for refinement.

        Nodes are grouped by tuples of node_attr_keys. If no node_attr_keys
        are provided, all nodes start in a single cell.

        :param G: Graph whose nodes are to be partitioned.
        :type G: nx.Graph
        :returns: List of partition cells, each a list of nodes.
        :rtype: List[List[Any]]
        """
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
        Compute the refinement signature for a single node.

        The signature combines node attributes, degree information, neighbor
        counts per partition cell, and a multiset of selected edge attributes.

        :param G: Graph on which refinement is performed.
        :type G: nx.DiGraph
        :param v: Node whose signature is computed.
        :type v: Any
        :param part: Current node partition.
        :type part: List[List[Any]]
        :returns: Signature tuple used to split cells.
        :rtype: Tuple[Any, ...]
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
            counts.append(sum(1 for n in nbrs if n in cell))
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
        """
        Refine a partition until no further splits occur.

        Each non-singleton cell is split by grouping nodes with identical
        signatures from :meth:`_sig`. Refinement continues until stable.

        :param G: Graph to refine.
        :type G: nx.DiGraph
        :param part: Current node partition.
        :type part: List[List[Any]]
        :returns: Refined partition.
        :rtype: List[List[Any]]
        """
        changed = True
        cache: Dict[Any, Tuple[Any, ...]] = {}
        while changed:
            changed = False
            new_part: List[List[Any]] = []
            for cell in part:
                if len(cell) <= 1:
                    new_part.append(cell)
                    continue
                sigs: Dict[Tuple[Any, ...], List[Any]] = {}
                for v in cell:
                    if v not in cache:
                        cache[v] = self._sig(G, v, part)
                    s = cache[v]
                    sigs.setdefault(s, []).append(v)
                if len(sigs) > 1:
                    changed = True
                    for s in sorted(sigs.keys()):
                        new_part.append(sorted(sigs[s]))
                else:
                    new_part.append(cell)
            part = new_part
        return part

    def _label(self, G: nx.DiGraph, perm: List[Any]) -> str:
        """
        Build a canonical label string for a full node permutation.

        The label encodes node attributes in order, plus the full adjacency
        pattern and selected edge attributes for all ordered node pairs.

        :param G: Graph to label.
        :type G: nx.DiGraph
        :param perm: Full node permutation (ordering).
        :type perm: List[Any]
        :returns: Canonical label string for this permutation.
        :rtype: str
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

    def _partial_label(self, G: nx.DiGraph, prefix: List[Any]) -> str:
        """
        Build a partial label for a permutation prefix.

        A long sequence of ``'{'`` characters is appended so that partial
        labels compare lexicographically larger than any full label with
        the same prefix, enabling safe pruning.

        :param G: Graph to label.
        :type G: nx.DiGraph
        :param prefix: Partial node permutation.
        :type prefix: List[Any]
        :returns: Partial label string.
        :rtype: str
        """
        node_seg = "|".join(
            ":".join(
                str(self._freeze(G.nodes[v].get(a, ""))) for a in self.node_attr_keys
            )
            for v in prefix
        )
        return node_seg + "{" * 1000

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
        Recursive backtracking search for minimal canonical labels.

        :param G: Graph to canonicalize.
        :type G: nx.DiGraph
        :param part: Current node partition.
        :type part: List[List[Any]]
        :param prefix: Current permutation prefix.
        :type prefix: List[Any]
        :param best: Dict with keys ``"label"`` and ``"perm"`` for the best
                     full label and permutation seen so far.
        :type best: Dict[str, Optional[str]]
        :param perms: List to collect permutations that achieve the best label.
        :type perms: List[List[Any]]
        :param depth: Current recursion depth.
        :type depth: int
        :param max_depth: Optional maximum recursion depth; None for unlimited.
        :type max_depth: Optional[int]
        :param start: Start time (:func:`time.time`) for timeout accounting.
        :type start: float
        :param timeout_sec: Optional wall-clock timeout in seconds.
        :type timeout_sec: Optional[float]
        :returns: True if the search stopped early due to timeout/depth.
        :rtype: bool
        """
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            return True
        if max_depth is not None and depth > max_depth:
            return True

        part = self._refine(G, part)

        # Fully discrete partition: construct final permutation.
        if all(len(c) == 1 for c in part):
            perm = prefix + [v for c in part for v in c]
            lab = self._label(G, perm)
            if best["label"] is None or lab < best["label"]:
                best["label"], best["perm"] = lab, perm
                perms.clear()
                perms.append(perm)
            elif lab == best["label"]:
                perms.append(perm)
            return False

        # Choose first non-singleton cell for branching.
        idx = next(i for i, c in enumerate(part) if len(c) > 1)
        cell = part[idx]
        cell_sorted = sorted(cell)

        for v in cell_sorted:
            rest = [w for w in cell if w != v]
            new_part = (
                part[:idx] + [[v]] + ([sorted(rest)] if rest else []) + part[idx + 1 :]
            )
            pref = prefix + [v]
            p_lab = self._partial_label(G, pref)
            if best["label"] is not None and p_lab > best["label"]:
                continue
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
        """
        Derive node orbits from a list of automorphism permutations.

        :param perms: List of permutations (each a list of nodes).
        :type perms: List[List[Any]]
        :returns: List of disjoint sets of nodes, one per orbit.
        :rtype: List[Set[Any]]
        """
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
        """
        Convert permutations into mapping dicts relative to a reference order.

        :param ref: Reference permutation (canonical ordering of nodes).
        :type ref: List[Any]
        :param perms: List of permutations to convert.
        :type perms: List[List[Any]]
        :returns: List of mappings {ref_node -> perm_node}.
        :rtype: List[Dict[Any, Any]]
        """
        maps: List[Dict[Any, Any]] = []
        n = len(ref)
        for p in perms:
            if len(p) != n:
                continue
            m = {ref[i]: p[i] for i in range(n)}
            maps.append(m)
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
        """
        Compute canonical graph, minimal permutations, orbits and mappings.

        :param max_depth: Optional maximum recursion depth; None for unlimited.
        :type max_depth: Optional[int]
        :param timeout_sec: Optional wall-clock timeout in seconds.
        :type timeout_sec: Optional[float]
        :returns: Tuple (G_can, perm, perms, orbits, maps, early_stop) where
                  G_can is the canonical graph, perm is a representative
                  minimal permutation, perms is the list of all minimal
                  permutations, orbits are node orbits, maps are mapping
                  dicts, and early_stop indicates early termination.
        :rtype: Tuple[nx.DiGraph, List[Any], List[List[Any]], List[Set[Any]], List[Dict[Any, Any]], bool]
        :raises RuntimeError: If no canonical form is found (e.g. due to early stop).
        """
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

        perm = best["perm"]
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
        """
        Return the canonical relabeled graph.

        :param max_depth: Optional maximum recursion depth; None for unlimited.
        :type max_depth: Optional[int]
        :param timeout_sec: Optional wall-clock timeout in seconds.
        :type timeout_sec: Optional[float]
        :returns: Canonically relabeled DiGraph with nodes 1..N.
        :rtype: nx.DiGraph
        """
        G_can, _, _, _, _, _ = self._canon(max_depth=max_depth, timeout_sec=timeout_sec)
        return G_can

    def has_nontrivial_automorphism(
        self,
        *,
        max_depth: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> bool:
        """
        Test whether the canonical graph has nontrivial automorphisms.

        :param max_depth: Optional maximum recursion depth; None for unlimited.
        :type max_depth: Optional[int]
        :param timeout_sec: Optional wall-clock timeout in seconds.
        :type timeout_sec: Optional[float]
        :returns: True if more than one minimal-label permutation exists.
        :rtype: bool
        """
        _, _, perms, _, _, _ = self._canon(max_depth=max_depth, timeout_sec=timeout_sec)
        return len(perms) > 1

    def summary(
        self,
        *,
        max_depth: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run canonicalization and return a summary dictionary.

        :param max_depth: Optional maximum recursion depth; None for unlimited.
        :type max_depth: Optional[int]
        :param timeout_sec: Optional wall-clock timeout in seconds.
        :type timeout_sec: Optional[float]
        :returns: Summary with permutations, mappings, orbits and diagnostics.
        :rtype: Dict[str, Any]
        """
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
        """
        Compute node orbits for the canonical form.

        :param max_depth: Optional maximum recursion depth; None for unlimited.
        :type max_depth: Optional[int]
        :param timeout_sec: Optional wall-clock timeout in seconds.
        :type timeout_sec: Optional[float]
        :returns: List of disjoint node sets representing orbits.
        :rtype: List[Set[Any]]
        """
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
    Run canonicalization and return a CRNCanonicalizer instance.

    The canonicalization is performed once (via :meth:`summary`) to
    populate internal state; the returned object can then be queried.

    :param hg: Hypergraph to canonicalize.
    :type hg: CRNHyperGraph
    :param include_rule: If True, use bipartite species→reaction→species view;
                         if False, use collapsed species graph.
    :type include_rule: bool
    :param node_attr_keys: Node attributes used in partition refinement and
                           canonical labels.
    :type node_attr_keys: Iterable[str]
    :param edge_attr_keys: Edge attributes included in canonical labels.
    :type edge_attr_keys: Iterable[str]
    :param integer_ids: If True, use integer node ids in bipartite view.
    :type integer_ids: bool
    :param include_stoich: If True, include stoichiometry on bipartite edges.
    :type include_stoich: bool
    :param max_depth: Optional maximum recursion depth; None for unlimited.
    :type max_depth: Optional[int]
    :param timeout_sec: Optional wall-clock timeout in seconds.
    :type timeout_sec: Optional[float]
    :returns: Canonicalizer instance with populated internal state.
    :rtype: CRNCanonicalizer
    """
    canon = CRNCanonicalizer(
        hg,
        include_rule=include_rule,
        node_attr_keys=node_attr_keys,
        edge_attr_keys=edge_attr_keys,
        integer_ids=integer_ids,
        include_stoich=include_stoich,
    )
    summary = canon.summary(max_depth=max_depth, timeout_sec=timeout_sec)
    return summary
