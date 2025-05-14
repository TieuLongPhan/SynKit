import networkx as nx
from collections import defaultdict
from typing import Any, Dict, List, Set, Union, Optional

######################################################################
# TurboISO – Optimised
#   • label‑bucket index => O(1) candidate lookup
#   • lazy, *radius‑bounded* distance checks (no full APSP)
#   • adaptive root & order selection
#   • optional distance filter skips when candidate set already tiny
######################################################################


class TurboISO:
    """TurboISO with pragmatic speed‑ups for **many small queries**.

    1.  Pre‑indexes the host graph by *node‑signature* → nodes bucket.
    2.  Uses **lazy, radius‑bounded BFS** instead of a pre‑computed all‑pairs
        matrix (saving both startup time and memory).
    3.  Skips distance consistency if the total candidate pool is already
        smaller than a configurable threshold (defaults to 5 000).
    """

    def __init__(
        self,
        graph: nx.Graph,
        node_label: Union[str, List[str]] = "label",
        edge_label: Union[str, List[str], None] = None,
        distance_threshold: int = 5000,
    ) -> None:
        self.G = graph
        # --- normalise attribute selectors -----------------------------------
        self.node_label = (
            [node_label] if isinstance(node_label, str) else list(node_label)
        )
        self.edge_label = (
            []
            if edge_label is None
            else ([edge_label] if isinstance(edge_label, str) else list(edge_label))
        )
        # --- signature cache + label buckets ---------------------------------
        self._sig: Dict[Any, str] = {}
        self._label_buckets: Dict[str, Set[Any]] = defaultdict(set)
        for v in self.G.nodes:
            sig = self._node_signature(v)
            self._sig[v] = sig
            self._label_buckets[sig].add(v)
        # degree cache (tiny speed‑up)
        self._deg = dict(self.G.degree())
        # limit at which we skip expensive distance filtering
        self.distance_threshold = distance_threshold

    # ------------------------------------------------------------------ util
    def _node_signature(self, v: Any, G: Optional[nx.Graph] = None) -> str:
        """Return cached signature for host graph, else compute for Q."""
        if G is None or G is self.G:
            # host graph – cached
            if v in self._sig:
                return self._sig[v]
            # fall back (shouldn’t happen)
            G = self.G
        parts = [str(G.nodes[v].get(a, "#")) for a in self.node_label]
        return "|".join(parts)

    def _edge_signature(self, u: Any, v: Any, H: nx.Graph) -> str:
        if not self.edge_label:
            return ""
        return "|".join(str(H[u][v].get(a, "#")) for a in self.edge_label)

    # --------------------------------------------------------- init filter
    def _init_candidates(self, Q: nx.Graph) -> Dict[Any, Set[Any]]:
        C: Dict[Any, Set[Any]] = {}
        for q in Q.nodes:
            sig = self._node_signature(q, Q)
            candidates = self._label_buckets.get(sig, set()).copy()  # label filter O(1)
            # degree filter
            qdeg = Q.degree(q)
            C[q] = {v for v in candidates if self._deg[v] >= qdeg}
        return C

    # ------------------------------------------------ distance consistency
    def _within_dist(self, src: Any, dsts: Set[Any], limit: int) -> bool:
        """Check whether *any* dst in *dsts* lies within *limit* hops of src.
        Stops BFS early once found. Returns True/False."""
        if not dsts:
            return False
        if limit == float("inf"):
            return True  # unconstrained
        if src in dsts:
            return True if 0 <= limit else False
        if limit <= 0:
            return False
        # BFS frontier
        visited = {src}
        frontier = {src}
        depth = 0
        while frontier and depth < limit:
            depth += 1
            next_frontier = set()
            for u in frontier:
                for nbr in self.G.neighbors(u):
                    if nbr in visited:
                        continue
                    if nbr in dsts:
                        return True
                    visited.add(nbr)
                    next_frontier.add(nbr)
            frontier = next_frontier
        return False

    def _distance_filter(self, Q: nx.Graph, C: Dict[Any, Set[Any]]) -> None:
        qdist = dict(nx.all_pairs_shortest_path_length(Q))
        changed = True
        while changed:
            changed = False
            for u in Q.nodes:
                to_remove = set()
                for v_host in C[u]:
                    ok = True
                    for x in Q.nodes:
                        if x == u:
                            continue
                        maxd = qdist[u].get(x, float("inf"))
                        if maxd == float("inf"):
                            continue  # disconnected
                        if not self._within_dist(v_host, C[x], maxd):
                            ok = False
                            break
                    if not ok:
                        to_remove.add(v_host)
                if to_remove:
                    C[u] -= to_remove
                    changed = True

    # ------------------------------------------------------------ search
    def search(
        self, Q: nx.Graph, prune: bool = False
    ) -> Union[List[Dict[Any, Any]], bool]:
        C = self._init_candidates(Q)
        # optional distance filter only if pool still large
        pool_size = sum(len(vs) for vs in C.values())
        if pool_size > self.distance_threshold and len(Q) > 1:
            self._distance_filter(Q, C)
        # choose root: fewest candidates / highest degree heuristics
        root = min(Q.nodes, key=lambda n: (len(C[n]), -Q.degree(n)))
        # compute query radius from root (for candidate region restriction)
        qdist_root = nx.single_source_shortest_path_length(Q, root)
        radius = max(qdist_root.values())

        # order query nodes: (candidates, degree)
        order = [root] + sorted(
            [n for n in Q.nodes if n != root], key=lambda n: (len(C[n]), -Q.degree(n))
        )
        mapping: Dict[Any, Any] = {}
        used: Set[Any] = set()
        results: List[Dict[Any, Any]] = []

        def backtrack(pos: int, region: Set[Any]) -> bool:
            if pos == len(order):
                results.append(mapping.copy())
                return prune
            q = order[pos]
            for h in C[q]:
                if h in used or h not in region:
                    continue
                # adjacency & edge label check
                good = True
                for qnbr in Q.neighbors(q):
                    if qnbr in mapping:
                        hnbr = mapping[qnbr]
                        if not self.G.has_edge(h, hnbr):
                            good = False
                            break
                        if self.edge_label and self._edge_signature(
                            q, qnbr, Q
                        ) != self._edge_signature(h, hnbr, self.G):
                            good = False
                            break
                if not good:
                    continue
                mapping[q] = h
                used.add(h)
                if backtrack(pos + 1, region):
                    return True
                used.remove(h)
                del mapping[q]
            return False

        # iterate over root candidates
        for hroot in C[root]:
            # build candidate region around hroot within radius
            region = set(
                nx.single_source_shortest_path_length(
                    self.G, hroot, cutoff=radius
                ).keys()
            )
            mapping[root] = hroot
            used.add(hroot)
            if backtrack(1, region):
                if prune:
                    return True
            used.remove(hroot)
            del mapping[root]
        return (len(results) > 0) if prune else results
