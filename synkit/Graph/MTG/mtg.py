import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Union

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
NodeID = int
Order = Tuple[float, float]
Node = Tuple[NodeID, Dict[str, Any]]
Edge = Tuple[NodeID, NodeID, Dict[str, Any]]

__all__ = ["MTG"]


class MTG:
    """Fuse two molecular graphs via a pair‑groupoid edge‑composition rule.

    Parameters
    ----------
    G1, G2
        Input :class:`networkx.Graph` (or *DiGraph*) objects.  Nodes must carry an
        ``"element"`` attribute; edges carry an ``"order"`` 2‑tuple *(x, y)*.
    mapping
        A partial node map **G1 → G2** indicating which atoms are chemically
        identical (intersection).  Keys are node IDs in *G1*, values in *G2*.

    Notes
    -----
    1. ``intersection_ids`` are created where mapping ``G1[i] → G2[j]``.
    2. Edges are inserted in two passes:
       * *Pass 1* – all edges from *G1* are copied unchanged.
       * *Pass 2* – edges from *G2* are remapped; when both endpoints are in
         ``intersection_ids`` **and** their bond orders satisfy the *pair‐
         groupoid* condition

         ``(a₁, a₂)  +  (b₁, b₂)   with   a₂ == b₁   →   (a₁, b₂)``,

         the edges are *composed* instead of duplicated.

    Examples
    --------
    >>> mtg = MTG(G1, G2, {1: 3, 4: 6, 5: 1})
    >>> fused = mtg.get_graph()
    >>> fused.nodes(data=True)
    ...
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        G1: Union[nx.Graph, nx.DiGraph],
        G2: Union[nx.Graph, nx.DiGraph],
        mapping: Dict[NodeID, NodeID],
    ) -> None:
        # Store originals
        self.G1 = G1
        self.G2 = G2
        self.mapping12 = mapping  # G1 → G2

        # ---- 1. Build fused node set ---------------------------------
        (
            self.product_nodes,  # list[(id, attrs)]  in fused graph
            self.map1,  # G1 id → fused id
            self.map2,  # G2 id → fused id
            self.intersection_ids,  # list[fused id]
        ) = self._fuse_nodes()

        # ---- 2. Fuse edges with groupoid rule ------------------------
        fused_edges_step1 = self._insert_edges_from(self.G1.edges(data=True), self.map1)
        self.product_edges = self._insert_edges_from(
            self.G2.edges(data=True), self.map2, existing=fused_edges_step1
        )

    # ------------------------------------------------------------------
    #  Node fusion
    # ------------------------------------------------------------------
    def _fuse_nodes(self):
        merged: Dict[NodeID, Dict[str, Any]] = {}
        map1: Dict[NodeID, NodeID] = {}
        map2: Dict[NodeID, NodeID] = {}
        used: Set[NodeID] = set()

        # --- copy G1 directly into fused graph ------------------------
        for v, attrs in self.G1.nodes(data=True):
            merged[v] = attrs.copy()
            map1[v] = v
            used.add(v)

        # inverse mapping: G2 node → G1 node it merges to
        inv_map = {g2: g1 for g1, g2 in self.mapping12.items()}
        intersection: List[NodeID] = []

        # --- process G2 nodes -----------------------------------------
        next_id = max(used) + 1 if used else 0
        for v, attrs in self.G2.nodes(data=True):
            if v in inv_map:  # merged node
                tgt = inv_map[v]
                map2[v] = tgt
                intersection.append(tgt)
            else:  # unique node from G2
                while next_id in used:
                    next_id += 1
                merged[next_id] = attrs.copy()
                map2[v] = next_id
                used.add(next_id)
                next_id += 1

        nodes_sorted = sorted(merged.items())  # list[(id, attrs)]
        return nodes_sorted, map1, map2, intersection

    # ------------------------------------------------------------------
    #  Edge insertion & groupoid composition
    # ------------------------------------------------------------------
    def _insert_edges_from(
        self, edge_iter, node_map: Dict[NodeID, NodeID], existing: List[Edge] = None
    ) -> List[Edge]:
        """Insert edges into *existing* applying the groupoid rule when possible."""
        existing = [] if existing is None else existing.copy()

        # Remap and append new edges
        for u, v, attrs in edge_iter:
            u3 = node_map[u]
            v3 = node_map[v]
            existing.append((u3, v3, attrs.copy()))

        # Canonicalize keys for undirected graphs
        def key(u, v):
            return (u, v) if isinstance(self.G1, nx.DiGraph) else tuple(sorted((u, v)))

        # Group edges by (u,v)
        buckets: Dict[Tuple[NodeID, NodeID], List[Order]] = defaultdict(list)
        bucket_src: Dict[Tuple[NodeID, NodeID], List[str]] = defaultdict(list)
        for idx, (u, v, attrs) in enumerate(existing):
            buckets[key(u, v)].append(tuple(attrs["order"]))
            bucket_src[key(u, v)].append("G1" if idx < len(self.G1.edges) else "G2")

        fused_edges: List[Edge] = []
        for (u, v), orders in buckets.items():
            # src = bucket_src[(u, v)]
            if (
                u in self.intersection_ids
                and v in self.intersection_ids
                and len(orders) >= 2
            ):
                # Attempt pair‑wise composition between G1 (first) and any G2 edge
                made_composite = False
                for idx2, ord2 in enumerate(orders[1:], start=1):
                    a1, a2 = orders[0]
                    b1, b2 = ord2
                    if a2 == b1:
                        fused_edges.append((u, v, {"order": (a1, b2)}))
                        made_composite = True
                        break
                if not made_composite:
                    # fall back to *all* distinct orders
                    for ord_ in orders:
                        fused_edges.append((u, v, {"order": ord_}))
            else:
                for ord_ in orders:
                    fused_edges.append((u, v, {"order": ord_}))

        return self._dedupe_edges(fused_edges)

    # ------------------------------------------------------------------
    @staticmethod
    def _dedupe_edges(edges: List[Edge]) -> List[Edge]:
        seen: Set[Tuple[int, int, Order]] = set()
        out: List[Edge] = []
        for u, v, attrs in edges:
            key = (min(u, v), max(u, v), tuple(attrs["order"]))
            if key not in seen:
                seen.add(key)
                out.append((u, v, attrs))
        return out

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------
    def get_nodes(self) -> List[Node]:
        """List of `(id, attrs)` for fused graph."""
        return self.product_nodes

    def get_edges(self) -> List[Edge]:
        """List of `(u, v, attrs)` for fused graph."""
        return self.product_edges

    def get_map1(self) -> Dict[NodeID, NodeID]:
        return self.map1

    def get_map2(self) -> Dict[NodeID, NodeID]:
        return self.map2

    def get_graph(self, *, directed: bool = False):
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(self.product_nodes)
        for u, v, attrs in self.product_edges:
            o = attrs["order"]
            attrs["standard_order"] = o[0] - o[1] if None not in o else None
            G.add_edge(u, v, **attrs)
        return G

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"MTG(|V|={len(self.product_nodes)}, |E|={len(self.product_edges)})"
