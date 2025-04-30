"""mtg_groupoid.py — Mechanistic Transition Graph (MTG)
======================================================

This module implements a class for fusing two molecular graphs via a groupoid-based
edge-composition rule.  Given two NetworkX graphs (G1, G2) and a partial
node-to-node mapping from G1 → G2, the `MTG` class builds a merged product graph
G3 as follows:

1. **Node fusion:** Nodes listed in the mapping are identified (intersection);
   all other nodes are preserved uniquely.  Produces `product_nodes`, plus
   `map1: G1→G3` and `map2: G2→G3` dictionaries and the `intersection_ids` list.
2. **Edge insertion & composition:**
   - Insert edges from G1 (first pass) with no merging.
   - Insert edges from G2, then apply pair-groupoid composition on edges whose
     endpoints lie in `intersection_ids`, using the rule:
       ```
       (u→v, order=(a1,a2)) + (u→v, order=(b1,b2)) = (u→v, order=(a1,b2))
       ```
     if `a2 == b1`.
   - Deduplicate edges at each step to ensure uniqueness.
3. **Visualization:** The fused graph can be extracted as a NetworkX `Graph` or
   `DiGraph` with attributes `order` and computed `standard_order = a1 - a2`.

Public API
----------

- `MTG(G1, G2, mapping)`: construct fused graph state.
- `mtg.get_nodes() -> List[Tuple[int, Dict]]`: list of fused nodes and attrs.
- `mtg.get_edges() -> List[Tuple[int,int,Dict]]`: list of fused edges and attrs.
- `mtg.get_map1()`, `mtg.get_map2()`: access node-mapping dicts.
- `mtg.get_graph(directed: bool=False) -> nx.Graph`: export as NetworkX graph.
- `MTG.help()`, `MTG.__doc__`, `help(MTG)`: built-in documentation.

Dependencies
------------
- Python 3.10+
- `networkx`

"""

from __future__ import annotations

import networkx as nx
from collections import defaultdict
from typing import (
    Dict,
    List,
    Tuple,
    Any,
    Set,
    Optional,
    Union,
)

# Type Aliases
Label = Tuple[int, int]
Node = Tuple[int, Dict[str, Any]]
Edge = Tuple[int, int, Dict[str, Any]]


class MTG:
    """
    A Molecular Topology Graph for fusing two graphs via groupoid edge-composition.

    Attributes
    ----------
    product_nodes : List[Node]
        Fused node list in G3, as (node_id, attrs).
    product_edges : List[Edge]
        Fused edge list in G3, as (u,v,attrs).
    map1 : Dict[int,int]
        Mapping from G1 node IDs → G3 node IDs.
    map2 : Dict[int,int]
        Mapping from G2 node IDs → G3 node IDs.
    intersection_ids : List[int]
        G3 node IDs that were merged (in both G1 and G2).

    Methods
    -------
    get_nodes() -> List[Node]
        Return fused node list.
    get_edges() -> List[Edge]
        Return fused edge list.
    get_map1() -> Dict[int,int]
        Return map1.
    get_map2() -> Dict[int,int]
        Return map2.
    get_graph(directed: bool=False) -> Union[nx.Graph, nx.DiGraph]
        Export fused graph as NetworkX Graph or DiGraph.
    help() -> None
        Print module and class documentation.
    """

    def __init__(
        self,
        G1: Union[nx.Graph, nx.DiGraph],
        G2: Union[nx.Graph, nx.DiGraph],
        mapping: Dict[int, int],
    ) -> None:
        """
        Initialize MTG by fusing G1 and G2 according to `mapping`.

        Parameters
        ----------
        G1 : networkx.Graph or DiGraph
            First input graph with node and edge attrs (`order`).
        G2 : networkx.Graph or DiGraph
            Second input graph.
        mapping : dict
            Partial mapping from G1 node IDs → G2 node IDs to identify.
        """
        # Store raw data
        self.G1_nodes = list(G1.nodes(data=True))
        self.G2_nodes = list(G2.nodes(data=True))
        self.G1_edges = [(u, v, data) for u, v, data in G1.edges(data=True)]
        self.G2_edges = [(u, v, data) for u, v, data in G2.edges(data=True)]

        # Fuse nodes
        (self.product_nodes, self.map1, self.map2, self.intersection_ids) = (
            self.build_product_graph_nodes(self.G1_nodes, self.G2_nodes, mapping)
        )

        # Fuse edges
        # Step 1: insert G1 edges (no merging)
        self.product_edges = self.merge_edges_groupoid(
            self.G1_edges, [], self.map1, merged_idx=set(self.intersection_ids)
        )
        # Step 2: insert G2 edges with groupoid merging on intersection
        self.product_edges = self.merge_edges_groupoid(
            self.G2_edges,
            self.product_edges,
            self.map2,
            merged_idx=set(self.intersection_ids),
        )

    @staticmethod
    def build_product_graph_nodes(
        graph1_nodes: List[Node], graph2_nodes: List[Node], match: Dict[int, int]
    ) -> Tuple[List[Node], Dict[int, int], Dict[int, int], List[int]]:
        """
        Construct fused node list and mapping dicts.

        Returns
        -------
        merged_nodes : List[(id, attrs)]
        map1         : G1→G3 mapping
        map2         : G2→G3 mapping
        intersection : G3 IDs of merged nodes
        """
        merged: Dict[int, Dict[str, Any]] = {}
        map1: Dict[int, int] = {}
        map2: Dict[int, int] = {}
        used: Set[int] = set()

        # Copy G1
        for v1, attr in graph1_nodes:
            merged[v1] = attr.copy()
            map1[v1] = v1
            used.add(v1)

        inv_match = {v2: v1 for v1, v2 in match.items()}
        intersection: List[int] = []

        # Merge or add G2
        for v2, attr in graph2_nodes:
            if v2 in inv_match:
                tgt = inv_match[v2]
                map2[v2] = tgt
                intersection.append(tgt)
            else:
                nid = v2 if v2 not in used else max(used) + 1
                merged[nid] = attr.copy()
                map2[v2] = nid
                used.add(nid)

        nodes: List[Node] = [(nid, merged[nid]) for nid in sorted(merged)]
        return nodes, map1, map2, intersection

    @staticmethod
    def merge_edges_groupoid(
        edges_new: List[Edge],
        edges_existing: List[Edge],
        map_new_to_prod: Dict[int, int],
        merged_idx: Optional[Set[int]] = None,
    ) -> List[Edge]:
        """
        Insert and optionally merge edges into the product edge list.

        Parameters
        ----------
        edges_new       : New edges to insert (u,v,data) in original G IDs
        edges_existing  : Existing product edges (u,v,data) in fused IDs
        map_new_to_prod : Mapping of new-edge node IDs → fused IDs
        merged_idx      : Nodes for which to apply groupoid merging

        Returns
        -------
        List of deduplicated fused edges
        """
        # Remap new edges into fused space
        remapped: List[Edge] = []
        for u, v, attrs in edges_new:
            u3 = map_new_to_prod.get(u, u)
            v3 = map_new_to_prod.get(v, v)
            remapped.append((u3, v3, attrs.copy()))

        combined: List[Edge] = edges_existing + remapped
        if merged_idx is None:
            return MTG._dedupe_edges(combined)

        # Group orders by key
        orders: Dict[Tuple[int, int], List[Label]] = defaultdict(list)
        for u, v, attrs in combined:
            orders[(u, v)].append(tuple(attrs["order"]))

        fused: List[Edge] = []
        for (u, v), lst in orders.items():
            if u in merged_idx and v in merged_idx and len(lst) > 1:
                # assume two lists: first G1, then G2
                a1, a2 = lst[0]
                for b1, b2 in lst[1:]:
                    if a2 == b1:
                        fused.append((u, v, {"order": (a1, b2)}))
            else:
                for bef, aft in lst:
                    fused.append((u, v, {"order": (bef, aft)}))

        return MTG._dedupe_edges(fused)

    @staticmethod
    def _dedupe_edges(edges: List[Edge]) -> List[Edge]:
        seen: Set[Tuple[int, int, Tuple[int, int]]] = set()
        out: List[Edge] = []
        for u, v, attrs in edges:
            key = (u, v, tuple(attrs["order"]))
            if key not in seen:
                seen.add(key)
                out.append((u, v, attrs.copy()))
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_nodes(self) -> List[Node]:
        """Return fused node list."""
        return self.product_nodes

    def get_edges(self) -> List[Edge]:
        """Return fused edge list."""
        return self.product_edges

    def get_map1(self) -> Dict[int, int]:
        """Return G1→G3 node mapping."""
        return self.map1

    def get_map2(self) -> Dict[int, int]:
        """Return G2→G3 node mapping."""
        return self.map2

    def get_graph(self, directed: bool = False) -> Union[nx.Graph, nx.DiGraph]:
        """
        Export the fused product as a NetworkX graph.

        Parameters
        ----------
        directed : bool
            If True, return a DiGraph; otherwise a simple Graph.

        Returns
        -------
        G : nx.Graph or nx.DiGraph
            Nodes and edges carry `order` and `standard_order` attributes.
        """
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(self.product_nodes)
        for u, v, attrs in self.product_edges:
            order = attrs.get("order", (None, None))
            attrs["standard_order"] = None if None in order else order[0] - order[1]
            G.add_edge(u, v, **attrs)
        return G

    def help(self) -> None:
        """Print this class's docstring and methods."""
        print(self.__doc__)
        for name in dir(self):
            if not name.startswith("_"):
                print(name)

    def __repr__(self) -> str:
        """Compact summary: MTG(|V|,|E|)."""
        return f"MTG(|V|={len(self.product_nodes)}, |E|={len(self.product_edges)})"


# EOF
