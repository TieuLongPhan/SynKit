from __future__ import annotations

"""mcs_matcher.py – Maximum Connected Common Subgraph finder with customizable matching
-----------------------------------------------------------------------

Finds the maximum connected common subgraph (MCS) between graphs, supporting:

* **Exact-attribute matching** via NetworkX’s generic matchers (wildcards, defaults).
* **WL-1 pre-filter** and **parallel subset checks** from GraphMatcherEngine.
* **Fallback** to custom node/edge match functions for full fidelity.
"""

import os
import concurrent.futures as _fut
from typing import Any, Iterable, List, Optional

import networkx as nx
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    generic_node_match,
    generic_edge_match,
)

from .graph_matcher import GraphMatcherEngine

__all__ = ["MCSMatcher", "HeuristicMCS"]


def _connected_subsets(g: nx.Graph, k: int) -> Iterable[frozenset[int]]:
    """Yield connected subsets of size k."""
    if k == 1:
        yield from (frozenset([n]) for n in g)
        return
    nodes = sorted(g)
    adj = g.adj

    def extend(partial: tuple[int, ...], cand: set[int]):
        if len(partial) == k:
            yield frozenset(partial)
            return
        for n in list(cand):
            cand2 = cand & set(adj[n])
            yield from extend(partial + (n,), cand2)

    for i, v in enumerate(nodes):
        # fmt: off
        yield from extend((v,), set(nodes[i + 1:]) & set(adj[v]))
        # fmt: on


class MCSMatcher:
    """
    MCS between two graphs with optional custom matchers.

    Parameters
    ----------
    node_attrs : List[str]
        Node attribute keys for matching (wildcard support).
    node_attr_defaults : List[Any]
        Defaults for missing node attributes (must correspond to node_attrs).
    edge_attrs : List[str]
        Edge attribute keys for matching.
    edge_attr_default : Any
        Default for missing edge attribute values.
    wl1_filter : bool
        Use WL-1 pre-filter from GraphMatcherEngine.
    max_mappings : Optional[int]
        Cap on embeddings to enumerate.
    workers : Optional[int]
        Number of threads for subset checks (None = cpu_count).
    """

    def __init__(
        self,
        *,
        node_attrs: Optional[List[str]] = None,
        node_attr_defaults: Optional[List[Any]] = None,
        edge_attrs: Optional[List[str]] = None,
        edge_attr_default: Any = None,
        wl1_filter: bool = False,
        max_mappings: Optional[int] = 1,
        workers: Optional[int] = None,
    ) -> None:
        # Build generic node matcher if attrs provided
        if node_attrs and node_attr_defaults:
            # use operator.eq for strict equality matching (no wildcards)
            from operator import eq

            comps = [eq] * len(node_attrs)
            self.node_match_fn = generic_node_match(
                node_attrs, node_attr_defaults, comps
            )
        else:
            self.node_match_fn = None
        # Build generic edge matcher if attrs provided
        if edge_attrs and edge_attr_default is not None:
            from operator import eq

            # For single-edge attribute, use generic_edge_match
            if len(edge_attrs) == 1:
                self.edge_match_fn = generic_edge_match(
                    edge_attrs[0], edge_attr_default, eq
                )
            else:
                # multiple edge attrs: build a composite matcher
                def edge_match_fn(eh, ep, _attrs=edge_attrs):
                    return all(
                        eh.get(k, edge_attr_default) == ep.get(k, edge_attr_default)
                        for k in _attrs
                    )

                self.edge_match_fn = edge_match_fn
        else:
            self.edge_match_fn = None

        # Structural engine ignores custom matchers
        self.engine = GraphMatcherEngine(
            backend="nx",
            node_attrs=node_attrs or [],
            edge_attrs=edge_attrs or [],
            wl1_filter=wl1_filter,
            max_mappings=max_mappings,
        )
        self.workers = workers or int(os.cpu_count() or 1)

    def _is_sub_iso(self, host: nx.Graph, sub: nx.Graph) -> bool:
        # Prefer custom matchers if provided
        if self.node_match_fn or self.edge_match_fn:
            gm = GraphMatcher(
                host,
                sub,
                node_match=self.node_match_fn,
                edge_match=self.edge_match_fn,
            )
            if (
                host.number_of_nodes() == sub.number_of_nodes()
                and host.number_of_edges() == sub.number_of_edges()
            ):
                return gm.is_isomorphic()
            return gm.subgraph_is_isomorphic()
        # Structural: allow WL pre-filter for speed
        return bool(self.engine.get_mappings(host, sub))

    def maximum_connected_common_subgraph(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
    ) -> nx.Graph:
        # ensure smaller first
        small, large = (
            (g1, g2) if g1.number_of_nodes() <= g2.number_of_nodes() else (g2, g1)
        )
        # quick full match
        if self._is_sub_iso(large, small):
            return small.copy()

        n = small.number_of_nodes()
        executor = _fut.ThreadPoolExecutor(self.workers)
        # search by decreasing size
        for k in range(n - 1, 0, -1):
            subsets = _connected_subsets(small, k)
            if self.workers > 1:
                results = executor.map(
                    lambda nodes: (
                        nodes
                        if self._is_sub_iso(large, small.subgraph(nodes))
                        else None
                    ),
                    subsets,
                )
            else:
                results = (
                    nodes if self._is_sub_iso(large, small.subgraph(nodes)) else None
                    for nodes in subsets
                )
            for hit in results:
                if hit:
                    return small.subgraph(hit).copy()
        return nx.Graph()


class HeuristicMCS:
    """Iteratively compute MCS over a list of graphs."""

    def __init__(
        self,
        *,
        kwargs: Any,
    ) -> None:
        self.matcher = MCSMatcher(**kwargs)

    def compute(self, graphs: List[nx.Graph]) -> nx.Graph:
        if not graphs:
            raise ValueError("Input list of graphs is empty.")
        if len(graphs) == 1:
            return graphs[0].copy()
        current = self.matcher.maximum_connected_common_subgraph(graphs[0], graphs[1])
        for g in graphs[2:]:
            if current.number_of_nodes() == 0:
                break
            current = self.matcher.maximum_connected_common_subgraph(current, g)
        return current
