"""canonicalize_graph.py
A lightweight, pure‑Python utility for **canonicalising NetworkX graphs**
without any chemistry‑specific dependencies.  It is designed as a drop‑in
alternative to toolkit‑heavy approaches that rely on canonical SMILES.

Key ideas
~~~~~~~~~
* **Canonical signature** – deterministically serialise the graph (after
  sorting nodes/edges by a user‑supplied key) and hash the result with
  SHA‑256.  The 32‑character hex digest acts as the *canonical ID*.
* **Value‑object semantics** – a thin wrapper (`CanonicalGraph`) exposes
  both the *original* and the *canonicalised* `networkx.Graph` alongside
  its signature, and implements `__eq__`, `__hash__`, etc., so it behaves
  well in sets/dicts.

Quick start
-----------
>>> import networkx as nx
>>> from canonicalize_graph import GraphCanonicaliser
>>> G = nx.Graph()
>>> G.add_node(1, element="C"); G.add_node(2, element="O")
>>> G.add_edge(1, 2, order=1)
>>> canon = GraphCanonicaliser().canonicalise_graph(G)
>>> canon.canonical_hash  # the SHA‑256 digest
'0df9e34a7c3cd9b35c0ba5f5cbe7598e'
>>> # Get the canonicalised graph (nodes renumbered & sorted):
>>> CG = canon.canonical_graph
>>> CG.nodes(data=True)
{1: {'element': 'C'}, 2: {'element': 'O'}}

"""

from __future__ import annotations

import hashlib
import inspect
from typing import Dict, Iterable, Tuple, Callable, Any

import networkx as nx

__all__ = [
    "CanonicalGraph",
    "GraphCanonicaliser",
]

###############################################################################
# Helper functions ############################################################
###############################################################################


def _node_key(node_id: Any, data: Dict[str, Any]) -> Tuple:
    """Default key used to sort nodes before serialisation."""
    return (
        data.get("element", ""),
        data.get("charge", 0),
        data.get("atom_map", 0),
        data.get("hcount", 0),
        tuple(data.get("typesGH", ())),
        node_id,
    )


def _edge_key(u: Any, v: Any, data: Dict[str, Any]) -> Tuple:
    """Deterministic key for sorting edges before serialisation."""
    ordered = tuple(sorted((u, v)))
    return (
        ordered,
        data.get("order", ()),
        data.get("standard_order", 0),
    )


def _digest(text: str) -> str:
    """Return the first 32 hex chars of the SHA‑256 digest."""
    return hashlib.sha256(text.encode()).hexdigest()[:32]


###############################################################################
# Core classes ################################################################
###############################################################################


class GraphCanonicaliser:
    """Factory for canonical signatures, graphs, and wrappers."""

    def __init__(
        self,
        node_sort_key: Callable[[Any, Dict[str, Any]], Tuple] = _node_key,
        edge_sort_key: Callable[[Any, Any, Dict[str, Any]], Tuple] = _edge_key,
    ) -> None:
        self._node_sort_key = node_sort_key
        self._edge_sort_key = edge_sort_key

    def canonical_signature(self, graph: nx.Graph) -> str:
        """Compute SHA‑256 signature after relabeling nodes/edges canonically."""
        # first create a canonical graph (reindexed nodes)
        canon = self.make_canonical_graph(graph)
        # serialize nodes and edges of the canonical graph
        nodes = sorted(canon.nodes(data=True), key=lambda x: self._node_sort_key(*x))
        edges = sorted(canon.edges(data=True), key=lambda x: self._edge_sort_key(*x))
        node_str = ";".join(
            f"{nid}:{self._node_sort_key(nid,data)}" for nid, data in nodes
        )
        edge_str = ";".join(
            f"{(u,v)}:{self._edge_sort_key(u,v,data)}" for u, v, data in edges
        )
        return _digest(f"N[{node_str}]|E[{edge_str}]")

    graph_canonical_hash = canonical_signature  # alias

    def make_canonical_graph(self, graph: nx.Graph) -> nx.Graph:
        """Return a new, re‑labelled graph whose nodes and edges are
        added in sorted order (1…N) to preserve canonical form."""
        nodes_sorted = sorted(
            graph.nodes(data=True), key=lambda x: self._node_sort_key(*x)
        )
        mapping = {old: idx + 1 for idx, (old, _) in enumerate(nodes_sorted)}
        G2 = type(graph)()
        for old, data in nodes_sorted:
            G2.add_node(mapping[old], **data)
        edges_sorted = sorted(
            graph.edges(data=True), key=lambda x: self._edge_sort_key(*x)
        )
        for u, v, data in edges_sorted:
            G2.add_edge(mapping[u], mapping[v], **data)
        return G2

    def canonicalise_graph(self, graph: nx.Graph) -> "CanonicalGraph":
        """Wrap *graph* into a value‑object with signature & canonical form."""
        return CanonicalGraph(graph, self)

    def canonicalise_graphs(
        self, graphs: Iterable[nx.Graph]
    ) -> Tuple["CanonicalGraph", ...]:
        """Canonicalise many graphs and return sorted wrappers."""
        return tuple(
            sorted(
                (self.canonicalise_graph(g) for g in graphs),
                key=lambda x: x.canonical_hash,
            )
        )

    def help(self) -> None:
        """Print public API overview."""
        print(inspect.getdoc(self))
        for name in (
            "canonical_signature",
            "make_canonical_graph",
            "canonicalise_graph",
        ):  # etc.
            print(f"  • {name}{inspect.signature(getattr(self,name))}")

    def __repr__(self) -> str:
        return f"<GraphCanonicaliser node_key={self._node_sort_key.__name__} edge_key={self._edge_sort_key.__name__}>"


class CanonicalGraph:
    """Wrapper that couples a graph with its canonical form & signature."""

    def __init__(self, graph: nx.Graph, canon: GraphCanonicaliser) -> None:
        self._original: nx.Graph = graph
        self._canonical_graph: nx.Graph = canon.make_canonical_graph(graph)
        self._canonical_hash: str = canon.canonical_signature(self._canonical_graph)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CanonicalGraph)
            and self.canonical_hash == other.canonical_hash
        )

    def __hash__(self) -> int:
        return hash(self.canonical_hash)

    def __str__(self) -> str:
        return (
            f"<CanonicalGraph nodes={self._canonical_graph.number_of_nodes()}"
            f" edges={self._canonical_graph.number_of_edges()}"
            f" sig={self.canonical_hash[:8]}>"
        )

    __repr__ = __str__

    @property
    def original_graph(self) -> nx.Graph:
        """The mutable, original NetworkX graph."""
        return self._original

    @property
    def canonical_graph(self) -> nx.Graph:
        """The new, relabelled NetworkX graph in canonical node/edge order."""
        return self._canonical_graph

    @property
    def canonical_hash(self) -> str:
        """32‑hex‑char SHA‑256 digest serving as canonical ID."""
        return self._canonical_hash

    def help(self) -> None:
        """Print original & canonical graphs' nodes/edges."""
        print("Original graph:")
        for n, d in self._original.nodes(data=True):
            print(f"  {n}: {d}")
        print("\nCanonical graph:")
        for n, d in self._canonical_graph.nodes(data=True):
            print(f"  {n}: {d}")
        print("\nEdges:")
        for u, v, d in self._canonical_graph.edges(data=True):
            print(f"  ({u},{v}): {d}")
