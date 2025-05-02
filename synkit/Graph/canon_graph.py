"""
canonicalize_graph.py
=====================

A **pure‑Python / pure‑NetworkX** library that assigns *canonical*,
deterministic identifiers to graphs **without any external toolchain**.
It can be dropped into cheminformatics, bio‑networks, knowledge graphs,
or any workflow that needs stable de‑duplication of isomorphic graphs.

Why canonicalise?
-----------------
* **De‑duplication** – hash the canonical form, then compare hashes
  instead of running an expensive isomorphism test for every pair.
* **Index keys** – store the 32‑hex digest in a database and use it as a
  primary key for sub‑structure search or provenance tracking.
* **Version control‑friendly** – serialise nodes/edges in a predictable,
  line‑ordered way so diffs stay minimal.

Two back‑ends
~~~~~~~~~~~~~
``backend="generic"`` (default)
    *Sort‑and‑hash* strategy identical to earlier releases.  Fastest on
    graphs where node/edge attributes already break most automorphisms.
``backend="wl"``
    Weisfeiler–Lehman colour‑refinement adds **structure awareness**
    without leaving pure Python.  Slightly slower but collapses many more
    isomorphic graphs to the same label.

Quick start
-----------
>>> import networkx as nx
>>> from canonicalize_graph import GraphCanonicaliser
>>>
>>> G = nx.Graph()
>>> G.add_node(1, element="C"); G.add_node(2, element="O")
>>> G.add_edge(1, 2, order=1)
>>>
>>> cg = GraphCanonicaliser(backend="wl").canonicalise_graph(G)
>>> cg.canonical_hash
'0df9e34a7c3cd9b35c0ba5f5cbe7598e'
>>> cg.canonical_graph.nodes(data=True)
[(1, {'element': 'C'}), (2, {'element': 'O'})]
"""

from __future__ import annotations

import hashlib
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Tuple,
)

from synkit.IO.chem_converter import gml_to_its, its_to_gml, rsmi_to_its, its_to_rsmi
import networkx as nx
from networkx.algorithms.graph_hashing import (
    weisfeiler_lehman_subgraph_hashes as _wl_hashes,
)

__all__: list[str] = ["CanonicalGraph", "GraphCanonicaliser", "CanonicalRule"]

###############################################################################
# Type aliases & helpers ######################################################
###############################################################################

NodeId = Hashable  # a node label accepted by NetworkX
NodeData = Dict[str, Any]
EdgeData = Dict[str, Any]
Digest = str

T_NodeSortKey = Callable[[NodeId, NodeData], Tuple[Any, ...]]
T_EdgeSortKey = Callable[[NodeId, NodeId, EdgeData], Tuple[Any, ...]]


def _default_node_key(node_id: NodeId, data: NodeData) -> Tuple[Any, ...]:
    """Fallback sort‑key if the user supplies none."""
    return (
        data.get("element", ""),
        data.get("charge", 0),
        data.get("atom_map", 0),
        data.get("hcount", 0),
        tuple(data.get("typesGH", ())),
        node_id,  # final tie‑breaker
    )


def _default_edge_key(u: NodeId, v: NodeId, data: EdgeData) -> Tuple[Any, ...]:
    """Fallback edge sort‑key (makes undirected edges order‑invariant)."""
    return (tuple(sorted((u, v))), data.get("order", 0), data.get("standard_order", 0))


def _digest(text: str) -> Digest:
    """First 32 hex chars of SHA‑256 – short *but* collision‑safe for up to 2¹²⁸ graphs."""
    return hashlib.sha256(text.encode()).hexdigest()[:32]


###############################################################################
# Public API ##################################################################
###############################################################################


class GraphCanonicaliser:
    """
    Factory that turns arbitrary ``networkx.Graph`` objects into their
    *canonical* twin plus a **stable 32‑hex digest**.

    Parameters
    ----------
    backend:
        ``"generic"`` or ``"wl"`` (structure‑aware Weisfeiler–Lehman).
    wl_iterations:
        Depth of WL refinement (ignored for ``generic``).  Three iterations
        distinguish nearly all real‑world chemical graphs; increase for
        very large or highly regular topologies.
    node_sort_key, edge_sort_key:
        Custom deterministic orderings.  They *must* treat their arguments
        as *read‑only* and return plain tuples for total ordering.

    Notes
    -----
    *All* returned graphs are of the *same class* as the input
    (``nx.Graph``, ``nx.DiGraph`` …), so multigraphs and digraphs are
    preserved.

    Examples
    --------
    >>> canon = GraphCanonicaliser(backend="generic")
    >>> sig   = canon.canonical_signature(G)
    >>> cg    = canon.canonicalise_graph(G)
    >>> cg.canonical_graph  # a relabelled copy, nodes 1…N
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        backend: Literal["generic", "wl"] = "generic",
        wl_iterations: int = 3,
        node_sort_key: T_NodeSortKey = _default_node_key,
        edge_sort_key: T_EdgeSortKey = _default_edge_key,
    ) -> None:
        if backend not in {"generic", "wl"}:
            raise ValueError("backend must be 'generic' or 'wl'")
        self.backend: Literal["generic", "wl"] = backend
        self._wl_k: int = wl_iterations
        self._node_key: T_NodeSortKey = node_sort_key
        self._edge_key: T_EdgeSortKey = edge_sort_key

    # ------------------------------------------------------------------ #
    # High‑level helpers                                                 #
    # ------------------------------------------------------------------ #
    def canonicalise_graph(self, graph: nx.Graph) -> "CanonicalGraph":
        """
        Return a :class:`CanonicalGraph` wrapper around *graph*.

        The wrapper exposes:

        * :pyattr:`~CanonicalGraph.canonical_graph` – relabelled 1…N
        * :pyattr:`~CanonicalGraph.canonical_hash`  – 32‑char digest
        """
        return CanonicalGraph(graph, self)

    def canonicalise_graphs(
        self,
        graphs: Iterable[nx.Graph],
    ) -> Tuple["CanonicalGraph", ...]:
        """
        Bulk helper that returns *all* wrappers **sorted by hash**.

        Useful when you want fast set comparison but need the canonical
        graphs as well:

        >>> wrappers = canon.canonicalise_graphs([G1, G2, G3])
        >>> unique   = {w.canonical_hash for w in wrappers}
        """
        return tuple(
            sorted(
                (self.canonicalise_graph(g) for g in graphs),
                key=lambda x: x.canonical_hash,
            )
        )

    # ------------------------------------------------------------------ #
    # Digest / core methods                                              #
    # ------------------------------------------------------------------ #
    def canonical_signature(self, graph: nx.Graph) -> Digest:
        """
        Return the *hash of the canonical form* of *graph*.

        Equal digests ⇒ graphs are guaranteed isomorphic **under the
        chosen back‑end and keys**.
        """
        return _digest(self._serialise(self._make_canonical_graph(graph)))

    # Alias kept for backward‑compatibility
    graph_canonical_hash: Callable[[nx.Graph], Digest] = canonical_signature

    # ------------------------------------------------------------------ #
    # Internal – make canonical graph                                    #
    # ------------------------------------------------------------------ #
    def _make_canonical_graph(self, g: nx.Graph) -> nx.Graph:
        """Dispatcher that calls the appropriate back‑end."""
        return (
            self._canon_generic(g) if self.backend == "generic" else self._canon_wl(g)
        )

    def _canon_generic(self, g: nx.Graph) -> nx.Graph:
        """Pure attribute‑sort strategy – O(|V| log |V| + |E| log |E|)."""
        nodes_sorted = sorted(g.nodes(data=True), key=lambda x: self._node_key(*x))
        mapping: Dict[NodeId, int] = {
            old: i + 1 for i, (old, _) in enumerate(nodes_sorted)
        }

        G2 = type(g)()
        for old, data in nodes_sorted:
            G2.add_node(mapping[old], **data)

        edges_sorted = sorted(g.edges(data=True), key=lambda x: self._edge_key(*x))
        for u, v, data in edges_sorted:
            G2.add_edge(mapping[u], mapping[v], **data)

        return G2

    def _canon_wl(self, g: nx.Graph) -> nx.Graph:
        """
        Weisfeiler–Lehman colour‑refinement back‑end (pure Python).

        The algorithm iteratively hashes each node’s neighbourhood up to
        *k* hops (``wl_iterations``) and orders nodes by
        ``(colour, degree, original_id)``.  The final mapping is stable
        for almost all non‑adversarial graphs.
        """
        wl_hash = _wl_hashes(
            g,
            node_attr=(
                "element"
                if any("element" in d for _, d in g.nodes(data=True))
                else None
            ),
            iterations=self._wl_k,
        )
        # use last hash as colour
        colour: Dict[NodeId, str] = {n: h[-1] for n, h in wl_hash.items()}
        order: List[NodeId] = sorted(g, key=lambda n: (colour[n], g.degree[n], n))
        mapping: Dict[NodeId, int] = {old: i + 1 for i, old in enumerate(order)}

        G2 = type(g)()
        for old in order:
            G2.add_node(mapping[old], **g.nodes[old])

        for u, v, data in sorted(
            g.edges(data=True),
            key=lambda e: tuple(sorted((mapping[e[0]], mapping[e[1]]))),
        ):
            G2.add_edge(mapping[u], mapping[v], **data)

        return G2

    # ------------------------------------------------------------------ #
    # Internal – serialisation                                           #
    # ------------------------------------------------------------------ #
    def _serialise(self, g: nx.Graph) -> str:
        """Stable plain‑text representation → fed into SHA‑256."""
        nodes = sorted(g.nodes(data=True), key=lambda x: self._node_key(*x))
        edges = sorted(g.edges(data=True), key=lambda x: self._edge_key(*x))

        node_str = ";".join(f"{n}:{self._node_key(n,d)}" for n, d in nodes)
        edge_str = ";".join(f"{(u,v)}:{self._edge_key(u,v,d)}" for u, v, d in edges)
        return f"N[{node_str}]|E[{edge_str}]"

    # ------------------------------------------------------------------ #
    # Misc utilities                                                     #
    # ------------------------------------------------------------------ #
    def help(self) -> None:  # pragma: no cover
        """Pretty‑print the public methods with their signatures."""
        print(inspect.getdoc(self))
        for meth in (
            "canonical_signature",
            "make_canonical_graph",
            "canonicalise_graph",
        ):
            print(f"  • {meth}{inspect.signature(getattr(self, meth))}")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<GraphCanonicaliser backend={self.backend!r} "
            f"node_key={self._node_key.__name__} "
            f"edge_key={self._edge_key.__name__}>"
        )

    # keep public attribute for backwards compatibility
    make_canonical_graph = _make_canonical_graph


# =============================================================================
# Value wrapper (unchanged surface – richer docs)
# =============================================================================
class CanonicalGraph:
    """
    *Value object* tying together:

    * the **original** NetworkX graph (mutable, user‑supplied);
    * its **canonical twin** (immutable copy, nodes relabelled 1…N);
    * a 32‑char **SHA‑256 digest**.

    Instances compare & hash **by digest only** – perfect for set/dict
    membership while still carrying the underlying graphs.

    Do **not** mutate :pyattr:`original_graph` in place if you need to
    rely on :pyattr:`canonical_hash`; repeat the canonicalisation after
    any structural change instead.
    """

    def __init__(self, g: nx.Graph, canon: GraphCanonicaliser) -> None:
        self._original: nx.Graph = g
        self._canonical_graph: nx.Graph = canon._make_canonical_graph(g)
        self._canonical_hash: Digest = canon.canonical_signature(self._canonical_graph)

    # ------------------------------------------------------------------ #
    # Dunder sugar                                                       #
    # ------------------------------------------------------------------ #
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CanonicalGraph)
            and self.canonical_hash == other.canonical_hash
        )

    def __hash__(self) -> int:
        return hash(self.canonical_hash)

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"<CanonicalGraph |V|={self._canonical_graph.number_of_nodes()} "
            f"|E|={self._canonical_graph.number_of_edges()} "
            f"hash={self.canonical_hash[:8]}>"
        )

    __repr__ = __str__

    # ------------------------------------------------------------------ #
    # Public read‑only views                                             #
    # ------------------------------------------------------------------ #
    @property
    def original_graph(self) -> nx.Graph:
        """A direct reference to **your** graph – **mutable**."""
        return self._original

    @property
    def canonical_graph(self) -> nx.Graph:
        """Immutable relabelled copy, nodes numbered 1 … |V|."""
        return self._canonical_graph

    @property
    def canonical_hash(self) -> Digest:
        """32‑hex‑char SHA‑256 digest (*lower‑case*, deterministic)."""
        return self._canonical_hash

    # ------------------------------------------------------------------ #
    # Convenience debug helper                                           #
    # ------------------------------------------------------------------ #
    def help(self) -> None:  # pragma: no cover
        """Print both graphs’ node/edge tables – handy for debugging."""
        print("Original graph:")
        for n, d in self._original.nodes(data=True):
            print(f"  {n}: {d}")
        print("\nCanonical graph:")
        for n, d in self._canonical_graph.nodes(data=True):
            print(f"  {n}: {d}")
        print("\nEdges:")
        for u, v, d in self._canonical_graph.edges(data=True):
            print(f"  ({u},{v}): {d}")


class CanonicalRule:
    """
    Value object that wraps a graph transformation rule in GML string form,
    providing a canonicalised GML output and a stable 32-character SHA-256 hash.

    Internally, the GML rule is parsed into a NetworkX graph via `gml_to_its`,
    canonicalised using a `GraphCanonicaliser`, and re-serialized back to GML
    with `its_to_gml`.

    Equality and hashing are based solely on the canonical hash, so
    isomorphic rules (under the chosen backend) compare equal.

    Attributes
    ----------
    original_rule : str
        The raw GML string supplied by the user.
    original_graph : nx.Graph
        The NetworkX graph parsed from `original_rule`.
    canonical_graph : nx.Graph
        The relabelled canonical graph (nodes renumbered 1…N).
    canonical_rule : str
        The canonical graph re-serialized to a GML string.
    canonical_hash : Digest
        32-hex-character SHA-256 digest of the canonical graph.
    """

    def __init__(
        self,
        rule: str,
        canon: GraphCanonicaliser = GraphCanonicaliser(),
    ) -> None:
        """
        Instantiate a CanonicalRule.

        Parameters
        ----------
        rule : str
            GML string of the transformation rule.
        canon : GraphCanonicaliser
            Initialized canonicaliser (generic or WL backend).
        """
        # Store raw inputs
        self._original_rule: str = rule
        # Parse into NetworkX
        self._original_graph: nx.Graph = gml_to_its(rule)
        # Canonicalise graph
        self._canonical_graph: nx.Graph = canon._make_canonical_graph(
            self._original_graph
        )
        # Serialize back to GML
        self._canonical_rule: str = its_to_gml(self._canonical_graph)
        # Compute hash
        self._canonical_hash: Digest = canon.canonical_signature(self._canonical_graph)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CanonicalRule)
            and self.canonical_hash == other.canonical_hash
        )

    def __hash__(self) -> int:
        return hash(self._canonical_hash)

    def __str__(self) -> str:
        n = self._canonical_graph.number_of_nodes()
        m = self._canonical_graph.number_of_edges()
        return f"<CanonicalRule |V|={n} |E|={m} hash={self._canonical_hash[:8]}>"

    __repr__ = __str__

    @property
    def original_rule(self) -> str:
        """Original GML rule string."""
        return self._original_rule

    @property
    def original_graph(self) -> nx.Graph:
        """Parsed NetworkX graph from `original_rule`."""
        return self._original_graph

    @property
    def canonical_graph(self) -> nx.Graph:
        """Relabelled canonical NetworkX graph (nodes 1…N)."""
        return self._canonical_graph

    @property
    def canonical_rule(self) -> str:
        """GML string of the canonical graph."""
        return self._canonical_rule

    @property
    def canonical_hash(self) -> Digest:
        """32-hex-character SHA-256 digest of the canonical graph."""
        return self._canonical_hash

    def help(self) -> None:
        """
        Print original and canonical rule texts and underlying graphs.
        """
        print("Original GML rule:")
        print(self._original_rule)
        print("\nCanonical GML rule:")
        print(self._canonical_rule)
        print("\nCanonical graph nodes and data:")
        for n, d in self._canonical_graph.nodes(data=True):
            print(f"  {n}: {d}")
        print("\nCanonical graph edges and data:")
        for u, v, d in self._canonical_graph.edges(data=True):
            print(f"  ({u},{v}): {d}")


def sync_atom_map_with_index(G):
    """
    Make each node’s 'atom_map' equal to its own node index.

    Parameters
    ----------
    G : networkx.Graph
        Your molecular graph. Node IDs are assumed to be the indices
        you want to copy into 'atom_map'.

    Returns
    -------
    None
        The graph is modified in place.
    """
    for node in G.nodes:
        G.nodes[node]["atom_map"] = node


def _canon_smart(
    rsmi: str,
    backend: str = "wl",
) -> str:
    """
    Return a canonicalised reaction‑SMILES.

    Parameters:
        rsmi: Reaction SMILES with atom‑mapping.
        backend: Canonicaliser backend (default ``"wl"``).

    Returns:
        Canonical rSMI in the form ``reactants > agents > products``.

    Raises:
        ValueError: If parsing or canonicalisation fails.
    """
    its: nx.Graph = rsmi_to_its(rsmi)
    its = GraphCanonicaliser(backend=backend).canonicalise_graph(its).canonical_graph
    sync_atom_map_with_index(its)
    return its_to_rsmi(its)
