"""
networkx_canonical_algorithms.py
================================

NetworkX-based canonical-labelling utilities for molecular graphs.
Each helper produces a deterministic ordering (or signature) for graph isomorphism
tasks and returns:
  - a relabelled NetworkX graph copy (where applicable),
  - a 32-hex SHA-256 digest.

Dependencies:
  * networkx
  * numpy
  * python-bliss (optional, for NAUTY/BLISS canonicalisation)
"""

import hashlib
from hashlib import sha256
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np


Digest = str


def _digest(text: str) -> Digest:
    """
    Compute a 32-character hexadecimal SHA-256 digest of the input string.

    Parameters
    ----------
    text : str
        Input text to be hashed.

    Returns
    -------
    Digest
        First 32 hex characters of the SHA-256 digest.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def ring_canonical_graph(g: nx.Graph) -> Tuple[nx.Graph, Digest]:
    """
    Generate a relabelled graph based on SSSR membership hierarchy and
    compute its canonical signature.

    Nodes are ordered by:
      1. Number of smallest rings they belong to (SSSR count).
      2. Node degree.
      3. Original node identifier.

    Parameters
    ----------
    g : nx.Graph
        Input molecular graph (nodes may have attributes).

    Returns
    -------
    Tuple[nx.Graph, Digest]
        - Relabelled graph with nodes numbered 1..N according to canonical order.
        - 32-hex digest based on node membership counts and ordering.
    """
    # Compute ring membership counts
    rings: List[List[Any]] = nx.cycle_basis(g)
    membership: Dict[Any, int] = {node: 0 for node in g.nodes()}
    for ring in rings:
        for node in ring:
            membership[node] += 1

    # Determine canonical node ordering
    order = sorted(g.nodes(), key=lambda n: (membership[n], g.degree[n], n))
    mapping: Dict[Any, int] = {old: idx + 1 for idx, old in enumerate(order)}

    # Build relabelled graph
    G2 = type(g)()
    if hasattr(g, "graph"):
        G2.graph.update(g.graph)
    for old in order:
        G2.add_node(mapping[old], **g.nodes[old])
    for u, v, data in g.edges(data=True):
        G2.add_edge(mapping[u], mapping[v], **data)

    # Compute signature text
    sig_text = ";".join(f"{mapping[node]}:{membership[node]}" for node in order)
    signature = _digest(sig_text)

    return G2, signature


def eigen_canonical_signature(g: nx.Graph) -> Digest:
    """
    Compute a graph signature from sorted eigenvalues of its weighted adjacency matrix.

    Edge weights are taken from the 'order' attribute (default=1).
    The adjacency matrix is symmetric for undirected graphs.

    Parameters
    ----------
    g : nx.Graph
        Input molecular graph.

    Returns
    -------
    Digest
        32-hex digest of sorted real parts of eigenvalues.
    """
    n = g.number_of_nodes()
    # Map nodes to matrix indices
    index_map: Dict[Any, int] = {node: i for i, node in enumerate(sorted(g.nodes()))}
    A = np.zeros((n, n), dtype=float)
    for u, v, data in g.edges(data=True):
        weight = data.get("order", 1)
        i, j = index_map[u], index_map[v]
        A[i, j] = A[j, i] = weight

    # Eigen decomposition
    eigvals = np.linalg.eigvals(A)
    eigvals_sorted = np.sort(eigvals)

    # Form digest text from real parts only
    text = ",".join(f"{ev.real:.8f}" for ev in eigvals_sorted)
    return _digest(text)


def pgraph_signature(g: nx.Graph, p: int = 4) -> Digest:
    """
    Generate a signature by hashing all simple paths up to length p.

    Each path is represented as a hyphen-separated sequence of node 'element'
    attributes (or '?' if missing), and the sorted list of these sequences
    is concatenated for hashing.

    Parameters
    ----------
    g : nx.Graph
        Input molecular graph.
    p : int, optional
        Maximum path length (number of edges), by default 4.

    Returns
    -------
    Digest
        32-hex digest of the concatenated sorted path strings.
    """
    paths: List[str] = []
    for src in g.nodes():
        for dst in g.nodes():
            if src == dst:
                continue
            for path in nx.all_simple_paths(g, src, dst, cutoff=p):
                seq = "-".join(str(g.nodes[a].get("element", "?")) for a in path)
                paths.append(seq)
    combined = "|".join(sorted(paths))
    return _digest(combined)


def canon_morgan(
    g: nx.Graph, morgan_radius: int = 2, node_attributes: List[str] = None
) -> Tuple[nx.Graph, Digest]:
    """
    Prime-based neighbourhood refinement analogous to Morgan fingerprinting.

    Each node is initially assigned a unique prime number; optionally,
    specified node attributes are incorporated into the seed label.
    For each iteration up to `morgan_radius`, node labels are updated by
    multiplying by the labels of neighboring nodes.

    Parameters
    ----------
    g : nx.Graph
        Input molecular graph.
    morgan_radius : int, optional
        Number of refinement iterations, by default 2.
    node_attributes : List[str], optional
        Node attribute keys to include in initial hashing; if None,
        only prime seeding is used.

    Returns
    -------
    Tuple[nx.Graph, Digest]
        - Relabelled graph with canonical node ordering.
        - 32-hex digest of the sequence of final labels per node.
    """
    nodes_sorted = sorted(g.nodes())
    # Generate unique primes
    primes: List[int] = []
    candidate = 2
    while len(primes) < len(nodes_sorted):
        if all(candidate % p for p in primes):
            primes.append(candidate)
        candidate += 1

    # Initial labels with optional node attributes
    labels: Dict[Any, int] = {}
    for idx, node in enumerate(nodes_sorted):
        label = primes[idx]
        if node_attributes:
            attr_text = "".join(
                str(g.nodes[node].get(attr, "")) for attr in node_attributes
            )
            attr_hash = int(_digest(attr_text), 16)
            label *= attr_hash
        labels[node] = label

    # Iterative refinement
    for _ in range(morgan_radius):
        new_labels: Dict[Any, int] = {}
        for node in g.nodes():
            prod = labels[node]
            for neighbor in g.neighbors(node):
                prod *= labels[neighbor]
            new_labels[node] = prod
        labels = new_labels

    # Determine canonical ordering
    order = sorted(g.nodes(), key=lambda n: (labels[n], g.degree[n], n))
    mapping: Dict[Any, int] = {old: idx + 1 for idx, old in enumerate(order)}

    # Build relabelled graph
    G2 = type(g)()
    if hasattr(g, "graph"):
        G2.graph.update(g.graph)
    for old in order:
        G2.add_node(mapping[old], **g.nodes[old])
    for u, v, data in sorted(
        g.edges(data=True), key=lambda e: tuple(sorted((mapping[e[0]], mapping[e[1]])))
    ):
        G2.add_edge(mapping[u], mapping[v], **data)

    return G2


# Utility to normalize and hash a node label with its neighbors and edge orders
# Utility to normalize and hash a node label with its neighbors and edge orders
def _hash_labels(node_label: int, neigh_info: List[Tuple[int, Any]]) -> int:
    """
    Combine a node's label with sorted neighbor labels and edge orders into a new hash.
    neigh_info is a list of tuples (neighbor_label, edge_order).
    """
    data = [str(node_label)]
    # Include edge order info in the string
    for nlabel, order in sorted(neigh_info):
        data.append(f"{nlabel}|{order}")
    digest_hex = sha256("|".join(data).encode()).hexdigest()
    # Truncate digest to fixed length and convert to integer
    return int(digest_hex[:16], 16)


__all__ = [
    "ring_canonical_graph",
    "eigen_canonical_signature",
    "pgraph_signature",
    "canon_morgan",
]
