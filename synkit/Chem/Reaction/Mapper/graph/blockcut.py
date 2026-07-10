"""
Block-cut-tree decomposition of a labeled graph.

A connected graph decomposes into *biconnected components* (blocks) joined at
*articulation points* (cut vertices). Contracting each block to a node and
linking it to its cut vertices yields the *block-cut tree*. Removing an
articulation point disconnects the graph, so any matching/automorphism question
factorises over this tree: the pieces interact only through the shared cut
vertices.

This module exposes the decomposition (:func:`block_cut_tree`) and a helper to
locate which block(s) each atom belongs to (:func:`node_block_map`). The exact
kernel solver uses them to split an uncertainty region that spans several
independent blocks into per-block sub-problems, turning one ``O(k!)`` search into
a sum of much smaller ones (see ``solve_kernel_blockwise`` in
:mod:`mapper.exact.branching`).

Built on :func:`networkx.biconnected_components` /
:func:`networkx.articulation_points`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set

try:
    import networkx as nx

    HAS_NX = True
except Exception:  # pragma: no cover
    HAS_NX = False


def _to_nx(graph, n):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i, nbrs in graph.items():
        for j in nbrs:
            if i != j:
                g.add_edge(i, j)
    return g


@dataclass
class BlockCutTree:
    """Block-cut-tree decomposition.

    Attributes
    ----------
    blocks : list[frozenset[int]]
        Biconnected components (node sets). A bridge edge is a block of size 2.
    articulation_points : set[int]
        Cut vertices shared between blocks.
    tree : networkx.Graph
        Bipartite block-cut tree with nodes ``("B", block_index)`` and
        ``("C", cut_vertex)``.
    """

    blocks: List[FrozenSet[int]]
    articulation_points: Set[int]
    tree: "nx.Graph"


def block_cut_tree(graph, n):
    """Compute the block-cut-tree decomposition of a labeled-graph adjacency dict.

    Parameters
    ----------
    graph : dict[int, dict[int, number]]
        Adjacency (edge weights ignored).
    n : int
        Number of nodes.

    Returns
    -------
    BlockCutTree
    """
    if not HAS_NX:
        raise ImportError("networkx is required for block-cut-tree decomposition")
    g = _to_nx(graph, n)
    blocks = [frozenset(b) for b in nx.biconnected_components(g)]
    arts = set(nx.articulation_points(g))

    tree = nx.Graph()
    for bi, block in enumerate(blocks):
        tree.add_node(("B", bi))
        for v in block:
            if v in arts:
                tree.add_node(("C", v))
                tree.add_edge(("B", bi), ("C", v))
    return BlockCutTree(blocks=blocks, articulation_points=arts, tree=tree)


def node_block_map(bct) -> Dict[int, List[int]]:
    """Map each atom index to the list of block indices containing it.

    Articulation points belong to several blocks; ordinary atoms to exactly one.
    """
    out: Dict[int, List[int]] = {}
    for bi, block in enumerate(bct.blocks):
        for v in block:
            out.setdefault(v, []).append(bi)
    return out
