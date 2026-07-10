"""
mapper.graph — labeled-graph data structure and graph algorithms.

Submodules
----------
labeled_graph
    :class:`LabeledGraph`: adjacency-dict graph with integer node labels.
refine
    1-WL and 2-FWL color refinement (:func:`wl_node_colors`,
    :func:`selective_two_wl_refine`).
automorphism
    Automorphism orbits via synkit (:func:`node_orbits`, :func:`n_automorphisms`).
synkit_adapter
    NetworkX/SynKit projection and canonical WL hashing helpers.
blockcut
    Block-cut-tree decomposition (:func:`block_cut_tree`, :class:`BlockCutTree`).
"""

from .labeled_graph import LabeledGraph
from .refine import (
    wl_node_colors,
    wl_graph_hash,
    two_wl_pair_colors,
    two_wl_graph_hash,
    two_wl_node_colors,
    selective_two_wl_refine,
)
from .automorphism import node_orbits, orbit_id_map, n_automorphisms
from .synkit_adapter import (
    canonical_graph_hash,
    graph_to_nx,
    synkit_wl_graph_hash,
    synkit_wl_node_labels,
)
from .blockcut import BlockCutTree, block_cut_tree, node_block_map

__all__ = [
    # labeled_graph
    "LabeledGraph",
    # refine
    "wl_node_colors",
    "wl_graph_hash",
    "two_wl_pair_colors",
    "two_wl_graph_hash",
    "two_wl_node_colors",
    "selective_two_wl_refine",
    # automorphism
    "node_orbits",
    "orbit_id_map",
    "n_automorphisms",
    # synkit_adapter
    "canonical_graph_hash",
    "graph_to_nx",
    "synkit_wl_graph_hash",
    "synkit_wl_node_labels",
    # blockcut
    "BlockCutTree",
    "block_cut_tree",
    "node_block_map",
]
