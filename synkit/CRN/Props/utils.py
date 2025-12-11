from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

from ..Hypergraph.conversion import _as_bipartite

LOGGER = logging.getLogger(__name__)


def _split_species_reactions(G: nx.Graph) -> Tuple[List[Any], List[Any]]:
    """
    Split nodes of a bipartite graph into species and reaction sets.

    Nodes are classified using:

    - ``kind``: ``\"species\"`` or ``\"reaction\"``, if present.
    - otherwise the ``bipartite`` flag: 0 for species, 1 for reactions.

    :param G: Bipartite NetworkX graph.
    :type G: networkx.Graph
    :returns: ``(species_nodes, reaction_nodes)`` lists of node IDs.
    :rtype: Tuple[List[Any], List[Any]]
    :raises ValueError: If graph does not appear to be bipartite.
    """
    species_nodes: List[Any] = []
    reaction_nodes: List[Any] = []

    for node, data in G.nodes(data=True):
        kind = data.get("kind")
        bflag = data.get("bipartite", None)

        if kind == "species" or bflag == 0:
            species_nodes.append(node)
        elif kind == "reaction" or bflag == 1:
            reaction_nodes.append(node)

    if not species_nodes or not reaction_nodes:
        raise ValueError(
            "Graph does not contain both species and reaction nodes with "
            "`kind`/`bipartite` attributes."
        )

    return species_nodes, reaction_nodes


def _species_order(
    G: nx.Graph,
) -> Tuple[List[Any], List[str], Dict[Any, int]]:
    """
    Determine a deterministic ordering of species nodes.

    Species nodes are sorted lexicographically by their ``label`` attribute
    if present, otherwise by the node identifier converted to string.

    :param G: Bipartite NetworkX graph.
    :type G: networkx.Graph
    :returns:
        - species_nodes_sorted: node IDs in order.
        - species_labels: list of species labels.
        - species_index: mapping node -> row index.
    :rtype: Tuple[List[Any], List[str], Dict[Any, int]]
    """
    species_nodes, _ = _split_species_reactions(G)
    species_nodes_sorted = sorted(
        species_nodes,
        key=lambda n: str(G.nodes[n].get("label", n)),
    )

    species_labels: List[str] = []
    species_index: Dict[Any, int] = {}
    for i, node in enumerate(species_nodes_sorted):
        label = str(G.nodes[node].get("label", node))
        species_labels.append(label)
        species_index[node] = i

    return species_nodes_sorted, species_labels, species_index


def _species_and_reaction_order(
    crn: Any,
) -> Tuple[List[str], List[str], Dict[Any, int], Dict[Any, int]]:
    """
    Determine ordering of species and reaction nodes and build index maps.

    Species and reactions are ordered lexicographically by their ``label``
    attribute if present, otherwise by their node identifier converted to
    string.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :returns:
        - species_labels: list of species names.
        - reaction_labels: list of reaction names (stringified node IDs).
        - species_index: mapping node -> species row index.
        - reaction_index: mapping node -> reaction column index.
    :rtype: Tuple[List[str], List[str], Dict[Any, int], Dict[Any, int]]
    """
    G = _as_bipartite(crn)
    species_nodes, reaction_nodes = _split_species_reactions(G)

    # Sort species/reactions deterministically
    species_nodes_sorted = sorted(
        species_nodes,
        key=lambda n: str(G.nodes[n].get("label", n)),
    )
    reaction_nodes_sorted = sorted(
        reaction_nodes,
        key=lambda n: str(G.nodes[n].get("label", n)),
    )

    species_labels: List[str] = []
    species_index: Dict[Any, int] = {}
    for i, node in enumerate(species_nodes_sorted):
        label = str(G.nodes[node].get("label", node))
        species_labels.append(label)
        species_index[node] = i

    reaction_labels: List[str] = []
    reaction_index: Dict[Any, int] = {}
    for j, node in enumerate(reaction_nodes_sorted):
        # Reaction labels are not heavily used; keep them simple and stable.
        label = str(G.nodes[node].get("label", node))
        reaction_labels.append(label)
        reaction_index[node] = j

    return species_labels, reaction_labels, species_index, reaction_index
