from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_GRAPH_TYPES = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)


def _as_graph(crn: Any) -> nx.Graph:
    """
    Return the NetworkX graph representing the SynCRN bipartite structure.

    Accepted inputs
    ---------------
    - a NetworkX graph directly
    - a SynCRN-like wrapper exposing one of:
      ``G``, ``nx_graph``, ``bipartite_graph``, ``bipartite``
    - an object exposing ``to_digraph()``
    """
    if isinstance(crn, _GRAPH_TYPES):
        return crn

    for attr in ("G", "nx_graph", "bipartite_graph", "bipartite"):
        G = getattr(crn, attr, None)
        if isinstance(G, _GRAPH_TYPES):
            return G

    to_digraph = getattr(crn, "to_digraph", None)
    if callable(to_digraph):
        for kwargs in ({}, {"include_rule": True}):
            try:
                G = to_digraph(**kwargs)
            except TypeError:
                continue
            if isinstance(G, _GRAPH_TYPES):
                return G

        raise TypeError(
            "`crn.to_digraph()` was found, but it did not return a NetworkX graph."
        )

    raise TypeError(
        "stoich expects either a NetworkX graph or a SynCRN-like object "
        "containing one at `.G`, `.nx_graph`, `.bipartite_graph`, "
        "`.bipartite`, or exposing `.to_digraph()`."
    )


def _node_kind(data: Dict[str, Any]) -> str:
    return str(data.get("kind", "")).strip().lower()


def _is_species_node(data: Dict[str, Any]) -> bool:
    return _node_kind(data) == "species"


def _is_rule_node(data: Dict[str, Any]) -> bool:
    return _node_kind(data) == "rule"


def _normalize_role(role: Any) -> str | None:
    if role is None:
        return None
    r = str(role).strip().lower()
    if r in {"reactant", "lhs", "educt", "substrate"}:
        return "reactant"
    if r in {"product", "rhs"}:
        return "product"
    return None


def _edge_coeff(data: Dict[str, Any]) -> float:
    return float(data.get("stoich", data.get("coeff", data.get("coefficient", 1.0))))


def _node_sort_key(item: Tuple[Any, Dict[str, Any]]) -> Tuple[Any, ...]:
    """
    Stable ordering for species and rule nodes.

    Species tend to have no `step` / `rule_index` / `app_index`, so they
    naturally sort before rule nodes when grouped by kind.
    """
    node, data = item
    kind_rank = 0 if _is_species_node(data) else 1 if _is_rule_node(data) else 2
    step = data.get("step", -1)
    rule_index = data.get("rule_index", -1)
    app_index = data.get("app_index", -1)
    label = str(data.get("label", ""))
    smiles = str(data.get("smiles", ""))
    return (kind_rank, step, rule_index, app_index, label, smiles, repr(node))


def _species_and_rule_order(
    crn: Any,
) -> Tuple[List[Any], List[Any], Dict[Any, int], Dict[Any, int]]:
    """
    Return stable ordering and indices for species rows and rule columns.
    """
    G = _as_graph(crn)

    species_nodes: List[Any] = []
    rule_nodes: List[Any] = []

    for node, data in sorted(G.nodes(data=True), key=_node_sort_key):
        if _is_species_node(data):
            species_nodes.append(node)
        elif _is_rule_node(data):
            rule_nodes.append(node)

    species_index = {node: i for i, node in enumerate(species_nodes)}
    rule_index = {node: j for j, node in enumerate(rule_nodes)}

    return species_nodes, rule_nodes, species_index, rule_index


# def _species_order(
#     G: nx.Graph,
# ) -> Tuple[List[Any], List[str], Dict[Any, int]]:
#     """
#     Determine a deterministic ordering of species nodes.

#     Species nodes are sorted lexicographically by their ``label`` attribute
#     if present, otherwise by the node identifier converted to string.

#     :param G: Bipartite NetworkX graph.
#     :type G: networkx.Graph
#     :returns:
#         - species_nodes_sorted: node IDs in order.
#         - species_labels: list of species labels.
#         - species_index: mapping node -> row index.
#     :rtype: Tuple[List[Any], List[str], Dict[Any, int]]
#     """
#     species_nodes, _ = _split_species_reactions(G)
#     species_nodes_sorted = sorted(
#         species_nodes,
#         key=lambda n: str(G.nodes[n].get("label", n)),
#     )

#     species_labels: List[str] = []
#     species_index: Dict[Any, int] = {}
#     for i, node in enumerate(species_nodes_sorted):
#         label = str(G.nodes[node].get("label", node))
#         species_labels.append(label)
#         species_index[node] = i

#     return species_nodes_sorted, species_labels, species_index


# def _species_and_reaction_order(
#     crn: Any,
# ) -> Tuple[List[str], List[str], Dict[Any, int], Dict[Any, int]]:
#     """
#     Determine ordering of species and reaction nodes and build index maps.

#     Species and reactions are ordered lexicographically by their ``label``
#     attribute if present, otherwise by their node identifier converted to
#     string.

#     :param crn: Hypergraph or bipartite NetworkX graph.
#     :type crn: Any
#     :returns:
#         - species_labels: list of species names.
#         - reaction_labels: list of reaction names (stringified node IDs).
#         - species_index: mapping node -> species row index.
#         - reaction_index: mapping node -> reaction column index.
#     :rtype: Tuple[List[str], List[str], Dict[Any, int], Dict[Any, int]]
#     """
#     G = _as_bipartite(crn)
#     species_nodes, reaction_nodes = _split_species_reactions(G)

#     # Sort species/reactions deterministically
#     species_nodes_sorted = sorted(
#         species_nodes,
#         key=lambda n: str(G.nodes[n].get("label", n)),
#     )
#     reaction_nodes_sorted = sorted(
#         reaction_nodes,
#         key=lambda n: str(G.nodes[n].get("label", n)),
#     )

#     species_labels: List[str] = []
#     species_index: Dict[Any, int] = {}
#     for i, node in enumerate(species_nodes_sorted):
#         label = str(G.nodes[node].get("label", node))
#         species_labels.append(label)
#         species_index[node] = i

#     reaction_labels: List[str] = []
#     reaction_index: Dict[Any, int] = {}
#     for j, node in enumerate(reaction_nodes_sorted):
#         # Reaction labels are not heavily used; keep them simple and stable.
#         label = str(G.nodes[node].get("label", node))
#         reaction_labels.append(label)
#         reaction_index[node] = j

#     return species_labels, reaction_labels, species_index, reaction_index
