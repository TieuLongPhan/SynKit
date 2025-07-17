import logging
from typing import Any, Callable, Dict, Optional, Union, Set

import networkx as nx
from networkx.algorithms import isomorphism

# Alias for any NetworkX graph type
GraphType = Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]


def find_wc_graph_isomorphism(
    G1: GraphType,
    G2: GraphType,
    node_match: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
    edge_match: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[Any, Any]]:
    """Wildcard‑aware sub‑graph isomorphism.  Returns a mapping from every node
    in the **smaller** graph to a node in the **larger** graph, allowing any
    node whose ``element == "*"`` to match *any* concrete node (or group of
    nodes) on the host side.

    :param G1: First input graph.
    :type  G1: nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph
    :param G2: Second input graph.
    :type  G2: nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph
    :param node_match: Optional node‑predicate; default treats “*” as a joker.
    :type  node_match: Callable[[dict, dict], bool] | None
    :param edge_match: Optional edge‑predicate; default ignores edge data.
    :type  edge_match: Callable[[dict, dict], bool] | None
    :param logger: Optional logger for diagnostics.
    :type  logger: logging.Logger | None
    :returns: Mapping *pattern‑node → host‑node* if a wildcard isomorphism
              exists; otherwise ``None``.
    :rtype: dict[Any, Any] | None
    """
    log = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------ helpers
    if node_match is None:

        def node_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:  # noqa: W shadow
            return (
                a.get("element") == "*"
                or b.get("element") == "*"
                or a.get("element") == b.get("element")
            )

    if edge_match is None:

        def edge_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            return True

    # ------------------------------------------------------------------ choose host|pattern
    if G1.number_of_nodes() >= G2.number_of_nodes():
        host, pattern = G1, G2
        invert = False
    else:
        host, pattern = G2, G1
        invert = True

    Matcher = (
        isomorphism.MultiGraphMatcher
        if isinstance(host, (nx.MultiGraph, nx.MultiDiGraph))
        else isomorphism.GraphMatcher
    )

    GM = Matcher(host, pattern, node_match=node_match, edge_match=edge_match)

    for mapping_host_to_pattern in GM.subgraph_isomorphisms_iter():
        log.debug("Wildcard match found: %s", mapping_host_to_pattern)
        return (
            {v: k for k, v in mapping_host_to_pattern.items()}  # pattern → host
            if invert
            else mapping_host_to_pattern
        )

    log.debug("No wildcard mapping exists between the two graphs.")
    return None


def fuse_wc_graphs(
    G1: GraphType,
    G2: GraphType,
    mapping: Dict[Any, Any],
    wildcard: str = "*",
    logger: Optional[logging.Logger] = None,
) -> GraphType:
    """Fuse a wildcard‑pattern graph *G1* into the concrete host *G2*.

    The result lives **entirely in G2’s node‑ID space** and contains:

    • every host‑node ``mapping[p]`` for a **non‑wildcard** node *p* in G1
      (and its attributes are overridden with those from G1, so no “*” leaks
      back in);

    • every host‑node ``mapping[w]`` for a **wildcard** node *w* in G1 **plus
      all one‑hop neighbours of that host** that were not already used by a
      non‑wildcard;

    • **all edges present in G2** among the nodes kept above.

    Parameters
    ----------
    G1, G2 : GraphType
        `G1` may contain nodes whose ``element`` is the wildcard marker
        (default ``"*"``, change via `wildcard`).
        `G2` is the concrete graph we will graft from.
    mapping : Dict[Any, Any]
        The full node–node map returned by `find_wc_graph_isomorphism`
        (must include *every* node of `G1`).
    wildcard : str, default "*"
        The value of the ``"element"`` attribute that marks a wildcard node.
    logger : logging.Logger or None
        Optional logger for debug output.

    Returns
    -------
    GraphType
        A new graph of the same class as G2 containing the fused structure.
    """
    log = logger or logging.getLogger(__name__)

    # ----------------------------------------------------------------- helpers
    def is_wildcard(node_id: Any) -> bool:
        return G1.nodes[node_id].get("element") == wildcard

    # 1) gather core (non‑wc) and wc nodes in G1 -------------------------------
    core_nodes = [n for n in G1 if not is_wildcard(n)]
    wc_nodes = [n for n in G1 if is_wildcard(n)]

    # 2) host nodes that *must* be present (mapping of core) -------------------
    host_nodes: Set[Any] = {mapping[p] for p in core_nodes}

    # 3) for every wildcard, keep its host anchor + all one‑hop extras ---------
    for w in wc_nodes:
        h_anchor = mapping[w]
        host_nodes.add(h_anchor)
        # neighbours of anchor not already used by the core
        for nbr in G2.neighbors(h_anchor):
            host_nodes.add(nbr)

    # 4) build H as the *induced* sub‑graph of G2 on that node‑set -------------
    H: GraphType = G2.subgraph(host_nodes).copy()

    # 5) overwrite attributes on core‑hosts with G1’s concrete data -----------
    for p in core_nodes:
        h = mapping[p]
        H.nodes[h].update(G1.nodes[p])

    log.debug(
        "Fused graph has %d nodes and %d edges",
        H.number_of_nodes(),
        H.number_of_edges(),
    )
    return H
