import networkx as nx
from typing import Any, Dict, List, Tuple


def print_graph_attributes(G: nx.Graph) -> None:
    """
    Print all node and edge attributes from a NetworkX graph.

    Parameters:
        G (nx.Graph): A NetworkX graph (Graph, DiGraph, MultiGraph, etc.).
    """
    print("🔹 Nodes and their attributes:")
    for node, attr in G.nodes(data=True):
        print(f"  Node {node}: {attr}")

    print("\n🔸 Edges and their attributes:")
    if G.is_multigraph():
        for u, v, key, attr in G.edges(data=True, keys=True):
            print(f"  Edge {u}-{v} (key={key}): {attr}")
    else:
        for u, v, attr in G.edges(data=True):
            print(f"  Edge {u}-{v}: {attr}")


def remove_wildcard_nodes(G: nx.Graph, inplace: bool = True) -> nx.Graph:
    """
    Remove all wildcard nodes from the graph.

    A wildcard node is identified by having its 'element' attribute equal to '*'.

    Parameters
    ----------
    G : nx.Graph
        The input graph from which wildcard nodes will be removed.
    inplace : bool, optional
        If True, modify the input graph in place and return it.
        If False (default), a copy of the graph is created and the removal is applied to the copy.

    Returns
    -------
    nx.Graph
        The graph after removing all wildcard nodes.
    """
    if not inplace:
        G = G.copy()

    # Identify and remove wildcard nodes
    wildcard_nodes = [
        node for node, data in G.nodes(data=True) if data.get("element") == "*"
    ]
    G.remove_nodes_from(wildcard_nodes)
    return G


def add_wildcard_subgraph_for_unmapped(
    G: nx.Graph,
    L: nx.Graph,
    mapping: Dict[Any, Any],
    edge_keys: List[str] = ["order"],
    inplace: bool = False,
) -> Tuple[nx.Graph, Dict[Any, Any]]:
    """
    Extend G with wildcard nodes/edges for every L-node not already mapped,
    preserving original L->G mapping and returning the full mapping.

    Parameters
    ----------
    G : nx.Graph
        Target graph. If inplace=False (default), operates on a shallow copy.
    L : nx.Graph
        Pattern/reference graph containing full nodes and edges.
    mapping : Dict[L_node, G_node]
        Partial mapping from pattern L nodes to graph G nodes.
    edge_keys : List[str], optional
        Edge attributes to copy (first element if list/tuple). Default ['order'].
    inplace : bool, optional
        If True, modify G in place; otherwise modify a copy.

    Returns
    -------
    G_ext : nx.Graph
        Extended graph with added wildcard nodes and edges.
    full_map : Dict[L_node, G_node]
        Combined L->G mapping, original plus newly added wildcard nodes.
    """
    # Use a copy if not in-place
    G_ext = G if inplace else G.copy()

    # Start from L->G mapping
    L_to_G: Dict[Any, Any] = mapping.copy()

    # Identify unmapped L nodes
    unmapped = set(L.nodes()) - set(L_to_G.keys())

    # Prepare new node IDs
    next_id = max(G_ext.nodes, default=-1) + 1

    # Add wildcard nodes for each unmapped L node
    for l_node in unmapped:
        attrs = L.nodes[l_node].copy()
        attrs["element"] = "*"
        attrs.setdefault("atom_map", next_id)
        G_ext.add_node(next_id, **attrs)
        L_to_G[l_node] = next_id
        next_id += 1

    # Add edges matching pattern L, mapping endpoints via L_to_G
    for u_l, v_l, data in L.edges(data=True):
        g_u = L_to_G.get(u_l)
        g_v = L_to_G.get(v_l)
        if g_u is None or g_v is None:
            continue
        edge_data: Dict[Any, Any] = {}
        for key in edge_keys:
            if key in data:
                val = data[key]
                edge_data[key] = val[0] if isinstance(val, (list, tuple)) else val
        G_ext.add_edge(g_u, g_v, **edge_data)

    # full mapping now includes original and new nodes
    full_map = L_to_G
    return G_ext, full_map


def clean_graph_keep_largest_component(graph: nx.Graph) -> nx.Graph:
    """
    Return a shallow copy of the input graph with all edges removed
    where the 'standard_order' attribute is exactly 0, then retain only
    the largest connected component.

    Parameters
    ----------
    graph : nx.Graph
        The input molecular graph.

    Returns
    -------
    nx.Graph
        A modified copy of the original graph with specified edges removed
        and only the largest connected component preserved.
    """
    # Work on a copy to avoid side effects
    G = graph.copy()

    # Remove edges with 'standard_order' == 0
    to_remove = [
        (u, v) for u, v, data in G.edges(data=True) if data.get("standard_order") == 0
    ]
    G.remove_edges_from(to_remove)

    # If no nodes remain, return the empty graph
    if G.number_of_nodes() == 0:
        return G

    # Identify the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)

    # Return the subgraph induced by the largest component
    return G.subgraph(largest_cc).copy()
