import networkx as nx
import copy


def normalize_hcount_and_typesGH(G):
    """
    Return a fresh copy of G where:
      - each node's `hcount` attribute is set to 0
      - in each tuple of `typesGH`, indices 1 and 2 are set to 0

    :param G: input NetworkX graph
    :type G: nx.Graph or nx.DiGraph or nx.MultiGraph or nx.MultiDiGraph
    :returns: a new graph with normalized hcount and typesGH
    :rtype: same type as G
    :raises: TypeError if G is not a NetworkX graph

    :example:
    >>> G = nx.Graph()
    >>> G.add_node(1, hcount=2, typesGH=(("C", 1, 2), ("O", 0, 1)))
    >>> H = normalize_hcount_and_typesGH(G)
    >>> H.nodes[1]['hcount']
    0
    >>> H.nodes[1]['typesGH']
    (('C', 0, 0), ('O', 0, 0))
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")

    # Create empty graph of same class and copy graph-level attributes
    H = G.__class__()
    H.graph.update(copy.deepcopy(G.graph))

    # Copy and normalize node data
    for node, data in G.nodes(data=True):
        new_data = data.copy()
        new_data["hcount"] = 0
        orig_th = data.get("typesGH", ())
        new_th = []
        for inner in orig_th:
            inner_list = list(inner)
            # Zero the aromatic slot (index 1) and the hcountGH slot (index 2)
            if len(inner_list) > 1:
                inner_list[1] = 0
            if len(inner_list) > 2:
                inner_list[2] = 0
            new_th.append(tuple(inner_list))
        new_data["typesGH"] = tuple(new_th)
        H.add_node(node, **new_data)

    # Copy edges (with keys for multigraphs)
    if G.is_multigraph():
        for u, v, key, edata in G.edges(keys=True, data=True):
            H.add_edge(u, v, key=key, **copy.deepcopy(edata))
    else:
        for u, v, edata in G.edges(data=True):
            H.add_edge(u, v, **copy.deepcopy(edata))

    return H


def extract_order_norm(order_tuple):
    """
    Given a sequence of four 2-element tuples (order data), return the normalized order:
      - left: first element of the first tuple that is not both sets
      - right: second element of the last tuple that is not both sets

    :param order_tuple: tuple of four 2-element tuples
    :type order_tuple: tuple(tuple, tuple, tuple, tuple)
    :returns: normalized (left, right) or None if not found
    :rtype: tuple or None
    :raises: ValueError if order_tuple is not length 4

    :example:
    >>> ot = (({1}, {2}), (3, 4), ({5}, {6}), (7, 8))
    >>> extract_order_norm(ot)
    (3, 8)
    """
    if not (isinstance(order_tuple, tuple) and len(order_tuple) == 4):
        raise ValueError("order_tuple must be a tuple of length 4")

    left = None
    right = None
    # Find first non-all-set tuple for left
    for a, b in order_tuple:
        if not (isinstance(a, set) and isinstance(b, set)):
            left = a
            break
    # Find last non-all-set tuple for right
    for a, b in reversed(order_tuple):
        if not (isinstance(a, set) and isinstance(b, set)):
            right = b
            break
    return (left, right) if (left is not None and right is not None) else None


def normalize_order(G):
    """
    Return a copy of G with edge attribute 'order' normalized.
    For each edge, if the 'order' attribute is a 4-tuple, replace it with the
    normalized 2-tuple returned by extract_order_norm.

    :param G: input NetworkX graph
    :type G: nx.Graph or nx.DiGraph or nx.MultiGraph or nx.MultiDiGraph
    :returns: a new graph with normalized edge orders
    :rtype: same type as G
    :raises: TypeError if G is not a NetworkX graph

    :example:
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, order=((1,2), ({3},{4}), ({5},{6}), (7,8)))
    >>> H = copy_and_normalize_order(G)
    >>> H.edges[1,2]['order']
    (1, 8)
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    H = G.copy()
    for _, _, _, attr in (
        H.edges(keys=True, data=True)
        if H.is_multigraph()
        else [(u, v, None, attr) for u, v, attr in H.edges(data=True)]
    ):
        order = attr.get("order")
        if isinstance(order, tuple) and len(order) == 4:
            norm = extract_order_norm(order)
            if norm is not None:
                attr["order"] = norm
    return H
