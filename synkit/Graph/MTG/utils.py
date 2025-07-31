import networkx as nx
import copy
from typing import Tuple, Any, Optional, Sequence, Set, Union

OrderPair = Tuple[float, float]
MissingOrder = Tuple[Set[float], Set[float]]

GraphType = Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
TypesGHTuple = Tuple[Any, ...]  # e.g., ('H', 0, 0, 0, ['C'])


def _extract_leaf_candidates(orig_th: Tuple[Any, ...]) -> Tuple[TypesGHTuple, ...]:
    """
    Flatten one level: if an element is a sequence whose elements are themselves sequences,
    treat those inner sequences as candidates; otherwise the element itself is a candidate.
    """
    candidates: list[TypesGHTuple] = []
    for item in orig_th:
        if (
            isinstance(item, (list, tuple))
            and item
            and all(isinstance(inner, (list, tuple)) for inner in item)
        ):
            # e.g., (('H',...), ('H',...)) -> take inner tuples
            for inner in item:
                if isinstance(inner, (list, tuple)):
                    candidates.append(tuple(inner))
        else:
            if isinstance(item, (list, tuple)):
                candidates.append(tuple(item))
            else:
                # non-sequence fallback, wrap to tuple for uniformity
                candidates.append((item,))
    return tuple(candidates)


def normalize_hcount_and_typesGH(G: GraphType) -> GraphType:
    """
    Return a fresh copy of G where:
      * each node's `hcount` attribute is set to 0
      * each node's `typesGH` is processed as follows:
          1. Flatten one level so that nested tuples-of-tuples are expanded.
          2. Drop any tuple that contains a `set` anywhere.
          3. From the remaining tuples, keep only the first and last (if more than two).
          4. Zero indices 1 and 2 in each kept tuple (if they exist).
          5. If nothing remains after dropping, result is an empty tuple.

    :param G: input NetworkX graph
    :type G: nx.Graph or nx.DiGraph or nx.MultiGraph or nx.MultiDiGraph
    :returns: a new graph with normalized hcount and typesGH
    :rtype: same type as G
    :raises: TypeError if G is not a supported NetworkX graph or if typesGH is malformed.
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected a NetworkX graph, got {type(G)!r}")

    H = G.__class__()
    H.graph.update(copy.deepcopy(G.graph))

    for node, data in G.nodes(data=True):
        new_data = data.copy()
        new_data["hcount"] = 0

        orig_th = data.get("typesGH", ())
        if not isinstance(orig_th, (list, tuple)):
            raise TypeError(
                f"Node {node} has typesGH of unexpected type {type(orig_th)}; expected sequence"
            )

        # Step 1: flatten one level of nested tuples-of-tuples
        candidates = _extract_leaf_candidates(tuple(orig_th))

        # Step 2: drop any tuple containing a set
        filtered = [
            t for t in candidates if not any(isinstance(elem, set) for elem in t)
        ]

        # Step 3: select first and last (or whatever remains)
        if not filtered:
            selected: Tuple[TypesGHTuple, ...] = ()
        elif len(filtered) > 2:
            selected = (filtered[0], filtered[-1])
        else:
            selected = tuple(filtered)  # 1 or 2 elements

        # Step 4: zero indices 1 and 2
        normalized: list[TypesGHTuple] = []
        for inner in selected:
            if not isinstance(inner, (list, tuple)):
                raise TypeError(
                    f"Inner element of typesGH for node {node} is not tuple-like: {inner!r}"
                )
            inner_list = list(inner)
            if len(inner_list) > 1:
                inner_list[1] = 0
            if len(inner_list) > 2:
                inner_list[2] = 0
            normalized.append(tuple(inner_list))

        new_data["typesGH"] = tuple(normalized)
        H.add_node(node, **new_data)

    # Copy edges appropriately
    if G.is_multigraph():
        for u, v, key, edata in G.edges(keys=True, data=True):
            H.add_edge(u, v, key=key, **copy.deepcopy(edata))
    else:
        for u, v, edata in G.edges(data=True):
            H.add_edge(u, v, **copy.deepcopy(edata))

    return H


def extract_order_norm(
    order_sequence: Sequence[Union[OrderPair, MissingOrder]],
) -> Optional[OrderPair]:
    """
    Given a sequence of order tuples and/or placeholders (MissingOrder),
    return the normalized bond order as a 2-tuple:
      - left: the first tuple element 'a' in the sequence where not both parts are sets
      - right: the second tuple element 'b' in the sequence where not both parts are sets, scanning from the end

    The input sequence must have length >= 2.

    :param order_sequence: A sequence of order tuples or placeholders
    :type order_sequence: Sequence[tuple[float, float]] or Sequence[MissingOrder]
    :returns: A 2-tuple (left, right) if found; otherwise None
    :rtype: tuple[float, float] or None
    :raises ValueError: If sequence length is less than 2

    :example:
    >>> seq = [({1}, {2}), (3.0, 4.0), ({5}, {6}), (7.0, 8.0)]
    >>> extract_order_norm(seq)
    (3.0, 8.0)
    """
    if not isinstance(order_sequence, Sequence) or len(order_sequence) < 2:
        raise ValueError("order_sequence must be a sequence of length >= 2")

    left: Any = None
    right: Any = None
    # Find first non-placeholder for left
    for entry in order_sequence:
        a, b = entry
        if not (isinstance(a, set) and isinstance(b, set)):
            left = a
            break
    # Find last non-placeholder for right
    for entry in reversed(order_sequence):
        a, b = entry
        if not (isinstance(a, set) and isinstance(b, set)):
            right = b
            break
    if left is not None and right is not None:
        return (left, right)
    return None


def normalize_order(G: nx.Graph) -> nx.Graph:
    """
    Return a copy of the graph with each edge's 'order' attribute normalized.
    If an edge has an 'order' attribute that is a sequence of length >= 2,
    it is replaced by the 2-tuple returned by :func:`extract_order_norm`,
    if that function returns a non-None result.

    :param G: Input NetworkX graph
    :type G: nx.Graph, nx.DiGraph, nx.MultiGraph, or nx.MultiDiGraph
    :returns: A new graph of the same type with normalized edge 'order'
    :rtype: same as G
    :raises TypeError: If G is not a NetworkX graph

    :example:
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, order=[(1,2), ({3},{4}), (5,6)])
    >>> H = normalize_order(G)
    >>> H.edges[1,2]['order']
    (1, 6)
    """
    from copy import deepcopy

    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected a NetworkX graph, got {type(G)}")

    H = deepcopy(G)
    # Iterate edges appropriately
    if H.is_multigraph():
        edge_iter = H.edges(keys=True, data=True)
        for _, _, _, attr in edge_iter:
            order = attr.get("order")
            if isinstance(order, (list, tuple)) and len(order) >= 2:
                norm = extract_order_norm(order)
                if norm is not None:
                    attr["order"] = norm
    else:
        for _, _, attr in H.edges(data=True):
            order = attr.get("order")
            if isinstance(order, (list, tuple)) and len(order) >= 2:
                norm = extract_order_norm(order)
                if norm is not None:
                    attr["order"] = norm
    return H


# def normalize_hcount_and_typesGH(G):
#     """
#     Return a fresh copy of G where:
#       - each node's `hcount` attribute is set to 0
#       - in each tuple of `typesGH`, indices 1 and 2 are set to 0

#     :param G: input NetworkX graph
#     :type G: nx.Graph or nx.DiGraph or nx.MultiGraph or nx.MultiDiGraph
#     :returns: a new graph with normalized hcount and typesGH
#     :rtype: same type as G
#     :raises: TypeError if G is not a NetworkX graph

#     :example:
#     >>> G = nx.Graph()
#     >>> G.add_node(1, hcount=2, typesGH=(("C", 1, 2), ("O", 0, 1)))
#     >>> H = normalize_hcount_and_typesGH(G)
#     >>> H.nodes[1]['hcount']
#     0
#     >>> H.nodes[1]['typesGH']
#     (('C', 0, 0), ('O', 0, 0))
#     """
#     if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
#         raise TypeError(f"Expected NetworkX graph, got {type(G)}")

#     # Create empty graph of same class and copy graph-level attributes
#     H = G.__class__()
#     H.graph.update(copy.deepcopy(G.graph))

#     # Copy and normalize node data
#     for node, data in G.nodes(data=True):
#         new_data = data.copy()
#         new_data["hcount"] = 0
#         orig_th = data.get("typesGH", ())
#         new_th = []
#         for inner in orig_th:
#             inner_list = list(inner)
#             # Zero the aromatic slot (index 1) and the hcountGH slot (index 2)
#             if len(inner_list) > 1:
#                 inner_list[1] = 0
#             if len(inner_list) > 2:
#                 inner_list[2] = 0
#             new_th.append(tuple(inner_list))
#         new_data["typesGH"] = tuple(new_th)
#         H.add_node(node, **new_data)

#     # Copy edges (with keys for multigraphs)
#     if G.is_multigraph():
#         for u, v, key, edata in G.edges(keys=True, data=True):
#             H.add_edge(u, v, key=key, **copy.deepcopy(edata))
#     else:
#         for u, v, edata in G.edges(data=True):
#             H.add_edge(u, v, **copy.deepcopy(edata))

#     return H


def label_mtg_edges(G: nx.Graph, inplace: bool = False) -> nx.Graph:
    """
    Label each edge in the MTG graph with a boolean 'is_mtg' attribute based on two criteria:
    1. There are at least two steps where the standard order (order[0] - order[1]) is non-zero.
    2. The sum of all non-None standard orders is zero.

    :param G: Input MTG graph with 'order' history per edge
    :type G: nx.Graph or nx.DiGraph
    :param inplace: If True, modify G in place; otherwise work on a copy
    :type inplace: bool
    :returns: Graph with 'is_mtg' boolean attribute on each edge
    :rtype: same type as G
    :raises TypeError: If G is not a NetworkX Graph

    :example:
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> # Single change only -> less than 2 non-zero steps => False
    >>> G.add_edge(7,3, order=((1.0,1.0),(1.0,0)))
    >>> H = label_mtg_edges(G)
    >>> H.edges[7,3]['is_mtg']
    False
    >>> # Two-step equal but opposite changes -> sum zero and count>=2 => True
    >>> G = nx.Graph()
    >>> G.add_edge(2,1, order=((1.0,2.0),(2.0,1.0)))
    >>> H = label_mtg_edges(G)
    >>> H.edges[2,1]['is_mtg']
    True
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        raise TypeError(f"Expected a NetworkX Graph, got {type(G)}")
    graph = G if inplace else G.copy()
    for u, v, attr in graph.edges(data=True):
        history = attr.get("order")
        # Extract numeric standard orders
        std_vals: list[float] = []
        if isinstance(history, (list, tuple)) and len(history) >= 2:
            for entry in history:
                if (
                    isinstance(entry, tuple)
                    and len(entry) == 2
                    and all(isinstance(x, (int, float)) for x in entry)
                ):
                    std_vals.append(entry[0] - entry[1])
        # Apply criteria
        non_zero_count = sum(1 for v in std_vals if v != 0)
        total = sum(std_vals)
        attr["is_mtg"] = non_zero_count >= 2 and total == 0
    return graph


def compute_standard_order(G: nx.Graph, inplace: bool = False) -> nx.Graph:
    """
    Compute and assign the 'standard_order' attribute for each edge in the graph.
    'standard_order' is defined as the difference order[0] - order[1]
    for edges whose 'order' attribute is a 2-tuple of numeric values.

    :param G: Input NetworkX graph
    :type G: nx.Graph, nx.DiGraph, nx.MultiGraph, or nx.MultiDiGraph
    :param inplace: If True, modify G in-place; otherwise operate on a copy
    :type inplace: bool
    :returns: Graph with 'standard_order' attributes set
    :rtype: same type as G
    :raises TypeError: If G is not a NetworkX graph

    :example:
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edge(7, 3, order=(1.0, 0))
    >>> H = compute_standard_order(G)
    >>> H.edges[7,3]['standard_order']
    1.0
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected a NetworkX graph, got {type(G)}")

    graph = G if inplace else G.copy()
    if graph.is_multigraph():
        for u, v, key, attr in graph.edges(keys=True, data=True):
            order = attr.get("order")
            if isinstance(order, tuple) and len(order) == 2:
                a, b = order
                try:
                    attr["standard_order"] = a - b
                except Exception:
                    attr["standard_order"] = None
    else:
        for u, v, attr in graph.edges(data=True):
            order = attr.get("order")
            if isinstance(order, tuple) and len(order) == 2:
                a, b = order
                try:
                    attr["standard_order"] = a - b
                except Exception:
                    attr["standard_order"] = None
    return graph


# def extract_order_norm(order_tuple):
#     """
#     Given a sequence of four 2-element tuples (order data), return the normalized order:
#       - left: first element of the first tuple that is not both sets
#       - right: second element of the last tuple that is not both sets

#     :param order_tuple: tuple of four 2-element tuples
#     :type order_tuple: tuple(tuple, tuple, tuple, tuple)
#     :returns: normalized (left, right) or None if not found
#     :rtype: tuple or None
#     :raises: ValueError if order_tuple is not length 4

#     :example:
#     >>> ot = (({1}, {2}), (3, 4), ({5}, {6}), (7, 8))
#     >>> extract_order_norm(ot)
#     (3, 8)
#     """
#     if not (isinstance(order_tuple, tuple) and len(order_tuple) == 4):
#         raise ValueError("order_tuple must be a tuple of length 4")

#     left = None
#     right = None
#     # Find first non-all-set tuple for left
#     for a, b in order_tuple:
#         if not (isinstance(a, set) and isinstance(b, set)):
#             left = a
#             break
#     # Find last non-all-set tuple for right
#     for a, b in reversed(order_tuple):
#         if not (isinstance(a, set) and isinstance(b, set)):
#             right = b
#             break
#     return (left, right) if (left is not None and right is not None) else None


# def normalize_order(G):
#     """
#     Return a copy of G with edge attribute 'order' normalized.
#     For each edge, if the 'order' attribute is a 4-tuple, replace it with the
#     normalized 2-tuple returned by extract_order_norm.

#     :param G: input NetworkX graph
#     :type G: nx.Graph or nx.DiGraph or nx.MultiGraph or nx.MultiDiGraph
#     :returns: a new graph with normalized edge orders
#     :rtype: same type as G
#     :raises: TypeError if G is not a NetworkX graph

#     :example:
#     >>> G = nx.Graph()
#     >>> G.add_edge(1, 2, order=((1,2), ({3},{4}), ({5},{6}), (7,8)))
#     >>> H = copy_and_normalize_order(G)
#     >>> H.edges[1,2]['order']
#     (1, 8)
#     """
#     if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
#         raise TypeError(f"Expected NetworkX graph, got {type(G)}")
#     H = G.copy()
#     for _, _, _, attr in (
#         H.edges(keys=True, data=True)
#         if H.is_multigraph()
#         else [(u, v, None, attr) for u, v, attr in H.edges(data=True)]
#     ):
#         order = attr.get("order")
#         if isinstance(order, tuple) and len(order) == 4:
#             norm = extract_order_norm(order)
#             if norm is not None:
#                 attr["order"] = norm
#     return H
