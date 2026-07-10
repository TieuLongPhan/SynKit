"""
Weisfeiler-Lehman colour refinement: 1-WL and selective 2-WL.

The SLAP engine relies on 1-dimensional Weisfeiler-Lehman (1-WL) colour
refinement to tell atoms apart before branching. 1-WL is fast but limited: there
are non-isomorphic local environments it cannot separate (the textbook example
is a 6-cycle versus two disjoint triangles -- 1-WL assigns every vertex the same
colour in both). When 1-WL leaves atoms in the same colour class that are not
actually equivalent, the matcher must branch over all of them, which is wasteful.

2-WL (here the 2-dimensional Folklore WL, 2-FWL) colours *ordered pairs* of
vertices and is strictly stronger. It is also more expensive: O(n^2) memory and
O(n^3) work per iteration. This module therefore applies it **selectively** --
only to the atoms 1-WL left ambiguous -- so the extra cost is paid only where it
can help.

Public functions
----------------
``wl_node_colors`` / ``wl_graph_hash``
    1-WL node colouring and graph-level invariant.
``two_wl_pair_colors`` / ``two_wl_graph_hash``
    2-FWL pair colouring and the strictly stronger graph-level invariant.
``selective_two_wl_refine``
    Run 1-WL, then refine only the still-ambiguous colour classes with 2-FWL.
    The result is never coarser than 1-WL.
"""

from collections import defaultdict


def _canonicalize(values):
    """Map arbitrary hashable values to ints 0,1,2,... in first-appearance order."""
    remap = {}
    out = []
    for v in values:
        if v not in remap:
            remap[v] = len(remap)
        out.append(remap[v])
    return out


def _num_classes(colors):
    return len(set(colors))


def _edge(graph, i, j):
    return graph.get(i, {}).get(j, 0)


def wl_node_colors(graph, n, init=None, max_iter=None):
    """1-WL stable node colours.

    Parameters
    ----------
    graph : dict[int, dict[int, number]]
        Adjacency with edge weights; ``graph[i][j]`` is the bond order.
    n : int
        Number of nodes (indices ``0..n-1``).
    init : sequence, optional
        Initial colours (defaults to all-equal).
    max_iter : int, optional
        Iteration cap (defaults to ``n``).

    Returns
    -------
    list[int]
        Canonical integer colours, stable under further 1-WL refinement.
    """
    colors = _canonicalize(init if init is not None else [0] * n)
    if max_iter is None:
        max_iter = n
    for _ in range(max_iter):
        sigs = []
        for i in range(n):
            nbr = sorted((colors[j], w) for j, w in graph.get(i, {}).items() if j != i)
            sigs.append((colors[i], tuple(nbr)))
        new = _canonicalize(sigs)
        if _num_classes(new) == _num_classes(colors):
            return colors
        colors = new
    return colors


def wl_graph_hash(graph, n, init=None):
    """Graph-level 1-WL invariant (sorted multiset of stable node colours)."""
    colors = wl_node_colors(graph, n, init)
    return hash(tuple(sorted(colors)))


def two_wl_pair_colors(graph, n, init=None, max_iter=None):
    """2-FWL stable colours for every ordered pair ``(i, j)``.

    Returns
    -------
    dict[tuple[int, int], int]
        Canonical integer colour for each ordered pair.
    """
    node_colors = wl_node_colors(graph, n, init)
    if max_iter is None:
        max_iter = n

    keys = [(i, j) for i in range(n) for j in range(n)]
    raw = [(node_colors[i], node_colors[j], _edge(graph, i, j)) for (i, j) in keys]
    pair = dict(zip(keys, _canonicalize(raw)))

    for _ in range(max_iter):
        raw = []
        for i, j in keys:
            agg = sorted((pair[(i, k)], pair[(k, j)]) for k in range(n))
            raw.append((pair[(i, j)], tuple(agg)))
        new_vals = _canonicalize(raw)
        new = dict(zip(keys, new_vals))
        if _num_classes(new_vals) == _num_classes(list(pair.values())):
            return new
        pair = new
    return pair


def two_wl_graph_hash(graph, n, init=None):
    """Graph-level 2-FWL invariant (strictly stronger than :func:`wl_graph_hash`)."""
    pair = two_wl_pair_colors(graph, n, init)
    return hash(tuple(sorted(pair.values())))


def two_wl_node_colors(graph, n, init=None, targets=None):
    """Refine node colours using 2-FWL pair colours.

    Each node's refined colour combines its 1-WL colour, its diagonal pair
    colour, and the multiset of pair colours in its row. If ``targets`` is given,
    only those nodes receive the refined colour (others keep their 1-WL colour),
    which is the "selective" application.

    The result is never coarser than the input 1-WL colouring.
    """
    node_colors = wl_node_colors(graph, n, init)
    pair = two_wl_pair_colors(graph, n, init)

    if targets is None:
        target_set = set(range(n))
    else:
        target_set = set(targets)

    sigs = []
    for i in range(n):
        if i in target_set:
            row = tuple(sorted(pair[(i, j)] for j in range(n)))
            sigs.append(("t", node_colors[i], pair[(i, i)], row))
        else:
            sigs.append(("o", node_colors[i]))

    refined = _canonicalize(sigs)
    # Combine with the 1-WL colouring so the result can only ever be finer.
    return _canonicalize(list(zip(node_colors, refined)))


def selective_two_wl_refine(graph, n, colors=None, max_nodes=200):
    """Refine 1-WL colours with 2-FWL, but only on still-ambiguous classes.

    Parameters
    ----------
    graph : dict[int, dict[int, number]]
        Adjacency with edge weights.
    n : int
        Number of nodes.
    colors : sequence, optional
        Initial colours (defaults to all-equal).
    max_nodes : int, optional
        Skip the (cubic) 2-FWL pass when ``n`` exceeds this, returning the plain
        1-WL colouring. Guards against pathological cost on large molecules.

    Returns
    -------
    list[int]
        Refined colours, never coarser than 1-WL.
    """
    c1 = wl_node_colors(graph, n, colors)

    class_sizes = defaultdict(int)
    for c in c1:
        class_sizes[c] += 1
    ambiguous = [i for i in range(n) if class_sizes[c1[i]] > 1]

    if not ambiguous or n > max_nodes:
        return c1

    return two_wl_node_colors(graph, n, init=c1, targets=ambiguous)
