from __future__ import annotations

from typing import Optional, Union, Tuple, Iterable, Dict

import networkx as nx
from synkit.IO import smiles_to_graph


GraphLike = Union["nx.Graph", "nx.DiGraph"]  # type: ignore[misc]


def _get_graph(obj: Union[str, GraphLike]) -> Optional[GraphLike]:
    """
    Convenience: accept either an RSMI string or a pre-built NetworkX graph.
    """

    if isinstance(obj, str):
        return smiles_to_graph(obj)

    # assume it's already a graph
    if hasattr(obj, "nodes") and hasattr(obj, "edges"):
        return obj  # type: ignore[return-value]

    return None


# ------------------------------
# Pure graph-based heuristics
# ------------------------------


def h_graph_size_diff(
    source: Union[str, GraphLike], target: Union[str, GraphLike]
) -> float:
    """
    Heuristic: difference in basic graph size (nodes + edges).

    Score = |V_s - V_t| + 0.5 * |E_s - E_t|

    Lower values = more similar in overall size/connectivity.
    """
    if nx is None:
        return 1e9

    gs = _get_graph(source)
    gt = _get_graph(target)
    if gs is None or gt is None:
        return 1e9

    vs, es = gs.number_of_nodes(), gs.number_of_edges()
    vt, et = gt.number_of_nodes(), gt.number_of_edges()

    return float(abs(vs - vt) + 0.5 * abs(es - et))


def _degree_histogram(g: GraphLike) -> Dict[int, int]:
    degs = [d for _, d in g.degree()]
    hist: Dict[int, int] = {}
    for d in degs:
        hist[d] = hist.get(d, 0) + 1
    return hist


def h_degree_hist_diff(
    source: Union[str, GraphLike], target: Union[str, GraphLike]
) -> float:
    """
    Heuristic: L1 distance between degree histograms.

    Builds histograms over node degrees and sums |count_s(d) - count_t(d)|.
    This captures global branching/connectivity differences.
    """
    if nx is None:
        return 1e9

    gs = _get_graph(source)
    gt = _get_graph(target)
    if gs is None or gt is None:
        return 1e9

    hs = _degree_histogram(gs)
    ht = _degree_histogram(gt)

    degrees = set(hs) | set(ht)
    diff = 0
    for d in degrees:
        diff += abs(hs.get(d, 0) - ht.get(d, 0))

    return float(diff)


def _element_multiset(g: GraphLike, element_key: str = "element") -> Dict[str, int]:
    """
    Count occurrences of atomic elements from node attributes.

    Falls back to empty dict if attribute is missing.
    """
    counts: Dict[str, int] = {}
    for _, data in g.nodes(data=True):
        el = data.get(element_key)
        if el is None:
            continue
        counts[el] = counts.get(el, 0) + 1
    return counts


def h_element_multiset_diff(
    source: Union[str, GraphLike],
    target: Union[str, GraphLike],
    element_key: str = "element",
) -> float:
    """
    Heuristic: L1 distance between element multisets.

    Uses node attribute ``element_key`` (default: ``"element"``) to build
    element count dictionaries and sums the absolute count differences.

    This is a graph-level analogue of heavy-atom/element composition.
    """
    if nx is None:
        return 1e9

    gs = _get_graph(source)
    gt = _get_graph(target)
    if gs is None or gt is None:
        return 1e9

    cs = _element_multiset(gs, element_key=element_key)
    ct = _element_multiset(gt, element_key=element_key)

    elements = set(cs) | set(ct)
    diff = 0
    for el in elements:
        diff += abs(cs.get(el, 0) - ct.get(el, 0))

    return float(diff)


def h_radius_diameter_diff(
    source: Union[str, GraphLike], target: Union[str, GraphLike]
) -> float:
    """
    Heuristic: difference in graph radius and diameter.

    For each graph, compute:

    - radius r(G)   (min eccentricity)
    - diameter D(G) (max eccentricity)

    Score = |r_s - r_t| + |D_s - D_t|

    For disconnected graphs, we work on the largest connected component.
    """
    if nx is None:
        return 1e9

    gs = _get_graph(source)
    gt = _get_graph(target)
    if gs is None or gt is None:
        return 1e9

    def _core(g: GraphLike) -> GraphLike:
        if nx.is_connected(g):  # type: ignore[arg-type]
            return g
        # largest CC
        components = list(nx.connected_components(g))  # type: ignore[arg-type]
        largest = max(components, key=len)
        return g.subgraph(largest).copy()

    try:
        cs = _core(gs)
        ct = _core(gt)

        rs = nx.radius(cs)
        rt = nx.radius(ct)

        ds = nx.diameter(cs)
        dt = nx.diameter(ct)
    except Exception:
        # If eccentricities fail for some reason (e.g., directed graphs),
        # fall back to a large penalty.
        return 1e9

    return float(abs(rs - rt) + abs(ds - dt))


def h_cycle_basis_diff(
    source: Union[str, GraphLike], target: Union[str, GraphLike]
) -> float:
    """
    Heuristic: difference in cycle basis (number + length distribution).

    Uses :func:`networkx.cycle_basis` on the largest connected component
    and compares:

    - number of cycles
    - histogram of cycle lengths

    Score = |#cycles_s - #cycles_t|
          + sum_L |count_s(L) - count_t(L)|
    """
    if nx is None:
        return 1e9

    gs = _get_graph(source)
    gt = _get_graph(target)
    if gs is None or gt is None:
        return 1e9

    def _cycle_stats(g: GraphLike) -> Dict[int, int]:
        if not nx.is_connected(g):  # type: ignore[arg-type]
            components = list(nx.connected_components(g))  # type: ignore[arg-type]
            largest = max(components, key=len)
            g = g.subgraph(largest).copy()
        cycles = nx.cycle_basis(g)
        hist: Dict[int, int] = {}
        for c in cycles:
            L = len(c)
            hist[L] = hist.get(L, 0) + 1
        # store total count in key 0 for convenience
        hist[0] = len(cycles)
        return hist

    hs = _cycle_stats(gs)
    ht = _cycle_stats(gt)

    lengths = set(hs) | set(ht)
    diff = 0
    for L in lengths:
        diff += abs(hs.get(L, 0) - ht.get(L, 0))

    return float(diff)


def h_spectral_l2(
    source: Union[str, GraphLike], target: Union[str, GraphLike], k: int = 8
) -> float:
    """
    Heuristic: L2 distance between the top-k eigenvalues of the
    normalized Laplacian spectrum.

    Steps:
      1. Build normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
      2. Compute sorted eigenvalues (ascending).
      3. Take the first ``k`` eigenvalues (padded with 1.0 if needed).
      4. Score = sqrt(sum_i (λ_s[i] - λ_t[i])^2).

    This captures more global shape/topology differences.
    """
    if nx is None:
        return 1e9

    gs = _get_graph(source)
    gt = _get_graph(target)
    if gs is None or gt is None:
        return 1e9

    try:
        import numpy as np
    except ImportError:
        return 1e9

    def _topk_eigs(g: GraphLike, k: int) -> np.ndarray:
        # Use largest connected component for stability.
        if not nx.is_connected(g):  # type: ignore[arg-type]
            components = list(nx.connected_components(g))  # type: ignore[arg-type]
            largest = max(components, key=len)
            g = g.subgraph(largest).copy()

        L = nx.normalized_laplacian_matrix(g).astype(float)
        # For small k relative to n, eigsh would be better; for now, numpy eig.
        vals = np.linalg.eigvalsh(L.A)
        vals = np.sort(vals)
        if len(vals) >= k:
            return vals[:k]
        # pad with 1.0 (max eigenvalue of normalized Laplacian) for short spectra
        padded = np.ones(k)
        padded[: len(vals)] = vals
        return padded

    try:
        evs = _topk_eigs(gs, k)
        evt = _topk_eigs(gt, k)
    except Exception:
        return 1e9

    diff = float(((evs - evt) ** 2).sum() ** 0.5)
    return diff


# ------------------------------
# Example combined graph heuristic
# ------------------------------


def h_graph_combo(
    source: Union[str, GraphLike], target: Union[str, GraphLike]
) -> float:
    """
    Combined graph-based heuristic.

    Currently:

      0.5 * h_graph_size_diff
    + 0.5 * h_degree_hist_diff
    + 0.5 * h_element_multiset_diff
    + 0.2 * h_cycle_basis_diff

    Adjust weights to taste; idea is to approximate global structural
    + element-composition similarity using only graph-level information.
    """
    size_term = h_graph_size_diff(source, target)
    deg_term = h_degree_hist_diff(source, target)
    elem_term = h_element_multiset_diff(source, target)
    cyc_term = h_cycle_basis_diff(source, target)

    # If any term is an obvious "failed to compute" (1e9), propagate that.
    if any(t >= 1e8 for t in (size_term, deg_term, elem_term, cyc_term)):
        return 1e9

    return float(0.5 * size_term + 0.5 * deg_term + 0.5 * elem_term + 0.2 * cyc_term)
