from __future__ import annotations
import networkx as nx
from collections import defaultdict
from typing import Iterable, Mapping, List, Dict, Any, Tuple, TypeVar, Optional, Set

# ==============================================================================
# Type Aliases
# ==============================================================================

NodeId = int
ChargeTuple = Tuple[int | None, int | None]
Node = Tuple[NodeId, Dict[str, Any]]  # (id, attribute-dict)
Edge = Tuple[NodeId, NodeId, Dict[str, Any]]  # (u, v, attribute-dict)
MappingList = List[Dict[NodeId, NodeId]]

# ==============================================================================
# Public Groupoid Operations
# ==============================================================================


def charge_tuple(attrs: Mapping[str, Any]) -> ChargeTuple:
    """Extract the 2-tuple charge signature from node attributes.

    Supports both:
      - attrs['charges'] as a tuple of two ints
      - attrs['typesGH'] as an iterable of two tuples where the 3rd element
        in each is an int charge.

    Returns
    -------
    (charge0, charge1) or (None, None) if unavailable
    """
    # Case 1: direct 'charges' field
    ch = attrs.get("charges")
    if isinstance(ch, tuple) and len(ch) == 2:
        return ch[0], ch[1]

    # Case 2: 'typesGH' field
    tg = attrs.get("typesGH")
    if isinstance(tg, (list, tuple)) and len(tg) >= 2:
        try:
            return tg[0][3], tg[1][3]
        except Exception:
            pass

    return None, None


def node_constraint(
    nodes1: Iterable[Node],
    nodes2: Iterable[Node],
) -> Dict[NodeId, List[NodeId]]:
    """Compute candidate node mappings based on element and groupoid charge rule.

    For each node v1 in nodes1 and v2 in nodes2, v2 is a candidate if:
      1. v1.attrs['element'] == v2.attrs['element'], and
      2. charge_tuple(v1)[1] == charge_tuple(v2)[0].

    Returns
    -------
    mapping : dict mapping each G1 node_id to a list of G2 node_ids
    """
    # Index G2 by (element, first_charge)
    idx_g2: Dict[Tuple[Any, Any], List[NodeId]] = defaultdict(list)
    for n2_id, attrs2 in nodes2:
        elem2 = attrs2.get("element")
        first_charge, _ = charge_tuple(attrs2)
        if elem2 is not None:
            idx_g2[(elem2, first_charge)].append(n2_id)

    # Build mapping for G1
    mapping: Dict[NodeId, List[NodeId]] = {}
    for n1_id, attrs1 in nodes1:
        elem1 = attrs1.get("element")
        _, second_charge = charge_tuple(attrs1)
        mapping[n1_id] = idx_g2.get((elem1, second_charge), [])

    return mapping


# def edge_constraint(
#     edges1: Iterable[Edge],
#     edges2: Iterable[Edge],
#     node_mapping: Mapping[NodeId, List[NodeId]] | None = None,
# ) -> MappingList:
#     """Find disjoint edge-merge mappings under the groupoid order rule.

#     For each edge e1=(u1,v1,attrs1) and e2=(u2,v2,attrs2):
#       - attrs1['order'][1] == attrs2['order'][0]
#       - if node_mapping provided: u2 in node_mapping[u1] and v2 in node_mapping[v1]

#     Performs a maximum set-packing via backtracking so that returned mappings
#     have no shared endpoints on either side.

#     Returns
#     -------
#     List of node mappings (dict G1_node -> G2_node) for each valid composition.
#     """
#     # 1. Build candidate pairs
#     candidates: List[Tuple[Edge, Edge]] = []
#     for u1, v1, a1 in edges1:
#         order1 = a1.get("order", (None, None))
#         if len(order1) < 2:
#             continue
#         a2_val = order1[1]
#         for u2, v2, a2 in edges2:
#             order2 = a2.get("order", (None, None))
#             if len(order2) < 2:
#                 continue
#             b1_val = order2[0]
#             if a2_val != b1_val:
#                 continue
#             if node_mapping is not None:
#                 if u2 not in node_mapping.get(u1, []):
#                     continue
#                 if v2 not in node_mapping.get(v1, []):
#                     continue
#             candidates.append(((u1, v1, a1), (u2, v2, a2)))

#     # 2. Backtracking for disjoint sets
#     pair_sets: List[List[Tuple[Edge, Edge]]] = []

#     def _backtrack(
#         chosen: List[Tuple[Edge, Edge]], remaining: List[Tuple[Edge, Edge]]
#     ) -> None:
#         if not remaining:
#             if chosen:
#                 pair_sets.append(chosen.copy())
#             return
#         first, *rest = remaining
#         e1, e2 = first
#         # include: drop conflicts
#         filtered = [p for p in rest if p[0] != e1 and p[1] != e2]
#         _backtrack(chosen + [first], filtered)
#         # exclude
#         _backtrack(chosen, rest)

#     _backtrack([], candidates)

#     # 3. Convert to mapping list and dedupe
#     mapping_list: MappingList = []
#     seen: Set[frozenset] = set()
#     for matching in pair_sets:
#         m: Dict[NodeId, NodeId] = {}
#         for (u1, v1, _), (u2, v2, _) in matching:
#             m[u1] = u2
#             m[v1] = v2
#         key = frozenset(m.items())
#         if key not in seen:
#             seen.add(key)
#             mapping_list.append(m)
#     return mapping_list


# from typing import Iterable, Mapping, List, Dict, Any, Tuple, TypeVar, Optional, Set

# # Type variables
# NodeId = TypeVar('NodeId')
# # Edge represented as (node1, node2, attributes)
# Edge = Tuple[NodeId, NodeId, Mapping[str, Any]]
# # Resulting list of node mappings
# MappingList = List[Dict[NodeId, NodeId]]


# def edge_constraint(
#     edges1: Iterable[Edge],
#     edges2: Iterable[Edge],
#     node_mapping: Optional[Mapping[NodeId, List[NodeId]]] = None,
# ) -> MappingList:
#     """
#     Compute maximum common subgraph mappings under groupoid order constraints.

#     Identifies edge-to-edge correspondences (e1 -> e2) satisfying:
#       - e1.attrs['order'][1] == e2.attrs['order'][0]
#       - If `node_mapping` is provided, then u2 in node_mapping[u1] and v2 in node_mapping[v1].

#     Seeks the largest set of disjoint edge pairs (no shared endpoints
#     on either graph) via backtracking, and returns only those mappings
#     that achieve maximal edge count.

#     Parameters
#     ----------
#     edges1 : Iterable[Edge]
#         Edges of first graph as (u, v, attrs).
#     edges2 : Iterable[Edge]
#         Edges of second graph as (u, v, attrs).
#     node_mapping : Optional[Mapping[NodeId, List[NodeId]]], optional
#         Pre-existing node correspondences (default: None).

#     Returns
#     -------
#     MappingList
#         A list of node-to-node mappings for each maximum matching.
#         Each mapping dict maps nodes of graph1 to nodes of graph2.
#     """
#     # Step 1: build all candidate edge-pairings
#     candidates: List[Tuple[Edge, Edge]] = []
#     for u1, v1, a1 in edges1:
#         order1 = a1.get('order', (None, None))
#         if len(order1) < 2:
#             continue
#         needed = order1[1]
#         for u2, v2, a2 in edges2:
#             order2 = a2.get('order', (None, None))
#             if len(order2) < 2 or order2[0] != needed:
#                 continue
#             if node_mapping:
#                 if u2 not in node_mapping.get(u1, []):
#                     continue
#                 if v2 not in node_mapping.get(v1, []):
#                     continue
#             candidates.append(((u1, v1, a1), (u2, v2, a2)))

#     # Step 2: backtracking to find disjoint sets of edge-pairs maximizing size
#     best_sets: List[List[Tuple[Edge, Edge]]] = []
#     max_size: int = 0

#     def _backtrack(
#         chosen: List[Tuple[Edge, Edge]],
#         remaining: List[Tuple[Edge, Edge]]
#     ) -> None:
#         nonlocal best_sets, max_size
#         if not remaining:
#             count = len(chosen)
#             if count > 0:
#                 if count > max_size:
#                     max_size = count
#                     best_sets = [chosen.copy()]
#                 elif count == max_size:
#                     best_sets.append(chosen.copy())
#             return
#         first, *rest = remaining
#         (u1, v1, _), (u2, v2, _) = first
#         # Include first: remove any pairs sharing endpoints
#         filtered = []
#         for (x1, y1, _), (x2, y2, _) in rest:
#             if x1 in (u1, v1) or y1 in (u1, v1) or x2 in (u2, v2) or y2 in (u2, v2):
#                 continue
#             filtered.append(((x1, y1, _), (x2, y2, _)))
#         _backtrack(chosen + [first], filtered)
#         # Exclude first
#         _backtrack(chosen, rest)

#     _backtrack([], candidates)

#     # Step 3: convert best_sets to unique node mappings
#     mapping_list: MappingList = []
#     seen: Set[frozenset] = set()
#     for match in best_sets:
#         m: Dict[NodeId, NodeId] = {}
#         for (u1, v1, _), (u2, v2, _) in match:
#             m[u1] = u2
#             m[v1] = v2
#         key = frozenset(m.items())
#         if key not in seen:
#             seen.add(key)
#             mapping_list.append(m)

#     return mapping_list


# ---------------------------------------------------------------------------
# Helper – original back‑tracking implementation (kept for reference / fallback)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Back‑tracking implementation (legacy / fallback)
# ---------------------------------------------------------------------------


def _edge_constraint_backtracking(
    edges1: Iterable[Edge],
    edges2: Iterable[Edge],
    node_mapping: Optional[Mapping[NodeId, List[NodeId]]] = None,
    *,
    mcs: bool = True,
) -> MappingList:
    """Explicit set‑packing search.

    Parameters
    ----------
    mcs : bool, default ``True``
        If ``True`` return **only** mappings that maximise the number of matched
        edges (MCS).  If ``False`` return *all* disjoint edge‑set mappings.
    """
    # 1. candidate edge pairs ------------------------------------------------
    candidates: List[Tuple[Edge, Edge]] = []
    for u1, v1, a1 in edges1:
        o1 = a1.get("order", (None, None))
        if len(o1) < 2:
            continue
        needed = o1[1]
        for u2, v2, a2 in edges2:
            o2 = a2.get("order", (None, None))
            if len(o2) < 2 or o2[0] != needed:
                continue
            if node_mapping and (
                u2 not in node_mapping.get(u1, []) or v2 not in node_mapping.get(v1, [])
            ):
                continue
            candidates.append(((u1, v1, a1), (u2, v2, a2)))

    # 2. DFS to enumerate *all* disjoint edge‑pair sets ----------------------
    pair_sets: List[List[Tuple[Edge, Edge]]] = []

    def _dfs(chosen: List[Tuple[Edge, Edge]], rem: List[Tuple[Edge, Edge]]):
        if not rem:
            if chosen:
                pair_sets.append(chosen.copy())
            return
        first, *rest = rem
        (u1, v1, _), (u2, v2, _) = first
        # include if disjoint on both graphs
        filt = [
            p
            for p in rest
            if p[0][0] not in (u1, v1)
            and p[0][1] not in (u1, v1)
            and p[1][0] not in (u2, v2)
            and p[1][1] not in (u2, v2)
        ]
        _dfs(chosen + [first], filt)  # include
        _dfs(chosen, rest)  # exclude

    _dfs([], candidates)

    # 3. select MCS (optional) ----------------------------------------------
    if mcs:
        max_sz = max((len(s) for s in pair_sets), default=0)
        pair_sets = [s for s in pair_sets if len(s) == max_sz]

    # 4. convert → mapping list & dedupe ------------------------------------
    mappings: MappingList = []
    seen: Set[frozenset] = set()
    for match_set in pair_sets:
        m: Dict[NodeId, NodeId] = {}
        for (u1, v1, _), (u2, v2, _) in match_set:
            m[u1] = u2
            m[v1] = v2
        key = frozenset(m.items())
        if key not in seen:
            seen.add(key)
            mappings.append(m)
    return mappings


# ---------------------------------------------------------------------------
# VF2 – full sub‑graph isomorphism via NetworkX
# ---------------------------------------------------------------------------


def _edge_constraint_vf2(
    edges1: Iterable[Edge],
    edges2: Iterable[Edge],
    node_mapping: Optional[Mapping[NodeId, List[NodeId]]] = None,
) -> MappingList:
    """Use NetworkX's VF2 to find maximum edge‑count isomorphisms."""
    g1, g2 = nx.Graph(), nx.Graph()
    for u, v, data in edges1:
        g1.add_edge(u, v, **dict(data))
    for u, v, data in edges2:
        g2.add_edge(u, v, **dict(data))

    def edge_match(a1: Mapping[str, Any], a2: Mapping[str, Any]) -> bool:
        o1 = a1.get("order", (None, None))
        o2 = a2.get("order", (None, None))
        return len(o1) >= 2 and len(o2) >= 2 and o1[1] == o2[0]

    gm = nx.algorithms.isomorphism.GraphMatcher(
        g1, g2, node_match=lambda *_: True, edge_match=edge_match
    )

    max_edges = 0
    best: MappingList = []

    for iso in gm.subgraph_isomorphisms_iter():
        # honour pre‑mapping constraint
        if node_mapping and any(
            iso[src] not in node_mapping.get(src, []) for src in iso
        ):
            continue
        # count matched edges (orientation‑independent)
        ecount = sum(
            1
            for u, v in g1.edges
            if u in iso and v in iso and g2.has_edge(iso[u], iso[v])
        )
        if ecount == 0:
            continue  # ignore isolated node maps
        if ecount > max_edges:
            max_edges = ecount
            best = [dict(iso)]
        elif ecount == max_edges:
            best.append(dict(iso))

    # dedupe (automorphisms may repeat)
    uniq: MappingList = []
    seen: Set[frozenset] = set()
    for m in best:
        k = frozenset(m.items())
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    return uniq


# ---------------------------------------------------------------------------
# VF3 – pairwise → grouped matching (hybrid)
# ---------------------------------------------------------------------------


def _edge_constraint_vf3(
    edges1: Iterable[Edge],
    edges2: Iterable[Edge],
    node_mapping: Optional[Mapping[NodeId, List[NodeId]]] = None,
) -> MappingList:
    """Hybrid strategy: single‑edge matches seeded, then grouped via DFS."""
    # 1. seed list
    seeds: List[Dict[NodeId, NodeId]] = []
    for u1, v1, a1 in edges1:
        o1 = a1.get("order", (None, None))
        if len(o1) < 2:
            continue
        need = o1[1]
        for u2, v2, a2 in edges2:
            o2 = a2.get("order", (None, None))
            if len(o2) < 2 or o2[0] != need:
                continue
            if node_mapping and (
                u2 not in node_mapping.get(u1, []) or v2 not in node_mapping.get(v1, [])
            ):
                continue
            seeds.append({u1: u2, v1: v2})
    if not seeds:
        return []

    # 2. DFS grouping
    best: List[Dict[NodeId, NodeId]] = []
    max_edges = 0

    def _dfs(idx: int, current: Dict[NodeId, NodeId]):
        nonlocal best, max_edges
        if idx == len(seeds):
            edges = len(current) // 2
            if edges == 0:
                return
            if edges > max_edges:
                max_edges = edges
                best = [current.copy()]
            elif edges == max_edges:
                best.append(current.copy())
            return
        cand = seeds[idx]
        if not (
            set(cand.keys()) & current.keys()
            or set(cand.values()) & set(current.values())
        ):
            _dfs(idx + 1, {**current, **cand})
        _dfs(idx + 1, current)

    _dfs(0, {})

    # dedupe
    uniq: MappingList = []
    seen: Set[frozenset] = set()
    for m in best:
        k = frozenset(m.items())
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    return uniq


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


def edge_constraint(
    edges1: Iterable[Edge],
    edges2: Iterable[Edge],
    node_mapping: Optional[Mapping[NodeId, List[NodeId]]] = None,
    *,
    algorithm: str = "vf2",
    mcs: bool = True,
) -> MappingList:
    """Return node‑mappings under the groupoid order rule.

    Parameters
    ----------
    algorithm : {'vf2', 'vf3', 'bt'}, default 'vf2'
        Which internal strategy to use.
    mcs : bool, default True
        Only for ``algorithm='bt'`` – if ``True`` keep maximum‑edge mappings, else
        return *all* disjoint mappings.
    """
    alg = algorithm.lower()
    if alg == "vf3":
        return _edge_constraint_vf3(edges1, edges2, node_mapping)
    if alg == "bt" or alg == "backtracking":
        return _edge_constraint_backtracking(edges1, edges2, node_mapping, mcs=mcs)
    # default VF2
    return _edge_constraint_vf2(edges1, edges2, node_mapping)
