from __future__ import annotations
import networkx as nx
from collections import defaultdict
from typing import Iterable, Mapping, List, Dict, Any, Tuple, Optional, Set, FrozenSet

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
    seen: Set[FrozenSet] = set()
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
# VF2
# ---------------------------------------------------------------------------


def _edge_constraint_vf2(
    edges1: Iterable[Edge],
    edges2: Iterable[Edge],
    node_mapping: Optional[Mapping[NodeId, List[NodeId]]] = None,
) -> MappingList:
    """
    VF2‐style routine, fully in Python (no NetworkX), seeded like VF3 but
    relaxed so it returns the same maximal‐common‐subgraph mappings.
    The returned dicts will always have their keys sorted ascending.
    """
    # --- build adjacency lists with valid 'order' tuples ---
    adj1: Dict[NodeId, Dict[NodeId, Tuple[int, int]]] = {}
    for u, v, data in edges1:
        o = data.get("order", ())
        if isinstance(o, tuple) and len(o) >= 2:
            adj1.setdefault(u, {})[v] = o
            adj1.setdefault(v, {})[u] = (o[1], o[0])
    adj2: Dict[NodeId, Dict[NodeId, Tuple[int, int]]] = {}
    for u, v, data in edges2:
        o = data.get("order", ())
        if isinstance(o, tuple) and len(o) >= 2:
            adj2.setdefault(u, {})[v] = o
            adj2.setdefault(v, {})[u] = (o[1], o[0])

    # --- seed exactly as VF3 does ---
    seeds: List[Dict[NodeId, NodeId]] = []
    for u1, v1, d1 in edges1:
        o1 = d1.get("order", ())
        if not (isinstance(o1, tuple) and len(o1) >= 2):
            continue
        need = o1[1]
        for u2, v2, d2 in edges2:
            o2 = d2.get("order", ())
            if not (isinstance(o2, tuple) and len(o2) >= 2) or o2[0] != need:
                continue
            if node_mapping and (
                u2 not in node_mapping.get(u1, []) or v2 not in node_mapping.get(v1, [])
            ):
                continue
            seeds.append({u1: u2, v1: v2})
    if not seeds:
        return []

    # --- DFS grouping by using state dict ---
    state: Dict[str, Any] = {"best": [], "max_edges": 0}

    def _dfs(
        idx: int,
        current: Dict[NodeId, NodeId],
        mapped1: Set[NodeId],
        mapped2: Set[NodeId],
        edge_count: int,
    ):
        # mutate state
        if idx == len(seeds):
            if edge_count > state["max_edges"]:
                state["max_edges"] = edge_count
                state["best"] = [current.copy()]
            elif edge_count == state["max_edges"]:
                state["best"].append(current.copy())
            return

        cand = seeds[idx]
        # try including if no node-ID conflict
        if not (set(cand.keys()) & mapped1 or set(cand.values()) & mapped2):
            _dfs(
                idx + 1,
                {**current, **cand},
                mapped1 | set(cand.keys()),
                mapped2 | set(cand.values()),
                edge_count + 1,
            )
        # try skipping this seed
        _dfs(idx + 1, current, mapped1, mapped2, edge_count)

    # kick off DFS from each seed
    for i, seed in enumerate(seeds):
        _dfs(i, seed.copy(), set(seed.keys()), set(seed.values()), 1)

    # --- dedupe automorphisms & sort keys ---
    uniq: MappingList = []
    seen: Set[FrozenSet] = set()
    for m in state["best"]:
        key = frozenset(m.items())
        if key in seen:
            continue
        seen.add(key)
        # rebuild with keys in ascending order
        sorted_map = {u: m[u] for u in sorted(m.keys())}
        uniq.append(sorted_map)

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

    # 2. DFS grouping by using a state dict
    state: Dict[str, Any] = {"best": [], "max_edges": 0}

    def _dfs(idx: int, current: Dict[NodeId, NodeId]):
        if idx == len(seeds):
            edges = len(current) // 2
            if edges == 0:
                return
            if edges > state["max_edges"]:
                state["max_edges"] = edges
                state["best"] = [current.copy()]
            elif edges == state["max_edges"]:
                state["best"].append(current.copy())
            return

        cand = seeds[idx]
        # include this seed if no conflicts
        if not (
            set(cand.keys()) & current.keys()
            or set(cand.values()) & set(current.values())
        ):
            _dfs(idx + 1, {**current, **cand})
        # always try skipping
        _dfs(idx + 1, current)

    _dfs(0, {})

    # 3. dedupe
    uniq: MappingList = []
    seen: Set[FrozenSet] = set()
    for m in state["best"]:
        key = frozenset(m.items())
        if key not in seen:
            seen.add(key)
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
    algorithm: str = "bt",
    mcs: bool = False,
) -> MappingList:
    """Return node‑mappings under the groupoid order rule.

    Parameters
    ----------
    algorithm : {'vf2', 'vf3', 'bt'}, default 'bt'
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
    return _edge_constraint_vf2(edges1, edges2, node_mapping)
