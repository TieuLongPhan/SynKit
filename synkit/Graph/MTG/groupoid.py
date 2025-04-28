from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Set

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


def edge_constraint(
    edges1: Iterable[Edge],
    edges2: Iterable[Edge],
    node_mapping: Mapping[NodeId, List[NodeId]] | None = None,
) -> MappingList:
    """Find disjoint edge-merge mappings under the groupoid order rule.

    For each edge e1=(u1,v1,attrs1) and e2=(u2,v2,attrs2):
      - attrs1['order'][1] == attrs2['order'][0]
      - if node_mapping provided: u2 in node_mapping[u1] and v2 in node_mapping[v1]

    Performs a maximum set-packing via backtracking so that returned mappings
    have no shared endpoints on either side.

    Returns
    -------
    List of node mappings (dict G1_node -> G2_node) for each valid composition.
    """
    # 1. Build candidate pairs
    candidates: List[Tuple[Edge, Edge]] = []
    for u1, v1, a1 in edges1:
        order1 = a1.get("order", (None, None))
        if len(order1) < 2:
            continue
        a2_val = order1[1]
        for u2, v2, a2 in edges2:
            order2 = a2.get("order", (None, None))
            if len(order2) < 2:
                continue
            b1_val = order2[0]
            if a2_val != b1_val:
                continue
            if node_mapping is not None:
                if u2 not in node_mapping.get(u1, []):
                    continue
                if v2 not in node_mapping.get(v1, []):
                    continue
            candidates.append(((u1, v1, a1), (u2, v2, a2)))

    # 2. Backtracking for disjoint sets
    pair_sets: List[List[Tuple[Edge, Edge]]] = []

    def _backtrack(
        chosen: List[Tuple[Edge, Edge]], remaining: List[Tuple[Edge, Edge]]
    ) -> None:
        if not remaining:
            if chosen:
                pair_sets.append(chosen.copy())
            return
        first, *rest = remaining
        e1, e2 = first
        # include: drop conflicts
        filtered = [p for p in rest if p[0] != e1 and p[1] != e2]
        _backtrack(chosen + [first], filtered)
        # exclude
        _backtrack(chosen, rest)

    _backtrack([], candidates)

    # 3. Convert to mapping list and dedupe
    mapping_list: MappingList = []
    seen: Set[frozenset] = set()
    for matching in pair_sets:
        m: Dict[NodeId, NodeId] = {}
        for (u1, v1, _), (u2, v2, _) in matching:
            m[u1] = u2
            m[v1] = v2
        key = frozenset(m.items())
        if key not in seen:
            seen.add(key)
            mapping_list.append(m)
    return mapping_list
