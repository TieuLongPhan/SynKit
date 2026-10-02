from __future__ import annotations

"""Chemistry-specific helpers for bond order and electron-flow semantics."""

from typing import Any, Dict, Optional, Sequence, Tuple

import networkx as nx

from .constants import transition_family
from .mapping import edge_amap_key_from_nodes, edge_nodes_from_amap_key
from .utils import canonical_bond_order, bond_symbol, tget, as_tuple


def other_in_bond(bond: Tuple[int, int], atom: int) -> Optional[int]:
    """Return the other atom in a 2-center bond tuple."""
    u, v = bond
    if atom == u:
        return v
    if atom == v:
        return u
    return None


def infer_shared_atom(
    src: Tuple[int, ...],
    dst: Tuple[int, ...],
    data: Dict[str, Any],
) -> Optional[int]:
    """Infer the shared atom in a source/destination bond transition."""
    shared = data.get("shared_atom", None)
    if shared is not None:
        return shared
    inter = set(src) & set(dst)
    if len(inter) == 1:
        return next(iter(inter))
    return None


def edge_order(edata: Dict[str, Any]) -> Optional[float]:
    """Read a bond order from an edge attribute dictionary."""
    bond_type = edata.get("bond_type", None)
    if bond_type == "AROMATIC":
        return 1.5

    kek = edata.get("kekule_order", None)
    if kek is not None and not isinstance(kek, (tuple, list)):
        return canonical_bond_order(kek)

    order = edata.get("order", None)
    if order is not None and not isinstance(order, (tuple, list)):
        return canonical_bond_order(order)

    return 1.0


def edge_symbol(edata: Dict[str, Any]) -> str:
    """Single-state symbolic bond label for a graph edge."""
    return bond_symbol(edge_order(edata))


def edge_order_from_amap_key(
    graph: Optional[nx.Graph],
    amap_edge: Tuple[int, int],
    atom_map_key: str = "atom_map",
) -> Optional[float]:
    """Resolve an atom-map edge and return its bond order."""
    node_edge = edge_nodes_from_amap_key(graph, amap_edge, atom_map_key=atom_map_key)
    if node_edge is None or graph is None:
        return None
    u, v = node_edge
    if not graph.has_edge(u, v):
        return None
    return edge_order(graph.get_edge_data(u, v))


def collect_rc_amap_sets(
    transitions: Sequence[Any],
    reactant_graph: nx.Graph,
    atom_map_key: str = "atom_map",
) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:
    """Collect broken and forming reaction-center edges in atom-map space."""
    broken: set[Tuple[int, int]] = set()
    forming: set[Tuple[int, int]] = set()

    for t in transitions:
        _raw_kind = tget(t, "kind")
        kind = transition_family(_raw_kind)
        src = as_tuple(tget(t, "src"))
        dst = as_tuple(tget(t, "dst"))

        if kind in {"B-/LP+", "B-/B+"} and len(src) == 2:
            broken.add(
                edge_amap_key_from_nodes(
                    reactant_graph,
                    (src[0], src[1]),
                    atom_map_key=atom_map_key,
                )
            )

        if kind in {"LP-/B+", "B-/B+", "H-/B+"} and len(dst) == 2:
            forming.add(
                edge_amap_key_from_nodes(
                    reactant_graph,
                    (dst[0], dst[1]),
                    atom_map_key=atom_map_key,
                )
            )

    return broken, forming


def its_pair_orders_from_edge(
    its_graph: nx.Graph,
    u: int,
    v: int,
    reactant_graph: Optional[nx.Graph] = None,
    product_graph: Optional[nx.Graph] = None,
    atom_map_key: str = "atom_map",
) -> Tuple[Optional[float], Optional[float]]:
    """Return (reactant_order, product_order) for an ITS edge."""
    edata = its_graph.get_edge_data(u, v, default={})

    for key in ("order", "kekule_order"):
        val = edata.get(key, None)
        if isinstance(val, (tuple, list)) and len(val) >= 2:
            return canonical_bond_order(val[0]), canonical_bond_order(val[1])

    amap_u = int(its_graph.nodes[u].get(atom_map_key, u))
    amap_v = int(its_graph.nodes[v].get(atom_map_key, v))
    amap_edge = tuple(sorted((amap_u, amap_v)))

    r_order = edge_order_from_amap_key(
        reactant_graph, amap_edge, atom_map_key=atom_map_key
    )
    p_order = edge_order_from_amap_key(
        product_graph, amap_edge, atom_map_key=atom_map_key
    )
    return canonical_bond_order(r_order), canonical_bond_order(p_order)


def its_pair_label(
    its_graph: nx.Graph,
    u: int,
    v: int,
    reactant_graph: Optional[nx.Graph] = None,
    product_graph: Optional[nx.Graph] = None,
    atom_map_key: str = "atom_map",
) -> str:
    """Return a label like ``(—,=)`` for an ITS edge."""
    r_order, p_order = its_pair_orders_from_edge(
        its_graph,
        u,
        v,
        reactant_graph=reactant_graph,
        product_graph=product_graph,
        atom_map_key=atom_map_key,
    )
    return f"({bond_symbol(r_order)},{bond_symbol(p_order)})"


def its_edge_change_sign(
    its_graph: nx.Graph,
    u: int,
    v: int,
    reactant_graph: Optional[nx.Graph] = None,
    product_graph: Optional[nx.Graph] = None,
    atom_map_key: str = "atom_map",
) -> int:
    """Return +1 for decrease/breaking, -1 for increase/forming, 0 unchanged."""
    edata = its_graph.get_edge_data(u, v, default={})
    std = edata.get("standard_order", None)
    if std is not None and not isinstance(std, (tuple, list)):
        try:
            val = float(std)
            if val > 0:
                return +1
            if val < 0:
                return -1
            return 0
        except Exception:
            pass

    r_order, p_order = its_pair_orders_from_edge(
        its_graph,
        u,
        v,
        reactant_graph=reactant_graph,
        product_graph=product_graph,
        atom_map_key=atom_map_key,
    )
    r_val = 0.0 if r_order is None else float(r_order)
    p_val = 0.0 if p_order is None else float(p_order)
    if p_val > r_val:
        return -1
    if p_val < r_val:
        return +1
    return 0
