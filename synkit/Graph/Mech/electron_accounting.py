from __future__ import annotations

from typing import Any

import networkx as nx
from rdkit import Chem

from synkit.IO.graph_to_mol import GraphToMol


def bond_order_sum(graph: nx.Graph, node: Any) -> float:
    """Return the sigma-plus-pi bond-order sum around one node."""
    total = 0.0
    for _, _, data in graph.edges(node, data=True):
        total += float(data.get("sigma_order", 0.0)) + float(data.get("pi_order", 0.0))
    return total


def recompute_charge(graph: nx.Graph, node: Any) -> int | float:
    """Recompute formal charge from stored electron-state fields."""
    attrs = graph.nodes[node]
    charge = float(attrs["valence_electrons"]) - (
        2 * float(attrs.get("lone_pairs", 0))
        + float(attrs.get("radical", 0))
        + float(attrs.get("hcount", 0))
        + bond_order_sum(graph, node)
    )
    return int(charge) if charge.is_integer() else charge


def refresh_electron_fields(graph: nx.Graph, *, in_place: bool = False) -> nx.Graph:
    """Refresh derived electron bookkeeping on a molecular graph.

    The graph is expected to store scalar ``sigma_order`` and ``pi_order`` edge
    fields plus node-level electron state. Presentation-facing ``order`` is not
    rewritten here; RDKit reconstruction remains responsible for aromatic
    re-perception at the product boundary.
    """
    target = graph if in_place else graph.copy()

    for _, _, data in target.edges(data=True):
        sigma = float(data.get("sigma_order", 0.0))
        pi = float(data.get("pi_order", 0.0))
        data["kekule_order"] = sigma + pi

    for node, attrs in target.nodes(data=True):
        attrs["bond_order_sum"] = bond_order_sum(target, node)
        if "valence_electrons" not in attrs:
            continue
        attrs["recomputed_charge"] = recompute_charge(target, node)
        represented_charge = float(attrs.get("charge", 0))
        attrs["charge_mismatch"] = represented_charge != attrs["recomputed_charge"]

    return target


def graph_to_sanitized_kekule_mol(graph: nx.Graph) -> Chem.Mol:
    """Reconstruct a product from ``kekule_order`` and let RDKit sanitize it."""
    refreshed = refresh_electron_fields(graph)
    return GraphToMol(edge_attributes={"order": "kekule_order"}).graph_to_mol(
        refreshed,
        sanitize=True,
        use_h_count=True,
    )
