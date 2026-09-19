from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
from rdkit import Chem


@dataclass(frozen=True)
class ChargeRefresh:
    """VE/NBE/B charge refresh report for one atom map."""

    atom_map: int
    node: Any
    previous_charge: int | float
    refreshed_charge: int | float
    valence_electrons: float
    nonbonding_electrons: float
    bond_electrons: float


@dataclass(frozen=True)
class ChargeEdit:
    """Incremental local formal-charge edit for one atom map."""

    atom_map: int
    node: Any
    delta: int | float
    previous_charge: int | float
    new_charge: int | float


def bond_order_sum(graph: nx.Graph, node: Any) -> float:
    """Return the sigma-plus-pi bond-order sum around one node."""
    total = 0.0
    for _, _, data in graph.edges(node, data=True):
        total += float(data.get("sigma_order", 0.0)) + float(data.get("pi_order", 0.0))
    return total


def nonbonding_electron_count(graph: nx.Graph, node: Any) -> float:
    """Return the nonbonding-electron count for one atom."""
    attrs = graph.nodes[node]
    return 2 * float(attrs.get("lone_pairs", 0)) + float(attrs.get("radical", 0))


def bond_electron_count(graph: nx.Graph, node: Any) -> float:
    """Return the atom's formal-charge bonding allocation.

    Each bond pair contributes one electron to this atom; this is therefore
    not the complete two-electron population of the bonds.
    """
    return float(graph.nodes[node].get("hcount", 0)) + bond_order_sum(graph, node)


def recompute_charge(graph: nx.Graph, node: Any) -> int | float:
    """Recompute formal charge from stored electron-state fields."""
    attrs = graph.nodes[node]
    charge = (
        float(attrs["valence_electrons"])
        - nonbonding_electron_count(
            graph,
            node,
        )
        - bond_electron_count(graph, node)
    )
    return int(charge) if charge.is_integer() else charge


def atom_map_to_node(graph: nx.Graph) -> dict[int, Any]:
    """Build a unique atom-map-to-node lookup for a molecular graph."""
    lookup: dict[int, Any] = {}
    duplicates: dict[int, list[Any]] = {}

    for node, attrs in graph.nodes(data=True):
        atom_map = attrs.get("atom_map", node)
        if atom_map in (None, 0, "0"):
            continue

        atom_map_int = int(atom_map)
        if atom_map_int in lookup:
            duplicates.setdefault(atom_map_int, [lookup[atom_map_int]]).append(node)
        else:
            lookup[atom_map_int] = node

    if duplicates:
        raise ValueError(f"Duplicate atom maps in graph: {duplicates}")

    return lookup


def refresh_changed_atom_charge(
    graph: nx.Graph,
    atom_maps: list[int] | tuple[int, ...] | set[int],
) -> list[ChargeRefresh]:
    """Refresh formal charges for selected mapped atoms in place."""
    lookup = atom_map_to_node(graph)
    reports: list[ChargeRefresh] = []

    for atom_map in sorted({int(value) for value in atom_maps}):
        if atom_map not in lookup:
            raise ValueError(f"Atom map {atom_map} is missing from graph.")

        node = lookup[atom_map]
        attrs = graph.nodes[node]
        if "valence_electrons" not in attrs:
            raise ValueError(f"Atom map {atom_map} has no valence_electrons field.")

        previous_charge = attrs.get("charge", 0)
        refreshed_charge = recompute_charge(graph, node)
        attrs["charge"] = refreshed_charge
        attrs["bond_order_sum"] = bond_order_sum(graph, node)
        attrs["recomputed_charge"] = refreshed_charge
        attrs["charge_mismatch"] = False
        reports.append(
            ChargeRefresh(
                atom_map=atom_map,
                node=node,
                previous_charge=previous_charge,
                refreshed_charge=refreshed_charge,
                valence_electrons=float(attrs["valence_electrons"]),
                nonbonding_electrons=nonbonding_electron_count(graph, node),
                bond_electrons=bond_electron_count(graph, node),
            )
        )

    return reports


def change_atom_charge(
    graph: nx.Graph,
    atom_maps: list[int] | tuple[int, ...] | set[int],
    *,
    delta: int | float,
) -> list[ChargeEdit]:
    """Apply a local formal-charge delta to selected mapped atoms."""
    lookup = atom_map_to_node(graph)
    reports: list[ChargeEdit] = []

    for atom_map in [int(value) for value in atom_maps]:
        if atom_map not in lookup:
            raise ValueError(f"Atom map {atom_map} is missing from graph.")

        node = lookup[atom_map]
        attrs = graph.nodes[node]
        previous_charge = attrs.get("charge", 0)
        new_charge = previous_charge + delta
        if isinstance(new_charge, float) and new_charge.is_integer():
            new_charge = int(new_charge)
        attrs["charge"] = new_charge
        reports.append(
            ChargeEdit(
                atom_map=atom_map,
                node=node,
                delta=delta,
                previous_charge=previous_charge,
                new_charge=new_charge,
            )
        )

    return reports


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
    from synkit.IO.graph_to_mol import GraphToMol

    refreshed = refresh_electron_fields(graph)
    return GraphToMol(edge_attributes={"order": "kekule_order"}).graph_to_mol(
        refreshed,
        sanitize=True,
        use_h_count=True,
    )
