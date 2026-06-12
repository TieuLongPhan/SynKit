from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
from rdkit import Chem


@dataclass(frozen=True)
class ChargeRefresh:
    """VE/NBE/B charge refresh report for one atom map.

    :param atom_map: Refreshed atom map.
    :type atom_map: int
    :param node: NetworkX node corresponding to ``atom_map``.
    :type node: Any
    :param previous_charge: Charge before refresh.
    :type previous_charge: int | float
    :param refreshed_charge: Charge computed by ``VE - NBE - B``.
    :type refreshed_charge: int | float
    :param valence_electrons: Atom valence electron count used as ``VE``.
    :type valence_electrons: float
    :param nonbonding_electrons: Nonbonding electron count used as ``NBE``.
    :type nonbonding_electrons: float
    :param bond_electrons: Bond electron count used as ``B``.
    :type bond_electrons: float
    """

    atom_map: int
    node: Any
    previous_charge: int | float
    refreshed_charge: int | float
    valence_electrons: float
    nonbonding_electrons: float
    bond_electrons: float


@dataclass(frozen=True)
class ChargeEdit:
    """Incremental local charge edit for one atom map.

    :param atom_map: Edited atom map.
    :type atom_map: int
    :param node: NetworkX node corresponding to ``atom_map``.
    :type node: Any
    :param delta: Applied formal-charge delta.
    :type delta: int | float
    :param previous_charge: Charge before the edit.
    :type previous_charge: int | float
    :param new_charge: Charge after the edit.
    :type new_charge: int | float
    """

    atom_map: int
    node: Any
    delta: int | float
    previous_charge: int | float
    new_charge: int | float


def bond_order_sum(graph: nx.Graph, node: Any) -> float:
    """Return the sigma-plus-pi bond-order sum around one node.

    :param graph: Lewis graph.
    :type graph: nx.Graph
    :param node: Node whose incident bond orders should be summed.
    :type node: Any
    :returns: Sum of incident ``sigma_order + pi_order`` values.
    :rtype: float
    """
    total = 0.0
    for _, _, data in graph.edges(node, data=True):
        total += float(data.get("sigma_order", 0.0)) + float(data.get("pi_order", 0.0))
    return total


def recompute_charge(graph: nx.Graph, node: Any) -> int | float:
    """Recompute formal charge from stored electron-state fields.

    This diagnostic helper uses ``Formal Charge = VE - NBE - B``. The
    :class:`synkit.Graph.Mech.lwg_editor.LWGEditor` production edit path uses
    local charge deltas instead.

    :param graph: Lewis graph.
    :type graph: nx.Graph
    :param node: Node whose charge should be recomputed.
    :type node: Any
    :returns: Recomputed formal charge.
    :rtype: int | float
    """
    attrs = graph.nodes[node]
    charge = (
        float(attrs["valence_electrons"])
        - nonbonding_electron_count(graph, node)
        - bond_electron_count(graph, node)
    )
    return int(charge) if charge.is_integer() else charge


def nonbonding_electron_count(graph: nx.Graph, node: Any) -> float:
    """Return NBE for one atom from stored lone-pair/radical fields.

    :param graph: Lewis graph.
    :type graph: nx.Graph
    :param node: Node whose nonbonding electrons should be counted.
    :type node: Any
    :returns: ``2 * lone_pairs + radical``.
    :rtype: float
    """
    attrs = graph.nodes[node]
    return 2 * float(attrs.get("lone_pairs", 0)) + float(attrs.get("radical", 0))


def bond_electron_count(graph: nx.Graph, node: Any) -> float:
    """Return B for one atom, including implicit hydrogen bonds.

    :param graph: Lewis graph.
    :type graph: nx.Graph
    :param node: Node whose bonding electrons should be counted.
    :type node: Any
    :returns: ``hcount`` plus incident sigma/pi bond-order sum.
    :rtype: float
    """
    attrs = graph.nodes[node]
    return float(attrs.get("hcount", 0)) + bond_order_sum(graph, node)


def atom_map_to_node(graph: nx.Graph) -> dict[int, Any]:
    """Build an atom-map to node lookup, ignoring unmapped atoms.

    :param graph: Graph with node ``atom_map`` attributes.
    :type graph: nx.Graph
    :returns: Mapping from atom-map number to graph node.
    :rtype: dict[int, Any]
    :raises ValueError: If a nonzero atom map appears on multiple nodes.
    """
    lookup: dict[int, Any] = {}
    duplicates: dict[int, list[Any]] = {}

    for node, attrs in graph.nodes(data=True):
        atom_map = attrs.get("atom_map", node)
        if atom_map in (None, 0, "0"):
            continue

        atom_map_int = int(atom_map)
        if atom_map_int in lookup:
            duplicates.setdefault(atom_map_int, [lookup[atom_map_int]]).append(node)
            continue
        lookup[atom_map_int] = node

    if duplicates:
        raise ValueError(f"Duplicate atom maps in graph: {duplicates}")

    return lookup


def refresh_changed_atom_charge(
    graph: nx.Graph,
    atom_maps: list[int] | tuple[int, ...] | set[int],
) -> list[ChargeRefresh]:
    """Refresh formal charges only for selected atom maps.

    Uses ``Formal Charge = VE - NBE - B`` where ``B`` includes implicit
    hydrogens stored as ``hcount`` plus incident sigma/pi order.

    :param graph: Lewis graph to mutate.
    :type graph: nx.Graph
    :param atom_maps: Atom maps whose ``charge`` field should be recomputed.
    :type atom_maps: list[int] | tuple[int, ...] | set[int]
    :returns: Per-atom refresh reports.
    :rtype: list[ChargeRefresh]
    :raises ValueError: If any atom map is missing or lacks
        ``valence_electrons``.
    """
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
    """Apply a local formal-charge delta to selected atom maps.

    This is the charge update primitive used by
    :class:`synkit.Graph.Mech.lwg_editor.LWGEditor`.

    :param graph: Lewis graph to mutate.
    :type graph: nx.Graph
    :param atom_maps: Atom maps whose formal charge should be edited.
    :type atom_maps: list[int] | tuple[int, ...] | set[int]
    :param delta: Formal-charge delta to add to each selected atom.
    :type delta: int | float
    :returns: Per-atom charge edit reports.
    :rtype: list[ChargeEdit]
    :raises ValueError: If any atom map is missing.
    """
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

    :param graph: Lewis graph to refresh.
    :type graph: nx.Graph
    :param in_place: If ``True``, mutate ``graph`` directly.
    :type in_place: bool
    :returns: Graph with refreshed ``kekule_order``, ``bond_order_sum``,
        ``recomputed_charge``, and ``charge_mismatch`` fields.
    :rtype: nx.Graph
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
    """Reconstruct a product from ``kekule_order`` and let RDKit sanitize it.

    :param graph: Lewis graph with ``kekule_order`` edge fields.
    :type graph: nx.Graph
    :returns: Sanitized RDKit molecule.
    :rtype: Chem.Mol
    """
    from synkit.IO.graph_to_mol import GraphToMol

    refreshed = refresh_electron_fields(graph)
    return GraphToMol(edge_attributes={"order": "kekule_order"}).graph_to_mol(
        refreshed,
        sanitize=True,
        use_h_count=True,
    )
