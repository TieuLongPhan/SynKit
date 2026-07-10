from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import networkx as nx

from synkit.Graph.Mech.electron_accounting import atom_map_to_node

BondField = Literal["sigma_order", "pi_order"]


@dataclass(frozen=True)
class EdgeChange:
    """Sigma/pi change report for one mapped edge.

    :param atom_maps: Two atom maps defining the edited edge.
    :type atom_maps: tuple[int, int]
    :param field: Edited edge field, either ``"sigma_order"`` or
        ``"pi_order"``.
    :type field: BondField
    :param delta: Applied numeric order change.
    :type delta: float
    :param previous_value: Field value before the edit.
    :type previous_value: float
    :param new_value: Field value after the edit.
    :type new_value: float
    :param removed: Whether the edge was removed after total order became zero.
    :type removed: bool
    """

    atom_maps: tuple[int, int]
    field: BondField
    delta: float
    previous_value: float
    new_value: float
    removed: bool = False


@dataclass(frozen=True)
class LonePairChange:
    """Lone-pair change report for one mapped atom.

    :param atom_map: Atom map whose lone-pair count changed.
    :type atom_map: int
    :param node: NetworkX node corresponding to ``atom_map``.
    :type node: Any
    :param delta: Applied lone-pair change.
    :type delta: float
    :param previous_value: Lone-pair count before the edit.
    :type previous_value: float
    :param new_value: Lone-pair count after the edit.
    :type new_value: float
    """

    atom_map: int
    node: Any
    delta: float
    previous_value: float
    new_value: float


def split_sigma_pi_order(order: float) -> tuple[float, float]:
    """Split a Kekule bond order into sigma and pi components.

    :param order: Numeric Kekule bond order.
    :type order: float
    :returns: ``(sigma_order, pi_order)``.
    :rtype: tuple[float, float]
    """
    value = max(0.0, float(order))
    if value <= 0.0:
        return 0.0, 0.0
    return 1.0, value - 1.0


def normalize_lwg_graph(graph: nx.Graph, *, in_place: bool = False) -> nx.Graph:
    """Ensure each edge has editable sigma/pi and exportable Kekule order.

    Missing sigma/pi fields are derived from ``kekule_order`` and then from
    ``order`` as a fallback. The function also mirrors
    ``kekule_order = sigma_order + pi_order`` into ``order`` for export.

    :param graph: Lewis graph to normalize.
    :type graph: nx.Graph
    :param in_place: If ``True``, mutate ``graph`` directly.
    :type in_place: bool
    :returns: Normalized graph.
    :rtype: nx.Graph
    """
    target = graph if in_place else graph.copy()

    for _, _, data in target.edges(data=True):
        if "sigma_order" not in data or "pi_order" not in data:
            source_order = data.get("kekule_order", data.get("order", 1.0))
            sigma_order, pi_order = split_sigma_pi_order(float(source_order))
            data.setdefault("sigma_order", sigma_order)
            data.setdefault("pi_order", pi_order)

        data["sigma_order"] = float(data.get("sigma_order", 0.0))
        data["pi_order"] = float(data.get("pi_order", 0.0))
        data["kekule_order"] = data["sigma_order"] + data["pi_order"]
        data["order"] = data["kekule_order"]

    return target


def resolve_atom_maps(graph: nx.Graph, atom_maps: Iterable[int]) -> list[Any]:
    """Resolve atom-map numbers to graph nodes.

    :param graph: Graph containing node ``atom_map`` attributes.
    :type graph: nx.Graph
    :param atom_maps: Atom maps to resolve.
    :type atom_maps: Iterable[int]
    :returns: Graph nodes in the same order as ``atom_maps``.
    :rtype: list[Any]
    :raises ValueError: If any atom map is missing or duplicated.
    """
    lookup = atom_map_to_node(graph)
    nodes: list[Any] = []

    for atom_map in atom_maps:
        atom_map_int = int(atom_map)
        if atom_map_int not in lookup:
            raise ValueError(f"Atom map {atom_map_int} is missing from graph.")
        nodes.append(lookup[atom_map_int])

    return nodes


def mapped_edge_nodes(
    graph: nx.Graph,
    atom_maps: Iterable[int],
    *,
    create: bool = False,
) -> tuple[Any, Any]:
    """Resolve a two-atom-map edge endpoint list.

    :param graph: Lewis graph.
    :type graph: nx.Graph
    :param atom_maps: Exactly two atom maps defining the edge.
    :type atom_maps: Iterable[int]
    :param create: If ``True``, create a zero-order edge when absent.
    :type create: bool
    :returns: Two graph nodes for the mapped edge.
    :rtype: tuple[Any, Any]
    :raises ValueError: If the atom-map list is not length two, an atom map is
        missing, or the edge is absent while ``create=False``.
    """
    atom_map_tuple = tuple(int(value) for value in atom_maps)
    if len(atom_map_tuple) != 2:
        raise ValueError(f"Expected exactly two atom maps for edge: {atom_map_tuple}")

    node_a, node_b = resolve_atom_maps(graph, atom_map_tuple)
    if graph.has_edge(node_a, node_b):
        return node_a, node_b

    if not create:
        raise ValueError(f"Graph has no edge for atom maps {atom_map_tuple}.")

    graph.add_edge(
        node_a,
        node_b,
        order=0.0,
        kekule_order=0.0,
        sigma_order=0.0,
        pi_order=0.0,
        bond_type="UNSPECIFIED",
        kekule_bond_type="UNSPECIFIED",
        aromatic=False,
    )
    return node_a, node_b


def normalize_edge(graph: nx.Graph, node_a: Any, node_b: Any) -> None:
    """Refresh combined order fields for one edge.

    :param graph: Lewis graph containing the edge.
    :type graph: nx.Graph
    :param node_a: First edge endpoint node.
    :type node_a: Any
    :param node_b: Second edge endpoint node.
    :type node_b: Any
    :returns: ``None``; the edge attributes are updated in place.
    :rtype: None
    """
    data = graph.edges[node_a, node_b]
    data["sigma_order"] = float(data.get("sigma_order", 0.0))
    data["pi_order"] = float(data.get("pi_order", 0.0))
    data["kekule_order"] = data["sigma_order"] + data["pi_order"]
    data["order"] = data["kekule_order"]


def change_edge_order(
    graph: nx.Graph,
    atom_maps: Iterable[int],
    *,
    field: BondField,
    delta: float,
    create: bool = False,
    remove_zero_edge: bool = True,
    tol: float = 1e-9,
) -> EdgeChange:
    """Increment/decrement one sigma/pi field on a mapped edge.

    :param graph: Lewis graph to edit.
    :type graph: nx.Graph
    :param atom_maps: Exactly two atom maps defining the edited edge.
    :type atom_maps: Iterable[int]
    :param field: Edge field to edit, either ``"sigma_order"`` or
        ``"pi_order"``.
    :type field: BondField
    :param delta: Amount to add to the selected edge field.
    :type delta: float
    :param create: Whether to create a zero-order edge before applying the edit.
    :type create: bool
    :param remove_zero_edge: Whether to remove the edge when sigma and pi order
        both become zero.
    :type remove_zero_edge: bool
    :param tol: Floating-point tolerance for zero/negative checks.
    :type tol: float
    :returns: Edge edit report.
    :rtype: EdgeChange
    :raises ValueError: If the edit would make the selected order negative.
    """
    atom_map_tuple = tuple(int(value) for value in atom_maps)
    node_a, node_b = mapped_edge_nodes(graph, atom_map_tuple, create=create)
    data = graph.edges[node_a, node_b]

    data.setdefault("sigma_order", 0.0)
    data.setdefault("pi_order", 0.0)

    previous_value = float(data.get(field, 0.0))
    new_value = previous_value + float(delta)
    if new_value < -tol:
        raise ValueError(
            f"Negative {field} for atom maps {atom_map_tuple}: "
            f"{previous_value} + {delta}"
        )

    if abs(new_value) < tol:
        new_value = 0.0
    data[field] = new_value
    normalize_edge(graph, node_a, node_b)

    removed = False
    if (
        remove_zero_edge
        and float(data.get("sigma_order", 0.0)) == 0.0
        and float(data.get("pi_order", 0.0)) == 0.0
    ):
        graph.remove_edge(node_a, node_b)
        removed = True

    return EdgeChange(
        atom_maps=atom_map_tuple,
        field=field,
        delta=float(delta),
        previous_value=previous_value,
        new_value=new_value,
        removed=removed,
    )


def change_lone_pairs(
    graph: nx.Graph,
    atom_maps: Iterable[int],
    *,
    delta: float,
    tol: float = 1e-9,
) -> list[LonePairChange]:
    """Increment/decrement lone pairs for mapped atoms.

    :param graph: Lewis graph to edit.
    :type graph: nx.Graph
    :param atom_maps: Atom maps whose lone-pair count should change.
    :type atom_maps: Iterable[int]
    :param delta: Amount to add to each atom's ``lone_pairs`` field.
    :type delta: float
    :param tol: Floating-point tolerance for zero/negative checks.
    :type tol: float
    :returns: Lone-pair edit reports.
    :rtype: list[LonePairChange]
    :raises ValueError: If any edit would make lone pairs negative.
    """
    atom_map_tuple = tuple(int(value) for value in atom_maps)
    nodes = resolve_atom_maps(graph, atom_map_tuple)
    changes: list[LonePairChange] = []

    for atom_map, node in zip(atom_map_tuple, nodes):
        attrs = graph.nodes[node]
        previous_value = float(attrs.get("lone_pairs", 0.0))
        new_value = previous_value + float(delta)
        if new_value < -tol:
            raise ValueError(
                f"Negative lone_pairs for atom map {atom_map}: "
                f"{previous_value} + {delta}"
            )
        if abs(new_value) < tol:
            new_value = 0.0

        attrs["lone_pairs"] = int(new_value) if new_value.is_integer() else new_value
        changes.append(
            LonePairChange(
                atom_map=atom_map,
                node=node,
                delta=float(delta),
                previous_value=previous_value,
                new_value=new_value,
            )
        )

    return changes
