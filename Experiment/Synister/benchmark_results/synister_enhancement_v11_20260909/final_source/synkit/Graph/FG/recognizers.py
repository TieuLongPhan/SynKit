from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from .model import FunctionalGroupPattern, Mapping
from .model import FunctionalGroupMatch


def _graph(
    nodes: Iterable[tuple[int, dict]],
    edges: Iterable[tuple[int, int, dict]],
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def _single_heavy_neighbors(
    graph: nx.Graph, node: int, *, exclude: set[int]
) -> list[int]:
    return [
        neighbor
        for neighbor in graph.neighbors(node)
        if neighbor not in exclude and graph.nodes[neighbor].get("element") != "H"
    ]


def _alcohol_carbon_heavy_degree(expected: int):
    def validator(graph: nx.Graph, mapping: Mapping) -> bool:
        carbon, oxygen = mapping[1], mapping[2]
        if not _alcohol_like(graph, mapping):
            return False
        return len(_single_heavy_neighbors(graph, carbon, exclude={oxygen})) == expected

    return validator


def _alcohol_like(graph: nx.Graph, mapping: Mapping) -> bool:
    carbon, oxygen = mapping[1], mapping[2]
    if graph.nodes[carbon].get("aromatic"):
        return False
    if graph.nodes[oxygen].get("hcount", 0) < 1:
        return False
    return all(
        graph.edges[carbon, neighbor].get("order") == 1.0
        for neighbor in graph.neighbors(carbon)
    )


def _aldehyde(graph: nx.Graph, mapping: Mapping) -> bool:
    carbon, oxygen = mapping[1], mapping[2]
    if graph.nodes[carbon].get("hcount", 0) < 1:
        return False
    others = _single_heavy_neighbors(graph, carbon, exclude={oxygen})
    return all(graph.nodes[node].get("element") == "C" for node in others)


def _amine(graph: nx.Graph, mapping: Mapping) -> bool:
    nitrogen = mapping[1]
    if graph.nodes[nitrogen].get("aromatic"):
        return False
    return all(
        graph.edges[nitrogen, neighbor].get("order") == 1.0
        for neighbor in graph.neighbors(nitrogen)
    )


def _phenol(graph: nx.Graph, mapping: Mapping) -> bool:
    carbon, oxygen = mapping[1], mapping[2]
    return (
        graph.nodes[carbon].get("aromatic") is True
        and graph.nodes[oxygen].get("hcount", 0) >= 1
    )


def _enol(graph: nx.Graph, mapping: Mapping) -> bool:
    return graph.nodes[mapping[3]].get("hcount", 0) >= 1


def _epoxide(graph: nx.Graph, mapping: Mapping) -> bool:
    return all(graph.nodes[mapping[node]].get("in_ring") for node in (1, 2, 3))


def _phosphite(graph: nx.Graph, mapping: Mapping) -> bool:
    phosphorus = mapping[1]
    return all(
        not (
            graph.nodes[neighbor].get("element") == "O"
            and graph.edges[phosphorus, neighbor].get("order") == 2.0
        )
        for neighbor in graph.neighbors(phosphorus)
    )


def _azide(graph: nx.Graph, mapping: Mapping) -> bool:
    middle, terminal = mapping[2], mapping[3]
    return (
        graph.nodes[middle].get("charge") == 1
        and graph.nodes[terminal].get("charge") == -1
    )


def _hydrazone(graph: nx.Graph, mapping: Mapping) -> bool:
    imine_nitrogen = mapping[2]
    hydrazine_nitrogen = mapping[3]
    return not (
        graph.nodes[imine_nitrogen].get("charge") == 1
        and graph.nodes[hydrazine_nitrogen].get("charge") == -1
    )


def _amidine(graph: nx.Graph, mapping: Mapping) -> bool:
    carbon = mapping[1]
    imine_nitrogen = mapping[2]
    amino_nitrogen = mapping[3]
    if any(
        graph.nodes[neighbor].get("element") == "O"
        for neighbor in graph.neighbors(imine_nitrogen)
        if neighbor != carbon
    ):
        return False
    if any(
        graph.nodes[neighbor].get("element") == "O"
        for neighbor in graph.neighbors(amino_nitrogen)
        if neighbor != carbon
    ):
        return False
    return True


def _imine(graph: nx.Graph, mapping: Mapping) -> bool:
    carbon = mapping[1]
    nitrogen = mapping[2]
    if graph.nodes[carbon].get("aromatic") or graph.nodes[nitrogen].get("aromatic"):
        return False
    carbon_neighbors = {
        graph.nodes[neighbor].get("element")
        for neighbor in graph.neighbors(carbon)
        if neighbor != nitrogen
    }
    nitrogen_neighbors = {
        graph.nodes[neighbor].get("element")
        for neighbor in graph.neighbors(nitrogen)
        if neighbor != carbon
    }
    if carbon_neighbors & {"O", "S", "N"}:
        return False
    if nitrogen_neighbors & {"O", "N"}:
        return False
    return True


def _aniline(graph: nx.Graph, mapping: Mapping) -> bool:
    return graph.nodes[mapping[1]].get("aromatic") is True and _amine(
        graph, {1: mapping[2]}
    )


def _aryl_halide(graph: nx.Graph, mapping: Mapping) -> bool:
    return graph.nodes[mapping[1]].get("aromatic") is True


def _oxygen_has_h(graph: nx.Graph, node: int) -> bool:
    return graph.nodes[node].get("hcount", 0) >= 1


def _carbon_substituent_count(graph: nx.Graph, carbon: int, excluded: set[int]) -> int:
    return len(_single_heavy_neighbors(graph, carbon, exclude=excluded))


def _aromatic_ring_nodes(graph: nx.Graph, mapping: Mapping) -> set[int]:
    return {mapping[node] for node in mapping}


def _all_aromatic(graph: nx.Graph, nodes: set[int]) -> bool:
    return all(graph.nodes[node].get("aromatic") for node in nodes)


def _single_node_recognizer(element: str):
    def recognize(
        graph: nx.Graph, pattern: FunctionalGroupPattern
    ) -> list[FunctionalGroupMatch]:
        matches: list[FunctionalGroupMatch] = []
        for node, data in graph.nodes(data=True):
            if data.get("element") != element:
                continue
            mapping = {1: node}
            if pattern.validator is not None and not pattern.validator(graph, mapping):
                continue
            matches.append(
                FunctionalGroupMatch(
                    name=pattern.name,
                    group_nodes=(node,),
                    mapping=mapping,
                    pattern=pattern,
                )
            )
        return matches

    return recognize


def _two_node_bond_recognizer(
    left_element: str,
    right_elements: tuple[str, ...],
    order: float,
):
    def recognize(
        graph: nx.Graph, pattern: FunctionalGroupPattern
    ) -> list[FunctionalGroupMatch]:
        matches: list[FunctionalGroupMatch] = []
        seen: set[tuple[int, ...]] = set()
        for left, right, data in graph.edges(data=True):
            if data.get("order") != order:
                continue
            pairs = ((left, right), (right, left))
            for first, second in pairs:
                if graph.nodes[first].get("element") != left_element:
                    continue
                if graph.nodes[second].get("element") not in right_elements:
                    continue
                mapping = {1: first, 2: second}
                if pattern.validator is not None and not pattern.validator(
                    graph, mapping
                ):
                    continue
                group_nodes = tuple(
                    sorted(mapping[node] for node in pattern.group_nodes)
                )
                if group_nodes in seen:
                    continue
                seen.add(group_nodes)
                matches.append(
                    FunctionalGroupMatch(
                        name=pattern.name,
                        group_nodes=group_nodes,
                        mapping=mapping,
                        pattern=pattern,
                    )
                )
        return matches

    return recognize


def _symmetric_two_node_bond_recognizer(element: str, order: float):
    def recognize(
        graph: nx.Graph, pattern: FunctionalGroupPattern
    ) -> list[FunctionalGroupMatch]:
        matches: list[FunctionalGroupMatch] = []
        for left, right, data in graph.edges(data=True):
            if data.get("order") != order:
                continue
            if graph.nodes[left].get("element") != element:
                continue
            if graph.nodes[right].get("element") != element:
                continue
            mapping = {1: left, 2: right}
            if pattern.validator is not None and not pattern.validator(graph, mapping):
                continue
            matches.append(
                FunctionalGroupMatch(
                    name=pattern.name,
                    group_nodes=tuple(sorted((left, right))),
                    mapping=mapping,
                    pattern=pattern,
                )
            )
        return matches

    return recognize
