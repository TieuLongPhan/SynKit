from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from .model import FunctionalGroupPattern, FunctionalGroupRegistry, Mapping
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


def default_registry() -> FunctionalGroupRegistry:
    """Build the default graph-native functional-group registry."""
    patterns = [
        FunctionalGroupPattern(
            "carbonyl",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "O"})],
                [(1, 2, {"order": 2.0})],
            ),
            (1, 2),
            anchor_node=2,
            priority=10,
            recognizer=_two_node_bond_recognizer("C", ("O",), 2.0),
        ),
        FunctionalGroupPattern(
            "aldehyde",
            _graph(
                [(1, {"element": "C", "hcount_min": 1}), (2, {"element": "O"})],
                [(1, 2, {"order": 2.0})],
            ),
            (1, 2),
            parents=("carbonyl",),
            requires=("carbonyl",),
            anchor_node=2,
            priority=30,
            validator=_aldehyde,
            recognizer=_two_node_bond_recognizer("C", ("O",), 2.0),
        ),
        FunctionalGroupPattern(
            "ketone",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "O"})],
                [(1, 2, {"order": 2.0})],
            ),
            (1, 2),
            parents=("carbonyl",),
            requires=("carbonyl",),
            anchor_node=2,
            priority=20,
            validator=lambda graph, mapping: graph.nodes[mapping[1]].get("hcount", 0)
            == 0,
            recognizer=_two_node_bond_recognizer("C", ("O",), 2.0),
        ),
        FunctionalGroupPattern(
            "carboxylic_acid",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "O"}),
                    (3, {"element": "O", "hcount_min": 1}),
                ],
                [(1, 2, {"order": 2.0}), (1, 3, {"order": 1.0})],
            ),
            (1, 2, 3),
            parents=("ester",),
            requires=("carbonyl",),
            anchor_node=3,
            priority=60,
        ),
        FunctionalGroupPattern(
            "amide",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "O"}), (3, {"element": "N"})],
                [(1, 2, {"order": 2.0}), (1, 3, {"order": 1.0})],
            ),
            (1, 2, 3),
            parents=("ketone", "amine"),
            requires=("carbonyl",),
            anchor_node=3,
            priority=50,
        ),
        FunctionalGroupPattern(
            "carbamate",
            _graph(
                [
                    (1, {"element": "O"}),
                    (2, {"element": "C"}),
                    (3, {"element": "O"}),
                    (4, {"element": "N"}),
                ],
                [
                    (1, 2, {"order": 1.0}),
                    (2, 3, {"order": 2.0}),
                    (2, 4, {"order": 1.0}),
                ],
            ),
            (1, 2, 3, 4),
            parents=("amide", "ester"),
            requires=("carbonyl",),
            anchor_node=4,
            priority=60,
        ),
        FunctionalGroupPattern(
            "ester",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "O"}),
                    (3, {"element": "O"}),
                    (4, {"element": "C"}),
                ],
                [
                    (1, 2, {"order": 2.0}),
                    (1, 3, {"order": 1.0}),
                    (3, 4, {"order": 1.0}),
                ],
            ),
            (1, 2, 3),
            parents=("ketone", "ether"),
            requires=("carbonyl",),
            anchor_node=3,
            priority=50,
        ),
        FunctionalGroupPattern(
            "alcohol",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "O", "hcount_min": 1})],
                [(1, 2, {"order": 1.0})],
            ),
            (1, 2),
            parents=("ether",),
            anchor_node=2,
            priority=20,
            validator=_alcohol_like,
            recognizer=_two_node_bond_recognizer("C", ("O",), 1.0),
        ),
        FunctionalGroupPattern(
            "primary_alcohol",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "O", "hcount_min": 1})],
                [(1, 2, {"order": 1.0})],
            ),
            (1, 2),
            parents=("alcohol",),
            anchor_node=2,
            priority=30,
            validator=_alcohol_carbon_heavy_degree(1),
            recognizer=_two_node_bond_recognizer("C", ("O",), 1.0),
        ),
        FunctionalGroupPattern(
            "secondary_alcohol",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "O", "hcount_min": 1})],
                [(1, 2, {"order": 1.0})],
            ),
            (1, 2),
            parents=("primary_alcohol",),
            requires=("alcohol",),
            anchor_node=2,
            priority=40,
            validator=_alcohol_carbon_heavy_degree(2),
            recognizer=_two_node_bond_recognizer("C", ("O",), 1.0),
        ),
        FunctionalGroupPattern(
            "tertiary_alcohol",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "O", "hcount_min": 1})],
                [(1, 2, {"order": 1.0})],
            ),
            (1, 2),
            parents=("secondary_alcohol",),
            requires=("alcohol",),
            anchor_node=2,
            priority=50,
            validator=_alcohol_carbon_heavy_degree(3),
            recognizer=_two_node_bond_recognizer("C", ("O",), 1.0),
        ),
        FunctionalGroupPattern(
            "oxygen_link",
            _graph(
                [(1, {"element": "O"}), (2, {"element": "C"})],
                [(1, 2, {"order": 1.0})],
            ),
            (1,),
            priority=0,
            recognizer=_two_node_bond_recognizer("O", ("C",), 1.0),
            public=False,
        ),
        FunctionalGroupPattern(
            "ether",
            _graph(
                [(1, {"element": "O"}), (2, {"element": "C"}), (3, {"element": "C"})],
                [(1, 2, {"order": 1.0}), (1, 3, {"order": 1.0})],
            ),
            (1,),
            requires=("oxygen_link",),
            anchor_node=1,
            priority=10,
        ),
        FunctionalGroupPattern(
            "acetal",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "O"}),
                    (3, {"element": "C"}),
                    (4, {"element": "O"}),
                    (5, {"element": "C"}),
                ],
                [
                    (1, 2, {"order": 1.0}),
                    (2, 3, {"order": 1.0}),
                    (1, 4, {"order": 1.0}),
                    (4, 5, {"order": 1.0}),
                ],
            ),
            (1, 2, 4),
            parents=("ketal",),
            requires=("oxygen_link",),
            anchor_node=1,
            priority=50,
            validator=lambda graph, mapping: graph.nodes[mapping[1]].get("hcount", 0)
            >= 1,
        ),
        FunctionalGroupPattern(
            "ketal",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "O"}),
                    (3, {"element": "C"}),
                    (4, {"element": "O"}),
                    (5, {"element": "C"}),
                ],
                [
                    (1, 2, {"order": 1.0}),
                    (2, 3, {"order": 1.0}),
                    (1, 4, {"order": 1.0}),
                    (4, 5, {"order": 1.0}),
                ],
            ),
            (1, 2, 4),
            parents=("ether",),
            requires=("oxygen_link",),
            anchor_node=1,
            priority=40,
            validator=lambda graph, mapping: graph.nodes[mapping[1]].get("hcount", 0)
            == 0
            and _carbon_substituent_count(graph, mapping[1], {mapping[2], mapping[4]})
            >= 2,
        ),
        FunctionalGroupPattern(
            "hemiacetal",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "O"}),
                    (3, {"element": "C"}),
                    (4, {"element": "O"}),
                ],
                [
                    (1, 2, {"order": 1.0}),
                    (2, 3, {"order": 1.0}),
                    (1, 4, {"order": 1.0}),
                ],
            ),
            (1, 2, 4),
            parents=("hemiketal",),
            requires=("oxygen_link",),
            suppresses=(
                "alcohol",
                "primary_alcohol",
                "secondary_alcohol",
                "tertiary_alcohol",
            ),
            anchor_node=1,
            priority=60,
            validator=lambda graph, mapping: graph.nodes[mapping[1]].get("hcount", 0)
            >= 1
            and _oxygen_has_h(graph, mapping[4]),
        ),
        FunctionalGroupPattern(
            "hemiketal",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "O"}),
                    (3, {"element": "C"}),
                    (4, {"element": "O"}),
                ],
                [
                    (1, 2, {"order": 1.0}),
                    (2, 3, {"order": 1.0}),
                    (1, 4, {"order": 1.0}),
                ],
            ),
            (1, 2, 4),
            parents=("ketal", "alcohol"),
            requires=("oxygen_link",),
            suppresses=(
                "alcohol",
                "primary_alcohol",
                "secondary_alcohol",
                "tertiary_alcohol",
            ),
            anchor_node=1,
            priority=60,
            validator=lambda graph, mapping: _oxygen_has_h(graph, mapping[4])
            and graph.nodes[mapping[1]].get("hcount", 0) == 0
            and _carbon_substituent_count(graph, mapping[1], {mapping[2], mapping[4]})
            >= 2,
        ),
        FunctionalGroupPattern(
            "amine",
            _graph([(1, {"element": "N"})], []),
            (1,),
            anchor_node=1,
            priority=10,
            validator=_amine,
            recognizer=_single_node_recognizer("N"),
        ),
        FunctionalGroupPattern(
            "aniline",
            _graph(
                [(1, {"element": "C", "aromatic": True}), (2, {"element": "N"})],
                [(1, 2, {"order": 1.0})],
            ),
            (2,),
            parents=("amine",),
            requires=("amine",),
            anchor_node=2,
            priority=30,
            validator=_aniline,
            recognizer=_two_node_bond_recognizer("C", ("N",), 1.0),
        ),
        FunctionalGroupPattern(
            "nitrile",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "N"})],
                [(1, 2, {"order": 3.0})],
            ),
            (1, 2),
            anchor_node=2,
            priority=30,
            recognizer=_two_node_bond_recognizer("C", ("N",), 3.0),
        ),
        FunctionalGroupPattern(
            "nitroso",
            _graph(
                [(1, {"element": "N"}), (2, {"element": "O"})],
                [(1, 2, {"order": 2.0})],
            ),
            (1, 2),
            anchor_node=1,
            priority=30,
            recognizer=_two_node_bond_recognizer("N", ("O",), 2.0),
        ),
        FunctionalGroupPattern(
            "nitro",
            _graph(
                [(1, {"element": "N"}), (2, {"element": "O"}), (3, {"element": "O"})],
                [(1, 2, {"order": 2.0}), (1, 3, {"order": 1.0})],
            ),
            (1, 2, 3),
            parents=("nitroso",),
            requires=("nitroso",),
            anchor_node=1,
            priority=40,
        ),
        FunctionalGroupPattern(
            "thioether",
            _graph(
                [(1, {"element": "S"}), (2, {"element": "C"}), (3, {"element": "C"})],
                [(1, 2, {"order": 1.0}), (1, 3, {"order": 1.0})],
            ),
            (1,),
            anchor_node=1,
            priority=20,
        ),
        FunctionalGroupPattern(
            "sulfoxide",
            _graph(
                [
                    (1, {"element": "S"}),
                    (2, {"element": "O"}),
                ],
                [(1, 2, {"order": 2.0})],
            ),
            (1, 2),
            suppresses=("thioether",),
            anchor_node=1,
            priority=30,
        ),
        FunctionalGroupPattern(
            "sulfone",
            _graph(
                [
                    (1, {"element": "S"}),
                    (2, {"element": "O"}),
                    (3, {"element": "O"}),
                ],
                [(1, 2, {"order": 2.0}), (1, 3, {"order": 2.0})],
            ),
            (1, 2, 3),
            parents=("sulfoxide",),
            requires=("sulfoxide",),
            suppresses=("thioether",),
            anchor_node=1,
            priority=40,
        ),
        FunctionalGroupPattern(
            "sulfonamide",
            _graph(
                [
                    (1, {"element": "S"}),
                    (2, {"element": "O"}),
                    (3, {"element": "O"}),
                    (4, {"element": "N"}),
                ],
                [
                    (1, 2, {"order": 2.0}),
                    (1, 3, {"order": 2.0}),
                    (1, 4, {"order": 1.0}),
                ],
            ),
            (1, 2, 3, 4),
            parents=("sulfone", "amine"),
            requires=("sulfone",),
            suppresses=("thioether",),
            anchor_node=1,
            priority=50,
        ),
        FunctionalGroupPattern(
            "thioester",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "O"}),
                    (3, {"element": "S"}),
                    (4, {"element": "C"}),
                ],
                [
                    (1, 2, {"order": 2.0}),
                    (1, 3, {"order": 1.0}),
                    (3, 4, {"order": 1.0}),
                ],
            ),
            (1, 2, 3),
            parents=("ketone", "thioether"),
            requires=("carbonyl",),
            anchor_node=3,
            priority=50,
        ),
        FunctionalGroupPattern(
            "phenol",
            _graph(
                [(1, {"element": "C", "aromatic": True}), (2, {"element": "O"})],
                [(1, 2, {"order": 1.0})],
            ),
            (2,),
            parents=("alcohol",),
            anchor_node=2,
            priority=50,
            validator=_phenol,
            recognizer=_two_node_bond_recognizer("C", ("O",), 1.0),
        ),
        FunctionalGroupPattern(
            "enol",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "C"}), (3, {"element": "O"})],
                [(1, 2, {"order": 2.0}), (2, 3, {"order": 1.0})],
            ),
            (1, 2, 3),
            parents=("alcohol",),
            anchor_node=3,
            priority=40,
            validator=_enol,
        ),
        FunctionalGroupPattern(
            "peroxide",
            _graph(
                [(1, {"element": "O"}), (2, {"element": "O"})],
                [(1, 2, {"order": 1.0})],
            ),
            (1, 2),
            parents=("ether",),
            anchor_node=1,
            priority=30,
            recognizer=_symmetric_two_node_bond_recognizer("O", 1.0),
        ),
        FunctionalGroupPattern(
            "peroxy_acid",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "O"}),
                    (3, {"element": "O"}),
                    (4, {"element": "O", "hcount_min": 1}),
                ],
                [
                    (1, 2, {"order": 2.0}),
                    (1, 3, {"order": 1.0}),
                    (3, 4, {"order": 1.0}),
                ],
            ),
            (1, 2, 3, 4),
            parents=("ester", "peroxide"),
            requires=("carbonyl",),
            anchor_node=3,
            priority=60,
        ),
        FunctionalGroupPattern(
            "anhydride",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "O"}),
                    (3, {"element": "O"}),
                    (4, {"element": "C"}),
                    (5, {"element": "O"}),
                ],
                [
                    (1, 2, {"order": 2.0}),
                    (1, 3, {"order": 1.0}),
                    (3, 4, {"order": 1.0}),
                    (4, 5, {"order": 2.0}),
                ],
            ),
            (1, 2, 3, 4, 5),
            parents=("ester",),
            requires=("carbonyl",),
            anchor_node=3,
            priority=60,
        ),
        FunctionalGroupPattern(
            "acyl_chloride",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "O"}), (3, {"element": "Cl"})],
                [(1, 2, {"order": 2.0}), (1, 3, {"order": 1.0})],
            ),
            (1, 2, 3),
            parents=("ketone",),
            requires=("carbonyl",),
            suppresses=("organohalide",),
            anchor_node=3,
            priority=50,
        ),
        FunctionalGroupPattern(
            "epoxide",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "C"}), (3, {"element": "O"})],
                [
                    (1, 2, {"order": 1.0}),
                    (1, 3, {"order": 1.0}),
                    (2, 3, {"order": 1.0}),
                ],
            ),
            (1, 2, 3),
            parents=("ether",),
            requires=("oxygen_link",),
            anchor_node=3,
            priority=40,
            validator=_epoxide,
        ),
        FunctionalGroupPattern(
            "boronic_acid",
            _graph(
                [
                    (1, {"element": "B"}),
                    (2, {"element": "O", "hcount_min": 1}),
                    (3, {"element": "O", "hcount_min": 1}),
                ],
                [(1, 2, {"order": 1.0}), (1, 3, {"order": 1.0})],
            ),
            (1, 2, 3),
            anchor_node=1,
            priority=40,
        ),
        FunctionalGroupPattern(
            "boronate_ester",
            _graph(
                [
                    (1, {"element": "B"}),
                    (2, {"element": "O"}),
                    (3, {"element": "C"}),
                    (4, {"element": "O"}),
                    (5, {"element": "C"}),
                ],
                [
                    (1, 2, {"order": 1.0}),
                    (2, 3, {"order": 1.0}),
                    (1, 4, {"order": 1.0}),
                    (4, 5, {"order": 1.0}),
                ],
            ),
            (1, 2, 4),
            anchor_node=1,
            priority=40,
        ),
        FunctionalGroupPattern(
            "silyl_ether",
            _graph(
                [(1, {"element": "O"}), (2, {"element": "Si"})],
                [(1, 2, {"order": 1.0})],
            ),
            (1, 2),
            anchor_node=2,
            priority=30,
            recognizer=_two_node_bond_recognizer("O", ("Si",), 1.0),
        ),
        FunctionalGroupPattern(
            "phosphate",
            _graph(
                [
                    (1, {"element": "P"}),
                    (2, {"element": "O"}),
                    (3, {"element": "O"}),
                    (4, {"element": "O"}),
                    (5, {"element": "O"}),
                ],
                [
                    (1, 2, {"order": 2.0}),
                    (1, 3, {"order": 1.0}),
                    (1, 4, {"order": 1.0}),
                    (1, 5, {"order": 1.0}),
                ],
            ),
            (1, 2, 3, 4, 5),
            anchor_node=1,
            priority=40,
        ),
        FunctionalGroupPattern(
            "phosphonate",
            _graph(
                [
                    (1, {"element": "P"}),
                    (2, {"element": "O"}),
                    (3, {"element": "O"}),
                    (4, {"element": "O"}),
                    (5, {"element": "C"}),
                ],
                [
                    (1, 2, {"order": 2.0}),
                    (1, 3, {"order": 1.0}),
                    (1, 4, {"order": 1.0}),
                    (1, 5, {"order": 1.0}),
                ],
            ),
            (1, 2, 3, 4, 5),
            anchor_node=1,
            priority=40,
        ),
        FunctionalGroupPattern(
            "phosphine_oxide",
            _graph(
                [
                    (1, {"element": "P"}),
                    (2, {"element": "O"}),
                    (3, {"element": "C"}),
                    (4, {"element": "C"}),
                    (5, {"element": "C"}),
                ],
                [
                    (1, 2, {"order": 2.0}),
                    (1, 3, {"order": 1.0}),
                    (1, 4, {"order": 1.0}),
                    (1, 5, {"order": 1.0}),
                ],
            ),
            (1, 2, 3, 4, 5),
            anchor_node=1,
            priority=40,
        ),
        FunctionalGroupPattern(
            "phosphite",
            _graph(
                [
                    (1, {"element": "P"}),
                    (2, {"element": "O"}),
                    (3, {"element": "O"}),
                    (4, {"element": "O"}),
                ],
                [
                    (1, 2, {"order": 1.0}),
                    (1, 3, {"order": 1.0}),
                    (1, 4, {"order": 1.0}),
                ],
            ),
            (1, 2, 3, 4),
            anchor_node=1,
            priority=30,
            validator=_phosphite,
        ),
        FunctionalGroupPattern(
            "isocyanate",
            _graph(
                [
                    (1, {"element": "O"}),
                    (2, {"element": "C"}),
                    (3, {"element": "N"}),
                ],
                [(1, 2, {"order": 2.0}), (2, 3, {"order": 2.0})],
            ),
            (1, 2, 3),
            suppresses=("carbonyl", "ketone"),
            anchor_node=2,
            priority=40,
        ),
        FunctionalGroupPattern(
            "oxime",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "N"}),
                    (3, {"element": "O"}),
                ],
                [(1, 2, {"order": 2.0}), (2, 3, {"order": 1.0})],
            ),
            (1, 2, 3),
            anchor_node=2,
            priority=40,
        ),
        FunctionalGroupPattern(
            "hydrazone",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "N"}),
                    (3, {"element": "N"}),
                ],
                [(1, 2, {"order": 2.0}), (2, 3, {"order": 1.0})],
            ),
            (1, 2, 3),
            suppresses=("amine",),
            anchor_node=2,
            priority=40,
            validator=_hydrazone,
        ),
        FunctionalGroupPattern(
            "imine",
            _graph(
                [(1, {"element": "C"}), (2, {"element": "N"})],
                [(1, 2, {"order": 2.0})],
            ),
            (1, 2),
            anchor_node=2,
            priority=20,
            validator=_imine,
            recognizer=_two_node_bond_recognizer("C", ("N",), 2.0),
        ),
        FunctionalGroupPattern(
            "amidine",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "N"}),
                    (3, {"element": "N"}),
                ],
                [(1, 2, {"order": 2.0}), (1, 3, {"order": 1.0})],
            ),
            (1, 2, 3),
            suppresses=("amine",),
            anchor_node=1,
            priority=40,
            validator=_amidine,
        ),
        FunctionalGroupPattern(
            "amidoxime",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "N"}),
                    (3, {"element": "N"}),
                    (4, {"element": "O"}),
                ],
                [
                    (1, 2, {"order": 2.0}),
                    (1, 3, {"order": 1.0}),
                    (2, 4, {"order": 1.0}),
                ],
            ),
            (1, 2, 3, 4),
            suppresses=("amine", "oxime"),
            anchor_node=1,
            priority=50,
        ),
        FunctionalGroupPattern(
            "azide",
            _graph(
                [
                    (1, {"element": "N"}),
                    (2, {"element": "N"}),
                    (3, {"element": "N"}),
                ],
                [(1, 2, {"order": 2.0}), (2, 3, {"order": 2.0})],
            ),
            (1, 2, 3),
            anchor_node=2,
            priority=40,
            validator=_azide,
        ),
        FunctionalGroupPattern(
            "azo",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "N"}),
                    (3, {"element": "N"}),
                    (4, {"element": "C"}),
                ],
                [
                    (1, 2, {"order": 1.0}),
                    (2, 3, {"order": 2.0}),
                    (3, 4, {"order": 1.0}),
                ],
            ),
            (2, 3),
            anchor_node=2,
            priority=40,
        ),
        FunctionalGroupPattern(
            "isothiocyanate",
            _graph(
                [
                    (1, {"element": "S"}),
                    (2, {"element": "C"}),
                    (3, {"element": "N"}),
                ],
                [(1, 2, {"order": 2.0}), (2, 3, {"order": 2.0})],
            ),
            (1, 2, 3),
            anchor_node=2,
            priority=50,
        ),
        FunctionalGroupPattern(
            "thiourea",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "S"}),
                    (3, {"element": "N"}),
                    (4, {"element": "N"}),
                ],
                [
                    (1, 2, {"order": 2.0}),
                    (1, 3, {"order": 1.0}),
                    (1, 4, {"order": 1.0}),
                ],
            ),
            (1, 2, 3, 4),
            suppresses=("amine", "thioamide"),
            anchor_node=1,
            priority=50,
        ),
        FunctionalGroupPattern(
            "thioamide",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": "S"}),
                    (3, {"element": "N"}),
                ],
                [(1, 2, {"order": 2.0}), (1, 3, {"order": 1.0})],
            ),
            (1, 2, 3),
            suppresses=("amine",),
            anchor_node=1,
            priority=40,
        ),
        FunctionalGroupPattern(
            "organohalide",
            _graph(
                [
                    (1, {"element": "C"}),
                    (2, {"element": ("F", "Cl", "Br", "I")}),
                ],
                [(1, 2, {"order": 1.0})],
            ),
            (1, 2),
            anchor_node=2,
            priority=20,
            recognizer=_two_node_bond_recognizer("C", ("F", "Cl", "Br", "I"), 1.0),
        ),
        FunctionalGroupPattern(
            "aryl_halide",
            _graph(
                [
                    (1, {"element": "C", "aromatic": True}),
                    (2, {"element": ("F", "Cl", "Br", "I")}),
                ],
                [(1, 2, {"order": 1.0})],
            ),
            (1, 2),
            parents=("organohalide",),
            requires=("organohalide",),
            anchor_node=2,
            priority=30,
            validator=_aryl_halide,
            recognizer=_two_node_bond_recognizer("C", ("F", "Cl", "Br", "I"), 1.0),
        ),
    ]
    return FunctionalGroupRegistry(patterns)
