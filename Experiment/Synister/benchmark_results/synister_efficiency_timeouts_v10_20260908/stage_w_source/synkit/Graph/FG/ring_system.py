from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class AromaticSubring:
    """One minimal aromatic cycle inside an aromatic ring system."""

    nodes: tuple[int, ...]
    element_counts: dict[str, int]
    hetero_sequence: tuple[str, ...]


@dataclass(frozen=True)
class AromaticRingSystem:
    """One connected aromatic ring system from a molecular graph."""

    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    hetero_nodes: tuple[int, ...]
    element_counts: dict[str, int]
    ring_sizes: tuple[int, ...]
    subrings: tuple[AromaticSubring, ...]
    is_fused: bool
    hetero_sequence: tuple[str, ...] | None
    hetero_pattern: str


class AromaticRingSystemDetector:
    """Extract aromatic connected components and characterize their rings."""

    @staticmethod
    def detect(graph: nx.Graph) -> list[AromaticRingSystem]:
        aromatic = nx.Graph()
        for node, data in graph.nodes(data=True):
            if data.get("aromatic"):
                aromatic.add_node(node, **data)
        for left, right, data in graph.edges(data=True):
            if left not in aromatic or right not in aromatic:
                continue
            if data.get("aromatic") or data.get("order") == 1.5:
                aromatic.add_edge(left, right, **data)

        systems: list[AromaticRingSystem] = []
        for component_nodes in nx.connected_components(aromatic):
            component = aromatic.subgraph(component_nodes).copy()
            cycles = nx.minimum_cycle_basis(component)
            if not cycles:
                continue
            nodes = tuple(sorted(component.nodes()))
            edges = tuple(sorted(tuple(sorted(edge)) for edge in component.edges()))
            hetero_nodes = tuple(
                sorted(
                    node
                    for node in component.nodes()
                    if component.nodes[node].get("element") not in {"C", "H"}
                )
            )
            element_counts = dict(
                sorted(
                    Counter(
                        component.nodes[node].get("element") for node in component
                    ).items()
                )
            )
            ring_sizes = tuple(sorted(len(cycle) for cycle in cycles))
            subrings = tuple(
                sorted(
                    (
                        AromaticSubring(
                            nodes=tuple(sorted(cycle)),
                            element_counts=dict(
                                sorted(
                                    Counter(
                                        component.nodes[node].get("element")
                                        for node in cycle
                                    ).items()
                                )
                            ),
                            hetero_sequence=AromaticRingSystemDetector._canonical_cycle_sequence(
                                component,
                                cycle,
                            ),
                        )
                        for cycle in cycles
                    ),
                    key=lambda ring: ring.nodes,
                )
            )
            hetero_sequence = None
            if len(cycles) == 1:
                hetero_sequence = subrings[0].hetero_sequence
            systems.append(
                AromaticRingSystem(
                    nodes=nodes,
                    edges=edges,
                    hetero_nodes=hetero_nodes,
                    element_counts=element_counts,
                    ring_sizes=ring_sizes,
                    subrings=subrings,
                    is_fused=len(cycles) > 1,
                    hetero_sequence=hetero_sequence,
                    hetero_pattern=AromaticRingSystemDetector._pattern(
                        element_counts,
                        ring_sizes,
                    ),
                )
            )
        return sorted(systems, key=lambda system: system.nodes)

    @staticmethod
    def _canonical_cycle_sequence(
        graph: nx.Graph,
        cycle_nodes: list[int],
    ) -> tuple[str, ...]:
        cycle = list(cycle_nodes)
        subgraph = graph.subgraph(cycle)
        start = min(cycle)
        neighbors = sorted(subgraph.neighbors(start))
        if len(neighbors) != 2:
            return tuple(graph.nodes[node].get("element") for node in sorted(cycle))

        candidates: list[tuple[str, ...]] = []
        for first in neighbors:
            order = [start, first]
            while len(order) < len(cycle):
                current = order[-1]
                previous = order[-2]
                nxt = [node for node in subgraph.neighbors(current) if node != previous]
                if not nxt:
                    break
                order.append(nxt[0])
            seq = tuple(graph.nodes[node].get("element") for node in order)
            candidates.append(seq)
        rotations: list[tuple[str, ...]] = []
        for seq in candidates:
            for index in range(len(seq)):
                rotations.append(seq[index:] + seq[:index])
        return min(rotations)

    @staticmethod
    def _pattern(element_counts: dict[str, int], ring_sizes: tuple[int, ...]) -> str:
        hetero = [
            f"{count}{element}"
            for element, count in element_counts.items()
            if element not in {"C", "H"}
        ]
        prefix = "-".join(hetero) if hetero else "carbocycle"
        sizes = "-".join(str(size) for size in ring_sizes)
        return f"{prefix}-{sizes}ring"
