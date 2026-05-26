from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import networkx as nx

from synkit.Chem.Reaction.standardize import Standardize
from synkit.IO.chem_converter import smiles_to_graph

from .detector import FunctionalGroupDetector
from .ring_system import AromaticRingSystemDetector


@dataclass(frozen=True)
class FunctionalGroupAudit:
    """Aggregated detector coverage over a reaction-SMILES corpus."""

    reactions: int
    molecules: int
    parse_failures: int
    elapsed_seconds: float
    label_counts: Counter[str]
    heteroaromatic_systems: int
    named_heteroaromatic_systems: int
    unnamed_heteroaromatic_systems: Counter[tuple]
    uncovered_atom_signatures: Counter[tuple]
    uncovered_edge_signatures: Counter[tuple]

    @property
    def unnamed_heteroaromatic_count(self) -> int:
        return self.heteroaromatic_systems - self.named_heteroaromatic_systems


def audit_reaction_smiles(
    reactions: Iterable[str],
    *,
    standardizer: Standardize | None = None,
) -> FunctionalGroupAudit:
    """Audit FG coverage for an iterable of reaction SMILES strings."""
    std = Standardize() if standardizer is None else standardizer
    detector = FunctionalGroupDetector()

    reaction_count = 0
    molecule_count = 0
    parse_failures = 0
    heteroaromatic_systems = 0
    named_heteroaromatic_systems = 0
    label_counts: Counter[str] = Counter()
    unnamed_systems: Counter[tuple] = Counter()
    uncovered_atoms: Counter[tuple] = Counter()
    uncovered_edges: Counter[tuple] = Counter()

    started = perf_counter()
    for reaction in reactions:
        reaction_count += 1
        standardized = std.fit(reaction, remove_aam=True)
        for side in standardized.split(">>"):
            for smiles in side.split("."):
                graph = smiles_to_graph(
                    smiles,
                    drop_non_aam=False,
                    use_index_as_atom_map=True,
                )
                if graph is None:
                    parse_failures += 1
                    continue
                molecule_count += 1
                matches = detector.matches(graph)
                label_counts.update(match.name for match in matches)
                covered = {node for match in matches for node in match.group_nodes}
                _count_uncovered_signatures(
                    graph,
                    covered,
                    uncovered_atoms,
                    uncovered_edges,
                )

                named_ring_nodes = {
                    match.group_nodes
                    for match in matches
                    if match.name != "heteroaromatic_ring"
                    and match.pattern.priority == 70
                }
                for system in AromaticRingSystemDetector.detect(graph):
                    if not system.hetero_nodes:
                        continue
                    heteroaromatic_systems += 1
                    has_named_subring = any(
                        set(nodes).issubset(system.nodes) for nodes in named_ring_nodes
                    )
                    if has_named_subring:
                        named_heteroaromatic_systems += 1
                        continue
                    unnamed_systems[
                        (
                            system.hetero_pattern,
                            system.is_fused,
                            system.ring_sizes,
                            tuple(sorted(system.element_counts.items())),
                        )
                    ] += 1

    return FunctionalGroupAudit(
        reactions=reaction_count,
        molecules=molecule_count,
        parse_failures=parse_failures,
        elapsed_seconds=perf_counter() - started,
        label_counts=label_counts,
        heteroaromatic_systems=heteroaromatic_systems,
        named_heteroaromatic_systems=named_heteroaromatic_systems,
        unnamed_heteroaromatic_systems=unnamed_systems,
        uncovered_atom_signatures=uncovered_atoms,
        uncovered_edge_signatures=uncovered_edges,
    )


def _count_uncovered_signatures(
    graph: nx.Graph,
    covered: set[int],
    atom_counts: Counter[tuple],
    edge_counts: Counter[tuple],
) -> None:
    for node, data in graph.nodes(data=True):
        if data.get("element") == "H" or node in covered:
            continue
        neighbors = tuple(
            sorted(
                graph.nodes[neighbor].get("element")
                for neighbor in graph.neighbors(node)
                if graph.nodes[neighbor].get("element") != "H"
            )
        )
        atom_counts[
            (
                data.get("element"),
                data.get("aromatic", False),
                data.get("hcount", 0),
                neighbors,
            )
        ] += 1

    for left, right, data in graph.edges(data=True):
        if left in covered or right in covered:
            continue
        left_element = graph.nodes[left].get("element")
        right_element = graph.nodes[right].get("element")
        if "H" in {left_element, right_element}:
            continue
        edge_counts[
            tuple(sorted((left_element, right_element)))
            + (data.get("order"), data.get("aromatic", False))
        ] += 1
