from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from .catalog import default_registry
from .model import FunctionalGroupMatch, FunctionalGroupPattern, FunctionalGroupRegistry
from .ring_system import AromaticRingSystemDetector


def _node_match(host: dict[str, Any], pattern: dict[str, Any]) -> bool:
    element = pattern.get("element")
    if element is not None:
        allowed = element if isinstance(element, tuple) else (element,)
        if host.get("element") not in allowed:
            return False

    for attr in ("aromatic", "charge", "radical", "in_ring"):
        if attr in pattern and host.get(attr) != pattern[attr]:
            return False

    hcount = host.get("hcount", 0)
    if "hcount_min" in pattern and hcount < pattern["hcount_min"]:
        return False
    if "hcount_max" in pattern and hcount > pattern["hcount_max"]:
        return False
    return True


def _edge_match(host: dict[str, Any], pattern: dict[str, Any]) -> bool:
    if "order" in pattern and host.get("order") != pattern["order"]:
        return False
    if "aromatic" in pattern and host.get("aromatic") != pattern["aromatic"]:
        return False
    return True


class FunctionalGroupDetector:
    """Detect functional groups from an input molecular ``nx.Graph``."""

    def __init__(self, registry: FunctionalGroupRegistry | None = None) -> None:
        self.registry = default_registry() if registry is None else registry
        self.profile_counts: Counter[str] = Counter()

    def raw_matches(
        self,
        graph: nx.Graph,
        *,
        include_internal: bool = False,
    ) -> list[FunctionalGroupMatch]:
        """Return raw public matches before hierarchy resolution."""
        matches: list[FunctionalGroupMatch] = []
        matched_names: set[str] = set()
        for pattern in self.registry.execution_order():
            if pattern.requires and not any(
                required in matched_names for required in pattern.requires
            ):
                self.profile_counts[f"skip:{pattern.name}"] += 1
                continue
            self.profile_counts[f"attempt:{pattern.name}"] += 1
            if pattern.recognizer is not None:
                found = pattern.recognizer(graph, pattern)
                matches.extend(found)
                if found:
                    matched_names.add(pattern.name)
                continue
            if pattern.anchor_node is not None:
                anchor_data = pattern.graph.nodes[pattern.anchor_node]
                anchor_candidates = [
                    node
                    for node, host_data in graph.nodes(data=True)
                    if _node_match(host_data, anchor_data)
                ]
                if not anchor_candidates:
                    continue
            matcher = GraphMatcher(
                graph,
                pattern.graph,
                node_match=_node_match,
                edge_match=_edge_match,
            )
            seen: set[tuple[int, ...]] = set()
            for host_to_pattern in matcher.subgraph_monomorphisms_iter():
                mapping = {
                    pattern_node: host_node
                    for host_node, pattern_node in host_to_pattern.items()
                }
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
            if any(match.name == pattern.name for match in matches):
                matched_names.add(pattern.name)
        matches.extend(self._heteroaromatic_matches(graph))
        if include_internal:
            return matches
        return [match for match in matches if match.pattern.public]

    def matches(self, graph: nx.Graph) -> list[FunctionalGroupMatch]:
        """Return hierarchy-resolved matches."""
        raw = self.raw_matches(graph)
        raw.sort(
            key=lambda match: (
                match.pattern.priority,
                len(match.group_nodes),
                match.name,
                match.group_nodes,
            ),
            reverse=True,
        )

        accepted: list[FunctionalGroupMatch] = []
        for candidate in raw:
            if any(self._suppressed_by(candidate, chosen) for chosen in accepted):
                continue
            accepted.append(candidate)
        return sorted(accepted, key=lambda match: (match.group_nodes, match.name))

    def detect(self, graph: nx.Graph) -> list[tuple[str, tuple[int, ...]]]:
        """Return simple ``(name, node_ids)`` functional-group labels."""
        return [(match.name, match.group_nodes) for match in self.matches(graph)]

    def _suppressed_by(
        self,
        candidate: FunctionalGroupMatch,
        chosen: FunctionalGroupMatch,
    ) -> bool:
        if not self.registry.is_ancestor(candidate.name, chosen.name):
            if candidate.name not in chosen.pattern.suppresses:
                return False
        return set(candidate.group_nodes).issubset(chosen.group_nodes)

    @staticmethod
    def _heteroaromatic_matches(graph: nx.Graph) -> list[FunctionalGroupMatch]:
        matches: list[FunctionalGroupMatch] = []
        for system in AromaticRingSystemDetector.detect(graph):
            if not system.hetero_nodes:
                continue
            pattern_graph = nx.Graph()
            pattern = FunctionalGroupPattern(
                name="heteroaromatic_ring",
                graph=pattern_graph,
                group_nodes=(),
                suppresses=("amine",),
                priority=60,
            )
            matches.append(
                FunctionalGroupMatch(
                    name="heteroaromatic_ring",
                    group_nodes=system.nodes,
                    mapping={},
                    pattern=pattern,
                )
            )
            for (
                classifier_name,
                group_nodes,
            ) in FunctionalGroupDetector._classify_ring_system(
                graph,
                system,
            ):
                named_pattern = FunctionalGroupPattern(
                    name=classifier_name,
                    graph=nx.Graph(),
                    group_nodes=(),
                    priority=70,
                )
                matches.append(
                    FunctionalGroupMatch(
                        name=classifier_name,
                        group_nodes=group_nodes,
                        mapping={},
                        pattern=named_pattern,
                    )
                )
        return matches

    @staticmethod
    def _classify_ring_system(
        graph: nx.Graph, system
    ) -> list[tuple[str, tuple[int, ...]]]:
        labels: list[tuple[str, tuple[int, ...]]] = []
        for ring in system.subrings:
            ring_size = len(ring.nodes)
            counts = ring.element_counts
            sequence = ring.hetero_sequence
            if counts == {"C": 5, "N": 1} and ring_size == 6:
                labels.append(("pyridine", ring.nodes))
            elif counts == {"C": 4, "N": 2} and ring_size == 6:
                labels.append(("diazine", ring.nodes))
            elif counts == {"C": 4, "N": 1} and ring_size == 5:
                labels.append(("pyrrole", ring.nodes))
            elif counts == {"C": 4, "O": 1} and ring_size == 5:
                labels.append(("furan", ring.nodes))
            elif counts == {"C": 4, "S": 1} and ring_size == 5:
                labels.append(("thiophene", ring.nodes))
            elif counts == {"C": 3, "N": 2} and ring_size == 5:
                if sequence == ("C", "C", "C", "N", "N"):
                    labels.append(("pyrazole", ring.nodes))
                elif sequence == ("C", "C", "N", "C", "N"):
                    labels.append(("imidazole", ring.nodes))
            elif counts == {"C": 3, "N": 1, "S": 1} and ring_size == 5:
                if sequence == ("C", "C", "C", "N", "S"):
                    labels.append(("isothiazole", ring.nodes))
                elif sequence == ("C", "C", "N", "C", "S"):
                    labels.append(("thiazole", ring.nodes))
            elif counts == {"C": 3, "N": 1, "O": 1} and ring_size == 5:
                if sequence == ("C", "C", "C", "N", "O"):
                    labels.append(("isoxazole", ring.nodes))
                elif sequence == ("C", "C", "N", "C", "O"):
                    labels.append(("oxazole", ring.nodes))
            elif counts == {"C": 2, "N": 3} and ring_size == 5:
                labels.append(("triazole", ring.nodes))
            elif counts == {"C": 2, "N": 2, "O": 1} and ring_size == 5:
                labels.append(("oxadiazole", ring.nodes))
            elif counts == {"C": 1, "N": 4} and ring_size == 5:
                labels.append(("tetrazole", ring.nodes))
            elif counts == {"C": 2, "N": 2, "S": 1} and ring_size == 5:
                labels.append(("thiadiazole", ring.nodes))
            elif counts == {"C": 3, "N": 3} and ring_size == 6:
                labels.append(("triazine", ring.nodes))
        labels.extend(FunctionalGroupDetector._classify_fused_ring_system(system))
        return labels

    @staticmethod
    def _classify_fused_ring_system(system) -> list[tuple[str, tuple[int, ...]]]:
        if not system.is_fused or system.ring_sizes != (5, 6):
            return []
        sequences = {ring.hetero_sequence for ring in system.subrings}
        if (
            system.element_counts == {"C": 8, "N": 1}
            and (
                "C",
                "C",
                "C",
                "C",
                "N",
            )
            in sequences
        ):
            return [("indole", system.nodes)]
        if system.element_counts == {"C": 7, "N": 2}:
            if ("C", "C", "N", "C", "N") in sequences:
                return [("benzimidazole", system.nodes)]
            if ("C", "C", "C", "N", "N") in sequences:
                return [("indazole", system.nodes)]
        if (
            system.element_counts == {"C": 7, "N": 1, "O": 1}
            and (
                "C",
                "C",
                "N",
                "C",
                "O",
            )
            in sequences
        ):
            return [("benzoxazole", system.nodes)]
        if (
            system.element_counts == {"C": 7, "N": 1, "S": 1}
            and (
                "C",
                "C",
                "N",
                "C",
                "S",
            )
            in sequences
        ):
            return [("benzothiazole", system.nodes)]
        return []
