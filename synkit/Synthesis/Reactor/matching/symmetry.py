"""Exact mapping quotients used by the synthesis reactor."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Tuple

import networkx as nx

from synkit.Graph.Matcher.subgraph_matcher import electron_aware_edge_match

MappingDict = Dict[Any, Any]


def deduplicate_port_witness_mappings(
    mappings: List[MappingDict],
    structural_pattern: nx.Graph,
) -> List[MappingDict]:
    """Project existential stereo-port witnesses from accepted mappings."""
    structural_nodes = frozenset(structural_pattern)
    if len(mappings) < 2 or not any(
        set(mapping) - structural_nodes for mapping in mappings
    ):
        return mappings
    seen = set()
    unique = []
    for mapping in mappings:
        signature = tuple(
            sorted(
                ((node, mapping[node]) for node in structural_nodes if node in mapping),
                key=repr,
            )
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(mapping)
    return unique


def deduplicate_exact_pattern_mappings(
    mappings: List[MappingDict],
    pattern: nx.Graph,
    *,
    node_attrs: List[str],
    edge_attrs: List[str],
    max_automorphisms: int = 256,
) -> List[MappingDict]:
    """Quotient complete mappings by the exact attributed group action."""
    if len(mappings) < 2 or pattern.is_multigraph():
        return mappings
    pattern_nodes = tuple(sorted(pattern, key=repr))
    if any(not set(pattern_nodes).issubset(mapping) for mapping in mappings):
        return mappings

    def node_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return all(
            left.get(attr, 0 if attr in {"hcount", "lone_pairs"} else None)
            == right.get(attr, 0 if attr in {"hcount", "lone_pairs"} else None)
            for attr in node_attrs
        )

    def edge_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return all(left.get(attr) == right.get(attr) for attr in edge_attrs)

    matcher_cls = (
        nx.algorithms.isomorphism.DiGraphMatcher
        if pattern.is_directed()
        else nx.algorithms.isomorphism.GraphMatcher
    )
    matcher = matcher_cls(
        pattern,
        pattern,
        node_match=node_match,
        edge_match=edge_match,
    )
    automorphisms = []
    for automorphism in matcher.isomorphisms_iter():
        automorphisms.append(automorphism)
        if len(automorphisms) > max_automorphisms:
            return mappings

    seen = set()
    unique = []
    for mapping in mappings:
        signature = min(
            (
                tuple(mapping[automorphism[node]] for node in pattern_nodes)
                for automorphism in automorphisms
            ),
            key=repr,
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(mapping)
    return unique


def deduplicate_free_host_component_mappings(
    mappings: List[MappingDict],
    host: nx.Graph,
    *,
    node_attrs: List[str],
    edge_attrs: List[str],
    max_automorphisms: int = 256,
) -> List[MappingDict]:
    """Quotient by exact automorphisms outside the largest host component."""
    if len(mappings) < 2 or host.is_multigraph():
        return mappings
    component_nodes = list(
        nx.weakly_connected_components(host)
        if host.is_directed()
        else nx.connected_components(host)
    )
    if len(component_nodes) < 2:
        return mappings
    anchor_index = max(
        range(len(component_nodes)),
        key=lambda index: (len(component_nodes[index]), -index),
    )
    component_index = {
        node: index
        for index, component in enumerate(component_nodes)
        for node in component
    }

    def node_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return all(
            left.get(attr, 0 if attr in {"hcount", "lone_pairs"} else None)
            == right.get(attr, 0 if attr in {"hcount", "lone_pairs"} else None)
            for attr in node_attrs
        )

    def edge_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return electron_aware_edge_match(
            left, right, edge_attrs
        ) and electron_aware_edge_match(right, left, edge_attrs)

    automorphisms = {}
    matcher_cls = (
        nx.algorithms.isomorphism.DiGraphMatcher
        if host.is_directed()
        else nx.algorithms.isomorphism.GraphMatcher
    )
    for index, nodes in enumerate(component_nodes):
        if index == anchor_index:
            continue
        component = host.subgraph(nodes)
        group = []
        matcher = matcher_cls(
            component,
            component,
            node_match=node_match,
            edge_match=edge_match,
        )
        for automorphism in matcher.isomorphisms_iter():
            group.append(automorphism)
            if len(group) > max_automorphisms:
                return mappings
        automorphisms[index] = group

    seen = set()
    unique = []
    for mapping in mappings:
        fixed = tuple(
            sorted(
                (
                    (pattern_node, host_node)
                    for pattern_node, host_node in mapping.items()
                    if component_index[host_node] == anchor_index
                ),
                key=repr,
            )
        )
        free = []
        for index, group in automorphisms.items():
            placements = tuple(
                (pattern_node, host_node)
                for pattern_node, host_node in mapping.items()
                if component_index[host_node] == index
            )
            variants = [
                tuple(
                    sorted(
                        (
                            (pattern_node, automorphism[host_node])
                            for pattern_node, host_node in placements
                        ),
                        key=repr,
                    )
                )
                for automorphism in group
            ]
            free.append(min(variants, key=repr))
        signature = fixed, tuple(free)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(mapping)
    return unique


def deduplicate_pure_coupling_mappings(  # noqa: C901
    mappings: List[MappingDict],
    pattern: nx.Graph,
    rule: Any,
    *,
    node_attrs: List[str],
    edge_attrs: List[str],
    max_automorphisms: int = 256,
) -> List[MappingDict]:
    """Deduplicate equivalent applications of a pure stereo coupling rule."""
    if (
        len(mappings) < 2
        or not rule.stereo_couplings
        or rule.stereo_guards
        or rule.stereo_effects
        or rule.stereo_outcomes
    ):
        return mappings

    pattern_by_map = {}
    for node, attrs in pattern.nodes(data=True):
        atom_map = attrs.get("atom_map", node)
        if isinstance(atom_map, int):
            pattern_by_map[atom_map] = node

    dependencies = {
        atom_map
        for coupling in rule.stereo_couplings.values()
        for atom_map in coupling.dependencies
    }
    changed_maps = set()
    for left, right, attrs in rule.rc.raw.edges(data=True):
        order = attrs.get("order")
        if not (isinstance(order, tuple) and len(order) == 2 and order[0] != order[1]):
            continue
        for node in (left, right):
            atom_map = rule.rc.raw.nodes[node].get("atom_map", node)
            if isinstance(atom_map, tuple):
                atom_map = atom_map[0]
            if isinstance(atom_map, int):
                changed_maps.add(atom_map)
    if not changed_maps or not changed_maps <= dependencies:
        return mappings
    if not dependencies <= set(pattern_by_map):
        return mappings

    decorated = pattern.copy()
    coupling_roles: Dict[Any, List[Tuple[str, str]]] = defaultdict(list)
    coupling_nodes = []
    for key, coupling in sorted(rule.stereo_couplings.items()):
        centers = tuple(pattern_by_map[value] for value in coupling.centers)
        ligands = tuple(pattern_by_map[value] for value in coupling.ligands)
        coupling_nodes.append((key, coupling, centers, ligands))
        for node in centers:
            coupling_roles[node].append((key, "center"))
        for node in ligands:
            coupling_roles[node].append((key, "ligand"))
    for node in decorated:
        decorated.nodes[node]["_coupling_role"] = tuple(
            sorted(coupling_roles.get(node, ()))
        )

    exact_node_attrs = [*node_attrs, "_coupling_role"]

    def node_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return all(left.get(attr) == right.get(attr) for attr in exact_node_attrs)

    def edge_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return all(left.get(attr) == right.get(attr) for attr in edge_attrs)

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        decorated,
        decorated,
        node_match=node_match,
        edge_match=edge_match,
    )
    automorphisms = []
    for automorphism in matcher.isomorphisms_iter():
        automorphisms.append(automorphism)
        if len(automorphisms) > max_automorphisms:
            return mappings

    seen = set()
    unique = []
    for mapping in mappings:
        variants = []
        for automorphism in automorphisms:
            coupling_signature = []
            for key, coupling, centers, ligands in coupling_nodes:
                mapped_locus = tuple(
                    (
                        mapping[automorphism[center]],
                        mapping[automorphism[ligand]],
                    )
                    for center, ligand in zip(centers, ligands)
                )
                coupling_signature.append(
                    (key, coupling.kind, coupling.relation, mapped_locus)
                )
            variants.append(tuple(coupling_signature))
        signature = min(variants, key=repr)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(mapping)
    return unique
