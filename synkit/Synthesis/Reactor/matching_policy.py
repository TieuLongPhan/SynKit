"""Narrow chemistry policies used during reactor mapping."""

from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value


def contextual_electron_pattern_graph(
    pattern: nx.Graph,
    *,
    reaction_center: nx.Graph,
    template_format: str | None,
) -> nx.Graph:
    """Record query attributes made unknown by stripped local context."""
    if template_format != "tuple":
        return pattern

    decorated = None
    for node, attrs in pattern.nodes(data=True):
        neighbors = attrs.get("neighbors", ())
        explicit_hydrogens = sum(
            pattern.nodes[neighbor].get("element") == "H"
            for neighbor in pattern.neighbors(node)
        )
        hidden_hydrogen = (
            isinstance(neighbors, (list, tuple))
            and neighbors.count("H") > explicit_hydrogens
        )
        rc_attrs = reaction_center.nodes.get(node, {})
        radical = rc_attrs.get("radical")
        radical_changes = (
            isinstance(radical, tuple)
            and len(radical) == 2
            and radical[0] != radical[1]
        )
        if not hidden_hydrogen or radical_changes:
            continue
        if decorated is None:
            decorated = pattern.copy()
        policies = dict(decorated.nodes[node].get("_query_attribute_policies", {}))
        policies["radical"] = "unknown"
        decorated.nodes[node]["_query_attribute_policies"] = policies
    return decorated or pattern


def has_heavy_cross_component_correlation(
    pattern: nx.Graph,
    reaction_center: nx.Graph,
) -> bool:
    """Return whether product bonds jointly couple heavy source components."""
    components = [
        frozenset(component) for component in nx.connected_components(pattern)
    ]
    if len(components) < 2:
        return False
    component_index = {
        node: index for index, component in enumerate(components) for node in component
    }
    heavy_components = {
        index
        for index, component in enumerate(components)
        if any(pattern.nodes[node].get("element") != "H" for node in component)
    }
    for left, right, attrs in reaction_center.edges(data=True):
        if left not in component_index or right not in component_index:
            continue
        left_component = component_index[left]
        right_component = component_index[right]
        if (
            left_component == right_component
            or left_component not in heavy_components
            or right_component not in heavy_components
        ):
            continue
        order: Any = attrs.get("order")
        if (
            isinstance(order, tuple)
            and len(order) == 2
            and float(order[0]) == 0.0
            and float(order[1]) > 0.0
        ):
            return True
    return False


def deduplicate_joint_rule_mappings(
    mappings: list[dict[Any, Any]],
    pattern: nx.Graph,
    reaction_center: nx.Graph,
    *,
    node_attrs: list[str],
    max_automorphisms: int = 256,
) -> list[dict[Any, Any]]:
    """Quotient mappings only by complete transition-graph automorphisms."""
    if len(mappings) < 2:
        return mappings
    pattern_nodes = frozenset(pattern)
    if any(not pattern_nodes.issubset(mapping) for mapping in mappings):
        # Partial-template mappings are not closed under whole-rule
        # automorphisms, so the group action below is undefined for them.
        return mappings

    transition = pattern.copy()
    transition_keys = (
        "order",
        "kekule_order",
        "sigma_order",
        "pi_order",
        "standard_order",
    )
    for left, right, attrs in reaction_center.edges(data=True):
        if left not in transition or right not in transition:
            continue
        role = _freeze(tuple((key, attrs.get(key)) for key in transition_keys))
        if transition.has_edge(left, right):
            transition.edges[left, right]["_transition_role"] = role
        else:
            transition.add_edge(
                left,
                right,
                _transition_role=role,
            )
    for _, _, attrs in transition.edges(data=True):
        attrs.setdefault(
            "_transition_role",
            _freeze(
                (
                    "source",
                    attrs.get("order"),
                    attrs.get("sigma_order"),
                    attrs.get("pi_order"),
                )
            ),
        )

    node_defaults = [0 if attr == "charge" else "*" for attr in node_attrs]
    matcher = GraphMatcher(
        transition,
        transition,
        node_match=categorical_node_match(node_attrs, node_defaults),
        edge_match=categorical_edge_match("_transition_role", ()),
    )
    automorphisms = []
    for automorphism in matcher.isomorphisms_iter():
        automorphisms.append(automorphism)
        if len(automorphisms) > max_automorphisms:
            return mappings

    nodes = sorted(pattern.nodes, key=repr)
    seen = set()
    unique = []
    for mapping in mappings:
        signature = min(
            (
                tuple(mapping[automorphism[node]] for node in nodes)
                for automorphism in automorphisms
            ),
            key=repr,
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(mapping)
    return unique
