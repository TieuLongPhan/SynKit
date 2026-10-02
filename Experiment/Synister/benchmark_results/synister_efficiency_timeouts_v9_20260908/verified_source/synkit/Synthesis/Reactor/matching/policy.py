"""Narrow chemistry policies used during reactor mapping."""

from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)

_PRIMITIVE_TYPES = frozenset((str, int, float, bool, type(None)))


def _freeze(
    value: Any,
    cache: dict[int, tuple[Any, Any]] | None = None,
) -> Any:
    if type(value) in _PRIMITIVE_TYPES:
        return value
    cache_key = id(value)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None and cached[0] is value:
            return cached[1]
    if isinstance(value, dict):
        result = tuple(
            sorted((key, _freeze(item, cache)) for key, item in value.items())
        )
    elif isinstance(value, (list, tuple)):
        shallow = tuple(value)
        try:
            hash(shallow)
        except TypeError:
            result = tuple(_freeze(item, cache) for item in value)
        else:
            # Every recursively mutable child makes the shallow tuple
            # unhashable. If hashing succeeds, recursive normalization would
            # preserve each child and produce exactly this tuple.
            result = shallow
    elif isinstance(value, set):
        result = tuple(sorted((_freeze(item, cache) for item in value), key=repr))
    else:
        result = value
    if cache is not None:
        cache[cache_key] = (value, result)
    return result


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


def _wl_proves_transition_asymmetric(transition: nx.Graph) -> bool:
    """Return whether exact color refinement proves a trivial automorphism group.

    Every attributed automorphism preserves each 1-WL color class. Therefore,
    if refinement makes every vertex color unique, only the identity
    automorphism can remain. A non-discrete partition proves nothing and keeps
    the complete GraphMatcher path.
    """
    if len(transition) < 2:
        return True

    node_palette: dict[Any, int] = {}
    colors = {}
    for node, attrs in transition.nodes(data=True):
        role = attrs.get("_transition_node_role", ())
        color = node_palette.get(role)
        if color is None:
            color = len(node_palette) + 1
            node_palette[role] = color
        colors[node] = color
    if len(set(colors.values())) == len(transition):
        return True

    edge_palette: dict[Any, int] = {}
    edge_colors = {}
    for left, right, attrs in transition.edges(data=True):
        role = attrs.get("_transition_role", ())
        color = edge_palette.get(role)
        if color is None:
            color = len(edge_palette) + 1
            edge_palette[role] = color
        edge_colors[frozenset((left, right))] = color

    for _ in range(len(transition)):
        signatures = {
            node: (
                colors[node],
                tuple(
                    sorted(
                        (
                            edge_colors[frozenset((node, neighbor))],
                            colors[neighbor],
                        )
                        for neighbor in transition.neighbors(node)
                    )
                ),
            )
            for node in transition
        }
        palette: dict[Any, int] = {}
        refined = {}
        for node, signature in signatures.items():
            color = palette.get(signature)
            if color is None:
                color = len(palette) + 1
                palette[signature] = color
            refined[node] = color
        if len(palette) == len(transition):
            return True
        if all(refined[node] == colors[node] for node in transition):
            return False
        if len(set(refined.values())) == len(set(colors.values())):
            return False
        colors = refined
    return False


def deduplicate_joint_rule_mappings(  # noqa: C901
    mappings: list[dict[Any, Any]],
    pattern: nx.Graph,
    reaction_center: nx.Graph,
    *,
    node_attrs: list[str],
    edge_attrs: list[str],
    fixed_nodes: frozenset[Any] = frozenset(),
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
    identity_cache: dict[int, tuple[Any, Any]] = {}
    transition_node_keys = (
        "element",
        "aromatic",
        "hcount",
        "charge",
        "radical",
        "neighbors",
        "lone_pairs",
        "valence_electrons",
        "present",
        "charge_model_consistent",
    )
    for node, attrs in transition.nodes(data=True):
        reaction_attrs = reaction_center.nodes.get(node, {})
        attrs["_transition_node_role"] = _freeze(
            (
                "source",
                tuple((key, attrs.get(key)) for key in node_attrs),
                "reaction_center",
                tuple((key, reaction_attrs.get(key)) for key in transition_node_keys),
            ),
            identity_cache,
        )

    # An attributed automorphism must preserve every node role. If those
    # roles already form a discrete partition, every chemical pattern node
    # is fixed. Any later virtual hydrogen-pair nodes can then permute only
    # among themselves and cannot change a mapping orbit.
    if len(
        {attrs["_transition_node_role"] for _, attrs in transition.nodes(data=True)}
    ) == len(transition):
        return mappings
    transition_edge_keys = (
        "order",
        "kekule_order",
        "sigma_order",
        "pi_order",
        "aromatic",
        "standard_order",
    )
    for left, right, attrs in reaction_center.edges(data=True):
        if left not in transition or right not in transition:
            continue
        role = _freeze(
            (
                "reaction_center",
                tuple((key, attrs.get(key)) for key in transition_edge_keys),
            ),
            identity_cache,
        )
        if transition.has_edge(left, right):
            transition.edges[left, right]["_transition_role"] = role
        else:
            transition.add_edge(
                left,
                right,
                _transition_role=role,
            )

    # Explicit-H transfer pair IDs are provenance labels, but their incidence
    # relation is chemical. Encode that relation with anonymous virtual nodes
    # so automorphisms may rename a pair while preserving its donor/recipient
    # connectivity exactly.
    pair_incidence: dict[Any, dict[Any, set[str]]] = {}
    for node, attrs in reaction_center.nodes(data=True):
        for side, key in (("left", "h_pairs_left"), ("right", "h_pairs_right")):
            for pair_id in attrs.get(key, ()):
                pair_incidence.setdefault(pair_id, {}).setdefault(node, set()).add(side)
    for ordinal, (_, incidence) in enumerate(
        sorted(pair_incidence.items(), key=lambda item: repr(item[0]))
    ):
        pair_node = ("_synkit_h_pair_role", ordinal)
        transition.add_node(
            pair_node,
            _transition_node_role=("hydrogen_pair",),
        )
        for node, sides in incidence.items():
            if node in transition:
                transition.add_edge(
                    pair_node,
                    node,
                    _transition_role=("hydrogen_pair", tuple(sorted(sides))),
                )
    for _, _, attrs in transition.edges(data=True):
        attrs.setdefault(
            "_transition_role",
            _freeze(
                (
                    "source",
                    tuple((key, attrs.get(key)) for key in edge_attrs),
                ),
                identity_cache,
            ),
        )

    if _wl_proves_transition_asymmetric(transition):
        return mappings

    matcher = GraphMatcher(
        transition,
        transition,
        node_match=categorical_node_match("_transition_node_role", ()),
        edge_match=categorical_edge_match("_transition_role", ()),
    )
    automorphisms = []
    for automorphism in matcher.isomorphisms_iter():
        if any(automorphism[node] != node for node in fixed_nodes):
            continue
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
