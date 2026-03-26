from __future__ import annotations

from collections import defaultdict
from math import cos, pi, sin
from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Tuple

import networkx as nx

Pos = Dict[Hashable, Tuple[float, float]]
LayoutFunc = Callable[..., Pos]


def _sort_nodes(graph: nx.DiGraph, nodes: Iterable[Hashable]) -> List[Hashable]:
    """
    Return nodes in a deterministic visualization order.

    Nodes are sorted by selected node attributes and finally by node identifier.

    :param graph: Directed graph containing node metadata.
    :type graph: nx.DiGraph
    :param nodes: Nodes to sort.
    :type nodes: Iterable[Hashable]
    :returns: Sorted node identifiers.
    :rtype: List[Hashable]
    """
    return sorted(
        nodes,
        key=lambda n: (
            graph.nodes[n].get("kind", ""),
            graph.nodes[n].get("step", 10**9),
            graph.nodes[n].get("rule_index", 10**9),
            graph.nodes[n].get("app_index", 10**9),
            str(graph.nodes[n].get("label", "")),
            str(n),
        ),
    )


def _stack_vertical(
    graph: nx.DiGraph,
    nodes: Iterable[Hashable],
    *,
    x: float,
    node_spacing: float,
) -> Pos:
    """
    Place nodes in a vertical stack centered around ``y = 0``.

    :param graph: Directed graph containing node metadata.
    :type graph: nx.DiGraph
    :param nodes: Nodes to place.
    :type nodes: Iterable[Hashable]
    :param x: Shared x-coordinate for the stack.
    :type x: float
    :param node_spacing: Distance between adjacent nodes along the y-axis.
    :type node_spacing: float
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    ordered = _sort_nodes(graph, nodes)
    n = len(ordered)
    if n == 0:
        return {}

    center = (n - 1) / 2.0
    pos: Pos = {}
    for i, node in enumerate(ordered):
        y = (center - i) * node_spacing
        pos[node] = (x, y)
    return pos


def _stack_horizontal(
    graph: nx.DiGraph,
    nodes: Iterable[Hashable],
    *,
    y: float,
    node_spacing: float,
) -> Pos:
    """
    Place nodes in a horizontal stack centered around ``x = 0``.

    :param graph: Directed graph containing node metadata.
    :type graph: nx.DiGraph
    :param nodes: Nodes to place.
    :type nodes: Iterable[Hashable]
    :param y: Shared y-coordinate for the stack.
    :type y: float
    :param node_spacing: Distance between adjacent nodes along the x-axis.
    :type node_spacing: float
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    ordered = _sort_nodes(graph, nodes)
    n = len(ordered)
    if n == 0:
        return {}

    center = (n - 1) / 2.0
    pos: Pos = {}
    for i, node in enumerate(ordered):
        x = (i - center) * node_spacing
        pos[node] = (x, y)
    return pos


def _rule_x_from_step(graph: nx.DiGraph, node: Hashable) -> float:
    """
    Compute the logical x-layer of a rule node from its step annotation.

    Rule layers are placed at odd indices: ``1, 3, 5, ...``.

    :param graph: Directed graph containing node metadata.
    :type graph: nx.DiGraph
    :param node: Rule node identifier.
    :type node: Hashable
    :returns: Logical x-layer for the rule node.
    :rtype: float
    """
    step = graph.nodes[node].get("step")
    return float(2 * int(step) - 1) if step is not None else 1.0


def _species_x_from_incidence(graph: nx.DiGraph, node: Hashable) -> float:
    """
    Infer the logical x-layer of a species node from adjacent rule nodes.

    The heuristic is:
    - if the species is a reactant of one or more rules, place it immediately
      before the earliest such rule
    - otherwise, if it is a product of one or more rules, place it immediately
      after the earliest such rule
    - otherwise place it at layer ``0``

    :param graph: Directed graph containing node and edge metadata.
    :type graph: nx.DiGraph
    :param node: Species node identifier.
    :type node: Hashable
    :returns: Logical x-layer for the species node.
    :rtype: float
    """
    reactant_steps: List[int] = []
    product_steps: List[int] = []

    for _, v, d in graph.out_edges(node, data=True):
        if d.get("role") == "reactant" and graph.nodes[v].get("kind") == "rule":
            step = graph.nodes[v].get("step", d.get("step"))
            if step is not None:
                reactant_steps.append(int(step))

    for u, _, d in graph.in_edges(node, data=True):
        if d.get("role") == "product" and graph.nodes[u].get("kind") == "rule":
            step = graph.nodes[u].get("step", d.get("step"))
            if step is not None:
                product_steps.append(int(step))

    if reactant_steps:
        return float(2 * (min(reactant_steps) - 1))
    if product_steps:
        return float(2 * min(product_steps))
    return 0.0


def _logical_layers(
    graph: nx.DiGraph,
    *,
    species_nodes: List[Hashable],
    rule_nodes: List[Hashable],
) -> Dict[int, List[Hashable]]:
    """
    Build integer logical layers for species and rule nodes.

    :param graph: Directed graph containing node and edge metadata.
    :type graph: nx.DiGraph
    :param species_nodes: List of species node identifiers.
    :type species_nodes: List[Hashable]
    :param rule_nodes: List of rule node identifiers.
    :type rule_nodes: List[Hashable]
    :returns: Mapping from integer layer index to nodes in that layer.
    :rtype: Dict[int, List[Hashable]]
    """
    layers: Dict[int, List[Hashable]] = defaultdict(list)

    for node in species_nodes:
        layers[int(_species_x_from_incidence(graph, node))].append(node)

    for node in rule_nodes:
        layers[int(_rule_x_from_step(graph, node))].append(node)

    return dict(layers)


def step_layout(
    graph: nx.DiGraph,
    *,
    species_nodes: List[Hashable],
    rule_nodes: List[Hashable],
    node_spacing: float = 1.4,
    layer_spacing: float = 2.5,
) -> Pos:
    """
    Compute a layered step-wise layout.

    Species and rule nodes are assigned to alternating logical x-layers inferred
    from rule steps and incidence relationships.

    :param graph: Directed graph containing species and rule nodes.
    :type graph: nx.DiGraph
    :param species_nodes: List of species node identifiers.
    :type species_nodes: List[Hashable]
    :param rule_nodes: List of rule node identifiers.
    :type rule_nodes: List[Hashable]
    :param node_spacing: Vertical spacing between nodes in the same layer.
    :type node_spacing: float
    :param layer_spacing: Horizontal spacing between adjacent layers.
    :type layer_spacing: float
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos

    .. code-block:: python

        pos = step_layout(
            graph,
            species_nodes=species_nodes,
            rule_nodes=rule_nodes,
            node_spacing=1.6,
            layer_spacing=3.0,
        )
    """
    groups: Dict[float, List[Hashable]] = defaultdict(list)

    for node in species_nodes:
        groups[_species_x_from_incidence(graph, node)].append(node)

    for node in rule_nodes:
        groups[_rule_x_from_step(graph, node)].append(node)

    pos: Pos = {}
    for raw_x in sorted(groups):
        pos.update(
            _stack_vertical(
                graph,
                groups[raw_x],
                x=raw_x * layer_spacing,
                node_spacing=node_spacing,
            )
        )
    return pos


def bipartite_layout(
    graph: nx.DiGraph,
    *,
    species_nodes: List[Hashable],
    rule_nodes: List[Hashable],
    node_spacing: float = 1.4,
    layer_spacing: float = 3.0,
    orientation: str = "vertical",
) -> Pos:
    """
    Compute a two-layer species-rule layout.

    :param graph: Directed graph containing species and rule nodes.
    :type graph: nx.DiGraph
    :param species_nodes: List of species node identifiers.
    :type species_nodes: List[Hashable]
    :param rule_nodes: List of rule node identifiers.
    :type rule_nodes: List[Hashable]
    :param node_spacing: Spacing between adjacent nodes within a layer.
    :type node_spacing: float
    :param layer_spacing: Distance between the species and rule layers.
    :type layer_spacing: float
    :param orientation: Layout orientation. Must be ``"vertical"`` or
        ``"horizontal"``.
    :type orientation: str
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    :raises ValueError: If ``orientation`` is not supported.
    """
    pos: Pos = {}

    if orientation == "vertical":
        pos.update(
            _stack_vertical(graph, species_nodes, x=0.0, node_spacing=node_spacing)
        )
        pos.update(
            _stack_vertical(
                graph,
                rule_nodes,
                x=layer_spacing,
                node_spacing=node_spacing,
            )
        )
        return pos

    if orientation == "horizontal":
        pos.update(
            _stack_horizontal(graph, species_nodes, y=0.0, node_spacing=node_spacing)
        )
        pos.update(
            _stack_horizontal(
                graph,
                rule_nodes,
                y=-layer_spacing,
                node_spacing=node_spacing,
            )
        )
        return pos

    raise ValueError("orientation must be 'vertical' or 'horizontal'")


def multipartite_step_layout(
    graph: nx.DiGraph,
    *,
    species_nodes: List[Hashable],
    rule_nodes: List[Hashable],
) -> Pos:
    """
    Compute a multipartite layout using inferred step layers.

    This layout is useful for larger step-annotated graphs where a simple custom
    step layout becomes crowded.

    :param graph: Directed graph containing species and rule nodes.
    :type graph: nx.DiGraph
    :param species_nodes: List of species node identifiers.
    :type species_nodes: List[Hashable]
    :param rule_nodes: List of rule node identifiers.
    :type rule_nodes: List[Hashable]
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    G = graph.copy()
    layers = _logical_layers(
        graph,
        species_nodes=species_nodes,
        rule_nodes=rule_nodes,
    )

    node_to_layer = {node: layer for layer, nodes in layers.items() for node in nodes}

    for node in G.nodes:
        G.nodes[node]["subset"] = node_to_layer.get(node, 0)

    return nx.multipartite_layout(G, subset_key="subset", align="vertical")


def radial_step_layout(
    graph: nx.DiGraph,
    *,
    species_nodes: List[Hashable],
    rule_nodes: List[Hashable],
    radius_step: float = 1.8,
) -> Pos:
    """
    Compute a radial step layout using concentric circles.

    Logical layers are mapped to increasing radii.

    :param graph: Directed graph containing species and rule nodes.
    :type graph: nx.DiGraph
    :param species_nodes: List of species node identifiers.
    :type species_nodes: List[Hashable]
    :param rule_nodes: List of rule node identifiers.
    :type rule_nodes: List[Hashable]
    :param radius_step: Radial increment between adjacent logical layers.
    :type radius_step: float
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    layers = _logical_layers(
        graph,
        species_nodes=species_nodes,
        rule_nodes=rule_nodes,
    )

    pos: Pos = {}
    for layer in sorted(layers):
        nodes = _sort_nodes(graph, layers[layer])
        n = len(nodes)
        radius = max(1.0, layer * radius_step)

        for i, node in enumerate(nodes):
            theta = (2.0 * pi * i) / max(n, 1)
            pos[node] = (radius * cos(theta), radius * sin(theta))

    return pos


def circular_bipartite_layout(
    graph: nx.DiGraph,
    *,
    species_nodes: List[Hashable],
    rule_nodes: List[Hashable],
    species_radius: float = 2.5,
    rule_radius: float = 4.0,
) -> Pos:
    """
    Compute a circular bipartite layout using two concentric circles.

    Species and rule nodes are placed on separate rings.

    :param graph: Directed graph containing species and rule nodes.
    :type graph: nx.DiGraph
    :param species_nodes: List of species node identifiers.
    :type species_nodes: List[Hashable]
    :param rule_nodes: List of rule node identifiers.
    :type rule_nodes: List[Hashable]
    :param species_radius: Radius of the species ring.
    :type species_radius: float
    :param rule_radius: Radius of the rule ring.
    :type rule_radius: float
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    pos: Pos = {}
    s_nodes = _sort_nodes(graph, species_nodes)
    r_nodes = _sort_nodes(graph, rule_nodes)

    for i, node in enumerate(s_nodes):
        theta = (2.0 * pi * i) / max(len(s_nodes), 1)
        pos[node] = (species_radius * cos(theta), species_radius * sin(theta))

    for i, node in enumerate(r_nodes):
        theta = (2.0 * pi * i) / max(len(r_nodes), 1)
        pos[node] = (rule_radius * cos(theta), rule_radius * sin(theta))

    return pos


def degree_shell_layout(
    graph: nx.DiGraph,
    *,
    species_nodes: List[Hashable],
    rule_nodes: List[Hashable],
) -> Pos:
    """
    Compute a degree-based shell layout.

    Nodes with the highest degree are placed in the inner shell, followed by
    medium-degree and lower-degree shells.

    :param graph: Directed graph containing species and rule nodes.
    :type graph: nx.DiGraph
    :param species_nodes: List of species node identifiers.
    :type species_nodes: List[Hashable]
    :param rule_nodes: List of rule node identifiers.
    :type rule_nodes: List[Hashable]
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    by_degree = sorted(graph.degree, key=lambda x: (-x[1], str(x[0])))
    n_total = len(by_degree)

    q1 = max(1, n_total // 4)
    q2 = max(2, n_total // 2)

    high = [node for node, _ in by_degree[:q1]]
    mid = [node for node, _ in by_degree[q1:q2]]
    rest = [node for node, _ in by_degree[q2:]]

    shells = [shell for shell in (high, mid, rest) if shell]
    return nx.shell_layout(graph, nlist=shells)


def choose_auto_layout(
    graph: nx.DiGraph,
    *,
    species_nodes: List[Hashable],
    rule_nodes: List[Hashable],
) -> str:
    """
    Choose a layout heuristically.

    Preference is given to step-aware layouts when rule step annotations are
    available. Larger or denser graphs are routed to layouts that typically
    reduce clutter.

    :param graph: Directed graph containing species and rule nodes.
    :type graph: nx.DiGraph
    :param species_nodes: List of species node identifiers.
    :type species_nodes: List[Hashable]
    :param rule_nodes: List of rule node identifiers.
    :type rule_nodes: List[Hashable]
    :returns: Selected layout name.
    :rtype: str
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    density = nx.density(graph) if n_nodes > 1 else 0.0
    has_step = any(graph.nodes[node].get("step") is not None for node in rule_nodes)

    if has_step and (n_nodes >= 35 or n_edges >= 55 or density >= 0.045):
        return "multipartite_step"
    if has_step:
        return "step"
    if n_nodes >= 40 or density >= 0.06:
        return "kamada_kawai"
    return "spring"


def _spring_layout(
    graph: nx.DiGraph,
    *,
    seed: int,
    **_: object,
) -> Pos:
    """
    Compute a spring layout with a slightly enlarged optimal distance.

    :param graph: Directed graph to layout.
    :type graph: nx.DiGraph
    :param seed: Random seed for the spring layout.
    :type seed: int
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    n_nodes = max(graph.number_of_nodes(), 1)
    k = 1.8 / (n_nodes**0.5)
    return nx.spring_layout(graph, seed=seed, k=k, iterations=200)


def _kamada_kawai_layout(
    graph: nx.DiGraph,
    **_: object,
) -> Pos:
    """
    Compute a Kamada-Kawai layout.

    :param graph: Directed graph to layout.
    :type graph: nx.DiGraph
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    return nx.kamada_kawai_layout(graph)


def _spectral_layout(
    graph: nx.DiGraph,
    **_: object,
) -> Pos:
    """
    Compute a spectral layout.

    :param graph: Directed graph to layout.
    :type graph: nx.DiGraph
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    return nx.spectral_layout(graph)


def _shell_layout(
    graph: nx.DiGraph,
    *,
    species_nodes: List[Hashable],
    rule_nodes: List[Hashable],
    **_: object,
) -> Pos:
    """
    Compute a shell layout with species and rule nodes grouped into shells.

    :param graph: Directed graph to layout.
    :type graph: nx.DiGraph
    :param species_nodes: List of species node identifiers.
    :type species_nodes: List[Hashable]
    :param rule_nodes: List of rule node identifiers.
    :type rule_nodes: List[Hashable]
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    shells: List[List[Hashable]] = []
    if species_nodes:
        shells.append(species_nodes)
    if rule_nodes:
        shells.append(rule_nodes)
    return nx.shell_layout(graph, nlist=shells) if shells else nx.shell_layout(graph)


def _spiral_layout(
    graph: nx.DiGraph,
    **_: object,
) -> Pos:
    """
    Compute a spiral layout.

    :param graph: Directed graph to layout.
    :type graph: nx.DiGraph
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    return nx.spiral_layout(graph)


def _random_layout(
    graph: nx.DiGraph,
    *,
    seed: int,
    **_: object,
) -> Pos:
    """
    Compute a random layout.

    :param graph: Directed graph to layout.
    :type graph: nx.DiGraph
    :param seed: Random seed for layout generation.
    :type seed: int
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    """
    return nx.random_layout(graph, seed=seed)


_LAYOUT_REGISTRY: Mapping[str, LayoutFunc] = {
    "step": step_layout,
    "bipartite": bipartite_layout,
    "multipartite_step": multipartite_step_layout,
    "radial_step": radial_step_layout,
    "circular_bipartite": circular_bipartite_layout,
    "degree_shell": degree_shell_layout,
    "spring": _spring_layout,
    "kamada_kawai": _kamada_kawai_layout,
    "spectral": _spectral_layout,
    "shell": _shell_layout,
    "spiral": _spiral_layout,
    "random": _random_layout,
}


def available_layouts() -> List[str]:
    """
    Return the list of supported layout names.

    :returns: Supported layout names, including ``"auto"``.
    :rtype: List[str]

    .. code-block:: python

        names = available_layouts()
        print(names)
    """
    return ["auto", *_LAYOUT_REGISTRY.keys()]


def compute_layout(
    graph: nx.DiGraph,
    *,
    species_nodes: List[Hashable],
    rule_nodes: List[Hashable],
    layout: str = "step",
    node_spacing: float = 1.4,
    layer_spacing: float = 2.5,
    seed: int = 0,
    orientation: str = "vertical",
) -> Pos:
    """
    Compute node positions for CRN visualization.

    :param graph: Directed graph containing species and rule nodes.
    :type graph: nx.DiGraph
    :param species_nodes: List of species node identifiers.
    :type species_nodes: List[Hashable]
    :param rule_nodes: List of rule node identifiers.
    :type rule_nodes: List[Hashable]
    :param layout: Layout name. Supported values are ``"auto"``, ``"step"``,
        ``"bipartite"``, ``"multipartite_step"``, ``"radial_step"``,
        ``"circular_bipartite"``, ``"degree_shell"``, ``"spring"``,
        ``"kamada_kawai"``, ``"spectral"``, ``"shell"``, ``"spiral"``,
        and ``"random"``.
    :type layout: str
    :param node_spacing: Spacing between nodes within a layer for layouts that
        support it.
    :type node_spacing: float
    :param layer_spacing: Spacing between layers for layouts that support it.
    :type layer_spacing: float
    :param seed: Random seed for stochastic layouts.
    :type seed: int
    :param orientation: Orientation for the bipartite layout. Must be
        ``"vertical"`` or ``"horizontal"``.
    :type orientation: str
    :returns: Mapping from node identifier to ``(x, y)`` coordinates.
    :rtype: Pos
    :raises ValueError: If the requested layout name is not supported.

    .. code-block:: python

        pos = compute_layout(
            graph,
            species_nodes=species_nodes,
            rule_nodes=rule_nodes,
            layout="auto",
        )
    """
    resolved_layout = (
        choose_auto_layout(
            graph,
            species_nodes=species_nodes,
            rule_nodes=rule_nodes,
        )
        if layout == "auto"
        else layout
    )

    try:
        layout_func = _LAYOUT_REGISTRY[resolved_layout]
    except KeyError as exc:
        supported = ", ".join(repr(name) for name in available_layouts())
        raise ValueError(
            f"Unsupported layout {layout!r}. Use one of: {supported}."
        ) from exc

    return layout_func(
        graph,
        species_nodes=species_nodes,
        rule_nodes=rule_nodes,
        node_spacing=node_spacing,
        layer_spacing=layer_spacing,
        seed=seed,
        orientation=orientation,
    )
