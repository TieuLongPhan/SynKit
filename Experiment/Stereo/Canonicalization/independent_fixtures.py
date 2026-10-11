"""Prospectively declared small typed constitutions for assignment validation.

These are mathematical fixtures, not experimentally annotated molecules or an
untouched chemical holdout. Carrier inventories are frozen before predictions.
"""

from dataclasses import dataclass

import networkx as nx

from Experiment.Stereo.Canonicalization.independent_model import State, transport


@dataclass(frozen=True)
class Fixture:
    name: str
    graph: nx.Graph
    seeds: tuple[State, ...]
    purpose: str


def _colored(graph):
    nx.set_node_attributes(graph, {node: f"atom:{node}" for node in graph}, "color")
    nx.set_edge_attributes(graph, "bond", "color")
    return graph


def _star(family, count, *, equivalent=False, virtual=None):
    graph = _colored(nx.star_graph(count - int(virtual is not None)))
    graph.nodes[0]["color"] = "center"
    if equivalent:
        for node in graph:
            if node:
                graph.nodes[node]["color"] = "ligand"
    references = tuple(range(1, count + 1))
    if virtual:
        references = (*references[:-1], f"@{virtual}:0")
        graph.nodes[0]["hcount" if virtual == "H" else "lone_pairs"] = 1
    parity = 0 if family == "square_planar" else 1
    state = State(family, (0, *references), parity)
    return Fixture(f"{family}:{'symmetric' if equivalent else 'distinct'}{':' + virtual if virtual else ''}",
                   graph, (state,), "Finite proper-rotation quotient and molecular symmetry")


def _bond(family, path_length=2):
    path = tuple(range(2, 2 + path_length))
    right_a, right_b = path[-1] + 1, path[-1] + 2
    graph = nx.Graph()
    graph.add_edges_from(zip(path, path[1:]))
    graph.add_edges_from(((0, path[0]), (1, path[0]), (path[-1], right_a), (path[-1], right_b)))
    _colored(graph)
    parity = 0 if family in {"planar_bond", "extended_cis_trans"} else 1
    state = State(family, (0, 1, path[0], path[-1], right_a, right_b), parity,
                  path if path_length > 2 else ())
    return Fixture(f"{family}:distinct", graph, (state,), "Terminal-frame exchange and mirror action")


def _meso(*, asymmetric=False, framework=None):
    graph = nx.Graph()
    graph.add_edges_from(((0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)))
    _colored(graph)
    for pair, color in [((0, 1), "center"), ((2, 5), "A"), ((3, 6), "B"), ((4, 7), "D")]:
        for node in pair:
            graph.nodes[node]["color"] = color
    if asymmetric:
        graph.nodes[7]["color"] = "E"
    frames = ((0, (1, 2, 3, 4), 1), (1, (0, 5, 6, 7), -1 if framework == "opposite" else 1))
    if framework:
        seeds = (State("framework", parity=1, support=tuple(graph), frames=frames),)
        name = f"framework:coupled:{framework}"
    else:
        seeds = tuple(State("tetrahedral", (center, *references), 1) for center, references, _ in frames)
        name = "tetrahedral:two:asymmetric" if asymmetric else "tetrahedral:two:meso"
    return Fixture(name, graph, seeds, "Exchange equivalent carriers or coupled frames; retain relative coupling")


def _path_plane(cyclic=False, plane=False):
    graph = nx.cycle_graph(5) if cyclic or plane else nx.path_graph(5)
    _colored(graph)
    if plane:
        graph.add_node(5, color="pilot")
        graph.add_edge(0, 5, color="pilot_link")
        seed = State("planar_chirality", parity=1, path=tuple(range(5)), pilot=5)
        name = "plane:distinct"
    else:
        seed = State("helical", parity=1, path=tuple(range(5)), cyclic=cyclic,
                     coupling_id="cycle-A" if cyclic else None)
        name = f"helical:{'cyclic' if cyclic else 'open'}"
    return Fixture(name, graph, (seed,), "Variable support and orientation transformations")


def _combine(left, right, *, name):
    mapping = {node: 100 + node for node in right.graph}
    graph = nx.compose(left.graph, nx.relabel_nodes(right.graph, mapping))
    # Connect peripheral atoms; preserve the existing center/reference degrees.
    graph.add_edge(max(left.graph), 100 + max(right.graph), color="connector")
    seeds = (*left.seeds, *(transport(seed, mapping) for seed in right.seeds))
    return Fixture(name, graph, seeds, "Mixed supported families; independent configuration product")


def fixtures():
    """Cover all eleven configured families, symmetry, virtual ports and mixtures."""
    stars = tuple(_star(family, arity) for family, arity in (
        ("tetrahedral", 4), ("square_planar", 4), ("trigonal_bipyramidal", 5), ("octahedral", 6),
    ))
    bonds = (_bond("planar_bond"), _bond("atrop_bond"),
             _bond("cumulene_axis", 3), _bond("extended_cis_trans", 4))
    extended = (_path_plane(), _path_plane(cyclic=True), _path_plane(plane=True))
    result = [*stars, *bonds, *extended]
    result += [_star("tetrahedral", 4, equivalent=True),
               _star("trigonal_bipyramidal", 5, equivalent=True),
               _star("octahedral", 6, equivalent=True),
               _star("tetrahedral", 4, virtual="H"), _star("tetrahedral", 4, virtual="LP")]
    result += [_meso(), _meso(asymmetric=True), _meso(framework="same"), _meso(framework="opposite")]
    virtual = _star("tetrahedral", 4, virtual="H")
    framework = State("framework", parity=1, support=tuple(virtual.graph),
                      frames=((0, virtual.seeds[0].atoms[1:], 1),))
    result.append(Fixture("framework:virtual", virtual.graph, (framework,), "Coupled virtual resource transport"))
    plane = _path_plane(plane=True)
    symmetric_plane = plane.graph.copy()
    for node in range(5):
        symmetric_plane.nodes[node]["color"] = "plane"
    result.append(Fixture("plane:reflection_symmetric", symmetric_plane, plane.seeds,
                          "A constitutional plane reflection collapses mirror states"))
    result += [_combine(stars[0], other, name=f"mixed:tetrahedral+{other.name}")
               for other in (*stars[1:], *bonds, *extended)]
    result.append(_combine(_meso(framework="opposite"), stars[0], name="mixed:framework+tetrahedral"))
    return tuple(result)
