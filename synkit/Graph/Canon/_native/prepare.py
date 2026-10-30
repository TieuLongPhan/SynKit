"""Snapshot exact typed colors and simple graph incidence for the native core."""

from dataclasses import dataclass
import json

import networkx as nx

from ..exact import _encode_color, _selector_value


@dataclass(frozen=True)
class PreparedGraph:
    """Immutable Python-owned CSR snapshot; input handles are output witnesses."""

    nodes: tuple
    offsets: tuple[int, ...]
    neighbors: tuple[int, ...]
    colors: tuple[int, ...]
    palette_bytes: bytes


def prepare_graph(graph, *, node_color="color", edge_color=None):
    """Prepare a simple undirected vertex-colored graph without normalization.

    Edge attributes are intentionally outside this prepared graph contract.
    An explicit edge-color selector is refused; encode typed edges as colored
    incidence vertices first, as the stereo expansion already does.
    """
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise TypeError("Native graph search requires a simple undirected graph")
    if edge_color is not None:
        raise NotImplementedError("Encode edge colors as incidence vertices before native search")
    if nx.number_of_selfloops(graph):
        raise ValueError("Native graph search does not support self loops")
    nodes = tuple(graph)
    if len(nodes) > 2**31 - 1:
        raise ValueError("Native graph dimensions exceed int32 vertex capacity")
    tokens = tuple(_encode_color(_selector_value(graph.nodes[v], node_color)) for v in nodes)
    palette = sorted(set(tokens))
    ranks = {token: i for i, token in enumerate(palette)}
    indices = {v: i for i, v in enumerate(nodes)}
    offsets, neighbors = [0], []
    for v in nodes:
        neighbors.extend(sorted(indices[w] for w in graph[v]))
        offsets.append(len(neighbors))
    return PreparedGraph(
        nodes, tuple(offsets), tuple(neighbors), tuple(ranks[t] for t in tokens),
        json.dumps(palette, separators=(",", ":"), ensure_ascii=True).encode("ascii"),
    )
