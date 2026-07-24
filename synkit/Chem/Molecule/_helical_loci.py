"""Topology-only perception of helicene-like molecular path carriers."""

from __future__ import annotations

import networkx as nx
from rdkit import Chem

from synkit.Graph.Stereo.supports import PathStereoSupport


def _aromatic_six_rings(molecule: Chem.Mol) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(ring)
        for ring in molecule.GetRingInfo().AtomRings()
        if len(ring) == 6
        and all(molecule.GetAtomWithIdx(atom).GetIsAromatic() for atom in ring)
    )


def _ring_edge_index(ring: tuple[int, ...], edge: frozenset[int]) -> int:
    for index, left in enumerate(ring):
        if frozenset((left, ring[(index + 1) % len(ring)])) == edge:
            return index
    raise ValueError("Fused aromatic-ring atoms do not form a ring edge.")


def _is_angular_chain(
    ring_order: tuple[int, ...],
    rings: tuple[tuple[int, ...], ...],
    fusion_edges: dict[frozenset[int], frozenset[int]],
) -> bool:
    for position, ring_index in enumerate(ring_order[1:-1], start=1):
        left_edge = fusion_edges[
            frozenset((ring_order[position - 1], ring_index))
        ]
        right_edge = fusion_edges[
            frozenset((ring_index, ring_order[position + 1]))
        ]
        if left_edge & right_edge:
            return False
        ring = rings[ring_index]
        left_index = _ring_edge_index(ring, left_edge)
        right_index = _ring_edge_index(ring, right_edge)
        separation = abs(left_index - right_index)
        separation = min(separation, len(ring) - separation)
        if separation != 2:
            return False
    return True


def _terminal_candidates(
    molecule_graph: nx.Graph,
    ring: tuple[int, ...],
    fusion_edge: frozenset[int],
) -> tuple[int, int] | None:
    left, right = tuple(fusion_edge)
    perimeter = molecule_graph.subgraph(ring).copy()
    perimeter.remove_edge(left, right)
    try:
        free_path = nx.shortest_path(perimeter, left, right)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if len(free_path) != 6:
        return None
    return free_path[1], free_path[-2]


def _unique_shortest_path(
    graph: nx.Graph,
    start: int,
    end: int,
) -> tuple[int, ...] | None:
    try:
        paths = nx.all_shortest_paths(graph, start, end)
        first = tuple(next(paths))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    try:
        next(paths)
    except StopIteration:
        return first
    return None


def _fusion_topology(
    molecule: Chem.Mol,
    rings: tuple[tuple[int, ...], ...],
) -> tuple[nx.Graph, dict[frozenset[int], frozenset[int]]]:
    """Build the ring-fusion graph and retain each material fusion edge."""
    fusion_graph = nx.Graph()
    fusion_graph.add_nodes_from(range(len(rings)))
    fusion_edges = {}
    for left in range(len(rings)):
        for right in range(left + 1, len(rings)):
            shared = frozenset(rings[left]) & frozenset(rings[right])
            if len(shared) != 2:
                continue
            first, second = tuple(shared)
            if molecule.GetBondBetweenAtoms(first, second) is None:
                continue
            fusion_graph.add_edge(left, right)
            fusion_edges[frozenset((left, right))] = shared
    return fusion_graph, fusion_edges


def detect_helicene_supports(
    molecule: Chem.Mol,
) -> tuple[PathStereoSupport, ...]:
    """Return curled, angular, fused-aromatic path carriers.

    A candidate is a chain of at least five edge-fused aromatic six-rings.
    Every internal fusion must be angular, and the two terminal rings must
    expose one unique closest pair joined by one unique shortest molecular
    path. This distinguishes a curled helicene-like topology from linear or
    ambiguous fused-ring chains without assigning handedness or stability.
    """
    if molecule is None:
        raise ValueError("Helical carrier detection requires a molecule.")
    rings = _aromatic_six_rings(molecule)
    if len(rings) < 5:
        return ()
    molecular_graph = nx.Graph(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in molecule.GetBonds()
    )
    fusion_graph, fusion_edges = _fusion_topology(molecule, rings)

    supports = set()
    for component in nx.connected_components(fusion_graph):
        chain = fusion_graph.subgraph(component)
        if (
            len(component) < 5
            or chain.number_of_edges() != len(component) - 1
            or any(degree > 2 for _, degree in chain.degree())
        ):
            continue
        terminals = sorted(node for node, degree in chain.degree() if degree == 1)
        if len(terminals) != 2:
            continue
        ring_order = tuple(nx.shortest_path(chain, *terminals))
        if not _is_angular_chain(ring_order, rings, fusion_edges):
            continue
        left_candidates = _terminal_candidates(
            molecular_graph,
            rings[ring_order[0]],
            fusion_edges[frozenset(ring_order[:2])],
        )
        right_candidates = _terminal_candidates(
            molecular_graph,
            rings[ring_order[-1]],
            fusion_edges[frozenset(ring_order[-2:])],
        )
        if left_candidates is None or right_candidates is None:
            continue
        distances = {
            (left, right): nx.shortest_path_length(molecular_graph, left, right)
            for left in left_candidates
            for right in right_candidates
        }
        minimum = min(distances.values())
        closest = [pair for pair, distance in distances.items() if distance == minimum]
        if len(closest) != 1:
            continue
        path = _unique_shortest_path(molecular_graph, *closest[0])
        if path is None or len(path) < len(component) + 1:
            continue
        canonical_path = min(path, tuple(reversed(path)))
        supports.add(PathStereoSupport(canonical_path))
    return tuple(sorted(supports, key=lambda support: support.path))


__all__ = ["detect_helicene_supports"]
