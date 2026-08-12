"""Deliberately slow reference enumerator for small typed RBL overlaps.

This module is an executable specification, not a production search path.
It enumerates literal partial injections and validates each with the public
fusion-interface contract.  Tests use it to detect omissions in the pruned,
incremental enumerator.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Any, Hashable, Mapping, Sequence

import networkx as nx

from synkit.Graph.Fusion import FusionInterface, FusionInterfaceError


def _scalar(value: Any) -> Any:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0] if value[0] == value[1] else None
    return value


def _key(mapping: Mapping[Hashable, Hashable]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((repr(left), repr(right)) for left, right in mapping.items()))


def brute_force_typed_overlaps(
    forward: nx.Graph,
    backward: nx.Graph,
    *,
    node_keys: Sequence[str],
    edge_keys: Sequence[str],
    element_key: str = "element",
    wildcard_element: Any = ("*", "*"),
    provenance_key: str = "atom_map",
    max_total_nodes: int = 12,
) -> tuple[Mapping[Hashable, Hashable], ...]:
    """Enumerate all admitted partial injections without search pruning.

    The guard prevents accidental benchmark use: factorial enumeration is
    intentional here because independence from production pruning is what
    makes this a useful oracle.
    """
    if forward.number_of_nodes() + backward.number_of_nodes() > max_total_nodes:
        raise ValueError("The exhaustive overlap oracle is limited to tiny graphs.")

    scalar_wildcard = (
        wildcard_element[0]
        if isinstance(wildcard_element, (tuple, list))
        else wildcard_element
    )
    wildcard_values = (wildcard_element, scalar_wildcard)
    left_by_map: dict[Any, list[Hashable]] = {}
    right_by_map: dict[Any, list[Hashable]] = {}
    for graph, buckets in ((forward, left_by_map), (backward, right_by_map)):
        for node, attributes in graph.nodes(data=True):
            if attributes.get(element_key) in wildcard_values:
                continue
            provenance = _scalar(attributes.get(provenance_key))
            if provenance not in {None, 0}:
                buckets.setdefault(provenance, []).append(node)
    anchors = {
        left_by_map[value][0]: right_by_map[value][0]
        for value in left_by_map.keys() & right_by_map.keys()
        if len(left_by_map[value]) == len(right_by_map[value]) == 1
    }

    left_nodes = tuple(forward.nodes)
    right_nodes = tuple(backward.nodes)
    admitted: list[dict[Hashable, Hashable]] = []
    for size in range(1, min(len(left_nodes), len(right_nodes)) + 1):
        for left_subset in combinations(left_nodes, size):
            if not set(anchors).issubset(left_subset):
                continue
            for right_subset in combinations(right_nodes, size):
                if not set(anchors.values()).issubset(right_subset):
                    continue
                for image in permutations(right_subset):
                    mapping = dict(zip(left_subset, image, strict=True))
                    if any(mapping[source] != target for source, target in anchors.items()):
                        continue
                    try:
                        FusionInterface.from_mapping(
                            forward,
                            backward,
                            mapping,
                            node_keys=node_keys,
                            edge_keys=edge_keys,
                            element_key=element_key,
                            wildcard_element=wildcard_element,
                        )
                    except FusionInterfaceError:
                        continue
                    admitted.append(mapping)
    admitted.sort(key=lambda mapping: (-len(mapping), _key(mapping)))
    return tuple(admitted)


__all__ = ["brute_force_typed_overlaps"]
