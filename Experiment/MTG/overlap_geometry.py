"""Explicit finite geometric tests on saturated total-endpoint atom overlaps.

Geometry does not decide guards, environment tokens, chemical coherence policies
or physical feasibility. Original rule metadata is retained by the caller.
"""

from dataclasses import dataclass

import networkx as nx

from synkit.Graph.Morphism import LewisLabelledGraph


@dataclass(frozen=True)
class OverlapGeometry:
    """A generated middle-world union and exact geometric obstruction names."""

    union: LewisLabelledGraph | None
    first_right: tuple
    second_left: tuple
    issues: tuple[str, ...]

    @property
    def geometrically_realizable(self):
        return not self.issues


def _union(first, second, mapping):
    inverse = {target: source for source, target in mapping.items()}
    right_map = {node: ("first", node) for node in first.node_ids}
    left_map = {
        node: ("first", inverse[node]) if node in inverse else ("second", node)
        for node in second.node_ids
    }
    graph = nx.Graph()
    for endpoint, transport in ((first, right_map), (second, left_map)):
        for node in endpoint.node_ids:
            labels = endpoint.node_labels(node, semantic=True)
            target = transport[node]
            if target in graph and dict(graph.nodes[target]) != labels:
                return None, right_map, left_map
            graph.add_node(target, **labels)
        for edge in endpoint.edge_keys:
            ends = tuple(transport[node] for node in edge)
            labels = endpoint.edge_labels(edge, semantic=True)
            if graph.has_edge(*ends) and dict(graph.edges[ends]) != labels:
                return None, right_map, left_map
            graph.add_edge(*ends, **labels)
    return LewisLabelledGraph.from_networkx(graph, first.schema), right_map, left_map


def _dangling(pattern, interface_arm, transport, union):
    deleted = {
        transport[node]
        for node in pattern.node_ids - set(interface_arm.mapping.values())
    }
    matched_edges = {
        frozenset(transport[node] for node in edge) for edge in pattern.edge_keys
    }
    return any(edge & deleted for edge in union.edge_keys - matched_edges)


def _collisions(before, after, before_arm, after_arm, transport, union):
    inverse = {node: interface for interface, node in after_arm.mapping.items()}
    for edge in after.edge_keys - set(after_arm.edge_mapping.values()):
        if not edge <= inverse.keys():
            continue
        original = frozenset(before_arm.mapping[inverse[node]] for node in edge)
        if original not in before.edge_keys:
            image = frozenset(transport[node] for node in original)
            if image in union.edge_keys:
                return True
    return False


def classify_overlap_geometry(first, second, mapping):
    """Check compatible union, reverse/forward dangling and restored/created bonds.

    Maps identify R1 atoms with L2 atoms and saturate common bonds and labels.
    No execution on a supplied complete host is used to decide these tests.
    """
    if first.left.schema != second.left.schema:
        raise ValueError("Both rules must use the same raw coordinate schema.")
    mapping = dict(mapping)
    if (
        not set(mapping) <= first.right.node_ids
        or not set(mapping.values()) <= second.left.node_ids
    ):
        raise ValueError("The overlap map must lie in R1 and L2.")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("The partial atom map must be injective.")
    union, right, left = _union(first.right, second.left, mapping)
    issues = []
    if union is None:
        issues.append("ENDPOINT_COORDINATE_CONFLICT")
    else:
        if _dangling(first.right, first.right_arm, right, union):
            issues.append("REVERSE_FIRST_DANGLING")
        if _dangling(second.left, second.left_arm, left, union):
            issues.append("FORWARD_SECOND_DANGLING")
        if _collisions(
            first.right, first.left, first.right_arm, first.left_arm, right, union
        ):
            issues.append("REVERSE_FIRST_BOND_COLLISION")
        if _collisions(
            second.left, second.right, second.left_arm, second.right_arm, left, union
        ):
            issues.append("FORWARD_SECOND_BOND_COLLISION")
    return OverlapGeometry(
        union, tuple(right.items()), tuple(left.items()), tuple(issues)
    )
