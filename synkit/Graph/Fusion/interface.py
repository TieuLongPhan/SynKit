"""Explicit two-arm interfaces for verified graph fusion."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Hashable, Mapping, Sequence

import networkx as nx

from synkit.Graph.Morphism import (
    GraphMorphism,
    NodeStateKind,
    WildcardConstraint,
    adapt_legacy_node_state,
)
from synkit.Graph.Morphism import StereoEffect

from .identity import FUSION_WL_ITERATIONS

DEFAULT_INTERFACE_NODE_KEYS = (
    "element",
    "aromatic",
    "charge",
    "radical",
)
DEFAULT_INTERFACE_EDGE_KEYS = ("order",)


class FusionInterfaceIssueCode(str, Enum):
    """Stable failures raised before graph construction."""

    SOURCE_MISMATCH = "FUSION_INTERFACE_SOURCE_MISMATCH"
    TARGET_MISMATCH = "FUSION_INTERFACE_TARGET_MISMATCH"
    NODE_CONFLICT = "FUSION_INTERFACE_NODE_CONFLICT"
    EDGE_CONFLICT = "FUSION_INTERFACE_EDGE_CONFLICT"
    WILDCARD_CONFLICT = "FUSION_INTERFACE_WILDCARD_CONFLICT"
    OWNER_OUTSIDE_INTERFACE = "FUSION_INTERFACE_OWNER_OUTSIDE_INTERFACE"
    INVALID_MAPPING = "FUSION_INTERFACE_INVALID_MAPPING"


@dataclass(frozen=True)
class FusionInterfaceIssue:
    code: FusionInterfaceIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class FusionInterfaceError(ValueError):
    """Raised when a proposed common interface is not compatible."""

    def __init__(self, *issues: FusionInterfaceIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((repr(key), _stable_value(item)) for key, item in value.items()),
                key=repr,
            )
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        items = tuple(_stable_value(item) for item in value)
        return (
            tuple(sorted(items, key=repr))
            if isinstance(value, (set, frozenset))
            else items
        )
    return value


def _attribute_signature(
    attributes: Mapping[str, Any], keys: Sequence[str]
) -> tuple[Any, ...]:
    return tuple((key, _stable_value(attributes.get(key))) for key in keys)


def _wildcard_values(wildcard_element: Any) -> tuple[Any, ...]:
    scalar = (
        wildcard_element[0]
        if isinstance(wildcard_element, (tuple, list))
        else wildcard_element
    )
    return (wildcard_element, scalar)


def _constraint_for_interface(
    graph: nx.Graph,
    node: Hashable,
    inverse_arm: Mapping[Hashable, Hashable],
    *,
    element_key: str,
    role_key: str,
    wildcard_element: Any,
) -> WildcardConstraint | None:
    state = adapt_legacy_node_state(
        graph.nodes[node],
        element_key=element_key,
        role_key=role_key,
        wildcard_values=_wildcard_values(wildcard_element),
    )
    if state.kind is not NodeStateKind.WILDCARD:
        return None
    constraint = state.constraint
    if constraint is None:
        return None
    if constraint.owner is not None and constraint.owner not in inverse_arm:
        raise FusionInterfaceError(
            FusionInterfaceIssue(
                FusionInterfaceIssueCode.OWNER_OUTSIDE_INTERFACE,
                "A wildcard owner is absent from the proposed interface.",
                {"node": repr(node), "owner": repr(constraint.owner)},
            )
        )
    return constraint.relabel_owner(inverse_arm)


def _validated_mapping_pairs(
    forward_graph: nx.Graph,
    backward_graph: nx.Graph,
    mapping: Mapping[Hashable, Hashable],
) -> tuple[tuple[Hashable, Hashable], ...]:
    pairs = tuple(
        sorted(mapping.items(), key=lambda item: (repr(item[0]), repr(item[1])))
    )
    if not pairs:
        raise FusionInterfaceError(
            FusionInterfaceIssue(
                FusionInterfaceIssueCode.INVALID_MAPPING,
                "A fusion interface cannot be empty.",
            )
        )
    if len({left for left, _ in pairs}) != len(pairs) or len(
        {right for _, right in pairs}
    ) != len(pairs):
        raise FusionInterfaceError(
            FusionInterfaceIssue(
                FusionInterfaceIssueCode.INVALID_MAPPING,
                "Interface mappings must be injective.",
            )
        )
    if any(
        left not in forward_graph or right not in backward_graph
        for left, right in pairs
    ):
        raise FusionInterfaceError(
            FusionInterfaceIssue(
                FusionInterfaceIssueCode.TARGET_MISMATCH,
                "Interface images must exist in both source graphs.",
            )
        )
    return pairs


def _compatible_node_label(
    forward_attrs: Mapping[str, Any],
    backward_attrs: Mapping[str, Any],
    *,
    node_keys: Sequence[str],
    element_key: str,
    wildcard_values: tuple[Any, ...],
    forward_node: Hashable,
    backward_node: Hashable,
) -> tuple[Any, ...]:
    forward_wildcard = forward_attrs.get(element_key) in wildcard_values
    backward_wildcard = backward_attrs.get(element_key) in wildcard_values
    if forward_wildcard or backward_wildcard:
        concrete = backward_attrs if forward_wildcard else forward_attrs
        return _attribute_signature(concrete, node_keys)
    left_label = _attribute_signature(forward_attrs, node_keys)
    right_label = _attribute_signature(backward_attrs, node_keys)
    if left_label != right_label:
        raise FusionInterfaceError(
            FusionInterfaceIssue(
                FusionInterfaceIssueCode.NODE_CONFLICT,
                "Mapped concrete nodes disagree on interface attributes.",
                {
                    "forward_node": repr(forward_node),
                    "backward_node": repr(backward_node),
                },
            )
        )
    return left_label


def _endpoint_scalar(value: Any) -> Any:
    if isinstance(value, (tuple, list)) and len(value) == 2 and value[0] == value[1]:
        return value[0]
    return value


def _constraint_accepts_concrete(
    constraint: WildcardConstraint,
    attributes: Mapping[str, Any],
) -> bool:
    concrete = {
        "element": _endpoint_scalar(attributes.get("element")),
        "charge": _endpoint_scalar(attributes.get("charge", 0)),
        "radical": _endpoint_scalar(attributes.get("radical", 0)),
    }
    if (
        constraint.elements is not None
        and concrete["element"] not in constraint.elements
    ):
        return False
    if constraint.charges is not None and concrete["charge"] not in constraint.charges:
        return False
    if (
        constraint.radicals is not None
        and concrete["radical"] not in constraint.radicals
    ):
        return False
    return constraint.virtual_kind is None


def _common_interface_edges(
    forward_graph: nx.Graph,
    backward_graph: nx.Graph,
    forward_arm: Mapping[Hashable, Hashable],
    backward_arm: Mapping[Hashable, Hashable],
    edge_keys: Sequence[str],
) -> tuple[tuple[Hashable, Hashable, tuple[Any, ...]], ...]:
    edges: list[tuple[Hashable, Hashable, tuple[Any, ...]]] = []
    inverse_forward = {target: source for source, target in forward_arm.items()}
    for f_left, f_right in forward_graph.edges:
        if f_left not in inverse_forward or f_right not in inverse_forward:
            continue
        left, right = inverse_forward[f_left], inverse_forward[f_right]
        b_left, b_right = backward_arm[left], backward_arm[right]
        if not backward_graph.has_edge(b_left, b_right):
            continue
        f_label = _attribute_signature(forward_graph.edges[f_left, f_right], edge_keys)
        b_label = _attribute_signature(backward_graph.edges[b_left, b_right], edge_keys)
        if f_label != b_label:
            raise FusionInterfaceError(
                FusionInterfaceIssue(
                    FusionInterfaceIssueCode.EDGE_CONFLICT,
                    "Mapped interface edges disagree on bond attributes.",
                    {"edge": (repr(left), repr(right))},
                )
            )
        edges.append((left, right, f_label))
    return tuple(sorted(edges, key=repr))


@dataclass(frozen=True)
class FusionInterface:
    """A common graph ``K`` with injective arms into two source graphs."""

    forward_morphism: GraphMorphism
    backward_morphism: GraphMorphism
    node_labels: tuple[tuple[Hashable, tuple[Any, ...]], ...] = ()
    edges: tuple[tuple[Hashable, Hashable, tuple[Any, ...]], ...] = ()
    stereo_effects: tuple[tuple[str, str, str, StereoEffect], ...] = ()

    def __post_init__(self) -> None:
        forward = self.forward_morphism
        backward = self.backward_morphism
        if (
            forward.source != backward.source
            or forward.source_nodes != backward.source_nodes
        ):
            raise FusionInterfaceError(
                FusionInterfaceIssue(
                    FusionInterfaceIssueCode.SOURCE_MISMATCH,
                    "Both fusion arms must have the same interface source.",
                )
            )
        labels = tuple(sorted(tuple(self.node_labels), key=lambda item: repr(item[0])))
        if labels and {node for node, _ in labels} != set(forward.source_nodes):
            raise FusionInterfaceError(
                FusionInterfaceIssue(
                    FusionInterfaceIssueCode.INVALID_MAPPING,
                    "Interface node labels must be total on K.",
                )
            )
        normalized_edges = tuple(
            sorted(
                (
                    (
                        (left, right, label)
                        if repr(left) <= repr(right)
                        else (right, left, label)
                    )
                    for left, right, label in self.edges
                ),
                key=repr,
            )
        )
        if any(
            left not in forward.source_nodes or right not in forward.source_nodes
            for left, right, _ in normalized_edges
        ):
            raise FusionInterfaceError(
                FusionInterfaceIssue(
                    FusionInterfaceIssueCode.INVALID_MAPPING,
                    "Interface edges must have endpoints in K.",
                )
            )
        object.__setattr__(self, "node_labels", labels)
        object.__setattr__(self, "edges", normalized_edges)
        effects = tuple(
            sorted(
                (
                    (
                        side,
                        layer,
                        descriptor_id,
                        (
                            effect
                            if isinstance(effect, StereoEffect)
                            else StereoEffect(effect)
                        ),
                    )
                    for side, layer, descriptor_id, effect in self.stereo_effects
                ),
                key=repr,
            )
        )
        if any(side not in {"forward", "backward"} for side, _, _, _ in effects):
            raise ValueError("Stereo effects require side 'forward' or 'backward'.")
        object.__setattr__(self, "stereo_effects", effects)

        f_theta = forward.substitutions
        b_theta = backward.substitutions
        for node in forward.source_nodes:
            left = f_theta.get(node)
            right = b_theta.get(node)
            if (
                left is not None
                and right is not None
                and not left.intersect(right).valid
            ):
                raise FusionInterfaceError(
                    FusionInterfaceIssue(
                        FusionInterfaceIssueCode.WILDCARD_CONFLICT,
                        "Wildcard substitutions contradict across the interface.",
                        {"interface_node": repr(node)},
                    )
                )

        edge_pairs = {frozenset((left, right)) for left, right, _ in normalized_edges}
        for node, constraint in self.substitutions.items():
            if (
                constraint.owner is not None
                and frozenset((node, constraint.owner)) not in edge_pairs
            ):
                raise FusionInterfaceError(
                    FusionInterfaceIssue(
                        FusionInterfaceIssueCode.OWNER_OUTSIDE_INTERFACE,
                        "Wildcard owner incidence is absent from K.",
                        {
                            "interface_node": repr(node),
                            "owner": repr(constraint.owner),
                        },
                    )
                )

    @property
    def interface_nodes(self) -> frozenset[Hashable]:
        return self.forward_morphism.source_nodes

    @property
    def substitutions(self) -> dict[Hashable, WildcardConstraint]:
        """Return the normalized cross-arm substitution environment."""
        result: dict[Hashable, WildcardConstraint] = {}
        forward = self.forward_morphism.substitutions
        backward = self.backward_morphism.substitutions
        for node in self.interface_nodes:
            left = forward.get(node)
            right = backward.get(node)
            if left is None and right is None:
                continue
            if left is None:
                result[node] = right  # type: ignore[assignment]
            elif right is None:
                result[node] = left
            else:
                unified = left.intersect(right)
                if unified.constraint is not None:
                    result[node] = unified.constraint
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return the interface and both morphism arms as proof data."""

        def morphism_payload(morphism: GraphMorphism) -> dict[str, Any]:
            return {
                "source": repr(morphism.source),
                "target": repr(morphism.target),
                "mapping": [(repr(left), repr(right)) for left, right in morphism.f],
                "theta": [
                    {
                        "node": repr(node),
                        "constraint": constraint.normalized(),
                    }
                    for node, constraint in morphism.theta
                ],
                "canonical_signature": morphism.canonical_signature(),
            }

        return {
            "nodes": [repr(node) for node in sorted(self.interface_nodes, key=repr)],
            "node_labels": [(repr(node), label) for node, label in self.node_labels],
            "edges": [
                (repr(left), repr(right), label) for left, right, label in self.edges
            ],
            "stereo_effects": [
                {
                    "side": side,
                    "layer": layer,
                    "descriptor_id": descriptor_id,
                    "effect": effect.value,
                }
                for side, layer, descriptor_id, effect in self.stereo_effects
            ],
            "forward_morphism": morphism_payload(self.forward_morphism),
            "backward_morphism": morphism_payload(self.backward_morphism),
            "canonical_signature": self.canonical_signature(),
        }

    def canonical_signature(self) -> tuple[Any, ...]:
        """Return a relabeling-invariant signature of K and both arms."""
        graph = nx.Graph()
        labels = dict(self.node_labels)
        substitutions = self.substitutions
        for node in self.interface_nodes:
            constraint = substitutions.get(node)
            graph.add_node(
                node,
                signature=repr(
                    (
                        labels.get(node, ()),
                        constraint.normalized() if constraint is not None else None,
                    )
                ),
            )
        for left, right, label in self.edges:
            graph.add_edge(left, right, signature=repr(label))
        digest = nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr="signature",
            edge_attr="signature",
            iterations=FUSION_WL_ITERATIONS,
            digest_size=16,
        )
        arms = tuple(
            sorted(
                (
                    self.forward_morphism.canonical_signature(),
                    self.backward_morphism.canonical_signature(),
                ),
                key=repr,
            )
        )
        effects = tuple(
            (side, layer, descriptor_id, effect.value)
            for side, layer, descriptor_id, effect in self.stereo_effects
        )
        return (digest, arms, effects)

    def with_stereo_effects(
        self,
        effects: Mapping[tuple[str, str, str], StereoEffect | str],
    ) -> "FusionInterface":
        """Return a copy carrying explicit rule-authoritative stereo effects."""
        return replace(
            self,
            stereo_effects=tuple(
                (side, layer, descriptor_id, StereoEffect(effect))
                for (side, layer, descriptor_id), effect in effects.items()
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        forward_graph: nx.Graph,
        backward_graph: nx.Graph,
        mapping: Mapping[Hashable, Hashable],
        *,
        node_keys: Sequence[str] = DEFAULT_INTERFACE_NODE_KEYS,
        edge_keys: Sequence[str] = DEFAULT_INTERFACE_EDGE_KEYS,
        element_key: str = "element",
        role_key: str = "wildcard_role",
        wildcard_element: Any = "*",
    ) -> "FusionInterface":
        """Validate a proposed forward-to-backward overlap and build K."""
        pairs = _validated_mapping_pairs(forward_graph, backward_graph, mapping)

        interface_nodes = frozenset(range(len(pairs)))
        forward_arm = {index: left for index, (left, _) in enumerate(pairs)}
        backward_arm = {index: right for index, (_, right) in enumerate(pairs)}
        inverse_forward = {value: key for key, value in forward_arm.items()}
        inverse_backward = {value: key for key, value in backward_arm.items()}
        forward_theta: dict[Hashable, WildcardConstraint] = {}
        backward_theta: dict[Hashable, WildcardConstraint] = {}
        labels: list[tuple[Hashable, tuple[Any, ...]]] = []
        wildcard_values = _wildcard_values(wildcard_element)

        for interface_node, (forward_node, backward_node) in enumerate(pairs):
            forward_attrs = forward_graph.nodes[forward_node]
            backward_attrs = backward_graph.nodes[backward_node]
            forward_wildcard = forward_attrs.get(element_key) in wildcard_values
            backward_wildcard = backward_attrs.get(element_key) in wildcard_values
            labels.append(
                (
                    interface_node,
                    _compatible_node_label(
                        forward_attrs,
                        backward_attrs,
                        node_keys=node_keys,
                        element_key=element_key,
                        wildcard_values=wildcard_values,
                        forward_node=forward_node,
                        backward_node=backward_node,
                    ),
                )
            )

            if forward_wildcard:
                constraint = _constraint_for_interface(
                    forward_graph,
                    forward_node,
                    inverse_forward,
                    element_key=element_key,
                    role_key=role_key,
                    wildcard_element=wildcard_element,
                )
                if constraint is not None:
                    forward_theta[interface_node] = constraint
            if backward_wildcard:
                constraint = _constraint_for_interface(
                    backward_graph,
                    backward_node,
                    inverse_backward,
                    element_key=element_key,
                    role_key=role_key,
                    wildcard_element=wildcard_element,
                )
                if constraint is not None:
                    backward_theta[interface_node] = constraint
            if forward_wildcard and not backward_wildcard:
                constraint = forward_theta.get(interface_node)
                if constraint is not None and not _constraint_accepts_concrete(
                    constraint, backward_attrs
                ):
                    raise FusionInterfaceError(
                        FusionInterfaceIssue(
                            FusionInterfaceIssueCode.WILDCARD_CONFLICT,
                            "Backward concrete state violates the forward wildcard domain.",
                            {"interface_node": repr(interface_node)},
                        )
                    )
            if backward_wildcard and not forward_wildcard:
                constraint = backward_theta.get(interface_node)
                if constraint is not None and not _constraint_accepts_concrete(
                    constraint, forward_attrs
                ):
                    raise FusionInterfaceError(
                        FusionInterfaceIssue(
                            FusionInterfaceIssueCode.WILDCARD_CONFLICT,
                            "Forward concrete state violates the backward wildcard domain.",
                            {"interface_node": repr(interface_node)},
                        )
                    )

        edges = _common_interface_edges(
            forward_graph,
            backward_graph,
            forward_arm,
            backward_arm,
            edge_keys,
        )

        forward = GraphMorphism(
            "K",
            "forward",
            interface_nodes,
            frozenset(forward_graph.nodes),
            forward_arm,
            forward_theta,
        )
        backward = GraphMorphism(
            "K",
            "backward",
            interface_nodes,
            frozenset(backward_graph.nodes),
            backward_arm,
            backward_theta,
        )
        return cls(forward, backward, tuple(labels), edges)


__all__ = [
    "DEFAULT_INTERFACE_EDGE_KEYS",
    "DEFAULT_INTERFACE_NODE_KEYS",
    "FusionInterface",
    "FusionInterfaceError",
    "FusionInterfaceIssue",
    "FusionInterfaceIssueCode",
]
