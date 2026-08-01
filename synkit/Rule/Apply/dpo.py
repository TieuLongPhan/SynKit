"""Typed DPO rule spans and application over finite Lewis-labelled graphs.

Rule interfaces carry identity labels and incidence only.  Endpoint state is
stored separately in the left and right LLG objects, so a preserved atom or
bond may change state without pretending that a strict LLG morphism preserves
two different values.  Matches into host graphs remain strict LLG morphisms.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Mapping

import networkx as nx

from synkit.Graph.Morphism import (
    ELECTRON_LLG_SCHEMA,
    LLGError,
    LLGIssueCode,
    LLGMorphism,
    LabelSchema,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
    llg_from_its,
)


class DPOIssueCode(str, Enum):
    """Stable refusal codes for rule construction and application."""

    SCHEMA_MISMATCH = "DPO_SCHEMA_MISMATCH"
    INTERFACE_SCHEMA = "DPO_INTERFACE_SCHEMA"
    INTERFACE_PARTIAL = "DPO_INTERFACE_PARTIAL"
    INTERFACE_NON_INJECTIVE = "DPO_INTERFACE_NON_INJECTIVE"
    INTERFACE_INCIDENCE = "DPO_INTERFACE_INCIDENCE"
    INTERFACE_LABEL = "DPO_INTERFACE_LABEL"
    MATCH_PARTIAL = "DPO_MATCH_PARTIAL"
    IDENTIFICATION = "DPO_IDENTIFICATION_CONDITION"
    MATCH_OUTSIDE_HOST = "DPO_MATCH_OUTSIDE_HOST"
    MATCH_INCIDENCE = "DPO_MATCH_INCIDENCE"
    MATCH_LABEL = "DPO_MATCH_LABEL"
    DANGLING = "DPO_DANGLING_CONDITION"
    RESULT_LABEL = "DPO_RESULT_LABEL_CONFLICT"
    CLOSED_MATERIAL = "DPO_CLOSED_MATERIAL_IMBALANCE"
    CLOSED_ELECTRON = "DPO_CLOSED_ELECTRON_IMBALANCE"
    OPEN_TOKEN_REQUIRED = "DPO_OPEN_TOKEN_REQUIRED"
    OPEN_TOKEN_MISMATCH = "DPO_OPEN_TOKEN_MISMATCH"
    ELECTRON_SCHEMA_REQUIRED = "DPO_ELECTRON_SCHEMA_REQUIRED"
    NONCOMMUTATIVE = "DPO_NONCOMMUTATIVE_SQUARE"
    SOURCE_MUTATED = "DPO_SOURCE_MUTATED"


@dataclass(frozen=True)
class DPOIssue:
    code: DPOIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class DPOError(ValueError):
    """Raised when a DPO premise or construction fails."""

    def __init__(self, *issues: DPOIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in issues))


class SystemBoundary(str, Enum):
    """Resource policy attached to a rule."""

    ABSTRACT = "abstract"
    CLOSED = "closed"
    OPEN = "open"


@dataclass(frozen=True)
class EnvironmentToken:
    """Explicit net resources supplied by an open environment.

    Deltas use ``right - left`` convention.  Positive values enter the graph
    system; negative values leave it.
    """

    name: str
    element_delta: tuple[tuple[str, int], ...] = ()
    electron_delta: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("An environment token requires a non-empty name.")
        normalized = tuple(
            sorted(
                (str(element), int(count))
                for element, count in dict(self.element_delta).items()
                if int(count) != 0
            )
        )
        object.__setattr__(self, "element_delta", normalized)
        object.__setattr__(self, "electron_delta", float(self.electron_delta))


def _identity_schema(schema: LabelSchema) -> LabelSchema:
    return LabelSchema(
        node_identity=schema.node_identity,
        edge_identity=schema.edge_identity,
        name=f"{schema.name}/partial-interface",
    )


@dataclass(frozen=True)
class RuleEmbedding:
    """An identity-labelled interface embedding into a valued endpoint."""

    interface: LewisLabelledGraph
    endpoint: LewisLabelledGraph
    f: tuple[tuple[Hashable, Hashable], ...]

    def __post_init__(self) -> None:
        pairs = tuple(self.f.items()) if isinstance(self.f, Mapping) else tuple(self.f)
        object.__setattr__(self, "f", tuple(sorted(pairs, key=repr)))
        expected = _identity_schema(self.endpoint.schema)
        if self.interface.schema != expected:
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.INTERFACE_SCHEMA,
                    "The rule interface must carry exactly endpoint identity labels.",
                )
            )
        source_keys = tuple(left for left, _ in pairs)
        target_values = tuple(right for _, right in pairs)
        missing = set(self.interface.node_ids) - set(source_keys)
        outside = set(target_values) - set(self.endpoint.node_ids)
        if len(set(source_keys)) != len(source_keys) or missing or outside:
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.INTERFACE_PARTIAL,
                    "An interface arm is total and lands in its endpoint.",
                    {
                        "missing": tuple(sorted(map(repr, missing))),
                        "outside": tuple(sorted(map(repr, outside))),
                    },
                )
            )
        if len(set(target_values)) != len(target_values):
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.INTERFACE_NON_INJECTIVE,
                    "An interface arm must be injective.",
                )
            )
        mapping = dict(pairs)
        issues: list[DPOIssue] = []
        for node, image in pairs:
            interface_labels = self.interface.node_labels(node, semantic=True)
            endpoint_labels = self.endpoint.node_labels(image, semantic=True)
            endpoint_identity = {
                key: endpoint_labels[key] for key in self.endpoint.schema.node_identity
            }
            if interface_labels != endpoint_identity:
                issues.append(
                    DPOIssue(
                        DPOIssueCode.INTERFACE_LABEL,
                        "An interface node identity label is not preserved.",
                        {"node": repr(node), "image": repr(image)},
                    )
                )
        for edge in self.interface.edge_keys:
            image = frozenset(mapping[node] for node in edge)
            if image not in self.endpoint.edge_keys:
                issues.append(
                    DPOIssue(
                        DPOIssueCode.INTERFACE_INCIDENCE,
                        "An interface edge image is absent from its endpoint.",
                        {"edge": tuple(sorted(map(repr, edge)))},
                    )
                )
                continue
            interface_labels = self.interface.edge_labels(edge, semantic=True)
            endpoint_labels = self.endpoint.edge_labels(image, semantic=True)
            endpoint_identity = {
                key: endpoint_labels[key] for key in self.endpoint.schema.edge_identity
            }
            if interface_labels != endpoint_identity:
                issues.append(
                    DPOIssue(
                        DPOIssueCode.INTERFACE_LABEL,
                        "An interface edge identity label is not preserved.",
                        {"edge": tuple(sorted(map(repr, edge)))},
                    )
                )
        if issues:
            raise DPOError(*issues)

    @property
    def mapping(self) -> dict[Hashable, Hashable]:
        return dict(self.f)

    @property
    def edge_mapping(self) -> dict[frozenset[Hashable], frozenset[Hashable]]:
        mapping = self.mapping
        return {
            edge: frozenset(mapping[node] for node in edge)
            for edge in self.interface.edge_keys
        }


@dataclass(frozen=True)
class ResourceDelta:
    """Rule-level material and electron delta using right-minus-left sign."""

    element_delta: tuple[tuple[str, int], ...]
    electron_delta: float


def _resource_delta(
    left: LewisLabelledGraph, right: LewisLabelledGraph
) -> ResourceDelta:
    if left.schema != ELECTRON_LLG_SCHEMA or right.schema != ELECTRON_LLG_SCHEMA:
        raise DPOError(
            DPOIssue(
                DPOIssueCode.ELECTRON_SCHEMA_REQUIRED,
                "Chemical resource policies require the electron-complete LLG schema.",
            )
        )

    def inventory(graph: LewisLabelledGraph) -> tuple[Counter[str], float]:
        elements: Counter[str] = Counter()
        electrons = 0.0
        for node in graph.node_ids:
            labels = graph.node_labels(node)
            elements[str(labels["element"])] += 1
            electrons += (
                float(labels["valence_electrons"])
                + float(labels["hcount"])
                - float(labels["charge"])
            )
        return elements, electrons

    left_elements, left_electrons = inventory(left)
    right_elements, right_electrons = inventory(right)
    keys = sorted(set(left_elements) | set(right_elements))
    element_delta = tuple(
        (key, right_elements[key] - left_elements[key])
        for key in keys
        if right_elements[key] != left_elements[key]
    )
    return ResourceDelta(element_delta, right_electrons - left_electrons)


@dataclass(frozen=True)
class RuleSpan:
    """A partially labelled DPO rule ``L <- K -> R``."""

    left: LewisLabelledGraph
    interface: LewisLabelledGraph
    right: LewisLabelledGraph
    left_arm: RuleEmbedding
    right_arm: RuleEmbedding
    boundary: SystemBoundary = SystemBoundary.ABSTRACT
    environment: EnvironmentToken | None = None
    name: str = "rule"

    def __post_init__(self) -> None:
        if self.left.schema != self.right.schema:
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.SCHEMA_MISMATCH,
                    "Rule endpoints must use the same LLG schema.",
                )
            )
        if (
            self.left_arm.interface != self.interface
            or self.right_arm.interface != self.interface
            or self.left_arm.endpoint != self.left
            or self.right_arm.endpoint != self.right
        ):
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.INTERFACE_SCHEMA,
                    "Rule arms must have the declared interface and endpoints.",
                )
            )
        boundary = SystemBoundary(self.boundary)
        object.__setattr__(self, "boundary", boundary)
        if boundary is SystemBoundary.ABSTRACT:
            if self.environment is not None:
                raise DPOError(
                    DPOIssue(
                        DPOIssueCode.OPEN_TOKEN_MISMATCH,
                        "Abstract rules do not consume chemical environment tokens.",
                    )
                )
            return
        delta = _resource_delta(self.left, self.right)
        if boundary is SystemBoundary.CLOSED:
            issues = []
            if delta.element_delta:
                issues.append(
                    DPOIssue(
                        DPOIssueCode.CLOSED_MATERIAL,
                        "A closed rule changes the elemental inventory.",
                        {"delta": delta.element_delta},
                    )
                )
            if delta.electron_delta != 0:
                issues.append(
                    DPOIssue(
                        DPOIssueCode.CLOSED_ELECTRON,
                        "A closed rule changes the electron inventory.",
                        {"delta": delta.electron_delta},
                    )
                )
            if self.environment is not None:
                issues.append(
                    DPOIssue(
                        DPOIssueCode.OPEN_TOKEN_MISMATCH,
                        "A closed rule cannot carry an environment token.",
                    )
                )
            if issues:
                raise DPOError(*issues)
            return
        if self.environment is None:
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.OPEN_TOKEN_REQUIRED,
                    "An open rule requires an explicit environment token.",
                )
            )
        if (
            self.environment.element_delta != delta.element_delta
            or self.environment.electron_delta != delta.electron_delta
        ):
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.OPEN_TOKEN_MISMATCH,
                    "The environment token does not equal the rule resource delta.",
                    {
                        "expected_elements": delta.element_delta,
                        "observed_elements": self.environment.element_delta,
                        "expected_electrons": delta.electron_delta,
                        "observed_electrons": self.environment.electron_delta,
                    },
                )
            )

    @classmethod
    def from_mapping(
        cls,
        left: LewisLabelledGraph,
        right: LewisLabelledGraph,
        preserved_nodes: Mapping[Hashable, Hashable],
        *,
        preserved_edges: set[frozenset[Hashable]] | None = None,
        boundary: SystemBoundary | str = SystemBoundary.ABSTRACT,
        environment: EnvironmentToken | None = None,
        name: str = "rule",
    ) -> "RuleSpan":
        """Build an explicit interface from a preserved endpoint mapping.

        ``preserved_edges`` contains left-side edge keys.  When omitted, every
        left edge whose mapped right edge exists is structurally preserved.
        """
        if left.schema != right.schema:
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.SCHEMA_MISMATCH,
                    "Rule endpoints must use the same LLG schema.",
                )
            )
        pairs = tuple(sorted(preserved_nodes.items(), key=repr))
        left_nodes = tuple(node for node, _ in pairs)
        if len(set(left_nodes)) != len(left_nodes):
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.INTERFACE_PARTIAL,
                    "Each preserved left node occurs once.",
                )
            )
        interface_schema = _identity_schema(left.schema)
        interface_graph = nx.Graph()
        for interface_node, (left_node, right_node) in enumerate(pairs):
            if left_node not in left.node_ids or right_node not in right.node_ids:
                raise DPOError(
                    DPOIssue(
                        DPOIssueCode.INTERFACE_PARTIAL,
                        "A preserved node must exist in both endpoints.",
                    )
                )
            left_identity = {
                key: left.node_labels(left_node, semantic=True)[key]
                for key in left.schema.node_identity
            }
            right_identity = {
                key: right.node_labels(right_node, semantic=True)[key]
                for key in right.schema.node_identity
            }
            if left_identity != right_identity:
                raise DPOError(
                    DPOIssue(
                        DPOIssueCode.INTERFACE_LABEL,
                        "Preserved nodes disagree on identity labels.",
                        {"left": repr(left_node), "right": repr(right_node)},
                    )
                )
            interface_graph.add_node(interface_node, **left_identity)

        left_to_interface = {
            left_node: interface_node
            for interface_node, (left_node, _) in enumerate(pairs)
        }
        right_by_left = dict(pairs)
        candidates = {
            edge
            for edge in left.edge_keys
            if edge <= set(left_to_interface)
            and frozenset(right_by_left[node] for node in edge) in right.edge_keys
        }
        selected = candidates if preserved_edges is None else set(preserved_edges)
        if not selected <= candidates:
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.INTERFACE_INCIDENCE,
                    "A preserved edge must exist under both endpoint arms.",
                )
            )
        for edge in selected:
            left_a, left_b = tuple(edge)
            right_edge = frozenset((right_by_left[left_a], right_by_left[left_b]))
            left_identity = {
                key: left.edge_labels(edge, semantic=True)[key]
                for key in left.schema.edge_identity
            }
            right_identity = {
                key: right.edge_labels(right_edge, semantic=True)[key]
                for key in right.schema.edge_identity
            }
            if left_identity != right_identity:
                raise DPOError(
                    DPOIssue(
                        DPOIssueCode.INTERFACE_LABEL,
                        "Preserved edges disagree on identity labels.",
                    )
                )
            interface_graph.add_edge(
                left_to_interface[left_a], left_to_interface[left_b], **left_identity
            )

        interface = LewisLabelledGraph.from_networkx(
            interface_graph, interface_schema, name=f"{name}:K"
        )
        left_map = {
            interface_node: left_node
            for interface_node, (left_node, _) in enumerate(pairs)
        }
        right_map = {
            interface_node: right_node
            for interface_node, (_, right_node) in enumerate(pairs)
        }
        return cls(
            left,
            interface,
            right,
            RuleEmbedding(interface, left, left_map),
            RuleEmbedding(interface, right, right_map),
            SystemBoundary(boundary),
            environment,
            name,
        )

    @property
    def resource_delta(self) -> ResourceDelta | None:
        if self.boundary is SystemBoundary.ABSTRACT:
            return None
        return _resource_delta(self.left, self.right)


@dataclass(frozen=True)
class DPOReplay:
    valid: bool
    issues: tuple[DPOIssue, ...] = ()


@dataclass(frozen=True)
class DPOCertificate:
    """Complete carrier-level certificate for the two DPO squares."""

    rule: RuleSpan
    host: LewisLabelledGraph
    context: LewisLabelledGraph
    result: LewisLabelledGraph
    match: tuple[tuple[Hashable, Hashable], ...]
    interface_to_context: tuple[tuple[Hashable, Hashable], ...]
    context_to_host: tuple[tuple[Hashable, Hashable], ...]
    context_to_result: tuple[tuple[Hashable, Hashable], ...]
    right_to_result: tuple[tuple[Hashable, Hashable], ...]
    deleted_nodes: tuple[Hashable, ...]
    deleted_edges: tuple[frozenset[Hashable], ...]
    added_nodes: tuple[Hashable, ...]
    added_edges: tuple[frozenset[Hashable], ...]
    resource_delta: ResourceDelta | None

    def replay(self) -> DPOReplay:
        issues: list[DPOIssue] = []
        match = dict(self.match)
        left_arm = self.rule.left_arm.mapping
        right_arm = self.rule.right_arm.mapping
        to_context = dict(self.interface_to_context)
        context_host = dict(self.context_to_host)
        context_result = dict(self.context_to_result)
        right_result = dict(self.right_to_result)
        domains = (
            ("match", set(match), set(self.rule.left.node_ids)),
            (
                "interface_to_context",
                set(to_context),
                set(self.rule.interface.node_ids),
            ),
            ("context_to_host", set(context_host), set(self.context.node_ids)),
            ("context_to_result", set(context_result), set(self.context.node_ids)),
            ("right_to_result", set(right_result), set(self.rule.right.node_ids)),
        )
        for name, observed, expected in domains:
            if observed != expected:
                issues.append(
                    DPOIssue(
                        DPOIssueCode.NONCOMMUTATIVE,
                        "A certificate carrier map has the wrong domain.",
                        {
                            "map": name,
                            "missing": tuple(sorted(map(repr, expected - observed))),
                            "extra": tuple(sorted(map(repr, observed - expected))),
                        },
                    )
                )
        missing_host = set(context_host.values()) - set(self.host.node_ids)
        missing_result = (
            set(context_result.values()) | set(right_result.values())
        ) - set(self.result.node_ids)
        if missing_host or missing_result:
            issues.append(
                DPOIssue(
                    DPOIssueCode.NONCOMMUTATIVE,
                    "A certificate map lands outside its declared object.",
                    {
                        "host": tuple(sorted(map(repr, missing_host))),
                        "result": tuple(sorted(map(repr, missing_result))),
                    },
                )
            )
        missing = object()
        for node in self.rule.interface.node_ids:
            left_path = match.get(left_arm[node], missing)
            context_node = to_context.get(node, missing)
            complement_path = context_host.get(context_node, missing)
            if left_path != complement_path:
                issues.append(
                    DPOIssue(
                        DPOIssueCode.NONCOMMUTATIVE,
                        "The pushout-complement square does not commute.",
                        {"interface_node": repr(node)},
                    )
                )
            right_path = right_result.get(right_arm[node], missing)
            pushout_path = context_result.get(context_node, missing)
            if right_path != pushout_path:
                issues.append(
                    DPOIssue(
                        DPOIssueCode.NONCOMMUTATIVE,
                        "The result pushout square does not commute.",
                        {"interface_node": repr(node)},
                    )
                )
        for edge in self.rule.interface.edge_keys:
            left_edge = _map_edge(left_arm, edge)
            matched_edge = _map_edge(match, left_edge)
            context_edge = _map_edge(to_context, edge)
            complement_edge = _map_edge(context_host, context_edge)
            if (
                matched_edge != complement_edge
                or matched_edge not in self.host.edge_keys
            ):
                issues.append(
                    DPOIssue(
                        DPOIssueCode.NONCOMMUTATIVE,
                        "The pushout-complement edge square does not commute.",
                        {"interface_edge": tuple(sorted(map(repr, edge)))},
                    )
                )
            right_edge = _map_edge(right_arm, edge)
            result_edge = _map_edge(right_result, right_edge)
            pushout_edge = _map_edge(context_result, context_edge)
            if result_edge != pushout_edge or result_edge not in self.result.edge_keys:
                issues.append(
                    DPOIssue(
                        DPOIssueCode.NONCOMMUTATIVE,
                        "The result edge square does not commute.",
                        {"interface_edge": tuple(sorted(map(repr, edge)))},
                    )
                )
        return DPOReplay(not issues, tuple(issues))


@dataclass(frozen=True)
class DPOApplication:
    result: LewisLabelledGraph
    certificate: DPOCertificate


def _translate_match_error(error: LLGError) -> DPOError:
    translated = []
    for issue in error.issues:
        if issue.code is LLGIssueCode.PARTIAL_MAPPING:
            code = DPOIssueCode.MATCH_PARTIAL
        elif issue.code is LLGIssueCode.NON_INJECTIVE:
            code = DPOIssueCode.IDENTIFICATION
        elif issue.code in {LLGIssueCode.OUTSIDE_SOURCE, LLGIssueCode.OUTSIDE_TARGET}:
            code = DPOIssueCode.MATCH_OUTSIDE_HOST
        elif issue.code is LLGIssueCode.MISSING_EDGE:
            code = DPOIssueCode.MATCH_INCIDENCE
        else:
            code = DPOIssueCode.MATCH_LABEL
        translated.append(DPOIssue(code, issue.message, issue.context))
    return DPOError(*translated)


def _map_edge(
    mapping: Mapping[Hashable, Hashable],
    edge: frozenset[Hashable] | None,
) -> frozenset[Hashable] | None:
    if edge is None or any(node not in mapping for node in edge):
        return None
    return frozenset(mapping[node] for node in edge)


def _fresh_nodes(existing: set[Hashable], count: int) -> tuple[tuple[str, int], ...]:
    fresh = []
    index = 0
    while len(fresh) < count:
        candidate = ("__dpo_fresh__", index)
        index += 1
        if candidate not in existing:
            fresh.append(candidate)
            existing.add(candidate)
    return tuple(fresh)


@dataclass(frozen=True)
class _DeletionPlan:
    nodes: frozenset[Hashable]
    edges: frozenset[frozenset[Hashable]]


@dataclass
class _ResultPlan:
    graph: nx.Graph
    interface_to_context: dict[Hashable, Hashable]
    right_to_result: dict[Hashable, Hashable]
    added_nodes: tuple[Hashable, ...]
    added_edges: set[frozenset[Hashable]]


def _validated_match(
    rule: RuleSpan,
    host: LewisLabelledGraph,
    match: Mapping[Hashable, Hashable],
) -> LLGMorphism:
    if rule.left.schema != host.schema:
        raise DPOError(
            DPOIssue(
                DPOIssueCode.SCHEMA_MISMATCH,
                "The rule left side and host must use the same schema.",
            )
        )
    try:
        return LLGMorphism(rule.left, host, tuple(match.items()))
    except LLGError as error:
        raise _translate_match_error(error) from error


def _deletion_plan(
    rule: RuleSpan,
    host: LewisLabelledGraph,
    match: Mapping[Hashable, Hashable],
) -> _DeletionPlan:
    preserved_nodes = set(rule.left_arm.mapping.values())
    deleted_nodes = frozenset(
        match[node] for node in set(rule.left.node_ids) - preserved_nodes
    )
    preserved_edges = set(rule.left_arm.edge_mapping.values())
    deleted_edges = frozenset(
        frozenset(match[node] for node in edge)
        for edge in set(rule.left.edge_keys) - preserved_edges
    )
    matched_edges = {
        frozenset(match[node] for node in edge) for edge in rule.left.edge_keys
    }
    dangling = {
        edge
        for edge in host.edge_keys
        if edge & deleted_nodes and edge not in matched_edges
    }
    if dangling:
        raise DPOError(
            DPOIssue(
                DPOIssueCode.DANGLING,
                "Deleting a matched node would leave an incident host edge dangling.",
                {
                    "edges": tuple(
                        sorted(
                            (tuple(sorted(map(repr, edge))) for edge in dangling),
                            key=repr,
                        )
                    )
                },
            )
        )
    return _DeletionPlan(deleted_nodes, deleted_edges)


def _construct_context(
    rule: RuleSpan,
    host: LewisLabelledGraph,
    deletion: _DeletionPlan,
) -> LewisLabelledGraph:
    graph = host.to_networkx()
    graph.remove_edges_from(tuple(tuple(edge) for edge in deletion.edges))
    graph.remove_nodes_from(deletion.nodes)
    return LewisLabelledGraph.from_networkx(graph, host.schema, name=f"{rule.name}:D")


def _initialize_result(
    rule: RuleSpan,
    context: LewisLabelledGraph,
    match: Mapping[Hashable, Hashable],
) -> _ResultPlan:
    graph = context.to_networkx()
    left_arm = rule.left_arm.mapping
    right_arm = rule.right_arm.mapping
    interface_to_context: dict[Hashable, Hashable] = {}
    right_to_result: dict[Hashable, Hashable] = {}
    for interface_node in rule.interface.node_ids:
        host_node = match[left_arm[interface_node]]
        interface_to_context[interface_node] = host_node
        right_to_result[right_arm[interface_node]] = host_node

    added_right = sorted(set(rule.right.node_ids) - set(right_arm.values()), key=repr)
    fresh = _fresh_nodes(set(graph.nodes), len(added_right))
    for right_node, result_node in zip(added_right, fresh):
        right_to_result[right_node] = result_node
        graph.add_node(result_node, **rule.right.node_labels(right_node))
    return _ResultPlan(
        graph,
        interface_to_context,
        right_to_result,
        tuple(fresh),
        set(),
    )


def _update_preserved_values(rule: RuleSpan, plan: _ResultPlan) -> None:
    right_arm = rule.right_arm.mapping
    node_keys = rule.right.schema.semantic_node_keys + rule.right.schema.node_derived
    for interface_node in rule.interface.node_ids:
        right_node = right_arm[interface_node]
        result_node = plan.right_to_result[right_node]
        right_labels = rule.right.node_labels(right_node)
        for key in node_keys:
            if key in right_labels:
                plan.graph.nodes[result_node][key] = right_labels[key]

    edge_keys = rule.right.schema.semantic_edge_keys + rule.right.schema.edge_derived
    right_edges = rule.right_arm.edge_mapping
    for interface_edge in rule.interface.edge_keys:
        right_edge = right_edges[interface_edge]
        result_edge = frozenset(plan.right_to_result[node] for node in right_edge)
        right_labels = rule.right.edge_labels(right_edge)
        left, right = tuple(result_edge)
        for key in edge_keys:
            if key in right_labels:
                plan.graph.edges[left, right][key] = right_labels[key]


def _add_right_edges(rule: RuleSpan, plan: _ResultPlan) -> None:
    preserved = set(rule.right_arm.edge_mapping.values())
    for right_edge in set(rule.right.edge_keys) - preserved:
        result_edge = frozenset(plan.right_to_result[node] for node in right_edge)
        left, right = tuple(result_edge)
        if plan.graph.has_edge(left, right):
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.RESULT_LABEL,
                    "An added rule edge collides with an unmatched host edge.",
                    {"edge": tuple(sorted(map(repr, result_edge)))},
                )
            )
        plan.graph.add_edge(left, right, **rule.right.edge_labels(right_edge))
        plan.added_edges.add(result_edge)


def _coherent_result(
    rule: RuleSpan,
    host: LewisLabelledGraph,
    plan: _ResultPlan,
) -> LewisLabelledGraph:
    graph = plan.graph
    if host.schema == ELECTRON_LLG_SCHEMA:
        for _, attrs in graph.nodes(data=True):
            for key in host.schema.node_derived:
                attrs.pop(key, None)
        for _, _, attrs in graph.edges(data=True):
            for key in host.schema.edge_derived:
                attrs.pop(key, None)
        try:
            graph = derive_electron_labeled_graph(graph)
        except LLGError as error:
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.RESULT_LABEL,
                    "The rewritten electron state is not coherent.",
                    {"issues": tuple(issue.to_dict() for issue in error.issues)},
                )
            ) from error
    try:
        return LewisLabelledGraph.from_networkx(
            graph, host.schema, name=f"{rule.name}:H"
        )
    except LLGError as error:
        raise DPOError(
            DPOIssue(
                DPOIssueCode.RESULT_LABEL,
                "The rewritten graph is outside the LLG object class.",
                {"issues": tuple(issue.to_dict() for issue in error.issues)},
            )
        ) from error


def _assert_unmutated(
    host: LewisLabelledGraph,
    rule: RuleSpan,
    before: tuple[nx.Graph, nx.Graph, nx.Graph],
) -> None:
    observed = (host.to_networkx(), rule.left.to_networkx(), rule.right.to_networkx())
    if any(
        not nx.utils.graphs_equal(current, original)
        for current, original in zip(observed, before)
    ):
        raise DPOError(
            DPOIssue(
                DPOIssueCode.SOURCE_MUTATED,
                "DPO application mutated a source object.",
            )
        )


def apply_dpo(
    rule: RuleSpan,
    host: LewisLabelledGraph,
    match: Mapping[Hashable, Hashable],
) -> DPOApplication:
    """Apply a rule at one explicit match or raise a typed DPO refusal."""
    match_morphism = _validated_match(rule, host, match)
    before = (host.to_networkx(), rule.left.to_networkx(), rule.right.to_networkx())
    match_map = match_morphism.mapping
    deletion = _deletion_plan(rule, host, match_map)
    context = _construct_context(rule, host, deletion)
    plan = _initialize_result(rule, context, match_map)
    _update_preserved_values(rule, plan)
    _add_right_edges(rule, plan)
    result = _coherent_result(rule, host, plan)
    _assert_unmutated(host, rule, before)

    context_identity = tuple((node, node) for node in context.node_ids)
    certificate = DPOCertificate(
        rule=rule,
        host=host,
        context=context,
        result=result,
        match=tuple(sorted(match_map.items(), key=repr)),
        interface_to_context=tuple(sorted(plan.interface_to_context.items(), key=repr)),
        context_to_host=context_identity,
        context_to_result=context_identity,
        right_to_result=tuple(sorted(plan.right_to_result.items(), key=repr)),
        deleted_nodes=tuple(sorted(deletion.nodes, key=repr)),
        deleted_edges=tuple(sorted(deletion.edges, key=repr)),
        added_nodes=plan.added_nodes,
        added_edges=tuple(sorted(plan.added_edges, key=repr)),
        resource_delta=rule.resource_delta,
    )
    replay = certificate.replay()
    if not replay.valid:
        raise DPOError(*replay.issues)
    return DPOApplication(result, certificate)


def rule_from_its(
    its: nx.Graph,
    *,
    electron_complete: bool = False,
    boundary: SystemBoundary | str = SystemBoundary.ABSTRACT,
    environment: EnvironmentToken | None = None,
    name: str = "its-rule",
) -> RuleSpan:
    """Adapt an ITS through two endpoint LLGs and an explicit carrier map."""
    left = llg_from_its(its, "reactant", electron_complete=electron_complete)
    right = llg_from_its(its, "product", electron_complete=electron_complete)
    shared = set(left.node_ids) & set(right.node_ids)
    return RuleSpan.from_mapping(
        left,
        right,
        {node: node for node in shared},
        boundary=boundary,
        environment=environment,
        name=name,
    )


def rule_from_synrule(
    rule: Any,
    *,
    boundary: SystemBoundary | str = SystemBoundary.ABSTRACT,
    environment: EnvironmentToken | None = None,
) -> RuleSpan:
    """Adapt ``SynRule`` endpoints without treating paired ITS as a rule span."""
    from synkit.Graph.ITS.its_construction import ITSConstruction
    from synkit.Rule.syn_rule import SynRule

    if not isinstance(rule, SynRule):
        raise TypeError("rule_from_synrule expects a SynRule instance.")
    its = ITSConstruction.construct(rule.left.raw, rule.right.raw)
    return rule_from_its(
        its,
        electron_complete=False,
        boundary=boundary,
        environment=environment,
        name=getattr(rule, "_name", "rule"),
    )


__all__ = [
    "DPOApplication",
    "DPOCertificate",
    "DPOError",
    "DPOIssue",
    "DPOIssueCode",
    "DPOReplay",
    "EnvironmentToken",
    "ResourceDelta",
    "RuleEmbedding",
    "RuleSpan",
    "SystemBoundary",
    "apply_dpo",
    "rule_from_its",
    "rule_from_synrule",
]
