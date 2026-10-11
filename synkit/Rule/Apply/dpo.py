"""Typed DPO rule spans and application over finite Lewis-labelled graphs.

Rule interfaces use the endpoint schema with explicitly undefined coordinates.
Contexts erase these coordinates; labels on the two squares are checked by
coordinate unions and agreement pullbacks. Matches into total hosts are exact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Any, Hashable, Mapping

import networkx as nx

from synkit.Graph.Morphism import (
    ISOTOPE_ELECTRON_LLG_SCHEMA,
    LLGError,
    LLGIssueCode,
    LLGMorphism,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
    is_electron_schema,
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
    APPLICATION_CONDITION = "DPO_APPLICATION_CONDITION"
    CLOSED_ISOTOPE = "DPO_CLOSED_ISOTOPE_IMBALANCE"


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
    isotope_delta: tuple[tuple[tuple[str, int], int], ...] = ()

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
        object.__setattr__(
            self, "isotope_delta", _normalize_isotope_delta(self.isotope_delta)
        )


@dataclass(frozen=True)
class RuleEmbedding:
    """An embedding preserving identity and every defined interface coordinate."""

    interface: LewisLabelledGraph
    endpoint: LewisLabelledGraph
    f: tuple[tuple[Hashable, Hashable], ...]

    def __post_init__(self) -> None:
        pairs = tuple(self.f.items()) if isinstance(self.f, Mapping) else tuple(self.f)
        object.__setattr__(self, "f", tuple(sorted(pairs, key=repr)))
        if (
            self.interface.schema != self.endpoint.schema
            or not self.interface.partial
            or self.endpoint.partial
        ):
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.INTERFACE_SCHEMA,
                    "The rule interface must be a partial pattern under the endpoint schema.",
                )
            )
        source_keys = tuple(left for left, _ in pairs)
        target_values = tuple(right for _, right in pairs)
        missing = set(self.interface.node_ids) - set(source_keys)
        outside = set(target_values) - set(self.endpoint.node_ids)
        if (
            len(set(source_keys)) != len(source_keys)
            or missing
            or outside
            or set(source_keys) != set(self.interface.node_ids)
        ):
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
            if not set(self.endpoint.schema.node_identity) <= set(
                interface_labels
            ) or any(
                endpoint_labels.get(key) != value
                for key, value in interface_labels.items()
            ):
                issues.append(
                    DPOIssue(
                        DPOIssueCode.INTERFACE_LABEL,
                        "An interface node omits identity or does not preserve a defined coordinate.",
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
            if not set(self.endpoint.schema.edge_identity) <= set(
                interface_labels
            ) or any(
                endpoint_labels.get(key) != value
                for key, value in interface_labels.items()
            ):
                issues.append(
                    DPOIssue(
                        DPOIssueCode.INTERFACE_LABEL,
                        "An interface edge omits identity or does not preserve a defined coordinate.",
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
    isotope_delta: tuple[tuple[tuple[str, int], int], ...] = ()


def _normalize_isotope_delta(pairs):
    pairs = tuple(pairs)
    for (element, mass), count in pairs:
        if (
            not str(element)
            or isinstance(mass, bool)
            or not isinstance(mass, Integral)
            or mass < 0
            or isinstance(count, bool)
            or not isinstance(count, Integral)
        ):
            raise ValueError(
                "Isotope resource deltas need an element, integral mass and integral count."
            )
    return tuple(
        sorted(
            ((str(element), int(mass)), int(count))
            for (element, mass), count in dict(pairs).items()
            if count
        )
    )


def _isotope_inventory(graph):
    if graph.schema != ISOTOPE_ELECTRON_LLG_SCHEMA:
        return Counter()
    values = Counter(
        (
            str(graph.node_labels(node)["element"]),
            int(graph.node_labels(node)["isotope"]),
        )
        for node in graph.node_ids
    )
    values[("H", 0)] += sum(int(graph.node_labels(n)["hcount"]) for n in graph.node_ids)
    return values


def _resource_delta(
    left: LewisLabelledGraph, right: LewisLabelledGraph
) -> ResourceDelta:
    if not is_electron_schema(left.schema) or not is_electron_schema(right.schema):
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
            hydrogen_count = float(labels["hcount"])
            if not hydrogen_count.is_integer() or hydrogen_count < 0:
                raise DPOError(
                    DPOIssue(
                        DPOIssueCode.CLOSED_MATERIAL,
                        "Unexpanded hydrogen inventory must be a nonnegative integer.",
                    )
                )
            elements["H"] += int(hydrogen_count)
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
    isotope_delta = _isotope_inventory(right)
    isotope_delta.subtract(_isotope_inventory(left))
    return ResourceDelta(
        element_delta,
        right_electrons - left_electrons,
        _normalize_isotope_delta(isotope_delta.items()),
    )


def _closed_boundary_issues(delta, environment):
    issues = []
    if delta.isotope_delta:
        issues.append(
            DPOIssue(
                DPOIssueCode.CLOSED_ISOTOPE,
                "A closed rule changes the isotope-resolved material inventory.",
                {"delta": delta.isotope_delta},
            )
        )
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
    if environment is not None:
        issues.append(
            DPOIssue(
                DPOIssueCode.OPEN_TOKEN_MISMATCH,
                "A closed rule cannot carry an environment token.",
            )
        )
    return issues


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
    condition: Any = None

    def __post_init__(self) -> None:
        if self.left.partial or self.right.partial:
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.INTERFACE_SCHEMA,
                    "Rule endpoints must have total semantic labels.",
                )
            )
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
            issues = _closed_boundary_issues(delta, self.environment)
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
            or self.environment.isotope_delta != delta.isotope_delta
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
                        "expected_isotopes": delta.isotope_delta,
                        "observed_isotopes": self.environment.isotope_delta,
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
        interface_schema = left.schema
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
            left_values = left.node_labels(left_node, semantic=True)
            right_values = right.node_labels(right_node, semantic=True)
            common = {
                key: value
                for key, value in left_values.items()
                if key in right_values and right_values[key] == value
            }
            interface_graph.add_node(interface_node, **common)

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
            left_values = left.edge_labels(edge, semantic=True)
            right_values = right.edge_labels(right_edge, semantic=True)
            common = {
                key: value
                for key, value in left_values.items()
                if key in right_values and right_values[key] == value
            }
            interface_graph.add_edge(
                left_to_interface[left_a], left_to_interface[left_b], **common
            )

        interface = LewisLabelledGraph.from_networkx(
            interface_graph, interface_schema, name=f"{name}:K", partial=True
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
        from .validation import validate_certificate

        issues = validate_certificate(self)
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
    match: Mapping[Hashable, Hashable],
) -> LewisLabelledGraph:
    graph = host.to_networkx()
    graph.remove_edges_from(tuple(tuple(edge) for edge in deletion.edges))
    graph.remove_nodes_from(deletion.nodes)
    # Naturality requires erasing precisely the coordinates undefined in K.
    for node in rule.interface.node_ids:
        image = match[rule.left_arm.mapping[node]]
        labels = rule.interface.node_labels(node, semantic=True)
        for key in host.schema.semantic_node_keys:
            if key not in labels:
                graph.nodes[image].pop(key, None)
    for edge in rule.interface.edge_keys:
        image = _map_edge(match, rule.left_arm.edge_mapping[edge])
        left, right = tuple(image)
        labels = rule.interface.edge_labels(edge, semantic=True)
        for key in host.schema.semantic_edge_keys:
            if key not in labels:
                graph.edges[left, right].pop(key, None)
    # Derived chemistry is meaningful on complete host states, not on D.
    for _, labels in graph.nodes(data=True):
        for key in host.schema.node_derived:
            labels.pop(key, None)
    for _, _, labels in graph.edges(data=True):
        for key in host.schema.edge_derived:
            labels.pop(key, None)
    return LewisLabelledGraph.from_networkx(
        graph, host.schema, name=f"{rule.name}:D", partial=True
    )


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
    if is_electron_schema(host.schema):
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
    if rule.condition is not None:
        rule.condition.validate(host, match_map)
    deletion = _deletion_plan(rule, host, match_map)
    context = _construct_context(rule, host, deletion, match_map)
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
    electron_complete: bool = False,
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
        electron_complete=electron_complete,
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
