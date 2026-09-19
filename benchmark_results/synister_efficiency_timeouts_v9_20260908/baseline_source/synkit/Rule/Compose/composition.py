"""Binary DPO rule composition along one explicit certified overlap."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Mapping

import networkx as nx

from synkit.Graph.Morphism import (
    ELECTRON_LLG_SCHEMA,
    LLGError,
    LLGMorphism,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
)
from synkit.Rule.Apply import (
    DPOApplication,
    DPOError,
    EnvironmentToken,
    RuleSpan,
    SystemBoundary,
    apply_dpo,
)


class CompositionIssueCode(str, Enum):
    """Stable failures for explicit binary rule composition."""

    SCHEMA_MISMATCH = "COMPOSITION_SCHEMA_MISMATCH"
    OVERLAP_PARTIAL = "COMPOSITION_OVERLAP_PARTIAL"
    OVERLAP_NON_INJECTIVE = "COMPOSITION_OVERLAP_NON_INJECTIVE"
    OVERLAP_NODE_LABEL = "COMPOSITION_OVERLAP_NODE_LABEL"
    OVERLAP_EDGE_LABEL = "COMPOSITION_OVERLAP_EDGE_LABEL"
    OVERLAP_EDGE_CLOSURE = "COMPOSITION_OVERLAP_EDGE_CLOSURE"
    PULLBACK_COMPLEMENT = "COMPOSITION_PULLBACK_COMPLEMENT"
    PARALLEL_EDGE_CONFLICT = "COMPOSITION_PARALLEL_EDGE_CONFLICT"
    FIRST_APPLICATION = "COMPOSITION_FIRST_APPLICATION"
    SECOND_APPLICATION = "COMPOSITION_SECOND_APPLICATION"
    RESOURCE_MISMATCH = "COMPOSITION_RESOURCE_MISMATCH"
    OUTER_REPLAY = "COMPOSITION_OUTER_REPLAY"
    NONCOMMUTATIVE = "COMPOSITION_NONCOMMUTATIVE"
    INCOMPLETE_PROVENANCE = "COMPOSITION_INCOMPLETE_PROVENANCE"
    SOURCE_MUTATED = "COMPOSITION_SOURCE_MUTATED"


@dataclass(frozen=True)
class CompositionIssue:
    code: CompositionIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class CompositionError(ValueError):
    """Raised when the declared overlap has no admitted composite."""

    def __init__(self, *issues: CompositionIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in issues))


@dataclass(frozen=True)
class RuleOverlap:
    """An explicit overlap span ``R1 <- D -> L2``."""

    interface: LewisLabelledGraph
    first_arm: LLGMorphism
    second_arm: LLGMorphism

    def __post_init__(self) -> None:
        if (
            self.first_arm.source != self.interface
            or self.second_arm.source != self.interface
        ):
            raise CompositionError(
                CompositionIssue(
                    CompositionIssueCode.OVERLAP_PARTIAL,
                    "Both overlap arms must start at the declared interface.",
                )
            )
        if self.first_arm.target.schema != self.second_arm.target.schema:
            raise CompositionError(
                CompositionIssue(
                    CompositionIssueCode.SCHEMA_MISMATCH,
                    "Overlap endpoints must use the same LLG schema.",
                )
            )

    @classmethod
    def from_mapping(
        cls,
        first_right: LewisLabelledGraph,
        second_left: LewisLabelledGraph,
        mapping: Mapping[Hashable, Hashable],
        *,
        overlap_edges: set[frozenset[Hashable]] | None = None,
        name: str = "overlap",
    ) -> "RuleOverlap":
        """Create an overlap; edge keys, when supplied, belong to ``R1``."""
        if first_right.schema != second_left.schema:
            raise CompositionError(
                CompositionIssue(
                    CompositionIssueCode.SCHEMA_MISMATCH,
                    "Overlap endpoints must use the same LLG schema.",
                )
            )
        pairs = tuple(sorted(mapping.items(), key=repr))
        first_nodes = tuple(left for left, _ in pairs)
        second_nodes = tuple(right for _, right in pairs)
        if len(set(first_nodes)) != len(first_nodes) or (
            set(first_nodes) - set(first_right.node_ids)
        ):
            raise CompositionError(
                CompositionIssue(
                    CompositionIssueCode.OVERLAP_PARTIAL,
                    "Every overlap source node must occur once in R1.",
                )
            )
        if len(set(second_nodes)) != len(second_nodes):
            raise CompositionError(
                CompositionIssue(
                    CompositionIssueCode.OVERLAP_NON_INJECTIVE,
                    "The overlap mapping must be injective.",
                )
            )
        if set(second_nodes) - set(second_left.node_ids):
            raise CompositionError(
                CompositionIssue(
                    CompositionIssueCode.OVERLAP_PARTIAL,
                    "Every overlap image must lie in L2.",
                )
            )

        interface_nx = nx.Graph()
        first_to_interface: dict[Hashable, int] = {}
        for interface_node, (first_node, second_node) in enumerate(pairs):
            first_labels = first_right.node_labels(first_node, semantic=True)
            second_labels = second_left.node_labels(second_node, semantic=True)
            if first_labels != second_labels:
                raise CompositionError(
                    CompositionIssue(
                        CompositionIssueCode.OVERLAP_NODE_LABEL,
                        "Mapped overlap nodes disagree on semantic labels.",
                        {"first": repr(first_node), "second": repr(second_node)},
                    )
                )
            interface_nx.add_node(interface_node, **first_right.node_labels(first_node))
            first_to_interface[first_node] = interface_node

        candidates = {
            edge
            for edge in first_right.edge_keys
            if edge <= set(first_nodes)
            and frozenset(mapping[node] for node in edge) in second_left.edge_keys
        }
        selected = candidates if overlap_edges is None else set(overlap_edges)
        if not selected <= candidates:
            raise CompositionError(
                CompositionIssue(
                    CompositionIssueCode.OVERLAP_EDGE_CLOSURE,
                    "Each declared overlap edge must exist under both arms.",
                )
            )
        for first_edge in selected:
            second_edge = frozenset(mapping[node] for node in first_edge)
            if first_right.edge_labels(
                first_edge, semantic=True
            ) != second_left.edge_labels(second_edge, semantic=True):
                raise CompositionError(
                    CompositionIssue(
                        CompositionIssueCode.OVERLAP_EDGE_LABEL,
                        "Mapped overlap edges disagree on semantic labels.",
                        {"edge": tuple(sorted(map(repr, first_edge)))},
                    )
                )
            left, right = tuple(first_edge)
            interface_nx.add_edge(
                first_to_interface[left],
                first_to_interface[right],
                **first_right.edge_labels(first_edge),
            )

        interface = LewisLabelledGraph.from_networkx(
            interface_nx, first_right.schema, name=name
        )
        first_map = {
            interface_node: first_node
            for interface_node, (first_node, _) in enumerate(pairs)
        }
        second_map = {
            interface_node: second_node
            for interface_node, (_, second_node) in enumerate(pairs)
        }
        try:
            return cls(
                interface,
                LLGMorphism(interface, first_right, first_map),
                LLGMorphism(interface, second_left, second_map),
            )
        except LLGError as error:
            raise CompositionError(
                CompositionIssue(
                    CompositionIssueCode.OVERLAP_EDGE_LABEL,
                    "The overlap arms are not strict LLG morphisms.",
                    {"issues": tuple(issue.to_dict() for issue in error.issues)},
                )
            ) from error


@dataclass(frozen=True)
class ProvenanceRef:
    rule: int
    side: str
    item: Hashable


@dataclass(frozen=True)
class CompositionProvenance:
    """Source-rule and overlap origins for every composite carrier item."""

    node_sources: tuple[tuple[str, Hashable, tuple[ProvenanceRef, ...]], ...]
    edge_sources: tuple[tuple[str, frozenset[Hashable], tuple[ProvenanceRef, ...]], ...]
    overlap_nodes: tuple[tuple[Hashable, Hashable, Hashable], ...]
    overlap_edges: tuple[
        tuple[frozenset[Hashable], frozenset[Hashable], frozenset[Hashable]], ...
    ]


@dataclass(frozen=True)
class CompositionReplay:
    valid: bool
    issues: tuple[CompositionIssue, ...] = ()


@dataclass(frozen=True)
class CompositionCertificate:
    """Replayable proof for one explicit-overlap composition."""

    first: RuleSpan
    second: RuleSpan
    overlap: RuleOverlap
    minimal_host: LewisLabelledGraph
    first_application: DPOApplication
    second_application: DPOApplication
    composite: RuleSpan
    outer_application: DPOApplication
    second_match: tuple[tuple[Hashable, Hashable], ...]
    provenance: CompositionProvenance

    def replay(self) -> CompositionReplay:
        issues: list[CompositionIssue] = []
        if not self.first_application.certificate.replay().valid:
            issues.append(
                CompositionIssue(
                    CompositionIssueCode.NONCOMMUTATIVE,
                    "The first DPO certificate no longer replays.",
                )
            )
        if not self.second_application.certificate.replay().valid:
            issues.append(
                CompositionIssue(
                    CompositionIssueCode.NONCOMMUTATIVE,
                    "The second DPO certificate no longer replays.",
                )
            )
        if not self.outer_application.certificate.replay().valid:
            issues.append(
                CompositionIssue(
                    CompositionIssueCode.OUTER_REPLAY,
                    "The outer DPO certificate no longer replays.",
                )
            )
        if not self.second_application.result.is_isomorphic(
            self.outer_application.result
        ):
            issues.append(
                CompositionIssue(
                    CompositionIssueCode.OUTER_REPLAY,
                    "Sequential and composed applications disagree.",
                )
            )
        left_node_keys = {
            (side, node)
            for side, graph in (
                ("left", self.composite.left),
                ("right", self.composite.right),
            )
            for node in graph.node_ids
        }
        observed_nodes = {
            (side, node) for side, node, refs in self.provenance.node_sources if refs
        }
        left_edge_keys = {
            (side, edge)
            for side, graph in (
                ("left", self.composite.left),
                ("right", self.composite.right),
            )
            for edge in graph.edge_keys
        }
        observed_edges = {
            (side, edge) for side, edge, refs in self.provenance.edge_sources if refs
        }
        if left_node_keys != observed_nodes or left_edge_keys != observed_edges:
            issues.append(
                CompositionIssue(
                    CompositionIssueCode.INCOMPLETE_PROVENANCE,
                    "Composite carrier provenance is incomplete.",
                )
            )
        return CompositionReplay(not issues, tuple(issues))


@dataclass(frozen=True)
class CompositionResult:
    rule: RuleSpan
    certificate: CompositionCertificate


@dataclass
class _HostPlan:
    graph: nx.Graph
    second_left_to_initial: dict[Hashable, Hashable | None]
    node_refs: dict[Hashable, set[ProvenanceRef]]
    edge_refs: dict[frozenset[Hashable], set[ProvenanceRef]]


def _fresh_context_nodes(existing: set[Hashable], count: int) -> tuple[Hashable, ...]:
    result = []
    index = 0
    while len(result) < count:
        candidate = ("__composition_context__", index)
        index += 1
        if candidate not in existing:
            result.append(candidate)
            existing.add(candidate)
    return tuple(result)


def _overlap_edge_closed(overlap: RuleOverlap) -> None:
    first = overlap.first_arm.target
    second = overlap.second_arm.target
    first_map = overlap.first_arm.mapping
    second_map = overlap.second_arm.mapping
    inverse_second = {node: source for source, node in second_map.items()}
    interface_edges = set(overlap.interface.edge_keys)
    for second_edge in second.edge_keys:
        if not second_edge <= set(inverse_second):
            continue
        interface_edge = frozenset(inverse_second[node] for node in second_edge)
        first_edge = frozenset(first_map[node] for node in interface_edge)
        if interface_edge not in interface_edges or first_edge not in first.edge_keys:
            raise CompositionError(
                CompositionIssue(
                    CompositionIssueCode.OVERLAP_EDGE_CLOSURE,
                    "The declared overlap omits an L2 edge induced by overlap nodes.",
                    {"edge": tuple(sorted(map(repr, second_edge)))},
                )
            )


def _initial_node_plan(
    first: RuleSpan,
    second: RuleSpan,
    overlap: RuleOverlap,
    graph: nx.Graph,
) -> tuple[
    dict[Hashable, Hashable | None],
    dict[Hashable, set[ProvenanceRef]],
    dict[frozenset[Hashable], set[ProvenanceRef]],
    dict[Hashable, Hashable],
]:
    node_refs = {node: {ProvenanceRef(1, "left", node)} for node in first.left.node_ids}
    edge_refs = {
        edge: {ProvenanceRef(1, "left", edge)} for edge in first.left.edge_keys
    }
    first_right_inverse = {
        node: interface for interface, node in first.right_arm.mapping.items()
    }
    inverse_second = {
        node: interface for interface, node in overlap.second_arm.mapping.items()
    }
    second_to_initial: dict[Hashable, Hashable | None] = {}
    for second_node, overlap_node in inverse_second.items():
        first_right_node = overlap.first_arm.mapping[overlap_node]
        first_interface = first_right_inverse.get(first_right_node)
        initial_node = (
            first.left_arm.mapping[first_interface]
            if first_interface is not None
            else None
        )
        second_to_initial[second_node] = initial_node
        if initial_node is not None:
            node_refs[initial_node].add(ProvenanceRef(2, "left", second_node))

    external = sorted(set(second.left.node_ids) - set(inverse_second), key=repr)
    fresh = _fresh_context_nodes(set(graph.nodes), len(external))
    for second_node, initial_node in zip(external, fresh):
        second_to_initial[second_node] = initial_node
        graph.add_node(initial_node, **second.left.node_labels(second_node))
        node_refs[initial_node] = {ProvenanceRef(2, "left", second_node)}
    return second_to_initial, node_refs, edge_refs, inverse_second


def _add_external_second_edge(
    second: RuleSpan,
    graph: nx.Graph,
    second_to_initial: Mapping[Hashable, Hashable | None],
    edge_refs: dict[frozenset[Hashable], set[ProvenanceRef]],
    second_edge: frozenset[Hashable],
) -> None:
    initial_endpoints = tuple(second_to_initial[node] for node in second_edge)
    if any(node is None for node in initial_endpoints):
        raise CompositionError(
            CompositionIssue(
                CompositionIssueCode.PULLBACK_COMPLEMENT,
                "L2 context is incident to an R1-created overlap node.",
                {"edge": tuple(sorted(map(repr, second_edge)))},
            )
        )
    initial_edge = frozenset(initial_endpoints)
    if len(initial_edge) != 2:
        raise CompositionError(
            CompositionIssue(
                CompositionIssueCode.PARALLEL_EDGE_CONFLICT,
                "The minimal host would contain a collapsed loop.",
                {"edge": tuple(sorted(map(repr, second_edge)))},
            )
        )
    left, right = tuple(initial_edge)
    if graph.has_edge(left, right):
        raise CompositionError(
            CompositionIssue(
                CompositionIssueCode.PARALLEL_EDGE_CONFLICT,
                "The minimal host would collapse two non-interface edges.",
                {"edge": tuple(sorted(map(repr, initial_edge)))},
            )
        )
    graph.add_edge(left, right, **second.left.edge_labels(second_edge))
    edge_refs[initial_edge] = {ProvenanceRef(2, "left", second_edge)}


def _add_second_context_edges(
    second: RuleSpan,
    overlap: RuleOverlap,
    graph: nx.Graph,
    second_to_initial: Mapping[Hashable, Hashable | None],
    inverse_second: Mapping[Hashable, Hashable],
    edge_refs: dict[frozenset[Hashable], set[ProvenanceRef]],
) -> None:
    interface_edges = set(overlap.interface.edge_keys)
    for second_edge in second.left.edge_keys:
        if not second_edge <= set(inverse_second):
            _add_external_second_edge(
                second, graph, second_to_initial, edge_refs, second_edge
            )
            continue
        interface_edge = frozenset(inverse_second[node] for node in second_edge)
        if interface_edge not in interface_edges:
            raise CompositionError(
                CompositionIssue(
                    CompositionIssueCode.OVERLAP_EDGE_CLOSURE,
                    "An overlap-node edge is absent from D.",
                )
            )
        initial_endpoints = tuple(second_to_initial[node] for node in second_edge)
        if all(node is not None for node in initial_endpoints):
            initial_edge = frozenset(initial_endpoints)
            if initial_edge in edge_refs:
                edge_refs[initial_edge].add(ProvenanceRef(2, "left", second_edge))


def _derive_initial_electron_state(graph: nx.Graph) -> nx.Graph:
    for _, attrs in graph.nodes(data=True):
        for key in ELECTRON_LLG_SCHEMA.node_derived:
            attrs.pop(key, None)
    for _, _, attrs in graph.edges(data=True):
        for key in ELECTRON_LLG_SCHEMA.edge_derived:
            attrs.pop(key, None)
    try:
        return derive_electron_labeled_graph(graph)
    except LLGError as error:
        raise CompositionError(
            CompositionIssue(
                CompositionIssueCode.PULLBACK_COMPLEMENT,
                "The minimal electron host is not coherent.",
                {"issues": tuple(issue.to_dict() for issue in error.issues)},
            )
        ) from error


def _minimal_host_plan(
    first: RuleSpan,
    second: RuleSpan,
    overlap: RuleOverlap,
) -> _HostPlan:
    if (
        overlap.first_arm.target != first.right
        or overlap.second_arm.target != second.left
    ):
        raise CompositionError(
            CompositionIssue(
                CompositionIssueCode.OVERLAP_PARTIAL,
                "The overlap endpoints must be exactly R1 and L2.",
            )
        )
    _overlap_edge_closed(overlap)
    graph = first.left.to_networkx()
    second_to_initial, node_refs, edge_refs, inverse_second = _initial_node_plan(
        first, second, overlap, graph
    )
    _add_second_context_edges(
        second,
        overlap,
        graph,
        second_to_initial,
        inverse_second,
        edge_refs,
    )
    if first.left.schema == ELECTRON_LLG_SCHEMA:
        graph = _derive_initial_electron_state(graph)
    return _HostPlan(graph, second_to_initial, node_refs, edge_refs)


def _first_match(first: RuleSpan, host: LewisLabelledGraph) -> dict[Hashable, Hashable]:
    return {node: node for node in first.left.node_ids}


def _second_match(
    second: RuleSpan,
    overlap: RuleOverlap,
    plan: _HostPlan,
    first_application: DPOApplication,
) -> dict[Hashable, Hashable]:
    first_right_result = dict(first_application.certificate.right_to_result)
    overlap_first = overlap.first_arm.mapping
    overlap_second_inverse = {
        node: interface for interface, node in overlap.second_arm.mapping.items()
    }
    result: dict[Hashable, Hashable] = {}
    for second_node in second.left.node_ids:
        if second_node in overlap_second_inverse:
            interface = overlap_second_inverse[second_node]
            result[second_node] = first_right_result[overlap_first[interface]]
        else:
            initial = plan.second_left_to_initial[second_node]
            if initial is None or initial not in first_application.result.node_ids:
                raise CompositionError(
                    CompositionIssue(
                        CompositionIssueCode.PULLBACK_COMPLEMENT,
                        "An L2 context item did not survive the first application.",
                        {"node": repr(second_node)},
                    )
                )
            result[second_node] = initial
    return result


def _apply_or_translate(
    rule: RuleSpan,
    host: LewisLabelledGraph,
    match: Mapping[Hashable, Hashable],
    code: CompositionIssueCode,
) -> DPOApplication:
    try:
        return apply_dpo(rule, host, match)
    except DPOError as error:
        raise CompositionError(
            CompositionIssue(
                code,
                "A component DPO application is not admissible.",
                {"issues": tuple(issue.to_dict() for issue in error.issues)},
            )
        ) from error


def _combine_boundary(
    first: RuleSpan, second: RuleSpan
) -> tuple[SystemBoundary, EnvironmentToken | None]:
    if SystemBoundary.ABSTRACT in {first.boundary, second.boundary}:
        return SystemBoundary.ABSTRACT, None
    if (
        first.boundary is SystemBoundary.CLOSED
        and second.boundary is SystemBoundary.CLOSED
    ):
        return SystemBoundary.CLOSED, None
    deltas = (first.resource_delta, second.resource_delta)
    elements: Counter[str] = Counter()
    electrons = 0.0
    names = []
    for rule, delta in zip((first, second), deltas):
        if delta is None:
            continue
        elements.update(dict(delta.element_delta))
        electrons += delta.electron_delta
        if rule.environment is not None:
            names.append(rule.environment.name)
    token = EnvironmentToken(
        " + ".join(names) or "composed environment",
        tuple((element, count) for element, count in elements.items() if count),
        electrons,
    )
    return SystemBoundary.OPEN, token


def _outer_rule(
    first: RuleSpan,
    second: RuleSpan,
    host: LewisLabelledGraph,
    first_application: DPOApplication,
    second_application: DPOApplication,
) -> RuleSpan:
    removed_first = set(first_application.certificate.deleted_nodes)
    removed_second = set(second_application.certificate.deleted_nodes)
    preserved = {
        node: node
        for node in host.node_ids
        if node not in removed_first
        and node not in removed_second
        and node in second_application.result.node_ids
    }
    boundary, environment = _combine_boundary(first, second)
    try:
        return RuleSpan.from_mapping(
            host,
            second_application.result,
            preserved,
            boundary=boundary,
            environment=environment,
            name=f"{first.name};{second.name}",
        )
    except DPOError as error:
        raise CompositionError(
            CompositionIssue(
                CompositionIssueCode.RESOURCE_MISMATCH,
                "The outer rule violates the composed resource contract.",
                {"issues": tuple(issue.to_dict() for issue in error.issues)},
            )
        ) from error


def _mapped_edge(
    mapping: Mapping[Hashable, Hashable], edge: frozenset[Hashable]
) -> frozenset[Hashable]:
    return frozenset(mapping[node] for node in edge)


def _provenance(
    first: RuleSpan,
    second: RuleSpan,
    overlap: RuleOverlap,
    host: LewisLabelledGraph,
    plan: _HostPlan,
    first_application: DPOApplication,
    second_application: DPOApplication,
) -> CompositionProvenance:
    final = second_application.result
    left_nodes = {node: set(refs) for node, refs in plan.node_refs.items()}
    left_edges = {edge: set(refs) for edge, refs in plan.edge_refs.items()}
    final_nodes: dict[Hashable, set[ProvenanceRef]] = {
        node: set(left_nodes.get(node, ())) for node in final.node_ids
    }
    final_edges: dict[frozenset[Hashable], set[ProvenanceRef]] = {
        edge: set(left_edges.get(edge, ())) for edge in final.edge_keys
    }

    first_right = dict(first_application.certificate.right_to_result)
    removed_second = set(second_application.certificate.deleted_nodes)
    for node in first.right.node_ids:
        result_node = first_right[node]
        if result_node in final.node_ids and result_node not in removed_second:
            final_nodes[result_node].add(ProvenanceRef(1, "right", node))
    for edge in first.right.edge_keys:
        result_edge = _mapped_edge(first_right, edge)
        if result_edge in final.edge_keys:
            final_edges[result_edge].add(ProvenanceRef(1, "right", edge))

    second_right = dict(second_application.certificate.right_to_result)
    for node in second.right.node_ids:
        final_nodes[second_right[node]].add(ProvenanceRef(2, "right", node))
    for edge in second.right.edge_keys:
        result_edge = _mapped_edge(second_right, edge)
        final_edges[result_edge].add(ProvenanceRef(2, "right", edge))

    node_sources = tuple(
        sorted(
            [
                (side, node, tuple(sorted(refs, key=repr)))
                for side, values in (("left", left_nodes), ("right", final_nodes))
                for node, refs in values.items()
            ],
            key=repr,
        )
    )
    edge_sources = tuple(
        sorted(
            [
                (side, edge, tuple(sorted(refs, key=repr)))
                for side, values in (("left", left_edges), ("right", final_edges))
                for edge, refs in values.items()
            ],
            key=repr,
        )
    )
    overlap_nodes = tuple(
        sorted(
            (
                (
                    node,
                    overlap.first_arm.mapping[node],
                    overlap.second_arm.mapping[node],
                )
                for node in overlap.interface.node_ids
            ),
            key=repr,
        )
    )
    overlap_edges = tuple(
        sorted(
            (
                (
                    edge,
                    _mapped_edge(overlap.first_arm.mapping, edge),
                    _mapped_edge(overlap.second_arm.mapping, edge),
                )
                for edge in overlap.interface.edge_keys
            ),
            key=repr,
        )
    )
    return CompositionProvenance(
        node_sources, edge_sources, overlap_nodes, overlap_edges
    )


def compose_rules(
    first: RuleSpan,
    second: RuleSpan,
    overlap: RuleOverlap,
) -> CompositionResult:
    """Compose exactly two rules along exactly one caller-supplied overlap."""
    if first.right.schema != second.left.schema:
        raise CompositionError(
            CompositionIssue(
                CompositionIssueCode.SCHEMA_MISMATCH,
                "Consecutive rule endpoints must use one LLG schema.",
            )
        )
    sources_before = (
        first.left.to_networkx(),
        first.right.to_networkx(),
        second.left.to_networkx(),
        second.right.to_networkx(),
    )
    plan = _minimal_host_plan(first, second, overlap)
    try:
        host = LewisLabelledGraph.from_networkx(
            plan.graph, first.left.schema, name=f"{first.name};{second.name}:G"
        )
    except LLGError as error:
        raise CompositionError(
            CompositionIssue(
                CompositionIssueCode.PULLBACK_COMPLEMENT,
                "The minimal initial context is not an LLG object.",
                {"issues": tuple(issue.to_dict() for issue in error.issues)},
            )
        ) from error
    first_application = _apply_or_translate(
        first, host, _first_match(first, host), CompositionIssueCode.FIRST_APPLICATION
    )
    second_match = _second_match(second, overlap, plan, first_application)
    second_application = _apply_or_translate(
        second,
        first_application.result,
        second_match,
        CompositionIssueCode.SECOND_APPLICATION,
    )
    composite = _outer_rule(first, second, host, first_application, second_application)
    outer_application = _apply_or_translate(
        composite,
        host,
        {node: node for node in host.node_ids},
        CompositionIssueCode.OUTER_REPLAY,
    )
    provenance = _provenance(
        first,
        second,
        overlap,
        host,
        plan,
        first_application,
        second_application,
    )
    certificate = CompositionCertificate(
        first,
        second,
        overlap,
        host,
        first_application,
        second_application,
        composite,
        outer_application,
        tuple(sorted(second_match.items(), key=repr)),
        provenance,
    )
    replay = certificate.replay()
    if not replay.valid:
        raise CompositionError(*replay.issues)
    sources_after = (
        first.left.to_networkx(),
        first.right.to_networkx(),
        second.left.to_networkx(),
        second.right.to_networkx(),
    )
    if any(
        not nx.utils.graphs_equal(before, after)
        for before, after in zip(sources_before, sources_after)
    ):
        raise CompositionError(
            CompositionIssue(
                CompositionIssueCode.SOURCE_MUTATED,
                "Composition mutated a source rule.",
            )
        )
    return CompositionResult(composite, certificate)


__all__ = [
    "CompositionCertificate",
    "CompositionError",
    "CompositionIssue",
    "CompositionIssueCode",
    "CompositionProvenance",
    "CompositionReplay",
    "CompositionResult",
    "ProvenanceRef",
    "RuleOverlap",
    "compose_rules",
]
