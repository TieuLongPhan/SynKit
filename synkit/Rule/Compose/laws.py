"""Proof-bearing identity, reversal, and parallel-independence laws."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Mapping

import networkx as nx

from synkit.Graph.Morphism import LLGMorphism, LewisLabelledGraph
from synkit.Rule.Apply import (
    DPOApplication,
    DPOError,
    EnvironmentToken,
    RuleSpan,
    SystemBoundary,
    apply_dpo,
)


class LawIssueCode(str, Enum):
    """Stable failed premises for concurrency and algebraic laws."""

    COMPONENT_APPLICATION = "LAW_COMPONENT_APPLICATION"
    DELETE_USE_NODE = "LAW_DELETE_USE_NODE"
    DELETE_USE_EDGE = "LAW_DELETE_USE_EDGE"
    LABEL_WRITE_NODE = "LAW_LABEL_WRITE_NODE"
    LABEL_WRITE_EDGE = "LAW_LABEL_WRITE_EDGE"
    ADD_ADD_EDGE = "LAW_ADD_ADD_EDGE"
    SEQUENTIAL_APPLICATION = "LAW_SEQUENTIAL_APPLICATION"
    NONCOMMUTATIVE = "LAW_NONCOMMUTATIVE"


@dataclass(frozen=True)
class LawIssue:
    code: LawIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class LawError(ValueError):
    """Raised when a named law premise is not satisfied."""

    def __init__(self, *issues: LawIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in issues))


@dataclass(frozen=True)
class RuleAccess:
    """Host-carrier read, delete, write, and create-edge footprint."""

    read_nodes: frozenset[Hashable]
    read_edges: frozenset[frozenset[Hashable]]
    delete_nodes: frozenset[Hashable]
    delete_edges: frozenset[frozenset[Hashable]]
    write_nodes: frozenset[Hashable]
    write_edges: frozenset[frozenset[Hashable]]
    add_edges: frozenset[frozenset[Hashable]]


@dataclass(frozen=True)
class ParallelIndependenceCertificate:
    """A symmetric decision with every failed carrier premise retained."""

    first_access: RuleAccess | None
    second_access: RuleAccess | None
    first_application: DPOApplication | None
    second_application: DPOApplication | None
    issues: tuple[LawIssue, ...] = ()

    @property
    def independent(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class CommutationCertificate:
    """Two linearizations and the explicit isomorphism between endpoints."""

    first_then_second: tuple[DPOApplication, DPOApplication]
    second_then_first: tuple[DPOApplication, DPOApplication]
    result_isomorphism: LLGMorphism
    independence: ParallelIndependenceCertificate

    def replay(self) -> bool:
        applications = self.first_then_second + self.second_then_first
        return (
            self.independence.independent
            and all(item.certificate.replay().valid for item in applications)
            and self.result_isomorphism.is_isomorphism
            and self.result_isomorphism.source == self.first_then_second[1].result
            and self.result_isomorphism.target == self.second_then_first[1].result
        )


def identity_rule(
    graph: LewisLabelledGraph,
    *,
    boundary: SystemBoundary | str = SystemBoundary.ABSTRACT,
    name: str = "identity",
) -> RuleSpan:
    """Return the identity span on one LLG object."""
    return RuleSpan.from_mapping(
        graph,
        graph,
        {node: node for node in graph.node_ids},
        boundary=boundary,
        name=name,
    )


def reverse_rule(rule: RuleSpan) -> RuleSpan:
    """Reverse a rule; open resource deltas change sign."""
    environment = rule.environment
    if environment is not None:
        environment = EnvironmentToken(
            environment.name,
            tuple((element, -count) for element, count in environment.element_delta),
            -environment.electron_delta,
        )
    return RuleSpan(
        rule.right,
        rule.interface,
        rule.left,
        rule.right_arm,
        rule.left_arm,
        boundary=rule.boundary,
        environment=environment,
        name=f"reverse({rule.name})",
    )


def _map_edge(
    mapping: Mapping[Hashable, Hashable], edge: frozenset[Hashable]
) -> frozenset[Hashable]:
    return frozenset(mapping[node] for node in edge)


def _rule_access(rule: RuleSpan, match: Mapping[Hashable, Hashable]) -> RuleAccess:
    left_preserved_nodes = set(rule.left_arm.mapping.values())
    right_to_interface = {
        node: interface for interface, node in rule.right_arm.mapping.items()
    }
    left_edges = set(rule.left.edge_keys)
    preserved_left_edges = set(rule.left_arm.edge_mapping.values())
    preserved_right_edges = set(rule.right_arm.edge_mapping.values())
    write_nodes = set()
    for interface in rule.interface.node_ids:
        left_node = rule.left_arm.mapping[interface]
        right_node = rule.right_arm.mapping[interface]
        if rule.left.node_labels(left_node, semantic=True) != rule.right.node_labels(
            right_node, semantic=True
        ):
            write_nodes.add(match[left_node])
    write_edges = set()
    for interface_edge in rule.interface.edge_keys:
        left_edge = rule.left_arm.edge_mapping[interface_edge]
        right_edge = rule.right_arm.edge_mapping[interface_edge]
        if rule.left.edge_labels(left_edge, semantic=True) != rule.right.edge_labels(
            right_edge, semantic=True
        ):
            write_edges.add(_map_edge(match, left_edge))
    added_host_edges = set()
    for right_edge in set(rule.right.edge_keys) - preserved_right_edges:
        if not right_edge <= set(right_to_interface):
            continue
        left_endpoints = frozenset(
            rule.left_arm.mapping[right_to_interface[node]] for node in right_edge
        )
        added_host_edges.add(_map_edge(match, left_endpoints))
    return RuleAccess(
        frozenset(match.values()),
        frozenset(_map_edge(match, edge) for edge in left_edges),
        frozenset(
            match[node] for node in set(rule.left.node_ids) - left_preserved_nodes
        ),
        frozenset(_map_edge(match, edge) for edge in left_edges - preserved_left_edges),
        frozenset(write_nodes),
        frozenset(write_edges),
        frozenset(added_host_edges),
    )


def _intersection_issue(
    code: LawIssueCode,
    message: str,
    first: frozenset[Any],
    second: frozenset[Any],
) -> LawIssue | None:
    conflict = first & second
    if not conflict:
        return None
    return LawIssue(code, message, {"carriers": tuple(sorted(map(repr, conflict)))})


def _access_issues(first: RuleAccess, second: RuleAccess) -> tuple[LawIssue, ...]:
    checks = (
        (
            LawIssueCode.DELETE_USE_NODE,
            "The first event deletes a node used by the second.",
            first.delete_nodes,
            second.read_nodes,
        ),
        (
            LawIssueCode.DELETE_USE_NODE,
            "The second event deletes a node used by the first.",
            second.delete_nodes,
            first.read_nodes,
        ),
        (
            LawIssueCode.DELETE_USE_EDGE,
            "The first event deletes an edge used by the second.",
            first.delete_edges,
            second.read_edges,
        ),
        (
            LawIssueCode.DELETE_USE_EDGE,
            "The second event deletes an edge used by the first.",
            second.delete_edges,
            first.read_edges,
        ),
        (
            LawIssueCode.LABEL_WRITE_NODE,
            "The first event changes a node label read by the second.",
            first.write_nodes,
            second.read_nodes,
        ),
        (
            LawIssueCode.LABEL_WRITE_NODE,
            "The second event changes a node label read by the first.",
            second.write_nodes,
            first.read_nodes,
        ),
        (
            LawIssueCode.LABEL_WRITE_EDGE,
            "The first event changes an edge label read by the second.",
            first.write_edges,
            second.read_edges,
        ),
        (
            LawIssueCode.LABEL_WRITE_EDGE,
            "The second event changes an edge label read by the first.",
            second.write_edges,
            first.read_edges,
        ),
        (
            LawIssueCode.ADD_ADD_EDGE,
            "Both events add the same simple host edge.",
            first.add_edges,
            second.add_edges,
        ),
    )
    return tuple(
        issue
        for code, message, left, right in checks
        if (issue := _intersection_issue(code, message, left, right)) is not None
    )


def check_parallel_independence(
    first: RuleSpan,
    first_match: Mapping[Hashable, Hashable],
    second: RuleSpan,
    second_match: Mapping[Hashable, Hashable],
    host: LewisLabelledGraph,
) -> ParallelIndependenceCertificate:
    """Decide the sufficient DPO carrier conditions for two host events."""
    applications: list[DPOApplication | None] = []
    issues: list[LawIssue] = []
    for position, (rule, match) in enumerate(
        ((first, first_match), (second, second_match)), start=1
    ):
        try:
            applications.append(apply_dpo(rule, host, match))
        except DPOError as error:
            applications.append(None)
            issues.append(
                LawIssue(
                    LawIssueCode.COMPONENT_APPLICATION,
                    "A component event is not applicable to the common host.",
                    {
                        "position": position,
                        "issues": tuple(issue.to_dict() for issue in error.issues),
                    },
                )
            )
    if issues:
        return ParallelIndependenceCertificate(
            None, None, applications[0], applications[1], tuple(issues)
        )
    first_access = _rule_access(first, first_match)
    second_access = _rule_access(second, second_match)
    return ParallelIndependenceCertificate(
        first_access,
        second_access,
        applications[0],
        applications[1],
        _access_issues(first_access, second_access),
    )


def find_llg_isomorphism(
    source: LewisLabelledGraph, target: LewisLabelledGraph
) -> LLGMorphism | None:
    """Return a strict LLG isomorphism witness, or ``None``."""
    if source.schema != target.schema:
        return None
    node_keys = source.schema.semantic_node_keys
    edge_keys = source.schema.semantic_edge_keys
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        source.to_networkx(),
        target.to_networkx(),
        node_match=lambda left, right: all(
            left[key] == right[key] for key in node_keys
        ),
        edge_match=lambda left, right: all(
            left[key] == right[key] for key in edge_keys
        ),
    )
    if not matcher.is_isomorphic():
        return None
    morphism = LLGMorphism(source, target, matcher.mapping)
    return morphism if morphism.is_isomorphism else None


def commute_independent(
    first: RuleSpan,
    first_match: Mapping[Hashable, Hashable],
    second: RuleSpan,
    second_match: Mapping[Hashable, Hashable],
    host: LewisLabelledGraph,
) -> CommutationCertificate:
    """Apply both independent linearizations and certify equal endpoints."""
    independence = check_parallel_independence(
        first, first_match, second, second_match, host
    )
    if not independence.independent:
        raise LawError(*independence.issues)
    try:
        first_step = apply_dpo(first, host, first_match)
        first_second = apply_dpo(second, first_step.result, second_match)
        second_step = apply_dpo(second, host, second_match)
        second_first = apply_dpo(first, second_step.result, first_match)
    except DPOError as error:
        raise LawError(
            LawIssue(
                LawIssueCode.SEQUENTIAL_APPLICATION,
                "A claimed independent event failed after its peer.",
                {"issues": tuple(issue.to_dict() for issue in error.issues)},
            )
        ) from error
    isomorphism = find_llg_isomorphism(first_second.result, second_first.result)
    if isomorphism is None:
        raise LawError(
            LawIssue(
                LawIssueCode.NONCOMMUTATIVE,
                "The two independent linearizations have non-isomorphic endpoints.",
            )
        )
    certificate = CommutationCertificate(
        (first_step, first_second),
        (second_step, second_first),
        isomorphism,
        independence,
    )
    if not certificate.replay():
        raise LawError(
            LawIssue(
                LawIssueCode.NONCOMMUTATIVE,
                "The commutation certificate does not replay.",
            )
        )
    return certificate


__all__ = [
    "CommutationCertificate",
    "LawError",
    "LawIssue",
    "LawIssueCode",
    "ParallelIndependenceCertificate",
    "RuleAccess",
    "check_parallel_independence",
    "commute_independent",
    "find_llg_isomorphism",
    "identity_rule",
    "reverse_rule",
]
