"""Bounded exhaustive overlap search and exact composite quotienting."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx

from synkit.Graph.Morphism import LewisLabelledGraph
from synkit.Rule.Apply import RuleSpan

from .composition import (
    CompositionError,
    CompositionIssue,
    CompositionResult,
    RuleOverlap,
    compose_rules,
)


class OverlapSearchIssueCode(str, Enum):
    """Stable failures for bounded overlap enumeration and quotienting."""

    SCHEMA_MISMATCH = "OVERLAP_SEARCH_SCHEMA_MISMATCH"
    STATE_LIMIT = "OVERLAP_SEARCH_STATE_LIMIT"
    OVERLAP_LIMIT = "OVERLAP_SEARCH_OVERLAP_LIMIT"
    NODE_LIMIT = "OVERLAP_SEARCH_NODE_LIMIT"
    MATRIX_LIMIT = "OVERLAP_SEARCH_MATRIX_LIMIT"
    CANONICAL_LIMIT = "OVERLAP_SEARCH_CANONICAL_LIMIT"


@dataclass(frozen=True)
class OverlapSearchIssue:
    code: OverlapSearchIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class OverlapSearchError(RuntimeError):
    """Raised instead of returning a silently truncated candidate family."""

    def __init__(
        self,
        issue: OverlapSearchIssue,
        *,
        partial_overlaps: Iterable[RuleOverlap] = (),
    ) -> None:
        self.issue = issue
        self.partial_overlaps = tuple(partial_overlaps)
        super().__init__(issue.message)


@dataclass(frozen=True)
class OverlapSearchLimits:
    """Explicit finite resource bounds for one search."""

    max_states: int = 1_000_000
    max_overlaps: int = 100_000
    max_overlap_nodes: int | None = None
    max_component_embeddings: int = 100_000
    max_canonical_permutations: int = 1_000_000

    def __post_init__(self) -> None:
        positive = {
            "max_states": self.max_states,
            "max_overlaps": self.max_overlaps,
            "max_component_embeddings": self.max_component_embeddings,
            "max_canonical_permutations": self.max_canonical_permutations,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("Search and canonicalization limits must be positive.")
        if self.max_overlap_nodes is not None and self.max_overlap_nodes < 0:
            raise ValueError("max_overlap_nodes cannot be negative.")


@dataclass(frozen=True)
class CompositionWitness:
    """One material overlap and its accepted construction proof."""

    overlap: RuleOverlap
    overlap_digest: str
    composition: CompositionResult


@dataclass(frozen=True)
class RejectedOverlap:
    """One enumerated overlap whose composition premise failed."""

    overlap: RuleOverlap
    overlap_digest: str
    issues: tuple[CompositionIssue, ...]


@dataclass(frozen=True)
class CompositionClass:
    """One exact labelled-rule isomorphism class retaining every witness."""

    canonical_id: str
    representative: RuleSpan
    witnesses: tuple[CompositionWitness, ...]


@dataclass(frozen=True)
class CompositionSearchResult:
    """A complete bounded search result; construction ambiguity is retained."""

    overlaps: tuple[RuleOverlap, ...]
    classes: tuple[CompositionClass, ...]
    rejected: tuple[RejectedOverlap, ...]
    match_matrix: "ExtendedMatchMatrix"
    explored_states: int

    @property
    def raw_overlap_count(self) -> int:
        return len(self.overlaps)

    @property
    def accepted_count(self) -> int:
        return sum(len(group.witnesses) for group in self.classes)

    @property
    def exact_class_count(self) -> int:
        return len(self.classes)


@dataclass(frozen=True)
class ExtendedMatchMatrix:
    """Full-component embedding counts plus one explicit empty column."""

    second_components: tuple[frozenset[Hashable], ...]
    first_components: tuple[frozenset[Hashable], ...]
    counts: tuple[tuple[int, ...], ...]

    @property
    def empty_column(self) -> int:
        return len(self.first_components)


@dataclass(frozen=True)
class RuleSpanIsomorphism:
    """Replayable layerwise isomorphism between two complete rule spans."""

    source: RuleSpan
    target: RuleSpan
    left_map: tuple[tuple[Hashable, Hashable], ...]
    interface_map: tuple[tuple[Hashable, Hashable], ...]
    right_map: tuple[tuple[Hashable, Hashable], ...]

    def replay(self) -> bool:
        mapping = {
            **{("L", left): ("L", right) for left, right in self.left_map},
            **{("K", left): ("K", right) for left, right in self.interface_map},
            **{("R", left): ("R", right) for left, right in self.right_map},
        }
        return _rule_metadata_key(self.source) == _rule_metadata_key(
            self.target
        ) and _mapping_is_colored_isomorphism(
            _rule_graph(self.source), _rule_graph(self.target), mapping
        )


def _labels(graph: LewisLabelledGraph, node: Hashable) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(graph.node_labels(node, semantic=True).items()))


def _edge_labels(
    graph: LewisLabelledGraph, edge: frozenset[Hashable]
) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(graph.edge_labels(edge, semantic=True).items()))


def _node_order_key(
    graph: LewisLabelledGraph, node: Hashable
) -> tuple[str, int, tuple[str, ...], str]:
    incident = tuple(
        sorted(
            repr(_edge_labels(graph, edge))
            for edge in graph.edge_keys
            if node in edge
        )
    )
    return repr(_labels(graph, node)), len(incident), incident, repr(node)


def _component_order_key(
    graph: LewisLabelledGraph, component: frozenset[Hashable]
) -> tuple[Any, ...]:
    edges = tuple(edge for edge in graph.edge_keys if edge <= component)
    return (
        len(component),
        len(edges),
        tuple(sorted(repr(_labels(graph, node)) for node in component)),
        tuple(sorted(repr(_edge_labels(graph, edge)) for edge in edges)),
        tuple(sorted(map(repr, component))),
    )


def _component_embedding_count(
    first: LewisLabelledGraph,
    first_component: frozenset[Hashable],
    second: LewisLabelledGraph,
    second_component: frozenset[Hashable],
    limit: int,
) -> int:
    first_graph = first.to_networkx().subgraph(first_component).copy()
    second_graph = second.to_networkx().subgraph(second_component).copy()
    node_keys = first.schema.semantic_node_keys
    edge_keys = first.schema.semantic_edge_keys
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        first_graph,
        second_graph,
        node_match=lambda left, right: all(left[key] == right[key] for key in node_keys),
        edge_match=lambda left, right: all(left[key] == right[key] for key in edge_keys),
    )
    count = 0
    for _ in matcher.subgraph_monomorphisms_iter():
        count += 1
        if count > limit:
            raise OverlapSearchError(
                OverlapSearchIssue(
                    OverlapSearchIssueCode.MATRIX_LIMIT,
                    "Extended match-matrix construction exceeded its embedding bound.",
                    {"limit": limit},
                )
            )
    return count


def extended_component_match_matrix(
    first_right: LewisLabelledGraph,
    second_left: LewisLabelledGraph,
    *,
    max_embeddings: int = 100_000,
) -> ExtendedMatchMatrix:
    """Build the native extended match matrix used to order exhaustive search."""
    if first_right.schema != second_left.schema:
        raise OverlapSearchError(
            OverlapSearchIssue(
                OverlapSearchIssueCode.SCHEMA_MISMATCH,
                "Match-matrix endpoints must use the same LLG schema.",
            )
        )
    first_components = tuple(
        sorted(
            (frozenset(part) for part in nx.connected_components(first_right.to_networkx())),
            key=lambda part: _component_order_key(first_right, part),
        )
    )
    second_components = tuple(
        sorted(
            (frozenset(part) for part in nx.connected_components(second_left.to_networkx())),
            key=lambda part: _component_order_key(second_left, part),
        )
    )
    remaining = max_embeddings
    rows = []
    for second_component in second_components:
        row = []
        for first_component in first_components:
            count = _component_embedding_count(
                first_right,
                first_component,
                second_left,
                second_component,
                remaining,
            )
            remaining -= count
            row.append(count)
        rows.append(tuple(row) + (1,))
    counts = tuple(rows)
    return ExtendedMatchMatrix(second_components, first_components, counts)


def _extension_is_compatible(
    source: LewisLabelledGraph,
    target: LewisLabelledGraph,
    mapping: Mapping[Hashable, Hashable],
    source_node: Hashable,
    target_node: Hashable,
) -> bool:
    for mapped_source, mapped_target in mapping.items():
        target_edge = frozenset((target_node, mapped_target))
        if target_edge not in target.edge_keys:
            continue
        source_edge = frozenset((source_node, mapped_source))
        if source_edge not in source.edge_keys:
            return False
        if _edge_labels(source, source_edge) != _edge_labels(target, target_edge):
            return False
    return True


@dataclass
class _EnumerationState:
    explored: int = 0
    overlaps: list[RuleOverlap] = field(default_factory=list)


def enumerate_overlaps(
    first_right: LewisLabelledGraph,
    second_left: LewisLabelledGraph,
    *,
    limits: OverlapSearchLimits | None = None,
    _match_matrix: ExtendedMatchMatrix | None = None,
) -> tuple[tuple[RuleOverlap, ...], int]:
    """Enumerate every admitted partial injective overlap within hard bounds."""
    active = limits or OverlapSearchLimits()
    if first_right.schema != second_left.schema:
        raise OverlapSearchError(
            OverlapSearchIssue(
                OverlapSearchIssueCode.SCHEMA_MISMATCH,
                "Overlap endpoints must use the same LLG schema.",
            )
        )
    maximum = min(len(first_right.node_ids), len(second_left.node_ids))
    if active.max_overlap_nodes is not None and active.max_overlap_nodes < maximum:
        raise OverlapSearchError(
            OverlapSearchIssue(
                OverlapSearchIssueCode.NODE_LIMIT,
                "The node bound excludes possible overlap cardinalities.",
                {"bound": active.max_overlap_nodes, "unbounded_maximum": maximum},
            )
        )
    max_nodes = maximum
    matrix = _match_matrix or extended_component_match_matrix(
        first_right,
        second_left,
        max_embeddings=active.max_component_embeddings,
    )
    first_component = {
        node: index
        for index, component in enumerate(matrix.first_components)
        for node in component
    }
    second_component = {
        node: index
        for index, component in enumerate(matrix.second_components)
        for node in component
    }

    def target_key(source: Hashable, target: Hashable) -> tuple[Any, ...]:
        count = matrix.counts[second_component[target]][first_component[source]]
        return -count, _node_order_key(second_left, target)

    compatible_targets = {
        node: tuple(
            sorted(
                (
                    target
                    for target in second_left.node_ids
                    if _labels(first_right, node) == _labels(second_left, target)
                ),
                key=lambda target: target_key(node, target),
            )
        )
        for node in first_right.node_ids
    }
    source_nodes = sorted(
        first_right.node_ids,
        key=lambda node: (
            len(compatible_targets[node]),
            _node_order_key(first_right, node),
        ),
    )
    state = _EnumerationState()

    def visit(
        index: int,
        mapping: dict[Hashable, Hashable],
        used_targets: set[Hashable],
    ) -> None:
        state.explored += 1
        if state.explored > active.max_states:
            raise OverlapSearchError(
                OverlapSearchIssue(
                    OverlapSearchIssueCode.STATE_LIMIT,
                    "Overlap enumeration exceeded max_states.",
                    {"limit": active.max_states},
                ),
                partial_overlaps=state.overlaps,
            )
        if index == len(source_nodes):
            if len(state.overlaps) >= active.max_overlaps:
                raise OverlapSearchError(
                    OverlapSearchIssue(
                        OverlapSearchIssueCode.OVERLAP_LIMIT,
                        "Overlap enumeration exceeded max_overlaps.",
                        {"limit": active.max_overlaps},
                    ),
                    partial_overlaps=state.overlaps,
                )
            state.overlaps.append(
                RuleOverlap.from_mapping(first_right, second_left, mapping)
            )
            return
        source_node = source_nodes[index]
        visit(index + 1, mapping, used_targets)
        if len(mapping) >= max_nodes:
            return
        for target_node in compatible_targets[source_node]:
            if target_node in used_targets or not _extension_is_compatible(
                first_right,
                second_left,
                mapping,
                source_node,
                target_node,
            ):
                continue
            mapping[source_node] = target_node
            used_targets.add(target_node)
            visit(index + 1, mapping, used_targets)
            used_targets.remove(target_node)
            del mapping[source_node]

    visit(0, {}, set())
    return tuple(state.overlaps), state.explored


def _add_layer(
    target: nx.Graph,
    layer: str,
    graph: LewisLabelledGraph,
) -> None:
    for node in graph.node_ids:
        target.add_node((layer, node), color=repr((layer, _labels(graph, node))))
    for edge in graph.edge_keys:
        left, right = tuple(edge)
        target.add_edge(
            (layer, left),
            (layer, right),
            color=repr(("incidence", _edge_labels(graph, edge))),
        )


def _rule_graph(rule: RuleSpan) -> nx.Graph:
    graph = nx.Graph()
    _add_layer(graph, "L", rule.left)
    _add_layer(graph, "K", rule.interface)
    _add_layer(graph, "R", rule.right)
    for node, image in rule.left_arm.mapping.items():
        graph.add_edge(("K", node), ("L", image), color="arm:left")
    for node, image in rule.right_arm.mapping.items():
        graph.add_edge(("K", node), ("R", image), color="arm:right")
    return graph


def _overlap_graph(overlap: RuleOverlap) -> nx.Graph:
    graph = nx.Graph()
    _add_layer(graph, "R1", overlap.first_arm.target)
    _add_layer(graph, "D", overlap.interface)
    _add_layer(graph, "L2", overlap.second_arm.target)
    for node, image in overlap.first_arm.mapping.items():
        graph.add_edge(("D", node), ("R1", image), color="arm:first")
    for node, image in overlap.second_arm.mapping.items():
        graph.add_edge(("D", node), ("L2", image), color="arm:second")
    return graph


def _boundary_key(rule: RuleSpan) -> tuple[Any, ...]:
    environment = rule.environment
    return (
        rule.boundary.value,
        None
        if environment is None
        else (
            environment.name,
            environment.element_delta,
            environment.electron_delta,
        ),
    )


def _rule_metadata_key(rule: RuleSpan) -> tuple[Any, ...]:
    return repr(rule.left.schema), _boundary_key(rule)


def _mapping_is_colored_isomorphism(
    left: nx.Graph,
    right: nx.Graph,
    mapping: Mapping[Hashable, Hashable],
) -> bool:
    if set(mapping) != set(left) or set(mapping.values()) != set(right):
        return False
    if len(set(mapping.values())) != len(mapping):
        return False
    for node, image in mapping.items():
        if left.nodes[node].get("color") != right.nodes[image].get("color"):
            return False
    if left.number_of_edges() != right.number_of_edges():
        return False
    for source, target, attrs in left.edges(data=True):
        image = (mapping[source], mapping[target])
        if not right.has_edge(*image):
            return False
        if attrs.get("color") != right.edges[image].get("color"):
            return False
    return True


def find_rule_span_isomorphism(
    left: RuleSpan, right: RuleSpan
) -> RuleSpanIsomorphism | None:
    """Return an exact replayable witness, or ``None`` when none exists."""
    if _rule_metadata_key(left) != _rule_metadata_key(right):
        return None
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        _rule_graph(left),
        _rule_graph(right),
        node_match=nx.algorithms.isomorphism.categorical_node_match("color", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("color", None),
    )
    if not matcher.is_isomorphic():
        return None
    mapping = matcher.mapping
    witness = RuleSpanIsomorphism(
        left,
        right,
        tuple(
            sorted(
                (
                    (source[1], target[1])
                    for source, target in mapping.items()
                    if source[0] == "L"
                ),
                key=repr,
            )
        ),
        tuple(
            sorted(
                (
                    (source[1], target[1])
                    for source, target in mapping.items()
                    if source[0] == "K"
                ),
                key=repr,
            )
        ),
        tuple(
            sorted(
                (
                    (source[1], target[1])
                    for source, target in mapping.items()
                    if source[0] == "R"
                ),
                key=repr,
            )
        ),
    )
    return witness if witness.replay() else None


def rule_spans_isomorphic(left: RuleSpan, right: RuleSpan) -> bool:
    """Decide exact rule-span isomorphism, including arms and resource policy."""
    return find_rule_span_isomorphism(left, right) is not None


def _canonical_code(graph: nx.Graph, permutation_limit: int) -> str:
    partitions: dict[str, list[Hashable]] = {}
    for node, attrs in graph.nodes(data=True):
        partitions.setdefault(str(attrs["color"]), []).append(node)
    colors = sorted(partitions)
    permutations = math.prod(math.factorial(len(partitions[color])) for color in colors)
    if permutations > permutation_limit:
        raise OverlapSearchError(
            OverlapSearchIssue(
                OverlapSearchIssueCode.CANONICAL_LIMIT,
                "Exact canonicalization exceeded its permutation bound.",
                {"required": permutations, "limit": permutation_limit},
            )
        )
    best: str | None = None
    choices = [tuple(itertools.permutations(partitions[color])) for color in colors]
    color_sequence = tuple(
        color for color in colors for _ in range(len(partitions[color]))
    )
    for selection in itertools.product(*choices):
        order = tuple(itertools.chain.from_iterable(selection))
        edges = []
        for index, left in enumerate(order):
            for right in order[index + 1 :]:
                edges.append(
                    graph.edges[left, right]["color"]
                    if graph.has_edge(left, right)
                    else None
                )
        code = json.dumps(
            (color_sequence, edges), separators=(",", ":"), sort_keys=True
        )
        if best is None or code < best:
            best = code
    return best or json.dumps((color_sequence, ()), separators=(",", ":"))


def canonical_rule_identity(rule: RuleSpan, *, permutation_limit: int) -> str:
    """Return an exact, carrier-map-invariant digest for a rule span."""
    payload = json.dumps(
        (
            _rule_metadata_key(rule),
            _canonical_code(_rule_graph(rule), permutation_limit),
        ),
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_overlap_digest(
    overlap: RuleOverlap, *, permutation_limit: int
) -> str:
    """Return a carrier-map-invariant digest of both overlap arms."""
    code = _canonical_code(_overlap_graph(overlap), permutation_limit)
    payload = json.dumps(
        (repr(overlap.first_arm.target.schema), code), separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _rule_bucket(rule: RuleSpan) -> tuple[Any, ...]:
    graph = _rule_graph(rule)
    digest = nx.weisfeiler_lehman_graph_hash(
        graph, node_attr="color", edge_attr="color"
    )
    return _rule_metadata_key(rule), len(graph), graph.number_of_edges(), digest


def _quotient_witnesses(
    witnesses: Iterable[CompositionWitness],
    limits: OverlapSearchLimits,
) -> tuple[CompositionClass, ...]:
    buckets: dict[tuple[Any, ...], list[list[CompositionWitness]]] = {}
    for witness in witnesses:
        rule = witness.composition.rule
        groups = buckets.setdefault(_rule_bucket(rule), [])
        for group in groups:
            if rule_spans_isomorphic(rule, group[0].composition.rule):
                group.append(witness)
                break
        else:
            groups.append([witness])
    classes = []
    for groups in buckets.values():
        for group in groups:
            representative = group[0].composition.rule
            canonical_id = canonical_rule_identity(
                representative,
                permutation_limit=limits.max_canonical_permutations,
            )
            classes.append(
                CompositionClass(
                    canonical_id,
                    representative,
                    tuple(sorted(group, key=lambda item: item.overlap_digest)),
                )
            )
    return tuple(sorted(classes, key=lambda group: group.canonical_id))


def search_compositions(
    first: RuleSpan,
    second: RuleSpan,
    *,
    limits: OverlapSearchLimits | None = None,
) -> CompositionSearchResult:
    """Enumerate, construct, retain, and exactly quotient all bounded overlaps."""
    active = limits or OverlapSearchLimits()
    match_matrix = extended_component_match_matrix(
        first.right,
        second.left,
        max_embeddings=active.max_component_embeddings,
    )
    overlaps, explored = enumerate_overlaps(
        first.right,
        second.left,
        limits=active,
        _match_matrix=match_matrix,
    )
    accepted: list[CompositionWitness] = []
    rejected: list[RejectedOverlap] = []
    for overlap in overlaps:
        digest = canonical_overlap_digest(
            overlap,
            permutation_limit=active.max_canonical_permutations,
        )
        try:
            composition = compose_rules(first, second, overlap)
        except CompositionError as error:
            rejected.append(RejectedOverlap(overlap, digest, error.issues))
        else:
            accepted.append(CompositionWitness(overlap, digest, composition))
    classes = _quotient_witnesses(accepted, active)
    return CompositionSearchResult(
        overlaps,
        classes,
        tuple(sorted(rejected, key=lambda item: item.overlap_digest)),
        match_matrix,
        explored,
    )


__all__ = [
    "CompositionClass",
    "CompositionSearchResult",
    "CompositionWitness",
    "ExtendedMatchMatrix",
    "OverlapSearchError",
    "OverlapSearchIssue",
    "OverlapSearchIssueCode",
    "OverlapSearchLimits",
    "RejectedOverlap",
    "RuleSpanIsomorphism",
    "canonical_overlap_digest",
    "canonical_rule_identity",
    "enumerate_overlaps",
    "extended_component_match_matrix",
    "find_rule_span_isomorphism",
    "rule_spans_isomorphic",
    "search_compositions",
]
