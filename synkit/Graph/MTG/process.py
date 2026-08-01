"""Finite occurrence processes with explicit material flow and concurrency."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx

from synkit.Graph.Morphism import LLGError, LLGMorphism, LewisLabelledGraph
from synkit.Rule.Apply import RuleSpan
from synkit.Rule.Compose import (
    CommutationCertificate,
    LawIssue,
    rule_spans_isomorphic,
)


class ProcessIssueCode(str, Enum):
    """Stable failures for occurrence values, flows, orders, and choices."""

    DUPLICATE_ID = "PROCESS_DUPLICATE_ID"
    UNKNOWN_MATERIAL = "PROCESS_UNKNOWN_MATERIAL"
    UNKNOWN_EVENT = "PROCESS_UNKNOWN_EVENT"
    ENDPOINT_PARTITION = "PROCESS_ENDPOINT_PARTITION"
    BINDING_NOT_ISOMORPHIC = "PROCESS_BINDING_NOT_ISOMORPHIC"
    MATERIAL_REUSED = "PROCESS_MATERIAL_REUSED"
    CAUSAL_CYCLE = "PROCESS_CAUSAL_CYCLE"
    MISSING_INDEPENDENCE = "PROCESS_MISSING_INDEPENDENCE"
    INVALID_INDEPENDENCE = "PROCESS_INVALID_INDEPENDENCE"
    EXTRA_INDEPENDENCE = "PROCESS_EXTRA_INDEPENDENCE"
    INVALID_EXTENSION = "PROCESS_INVALID_LINEAR_EXTENSION"
    EXTENSION_LIMIT = "PROCESS_LINEAR_EXTENSION_LIMIT"
    SERIES_PARALLEL_LIMIT = "PROCESS_SERIES_PARALLEL_LIMIT"
    INVALID_ALTERNATIVE = "PROCESS_INVALID_ALTERNATIVE"


@dataclass(frozen=True)
class ProcessIssue:
    code: ProcessIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class ProcessError(ValueError):
    """Raised when occurrence-process data do not prove their semantics."""

    def __init__(self, *issues: ProcessIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in issues))


@dataclass(frozen=True)
class MaterialOccurrence:
    """One material copy; isomorphic values do not identify copy IDs."""

    occurrence_id: str
    value: LewisLabelledGraph

    def __post_init__(self) -> None:
        if not self.occurrence_id:
            raise ValueError("A material occurrence requires a non-empty ID.")


@dataclass(frozen=True)
class MaterialBinding:
    """An endpoint component isomorphism to one material occurrence."""

    material_id: str
    endpoint_nodes: frozenset[Hashable]
    mapping: tuple[tuple[Hashable, Hashable], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint_nodes", frozenset(self.endpoint_nodes))
        pairs = self.mapping.items() if isinstance(self.mapping, Mapping) else self.mapping
        object.__setattr__(self, "mapping", tuple(sorted(pairs, key=repr)))


@dataclass(frozen=True)
class RuleOccurrence:
    """One event identity, independently of equality of its rule value."""

    occurrence_id: str
    rule: RuleSpan
    inputs: tuple[MaterialBinding, ...] = ()
    outputs: tuple[MaterialBinding, ...] = ()

    def __post_init__(self) -> None:
        if not self.occurrence_id:
            raise ValueError("A rule occurrence requires a non-empty ID.")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))


@dataclass(frozen=True)
class IndependenceWitness:
    """One incomparable event pair and its executable commutation proof."""

    first_event: str
    second_event: str
    certificate: CommutationCertificate

    @property
    def key(self) -> frozenset[str]:
        return frozenset((self.first_event, self.second_event))


@dataclass(frozen=True)
class LinearExtensionEquivalence:
    """A path of adjacent certified-independent swaps between extensions."""

    source: tuple[str, ...]
    target: tuple[str, ...]
    swaps: tuple[tuple[str, str], ...]

    def replay(self, process: "OccurrenceProcess") -> bool:
        if not process.is_linear_extension(self.source) or not process.is_linear_extension(
            self.target
        ):
            return False
        current = list(self.source)
        witnesses = process.independence_by_pair
        for left, right in self.swaps:
            try:
                index = next(
                    position
                    for position in range(len(current) - 1)
                    if current[position : position + 2] == [left, right]
                )
            except StopIteration:
                return False
            witness = witnesses.get(frozenset((left, right)))
            if witness is None or not witness.certificate.replay():
                return False
            current[index], current[index + 1] = right, left
        return tuple(current) == self.target


def _component_graph(
    endpoint: LewisLabelledGraph, nodes: frozenset[Hashable]
) -> LewisLabelledGraph:
    graph = endpoint.to_networkx().subgraph(nodes).copy()
    return LewisLabelledGraph.from_networkx(
        graph, endpoint.schema, name=(endpoint.name, "component")
    )


def _components(endpoint: LewisLabelledGraph) -> frozenset[frozenset[Hashable]]:
    return frozenset(
        frozenset(component)
        for component in nx.connected_components(endpoint.to_networkx())
    )


@dataclass(frozen=True)
class OccurrenceProcess:
    """A finite causal configuration with explicit material-copy flow."""

    process_id: str
    materials: tuple[MaterialOccurrence, ...]
    events: tuple[RuleOccurrence, ...]
    independence: tuple[IndependenceWitness, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "materials", tuple(self.materials))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "independence", tuple(self.independence))
        issues = self._validation_issues()
        if issues:
            raise ProcessError(*issues)

    @property
    def material_by_id(self) -> dict[str, MaterialOccurrence]:
        return {item.occurrence_id: item for item in self.materials}

    @property
    def event_by_id(self) -> dict[str, RuleOccurrence]:
        return {item.occurrence_id: item for item in self.events}

    @property
    def independence_by_pair(self) -> dict[frozenset[str], IndependenceWitness]:
        return {item.key: item for item in self.independence}

    def _binding_issues(
        self,
        event: RuleOccurrence,
        bindings: tuple[MaterialBinding, ...],
        endpoint: LewisLabelledGraph,
        side: str,
    ) -> list[ProcessIssue]:
        issues: list[ProcessIssue] = []
        materials = self.material_by_id
        expected = _components(endpoint)
        observed = frozenset(binding.endpoint_nodes for binding in bindings)
        if observed != expected or len(observed) != len(bindings):
            issues.append(
                ProcessIssue(
                    ProcessIssueCode.ENDPOINT_PARTITION,
                    "Bindings must partition an endpoint into connected components.",
                    {"event": event.occurrence_id, "side": side},
                )
            )
        material_ids = tuple(binding.material_id for binding in bindings)
        if len(set(material_ids)) != len(material_ids):
            issues.append(
                ProcessIssue(
                    ProcessIssueCode.MATERIAL_REUSED,
                    "One material copy cannot fill two endpoint components.",
                    {"event": event.occurrence_id, "side": side},
                )
            )
        for binding in bindings:
            material = materials.get(binding.material_id)
            if material is None:
                issues.append(
                    ProcessIssue(
                        ProcessIssueCode.UNKNOWN_MATERIAL,
                        "A binding names an unknown material occurrence.",
                        {"event": event.occurrence_id, "material": binding.material_id},
                    )
                )
                continue
            if binding.endpoint_nodes not in expected:
                continue
            component = _component_graph(endpoint, binding.endpoint_nodes)
            try:
                morphism = LLGMorphism(component, material.value, binding.mapping)
            except LLGError as error:
                issues.append(
                    ProcessIssue(
                        ProcessIssueCode.BINDING_NOT_ISOMORPHIC,
                        "A material binding is not a strict component isomorphism.",
                        {
                            "event": event.occurrence_id,
                            "material": binding.material_id,
                            "issues": tuple(
                                issue.to_dict() for issue in error.issues
                            ),
                        },
                    )
                )
            else:
                if not morphism.is_isomorphism:
                    issues.append(
                        ProcessIssue(
                            ProcessIssueCode.BINDING_NOT_ISOMORPHIC,
                            "A binding must cover the whole material occurrence.",
                            {
                                "event": event.occurrence_id,
                                "material": binding.material_id,
                            },
                        )
                    )
        return issues

    def _flow_graph(self) -> tuple[nx.DiGraph, list[ProcessIssue]]:
        graph = nx.DiGraph()
        graph.add_nodes_from(event.occurrence_id for event in self.events)
        producers: dict[str, str] = {}
        consumers: dict[str, str] = {}
        issues: list[ProcessIssue] = []
        for event in self.events:
            for binding in event.outputs:
                if binding.material_id in producers:
                    issues.append(
                        ProcessIssue(
                            ProcessIssueCode.MATERIAL_REUSED,
                            "A material occurrence has more than one producer.",
                            {"material": binding.material_id},
                        )
                    )
                producers[binding.material_id] = event.occurrence_id
            for binding in event.inputs:
                if binding.material_id in consumers:
                    issues.append(
                        ProcessIssue(
                            ProcessIssueCode.MATERIAL_REUSED,
                            "A material occurrence has more than one consumer.",
                            {"material": binding.material_id},
                        )
                    )
                consumers[binding.material_id] = event.occurrence_id
        for material in set(producers) & set(consumers):
            graph.add_edge(producers[material], consumers[material], material=material)
        if not nx.is_directed_acyclic_graph(graph):
            issues.append(
                ProcessIssue(
                    ProcessIssueCode.CAUSAL_CYCLE,
                    "Occurrence causality must be a finite DAG; unfold repeated cycles.",
                )
            )
        return graph, issues

    def _independence_issues(self, graph: nx.DiGraph) -> list[ProcessIssue]:
        if not nx.is_directed_acyclic_graph(graph):
            return []
        closure = nx.transitive_closure_dag(graph)
        event_ids = set(graph)
        incomparable = {
            frozenset((left, right))
            for left in event_ids
            for right in event_ids
            if repr(left) < repr(right)
            and not closure.has_edge(left, right)
            and not closure.has_edge(right, left)
        }
        witnesses = self.independence_by_pair
        issues: list[ProcessIssue] = []
        missing = incomparable - set(witnesses)
        extra = set(witnesses) - incomparable
        if missing:
            issues.append(
                ProcessIssue(
                    ProcessIssueCode.MISSING_INDEPENDENCE,
                    "Every incomparable event pair needs a commutation certificate.",
                    {"pairs": tuple(sorted(map(repr, missing)))},
                )
            )
        if extra:
            issues.append(
                ProcessIssue(
                    ProcessIssueCode.EXTRA_INDEPENDENCE,
                    "A causal pair cannot also be declared independent.",
                    {"pairs": tuple(sorted(map(repr, extra)))},
                )
            )
        events = self.event_by_id
        for pair, witness in witnesses.items():
            if len(pair) != 2 or not pair <= event_ids:
                issues.append(
                    ProcessIssue(
                        ProcessIssueCode.UNKNOWN_EVENT,
                        "An independence witness names unknown or identical events.",
                        {"pair": repr(pair)},
                    )
                )
                continue
            first = events[witness.first_event]
            second = events[witness.second_event]
            certificate = witness.certificate
            first_rule = certificate.first_then_second[0].certificate.rule
            second_rule = certificate.second_then_first[0].certificate.rule
            if (
                not certificate.replay()
                or not rule_spans_isomorphic(first.rule, first_rule)
                or not rule_spans_isomorphic(second.rule, second_rule)
            ):
                issues.append(
                    ProcessIssue(
                        ProcessIssueCode.INVALID_INDEPENDENCE,
                        "The independence certificate does not prove its event pair.",
                        {"pair": tuple(sorted(pair))},
                    )
                )
        return issues

    def _validation_issues(self) -> list[ProcessIssue]:
        issues: list[ProcessIssue] = []
        material_ids = [item.occurrence_id for item in self.materials]
        event_ids = [item.occurrence_id for item in self.events]
        if len(set(material_ids)) != len(material_ids) or len(set(event_ids)) != len(
            event_ids
        ):
            issues.append(
                ProcessIssue(
                    ProcessIssueCode.DUPLICATE_ID,
                    "Material and event IDs must be unique within their namespaces.",
                )
            )
            return issues
        for event in self.events:
            issues.extend(
                self._binding_issues(event, event.inputs, event.rule.left, "input")
            )
            issues.extend(
                self._binding_issues(event, event.outputs, event.rule.right, "output")
            )
        graph, flow_issues = self._flow_graph()
        issues.extend(flow_issues)
        issues.extend(self._independence_issues(graph))
        return issues

    @property
    def causal_graph(self) -> nx.DiGraph:
        graph, issues = self._flow_graph()
        if issues:
            raise ProcessError(*issues)
        return graph

    @property
    def causal_pairs(self) -> tuple[tuple[str, str], ...]:
        closure = nx.transitive_closure_dag(self.causal_graph)
        return tuple(sorted(closure.edges, key=repr))

    @property
    def cover_pairs(self) -> tuple[tuple[str, str], ...]:
        reduction = nx.transitive_reduction(self.causal_graph)
        return tuple(sorted(reduction.edges, key=repr))

    @property
    def incomparable_pairs(self) -> tuple[frozenset[str], ...]:
        causal = set(self.causal_pairs)
        result = []
        event_ids = sorted(self.event_by_id, key=repr)
        for index, left in enumerate(event_ids):
            for right in event_ids[index + 1 :]:
                if (left, right) not in causal and (right, left) not in causal:
                    result.append(frozenset((left, right)))
        return tuple(result)

    def linear_extensions(self, *, max_extensions: int = 10_000) -> tuple[tuple[str, ...], ...]:
        """Enumerate all extensions or raise instead of returning a prefix."""
        if max_extensions <= 0:
            raise ValueError("max_extensions must be positive.")
        result = []
        for extension in nx.all_topological_sorts(self.causal_graph):
            if len(result) >= max_extensions:
                raise ProcessError(
                    ProcessIssue(
                        ProcessIssueCode.EXTENSION_LIMIT,
                        "Linear-extension enumeration exceeded its explicit bound.",
                        {"limit": max_extensions},
                    )
                )
            result.append(tuple(extension))
        return tuple(sorted(result, key=repr))

    def is_linear_extension(self, extension: Iterable[str]) -> bool:
        order = tuple(extension)
        if set(order) != set(self.event_by_id) or len(order) != len(self.events):
            return False
        position = {event: index for index, event in enumerate(order)}
        return all(position[left] < position[right] for left, right in self.causal_pairs)

    def extension_equivalence(
        self, source: Iterable[str], target: Iterable[str]
    ) -> LinearExtensionEquivalence:
        """Construct adjacent-independent swaps between two valid extensions."""
        source_order = tuple(source)
        target_order = tuple(target)
        if not self.is_linear_extension(source_order) or not self.is_linear_extension(
            target_order
        ):
            raise ProcessError(
                ProcessIssue(
                    ProcessIssueCode.INVALID_EXTENSION,
                    "Both orders must be linear extensions of this process.",
                )
            )
        current = list(source_order)
        swaps = []
        for target_index, desired in enumerate(target_order):
            position = current.index(desired)
            while position > target_index:
                left, right = current[position - 1], current[position]
                if frozenset((left, right)) not in self.independence_by_pair:
                    raise ProcessError(
                        ProcessIssue(
                            ProcessIssueCode.INVALID_EXTENSION,
                            "A required adjacent swap lacks independence evidence.",
                            {"pair": (left, right)},
                        )
                    )
                current[position - 1], current[position] = right, left
                swaps.append((left, right))
                position -= 1
        witness = LinearExtensionEquivalence(
            source_order, target_order, tuple(swaps)
        )
        if not witness.replay(self):
            raise ProcessError(
                ProcessIssue(
                    ProcessIssueCode.INVALID_EXTENSION,
                    "The adjacent-swap equivalence does not replay.",
                )
            )
        return witness


@dataclass(frozen=True)
class ProcessAlternative:
    alternative_id: str
    process: OccurrenceProcess


@dataclass(frozen=True)
class ChoiceWitness:
    """Why two processes remain alternatives rather than one quotient class."""

    first_alternative: str
    second_alternative: str
    kind: str
    reason: str
    conflicts: tuple[LawIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"conflict", "material_assignment"}:
            raise ValueError("Choice kind must be conflict or material_assignment.")
        if self.kind == "conflict" and not self.conflicts:
            raise ValueError("A conflict choice requires failed law premises.")


@dataclass(frozen=True)
class OccurrenceProcessFamily:
    """A retained family of conflicting or materially ambiguous processes."""

    alternatives: tuple[ProcessAlternative, ...]
    choices: tuple[ChoiceWitness, ...]

    def __post_init__(self) -> None:
        alternatives = tuple(self.alternatives)
        choices = tuple(self.choices)
        object.__setattr__(self, "alternatives", alternatives)
        object.__setattr__(self, "choices", choices)
        identifiers = tuple(item.alternative_id for item in alternatives)
        issues = []
        if len(set(identifiers)) != len(identifiers):
            issues.append(
                ProcessIssue(
                    ProcessIssueCode.DUPLICATE_ID,
                    "Process alternative IDs must be unique.",
                )
            )
        known = set(identifiers)
        for choice in choices:
            pair = {choice.first_alternative, choice.second_alternative}
            if len(pair) != 2 or not pair <= known:
                issues.append(
                    ProcessIssue(
                        ProcessIssueCode.INVALID_ALTERNATIVE,
                        "A choice must connect two known distinct alternatives.",
                        {"pair": tuple(sorted(pair))},
                    )
                )
        if issues:
            raise ProcessError(*issues)


__all__ = [
    "ChoiceWitness",
    "IndependenceWitness",
    "LinearExtensionEquivalence",
    "MaterialBinding",
    "MaterialOccurrence",
    "OccurrenceProcess",
    "OccurrenceProcessFamily",
    "ProcessAlternative",
    "ProcessError",
    "ProcessIssue",
    "ProcessIssueCode",
    "RuleOccurrence",
]
