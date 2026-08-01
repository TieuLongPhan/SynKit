"""Occurrence-aware MTG histories with lossless process reconstruction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx

from synkit.Graph.Morphism import LewisLabelledGraph
from synkit.Rule.Apply import DPOError, RuleSpan

from .process import (
    ChoiceWitness,
    IndependenceWitness,
    MaterialOccurrence,
    OccurrenceProcess,
    OccurrenceProcessFamily,
    ProcessAlternative,
    ProcessError,
    RuleOccurrence,
)

Carrier = tuple[str, Hashable]
EdgeCarrier = tuple[str, frozenset[Hashable]]


class HistoryIssueCode(str, Enum):
    """Stable failures for occurrence-aware history derivation and replay."""

    EMPTY = "HISTORY_EMPTY"
    SCHEMA_MISMATCH = "HISTORY_SCHEMA_MISMATCH"
    FLOW_LOOKUP = "HISTORY_FLOW_LOOKUP"
    LINEAGE_CONFLICT = "HISTORY_LINEAGE_CONFLICT"
    INVALID_EXTENSION = "HISTORY_INVALID_EXTENSION"
    INACTIVE_INPUT = "HISTORY_INACTIVE_INPUT"
    ACTIVE_OUTPUT = "HISTORY_ACTIVE_OUTPUT"
    OUTER_RULE = "HISTORY_OUTER_RULE"
    ROUNDTRIP = "HISTORY_ROUNDTRIP"
    ALTERNATIVE_MISMATCH = "HISTORY_ALTERNATIVE_MISMATCH"


@dataclass(frozen=True)
class HistoryIssue:
    code: HistoryIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)


class HistoryError(ValueError):
    """Raised when a process cannot yield a lossless history object."""

    def __init__(self, *issues: HistoryIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in issues))


@dataclass(frozen=True)
class CarrierState:
    """One lineage state at a process cut; absence is explicit."""

    present: bool
    material_id: str | None = None
    carrier: Hashable | frozenset[Hashable] | None = None
    labels: tuple[tuple[str, Any], ...] = ()


ABSENT = CarrierState(False)


@dataclass(frozen=True)
class NodeLineage:
    lineage_id: str
    members: tuple[Carrier, ...]


@dataclass(frozen=True)
class EdgeLineage:
    lineage_id: str
    endpoints: frozenset[str]
    members: tuple[EdgeCarrier, ...]


@dataclass(frozen=True)
class HistoryReplay:
    valid: bool
    issues: tuple[HistoryIssue, ...] = ()


class _UnionFind:
    def __init__(self, values: Iterable[Hashable]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: Hashable) -> Hashable:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: Hashable, right: Hashable) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if repr(left_root) <= repr(right_root):
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root

    def groups(self) -> tuple[tuple[Hashable, ...], ...]:
        grouped: dict[Hashable, list[Hashable]] = {}
        for value in self.parent:
            grouped.setdefault(self.find(value), []).append(value)
        return tuple(
            sorted(
                (tuple(sorted(group, key=repr)) for group in grouped.values()),
                key=repr,
            )
        )


def _binding_carrier(
    bindings: tuple[Any, ...], endpoint_node: Hashable
) -> Carrier:
    for binding in bindings:
        mapping = dict(binding.mapping)
        if endpoint_node in mapping:
            return binding.material_id, mapping[endpoint_node]
    raise HistoryError(
        HistoryIssue(
            HistoryIssueCode.FLOW_LOOKUP,
            "No material binding covers a rule endpoint carrier.",
            {"node": repr(endpoint_node)},
        )
    )


def _binding_edge(
    bindings: tuple[Any, ...], endpoint_edge: frozenset[Hashable]
) -> EdgeCarrier:
    carriers = tuple(_binding_carrier(bindings, node) for node in endpoint_edge)
    material_ids = {material for material, _ in carriers}
    if len(material_ids) != 1:
        raise HistoryError(
            HistoryIssue(
                HistoryIssueCode.FLOW_LOOKUP,
                "One endpoint edge must belong to one material component.",
            )
        )
    return next(iter(material_ids)), frozenset(node for _, node in carriers)


def _lineages(
    materials: tuple[MaterialOccurrence, ...], events: tuple[RuleOccurrence, ...]
) -> tuple[tuple[NodeLineage, ...], tuple[EdgeLineage, ...]]:
    node_values = tuple(
        (material.occurrence_id, node)
        for material in materials
        for node in material.value.node_ids
    )
    edge_values = tuple(
        (material.occurrence_id, edge)
        for material in materials
        for edge in material.value.edge_keys
    )
    node_union = _UnionFind(node_values)
    edge_union = _UnionFind(edge_values)
    for event in events:
        for interface in event.rule.interface.node_ids:
            left_node = event.rule.left_arm.mapping[interface]
            right_node = event.rule.right_arm.mapping[interface]
            node_union.union(
                _binding_carrier(event.inputs, left_node),
                _binding_carrier(event.outputs, right_node),
            )
        for interface_edge in event.rule.interface.edge_keys:
            left_edge = event.rule.left_arm.edge_mapping[interface_edge]
            right_edge = event.rule.right_arm.edge_mapping[interface_edge]
            edge_union.union(
                _binding_edge(event.inputs, left_edge),
                _binding_edge(event.outputs, right_edge),
            )
    node_lineages = tuple(
        NodeLineage(f"node:{index}", group)  # type: ignore[arg-type]
        for index, group in enumerate(node_union.groups())
    )
    node_by_carrier = {
        carrier: lineage.lineage_id
        for lineage in node_lineages
        for carrier in lineage.members
    }
    edge_lineages = []
    material_by_id = {material.occurrence_id: material for material in materials}
    for index, group in enumerate(edge_union.groups()):
        endpoints: frozenset[str] | None = None
        for material_id, edge in group:
            candidate = frozenset(
                node_by_carrier[(material_id, node)] for node in edge
            )
            if endpoints is None:
                endpoints = candidate
            elif endpoints != candidate:
                raise HistoryError(
                    HistoryIssue(
                        HistoryIssueCode.LINEAGE_CONFLICT,
                        "A preserved edge lineage changes node lineage endpoints.",
                        {"material": material_id},
                    )
                )
            if edge not in material_by_id[material_id].value.edge_keys:
                raise HistoryError(
                    HistoryIssue(
                        HistoryIssueCode.FLOW_LOOKUP,
                        "An edge lineage names an absent material edge.",
                    )
                )
        edge_lineages.append(
            EdgeLineage(
                f"edge:{index}",
                endpoints or frozenset(),
                group,  # type: ignore[arg-type]
            )
        )
    return node_lineages, tuple(edge_lineages)


@dataclass(frozen=True)
class OccurrenceMTG:
    """A lossless event/material history plus compact carrier lineages."""

    process_id: str
    materials: tuple[MaterialOccurrence, ...]
    events: tuple[RuleOccurrence, ...]
    independence: tuple[IndependenceWitness, ...]
    causal_pairs: tuple[tuple[str, str], ...]
    node_lineages: tuple[NodeLineage, ...]
    edge_lineages: tuple[EdgeLineage, ...]

    @classmethod
    def from_process(cls, process: OccurrenceProcess) -> "OccurrenceMTG":
        if not process.materials:
            raise HistoryError(
                HistoryIssue(
                    HistoryIssueCode.EMPTY,
                    "An occurrence MTG requires at least one material occurrence.",
                )
            )
        schemas = {material.value.schema for material in process.materials}
        if len(schemas) != 1:
            raise HistoryError(
                HistoryIssue(
                    HistoryIssueCode.SCHEMA_MISMATCH,
                    "All material occurrences in one MTG need one LLG schema.",
                )
            )
        nodes, edges = _lineages(process.materials, process.events)
        result = cls(
            process.process_id,
            process.materials,
            process.events,
            process.independence,
            process.causal_pairs,
            nodes,
            edges,
        )
        replay = result.replay()
        if not replay.valid:
            raise HistoryError(*replay.issues)
        return result

    @property
    def material_by_id(self) -> dict[str, MaterialOccurrence]:
        return {material.occurrence_id: material for material in self.materials}

    @property
    def event_by_id(self) -> dict[str, RuleOccurrence]:
        return {event.occurrence_id: event for event in self.events}

    @property
    def node_lineage_by_carrier(self) -> dict[Carrier, str]:
        return {
            carrier: lineage.lineage_id
            for lineage in self.node_lineages
            for carrier in lineage.members
        }

    def to_process(self) -> OccurrenceProcess:
        """Reconstruct the complete process, including occurrence bindings."""
        return OccurrenceProcess(
            self.process_id, self.materials, self.events, self.independence
        )

    def replay(self) -> HistoryReplay:
        issues = []
        try:
            process = self.to_process()
        except ProcessError as error:
            issues.append(
                HistoryIssue(
                    HistoryIssueCode.ROUNDTRIP,
                    "The retained occurrence process no longer validates.",
                    {"issues": tuple(issue.to_dict() for issue in error.issues)},
                )
            )
            return HistoryReplay(False, tuple(issues))
        observed_nodes = Counter(
            carrier for lineage in self.node_lineages for carrier in lineage.members
        )
        expected_nodes = Counter(
            (material.occurrence_id, node)
            for material in self.materials
            for node in material.value.node_ids
        )
        observed_edges = Counter(
            carrier for lineage in self.edge_lineages for carrier in lineage.members
        )
        expected_edges = Counter(
            (material.occurrence_id, edge)
            for material in self.materials
            for edge in material.value.edge_keys
        )
        if observed_nodes != expected_nodes or observed_edges != expected_edges:
            issues.append(
                HistoryIssue(
                    HistoryIssueCode.ROUNDTRIP,
                    "Carrier lineages do not cover every material carrier exactly.",
                )
            )
        if process.causal_pairs != self.causal_pairs:
            issues.append(
                HistoryIssue(
                    HistoryIssueCode.ROUNDTRIP,
                    "Causal dependencies changed during reconstruction.",
                )
            )
        return HistoryReplay(not issues, tuple(issues))

    def _flow_roles(self) -> tuple[set[str], set[str]]:
        produced = {
            binding.material_id for event in self.events for binding in event.outputs
        }
        consumed = {
            binding.material_id for event in self.events for binding in event.inputs
        }
        return produced, consumed

    @property
    def initial_material_ids(self) -> frozenset[str]:
        produced, _ = self._flow_roles()
        return frozenset(set(self.material_by_id) - produced)

    @property
    def final_material_ids(self) -> frozenset[str]:
        _, consumed = self._flow_roles()
        return frozenset(set(self.material_by_id) - consumed)

    def _active_cuts(self, extension: Iterable[str]) -> tuple[frozenset[str], ...]:
        process = self.to_process()
        order = tuple(extension)
        if not process.is_linear_extension(order):
            raise HistoryError(
                HistoryIssue(
                    HistoryIssueCode.INVALID_EXTENSION,
                    "A history timeline requires a valid process linear extension.",
                )
            )
        active = set(self.initial_material_ids)
        cuts = [frozenset(active)]
        for event_id in order:
            event = self.event_by_id[event_id]
            inputs = {binding.material_id for binding in event.inputs}
            outputs = {binding.material_id for binding in event.outputs}
            if not inputs <= active:
                raise HistoryError(
                    HistoryIssue(
                        HistoryIssueCode.INACTIVE_INPUT,
                        "An event consumes material absent at its process cut.",
                        {"event": event_id, "materials": tuple(sorted(inputs - active))},
                    )
                )
            if outputs & active:
                raise HistoryError(
                    HistoryIssue(
                        HistoryIssueCode.ACTIVE_OUTPUT,
                        "An event produces an already active material occurrence.",
                        {"event": event_id},
                    )
                )
            active.difference_update(inputs)
            active.update(outputs)
            cuts.append(frozenset(active))
        return tuple(cuts)

    def _node_state(self, lineage: NodeLineage, active: frozenset[str]) -> CarrierState:
        members = tuple(member for member in lineage.members if member[0] in active)
        if not members:
            return ABSENT
        if len(members) != 1:
            raise HistoryError(
                HistoryIssue(
                    HistoryIssueCode.LINEAGE_CONFLICT,
                    "A process cut contains two active copies of one node lineage.",
                    {"lineage": lineage.lineage_id},
                )
            )
        material_id, node = members[0]
        labels = tuple(
            sorted(
                self.material_by_id[material_id]
                .value.node_labels(node, semantic=True)
                .items()
            )
        )
        return CarrierState(True, material_id, node, labels)

    def _edge_state(self, lineage: EdgeLineage, active: frozenset[str]) -> CarrierState:
        members = tuple(member for member in lineage.members if member[0] in active)
        if not members:
            return ABSENT
        if len(members) != 1:
            raise HistoryError(
                HistoryIssue(
                    HistoryIssueCode.LINEAGE_CONFLICT,
                    "A process cut contains two active copies of one edge lineage.",
                    {"lineage": lineage.lineage_id},
                )
            )
        material_id, edge = members[0]
        labels = tuple(
            sorted(
                self.material_by_id[material_id]
                .value.edge_labels(edge, semantic=True)
                .items()
            )
        )
        return CarrierState(True, material_id, edge, labels)

    def full_history(self, extension: Iterable[str]) -> nx.MultiGraph:
        """Project all node/edge lineages over one certified linear extension."""
        order = tuple(extension)
        cuts = self._active_cuts(order)
        graph = nx.MultiGraph()
        graph.graph.update(
            process_id=self.process_id,
            event_order=order,
            causal_pairs=self.to_process().causal_pairs,
            cut_count=len(cuts),
        )
        for lineage in self.node_lineages:
            graph.add_node(
                lineage.lineage_id,
                members=lineage.members,
                history=tuple(self._node_state(lineage, cut) for cut in cuts),
            )
        for lineage in self.edge_lineages:
            if len(lineage.endpoints) != 2:
                raise HistoryError(
                    HistoryIssue(
                        HistoryIssueCode.LINEAGE_CONFLICT,
                        "A simple edge lineage must have two node endpoints.",
                    )
                )
            left, right = tuple(lineage.endpoints)
            graph.add_edge(
                left,
                right,
                key=lineage.lineage_id,
                lineage_id=lineage.lineage_id,
                members=lineage.members,
                history=tuple(self._edge_state(lineage, cut) for cut in cuts),
            )
        return graph

    def _aggregate(self, material_ids: frozenset[str], name: str) -> LewisLabelledGraph:
        schema = self.materials[0].value.schema
        graph = nx.Graph()
        for material_id in sorted(material_ids):
            material = self.material_by_id[material_id]
            for node in material.value.node_ids:
                graph.add_node(
                    (material_id, node), **material.value.node_labels(node)
                )
            for edge in material.value.edge_keys:
                left, right = tuple(edge)
                graph.add_edge(
                    (material_id, left),
                    (material_id, right),
                    **material.value.edge_labels(edge),
                )
        return LewisLabelledGraph.from_networkx(graph, schema, name=name)

    def outer_rule(self) -> RuleSpan:
        """Return the initial-to-final transformation induced by carrier lineages."""
        left = self._aggregate(self.initial_material_ids, f"{self.process_id}:initial")
        right = self._aggregate(self.final_material_ids, f"{self.process_id}:final")
        preserved = {}
        for lineage in self.node_lineages:
            initial = tuple(
                member for member in lineage.members if member[0] in self.initial_material_ids
            )
            final = tuple(
                member for member in lineage.members if member[0] in self.final_material_ids
            )
            if len(initial) == len(final) == 1:
                preserved[initial[0]] = final[0]
            elif len(initial) > 1 or len(final) > 1:
                raise HistoryError(
                    HistoryIssue(
                        HistoryIssueCode.LINEAGE_CONFLICT,
                        "Outer projection has multiple boundary copies in one lineage.",
                    )
                )
        preserved_edges = set()
        for lineage in self.edge_lineages:
            initial = tuple(
                member for member in lineage.members if member[0] in self.initial_material_ids
            )
            final = tuple(
                member for member in lineage.members if member[0] in self.final_material_ids
            )
            if len(initial) == len(final) == 1:
                material_id, edge = initial[0]
                preserved_edges.add(
                    frozenset((material_id, node) for node in edge)
                )
            elif len(initial) > 1 or len(final) > 1:
                raise HistoryError(
                    HistoryIssue(
                        HistoryIssueCode.LINEAGE_CONFLICT,
                        "Outer projection has multiple boundary edges in one lineage.",
                    )
                )
        try:
            return RuleSpan.from_mapping(
                left,
                right,
                preserved,
                preserved_edges=preserved_edges,
                name=f"{self.process_id}:outer",
            )
        except DPOError as error:
            raise HistoryError(
                HistoryIssue(
                    HistoryIssueCode.OUTER_RULE,
                    "The history boundary does not form an admitted outer rule.",
                    {"issues": tuple(issue.to_dict() for issue in error.issues)},
                )
            ) from error

    def composed_change_graph(self) -> nx.MultiGraph:
        """Project only initial/final lineage states, analogous to a composed ITS."""
        process = self.to_process()
        extension = tuple(
            nx.lexicographical_topological_sort(process.causal_graph, key=repr)
        )
        full = self.full_history(extension)
        result = nx.MultiGraph()
        result.graph.update(process_id=self.process_id, projection="composed")
        for node, attrs in full.nodes(data=True):
            history = attrs["history"]
            result.add_node(node, left=history[0], right=history[-1])
        for left, right, key, attrs in full.edges(keys=True, data=True):
            history = attrs["history"]
            if history[0].present or history[-1].present:
                result.add_edge(
                    left,
                    right,
                    key=key,
                    left=history[0],
                    right=history[-1],
                )
        return result

    def minimal_changed_core(self, extension: Iterable[str]) -> nx.MultiGraph:
        """Retain every carrier whose full history changes, including transients."""
        full = self.full_history(extension)
        changed_edges = []
        changed_nodes = set()
        for left, right, key, attrs in full.edges(keys=True, data=True):
            signatures = {(state.present, state.labels) for state in attrs["history"]}
            if len(signatures) > 1:
                changed_edges.append((left, right, key, attrs))
                changed_nodes.update((left, right))
        for node, attrs in full.nodes(data=True):
            signatures = {(state.present, state.labels) for state in attrs["history"]}
            if len(signatures) > 1:
                changed_nodes.add(node)
        result = nx.MultiGraph()
        result.graph.update(process_id=self.process_id, projection="minimal-changed-core")
        for node in sorted(changed_nodes):
            result.add_node(node, **full.nodes[node])
        for left, right, key, attrs in changed_edges:
            result.add_edge(left, right, key=key, **attrs)
        return result


@dataclass(frozen=True)
class OccurrenceMTGAlternative:
    alternative_id: str
    history: OccurrenceMTG


@dataclass(frozen=True)
class OccurrenceMTGFamily:
    """One MTG per process alternative; no material-flow quotient is applied."""

    alternatives: tuple[OccurrenceMTGAlternative, ...]
    choices: tuple[ChoiceWitness, ...]

    @classmethod
    def from_process_family(
        cls, family: OccurrenceProcessFamily
    ) -> "OccurrenceMTGFamily":
        alternatives = tuple(
            OccurrenceMTGAlternative(
                alternative.alternative_id,
                OccurrenceMTG.from_process(alternative.process),
            )
            for alternative in family.alternatives
        )
        result = cls(alternatives, family.choices)
        if {item.alternative_id for item in alternatives} != {
            item.alternative_id for item in family.alternatives
        }:
            raise HistoryError(
                HistoryIssue(
                    HistoryIssueCode.ALTERNATIVE_MISMATCH,
                    "MTG derivation changed the process alternative family.",
                )
            )
        return result

    def to_process_family(self) -> OccurrenceProcessFamily:
        return OccurrenceProcessFamily(
            tuple(
                ProcessAlternative(item.alternative_id, item.history.to_process())
                for item in self.alternatives
            ),
            self.choices,
        )


__all__ = [
    "ABSENT",
    "CarrierState",
    "EdgeLineage",
    "HistoryError",
    "HistoryIssue",
    "HistoryIssueCode",
    "HistoryReplay",
    "NodeLineage",
    "OccurrenceMTG",
    "OccurrenceMTGAlternative",
    "OccurrenceMTGFamily",
]
