"""Finite bond-absence reads and ideal composites on fixed atom carriers.

The retained source uses every original guard. Shape checks and access
extraction share the existing unguarded primitives; no guarded execution is
silently replaced. This is an experimental supported fragment, not public API.
"""

from dataclasses import dataclass, replace
from itertools import combinations

import networkx as nx

from Experiment.MTG.compositional_condition_compiler import _restrict
from Experiment.MTG.frame_condition_compiler import AbsenceCondition, _risk_pairs
from synkit.Graph.Morphism import (
    LewisLabelledGraph,
    derive_electron_labeled_graph,
    is_electron_schema,
)
from synkit.Graph.MTG.local import (
    CoordinateCut,
    LocalEventRecord,
    LocalHistoryError,
    _access,
    _conflict,
    _coordinate_state,
    _declared_precedence,
    _decode,
    _present,
    _profile,
    _require_raw_match,
)
from synkit.Rule.Apply import RuleEmbedding, RuleSpan, SystemBoundary, apply_dpo


def _guard_pairs(event):
    condition = event.rule.condition
    if condition is None:
        return ()
    if type(condition) is not AbsenceCondition or any(
        not isinstance(pair, frozenset)
        or len(pair) != 2
        or not pair <= event.rule.left.node_ids
        for pair in condition.pairs
    ):
        raise LocalHistoryError(
            "Only well-formed named-pair absence guards are supported."
        )
    return condition.pairs


def _plan(event, initial):
    _guard_pairs(event)
    # _profile checks only schema, atom preservation and injective match shape.
    # It does not execute a rule. The original guarded event stays authoritative.
    shape_descriptor = (
        event
        if event.rule.condition is None
        else replace(event, rule=replace(event.rule, condition=None))
    )
    return _profile(shape_descriptor, initial)


@dataclass(frozen=True)
class IdealComposite:
    rule: RuleSpan
    source: object
    ideal: frozenset
    read_nodes: frozenset
    read_edges: frozenset
    retained_nodes: frozenset
    retained_edges: frozenset


@dataclass(frozen=True)
class FiniteGuardedHistory:
    initial: LewisLabelledGraph
    records: tuple
    bond_pairs: frozenset
    causal_pairs: tuple
    declared_precedence: tuple

    @property
    def causal_graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(record.event.event_id for record in self.records)
        graph.add_edges_from(self.causal_pairs)
        return graph

    def replay(self):
        try:
            return (
                construct(
                    self.initial,
                    tuple(record.event for record in self.records),
                    precedence=self.declared_precedence,
                )
                == self
            )
        except (ValueError, TypeError, KeyError):
            return False

    def _ideal(self, completed):
        completed = frozenset(completed)
        graph = self.causal_graph
        if not completed <= set(graph) or any(
            b in completed and a not in completed for a, b in graph.edges
        ):
            raise LocalHistoryError("A cut must name a known downward-closed ideal.")
        return completed

    def _cut(self, completed):
        state = _coordinate_state(self.initial, self.bond_pairs)
        versions = {key: None for key in state}
        for record in self.records:
            if record.event.event_id in completed:
                state.update(record.assignments)
                versions.update((key, record.event.event_id) for key in record.writes)
        return CoordinateCut(
            completed,
            _decode(self.initial, state, self.bond_pairs),
            tuple(sorted(versions.items(), key=repr)),
        )

    def configuration(self, completed):
        if not self.replay():
            raise LocalHistoryError("The complete original guarded source must replay.")
        return self._cut(self._ideal(completed))

    def compile_ideal(self, completed, *, retain_nodes=(), retain_edges=()):
        """Synthesize from fixed accesses and ideal assignments, without a schedule.

        Complete original source replay remains a validation cost. Span synthesis
        does not select or execute a new linearization of the requested ideal.
        """
        if not self.replay():
            raise LocalHistoryError("The complete original guarded source must replay.")
        completed = self._ideal(completed)
        records = tuple(r for r in self.records if r.event.event_id in completed)
        retained_nodes = frozenset(retain_nodes)
        retained_edges = frozenset(frozenset(edge) for edge in retain_edges)
        if (
            not retained_nodes <= self.initial.node_ids
            or not retained_edges <= self.initial.edge_keys
        ):
            raise LocalHistoryError(
                "Retained material carriers must belong to the initial world."
            )
        retained_nodes |= frozenset(n for e in retained_edges for n in e)
        read_nodes, read_edges, output_edges, obligations, writes = (
            set(),
            set(),
            set(),
            set(),
            set(),
        )
        for record in records:
            event = record.event
            match, right = _plan(event, self.initial)
            read_nodes.update(match.values())
            read_edges.update(
                frozenset(match[n] for n in pair)
                for pair in event.rule.left.edge_keys
                if frozenset(match[n] for n in pair) in self.initial.edge_keys
            )
            output_edges.update(
                frozenset(right[n] for n in pair) for pair in event.rule.right.edge_keys
            )
            obligations.update(
                frozenset(match[n] for n in pair)
                for pair in (*_guard_pairs(event), *_risk_pairs(event.rule))
            )
            writes.update(record.writes)
        nodes, edges = read_nodes | set(retained_nodes), read_edges | set(
            retained_edges
        )
        final = self._cut(completed).world
        left = _restrict(self.initial, nodes, edges)
        right = _restrict(final, nodes, (edges | output_edges) & final.edge_keys)
        preserved_edges = {
            pair
            for pair in left.edge_keys & right.edge_keys
            if _present(pair) not in writes
        }
        boundary = (
            SystemBoundary.CLOSED
            if is_electron_schema(left.schema)
            and all(r.event.rule.boundary is SystemBoundary.CLOSED for r in records)
            else SystemBoundary.ABSTRACT
        )
        span = RuleSpan.from_mapping(
            left,
            right,
            {n: n for n in nodes},
            preserved_edges=preserved_edges,
            boundary=boundary,
            name="fixed-ideal finite composite",
        )
        graph = span.interface.to_networkx()
        for n, carrier in span.left_arm.mapping.items():
            for key in left.schema.semantic_node_keys:
                if ("node", carrier, key) in writes:
                    graph.nodes[n].pop(key, None)
        for edge, carrier in span.left_arm.edge_mapping.items():
            for key in left.schema.semantic_edge_keys:
                if ("bond", carrier, key) in writes:
                    graph.edges[tuple(edge)].pop(key, None)
        interface = LewisLabelledGraph.from_networkx(graph, left.schema, partial=True)
        extra = tuple(
            sorted(
                (
                    pair
                    for pair in obligations
                    if pair not in self.initial.edge_keys
                    and pair not in right.edge_keys
                ),
                key=repr,
            )
        )
        rule = replace(
            span,
            interface=interface,
            left_arm=RuleEmbedding(interface, left, span.left_arm.mapping),
            right_arm=RuleEmbedding(interface, right, span.right_arm.mapping),
            condition=AbsenceCondition(extra),
        )
        return IdealComposite(
            rule,
            self,
            completed,
            frozenset(read_nodes),
            frozenset(read_edges),
            retained_nodes,
            retained_edges,
        )


def construct(initial, events, *, precedence=()):
    """Retain actual guarded applications and derive sufficient access order."""
    events = tuple(events)
    if initial.partial or len({e.event_id for e in events}) != len(events):
        raise LocalHistoryError(
            "A total initial world and distinct event IDs are required."
        )
    if is_electron_schema(initial.schema):
        for graph in (
            initial,
            *(g for e in events for g in (e.rule.left, e.rule.right)),
        ):
            if any(
                graph.node_labels(n, semantic=True)["aromatic"] for n in graph.node_ids
            ):
                raise LocalHistoryError(
                    "Original aromatic phase domains remain unsupported."
                )
            derive_electron_labeled_graph(graph.to_networkx())
    precedence = _declared_precedence(events, tuple(precedence))
    plans = tuple((e, *_plan(e, initial)) for e in events)
    pairs = set(initial.edge_keys)
    for event, match, right in plans:
        pairs.update(frozenset(match[n] for n in p) for p in event.rule.left.edge_keys)
        pairs.update(frozenset(right[n] for n in p) for p in event.rule.right.edge_keys)
        pairs.update(frozenset(match[n] for n in p) for p in _guard_pairs(event))
    pairs = frozenset(pairs)
    coordinates = frozenset(_coordinate_state(initial, pairs))
    world, records = initial, []
    for event, match, right in plans:
        reads, writes = _access(event, match, right, coordinates)
        reads |= frozenset(
            _present(frozenset(match[n] for n in p)) for p in _guard_pairs(event)
        )
        _require_raw_match(event, world)
        app = apply_dpo(event.rule, world, match)
        if (
            not app.certificate.replay().valid
            or app.result.node_ids != initial.node_ids
        ):
            raise LocalHistoryError(
                "Every actual guarded application must preserve the fixed atoms."
            )
        before, after = (_coordinate_state(g, pairs) for g in (world, app.result))
        assignments = tuple(sorted(((key, after[key]) for key in writes), key=repr))
        before.update(assignments)
        if before != after:
            raise LocalHistoryError(
                "Extracted assignments do not cover the guarded source edit."
            )
        records.append(LocalEventRecord(event, reads, writes, assignments, app))
        world = app.result
    dependencies = tuple(
        (a.event.event_id, b.event.event_id)
        for a, b in combinations(records, 2)
        if _conflict(a, b)
    )
    dependencies += tuple(p for p in precedence if p not in dependencies)
    return FiniteGuardedHistory(
        initial, tuple(records), pairs, dependencies, precedence
    )
