"""Grounded partial-order histories for local edits on fixed atom carriers.

This profile derives semantic reads and preservation writes from unguarded,
atom-preserving source spans. It retains an admitted DPO trace and permits
shared reads without assigning a whole molecule multiple consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx

from synkit.Graph.Morphism import (
    LewisLabelledGraph,
    derive_electron_labeled_graph,
    is_electron_schema,
)
from synkit.Rule.Apply import DPOApplication, RuleSpan, SystemBoundary, apply_dpo

Coordinate = tuple[str, Hashable, str | None]


class LocalHistoryError(ValueError):
    """Raised when an event or cut is outside the grounded coordinate profile."""


@dataclass(frozen=True)
class GroundedLocalEvent:
    """One rule occurrence with its explicit atom match into the fixed world."""

    event_id: str
    rule: RuleSpan
    match: tuple[tuple[Hashable, Hashable], ...]

    def __post_init__(self) -> None:
        pairs = self.match.items() if isinstance(self.match, Mapping) else self.match
        object.__setattr__(self, "match", tuple(sorted(pairs, key=repr)))
        if not self.event_id:
            raise LocalHistoryError("A local event requires a nonempty ID.")


@dataclass(frozen=True)
class LocalEventRecord:
    """A retained source execution and its semantic coordinate access sets."""

    event: GroundedLocalEvent
    reads: frozenset[Coordinate]
    writes: frozenset[Coordinate]
    assignments: tuple[tuple[Coordinate, Any], ...]
    application: DPOApplication


@dataclass(frozen=True)
class CoordinateCut:
    """An ideal's decoded world and last-writer versions, without a schedule."""

    executed_events: frozenset[str]
    world: LewisLabelledGraph
    versions: tuple[tuple[Coordinate, str | None], ...]


def _present(pair):
    return "presence", pair, None


def _node(node, key):
    return "node", node, key


def _bond(pair, key):
    return "bond", pair, key


def _coordinate_state(world, pairs):
    state = {
        _node(node, key): world.node_labels(node, semantic=True)[key]
        for node in world.node_ids
        for key in world.schema.semantic_node_keys
    }
    for pair in pairs:
        present = pair in world.edge_keys
        state[_present(pair)] = present
        labels = world.edge_labels(pair, semantic=True) if present else {}
        for key in world.schema.semantic_edge_keys:
            # Definedness is tagged independently of the raw value (including None).
            state[_bond(pair, key)] = (True, labels[key]) if present else (False, None)
    return state


def _decode(initial, state, pairs):
    graph = initial.to_networkx()
    graph.remove_edges_from(tuple(graph.edges))
    for node in graph:
        for key in initial.schema.semantic_node_keys:
            graph.nodes[node][key] = state[_node(node, key)]
        for key in initial.schema.node_derived:
            graph.nodes[node].pop(key, None)
    for pair in pairs:
        if state[_present(pair)]:
            labels = {
                key: state[_bond(pair, key)][1]
                for key in initial.schema.semantic_edge_keys
            }
            if any(
                not state[_bond(pair, key)][0]
                for key in initial.schema.semantic_edge_keys
            ):
                raise LocalHistoryError("A present bond has an absent coordinate.")
            graph.add_edge(*pair, **labels)
        elif any(
            state[_bond(pair, key)][0] for key in initial.schema.semantic_edge_keys
        ):
            raise LocalHistoryError("An absent bond retains a coordinate.")
    if is_electron_schema(initial.schema):
        graph = derive_electron_labeled_graph(graph)
    return LewisLabelledGraph.from_networkx(graph, initial.schema)


def _profile(event, initial):
    rule = event.rule
    if rule.left.schema != initial.schema or rule.right.schema != initial.schema:
        raise LocalHistoryError("Every local rule must use the initial world's schema.")
    if rule.condition is not None:
        raise LocalHistoryError("Arbitrary guards have no inferred locality contract.")
    if rule.boundary is SystemBoundary.OPEN:
        raise LocalHistoryError(
            "Open environmental resources require a separate locality contract."
        )
    if rule.left.partial or rule.right.partial or initial.partial:
        raise LocalHistoryError("Local worlds and rule endpoints must be total.")
    if set(rule.left_arm.mapping.values()) != set(rule.left.node_ids) or set(
        rule.right_arm.mapping.values()
    ) != set(rule.right.node_ids):
        raise LocalHistoryError(
            "This coordinate profile requires preservation of every rule atom."
        )
    match = dict(event.match)
    if len(match) != len(event.match) or set(match) != set(rule.left.node_ids):
        raise LocalHistoryError(
            "The declared atom match must cover the left pattern exactly."
        )
    if len(set(match.values())) != len(match) or not set(match.values()) <= set(
        initial.node_ids
    ):
        raise LocalHistoryError(
            "The declared atom match must be injective into the fixed atom universe."
        )
    return match, {
        rule.right_arm.mapping[interface]: match[left]
        for interface, left in rule.left_arm.mapping.items()
    }


def _access(event, match, right_match, all_coordinates):
    rule = event.rule
    schema = rule.left.schema
    reads = {
        _node(match[node], key)
        for node in rule.left.node_ids
        for key in schema.semantic_node_keys
    }
    if any(
        endpoint.node_labels(node).get("aromatic", False)
        for endpoint in (rule.left, rule.right)
        for node in endpoint.node_ids
    ):
        # Phase-aware source matching can inspect more than the local pattern.
        # A whole-world read is sufficient; no extra aromatic concurrency is inferred.
        reads.update(all_coordinates)
    writes = set()
    for interface, left in rule.left_arm.mapping.items():
        retained = rule.interface.node_labels(interface, semantic=True)
        writes.update(
            _node(match[left], key)
            for key in schema.semantic_node_keys
            if key not in retained
        )
    left_edges = {frozenset(match[n] for n in pair) for pair in rule.left.edge_keys}
    right_edges = {
        frozenset(right_match[n] for n in pair) for pair in rule.right.edge_keys
    }
    retained_edges = {
        frozenset(match[rule.left_arm.mapping[n]] for n in pair): pair
        for pair in rule.interface.edge_keys
    }
    for pair in left_edges:
        reads.add(_present(pair))
        reads.update(_bond(pair, key) for key in schema.semantic_edge_keys)
    for pair in right_edges - left_edges:
        # Simple-graph creation requires absence, including every new risk pair.
        reads.add(_present(pair))
    for pair in left_edges | right_edges:
        if pair not in retained_edges:
            writes.add(_present(pair))
            writes.update(_bond(pair, key) for key in schema.semantic_edge_keys)
        else:
            labels = rule.interface.edge_labels(retained_edges[pair], semantic=True)
            writes.update(
                _bond(pair, key)
                for key in schema.semantic_edge_keys
                if key not in labels
            )
    return frozenset(reads), frozenset(writes)


def _conflict(first, second):
    return bool(
        first.writes & (second.reads | second.writes)
        or second.writes & (first.reads | first.writes)
    )


def _declared_precedence(events, precedence):
    positions = {event.event_id: index for index, event in enumerate(events)}
    try:
        precedence = tuple(tuple(pair) for pair in precedence)
        valid = all(
            len(pair) == 2
            and pair[0] in positions
            and pair[1] in positions
            and positions[pair[0]] < positions[pair[1]]
            for pair in precedence
        )
    except TypeError:
        valid = False
    if not valid:
        raise LocalHistoryError(
            "Declared precedence must name known events in the supplied trace order."
        )
    return tuple(sorted(set(precedence), key=repr))


def _construct(initial, events, precedence=()):
    if initial.partial:
        raise LocalHistoryError(
            "The initial world must be total, including for an empty trace."
        )
    if is_electron_schema(initial.schema):
        initial = LewisLabelledGraph.from_networkx(
            derive_electron_labeled_graph(initial.to_networkx()), initial.schema
        )
    if len({event.event_id for event in events}) != len(events):
        raise LocalHistoryError("Local event IDs must be distinct.")
    precedence = _declared_precedence(events, precedence)
    plans = tuple((event, *_profile(event, initial)) for event in events)
    pairs = set(initial.edge_keys)
    for event, match, right_match in plans:
        pairs.update(
            frozenset(match[n] for n in pair) for pair in event.rule.left.edge_keys
        )
        pairs.update(
            frozenset(right_match[n] for n in pair)
            for pair in event.rule.right.edge_keys
        )
    pairs = frozenset(pairs)
    world = initial
    records = []
    all_coordinates = frozenset(_coordinate_state(initial, pairs))
    for event, match, right_match in plans:
        reads, writes = _access(event, match, right_match, all_coordinates)
        _require_raw_match(event, world)
        application = apply_dpo(event.rule, world, match)
        if (
            not application.certificate.replay().valid
            or application.result.node_ids != initial.node_ids
        ):
            raise LocalHistoryError(
                "The retained source execution does not preserve the fixed atoms."
            )
        before = _coordinate_state(world, pairs)
        after = _coordinate_state(application.result, pairs)
        assignments = tuple(sorted(((key, after[key]) for key in writes), key=repr))
        before.update(assignments)
        if before != after:
            raise LocalHistoryError(
                "The inferred writes do not cover the source execution."
            )
        records.append(LocalEventRecord(event, reads, writes, assignments, application))
        world = application.result
    dependencies = tuple(
        (first.event.event_id, second.event.event_id)
        for first, second in combinations(records, 2)
        if _conflict(first, second)
    )
    dependencies += tuple(pair for pair in precedence if pair not in dependencies)
    return LocalCoordinateHistory(
        initial, tuple(records), pairs, dependencies, precedence
    )


def _require_raw_match(event, world):
    match = dict(event.match)
    if any(
        event.rule.left.node_labels(node, semantic=True)
        != world.node_labels(image, semantic=True)
        for node, image in match.items()
    ):
        raise LocalHistoryError(
            "The supplied event requires exact raw atom coordinates, not phase equivalence."
        )
    for edge in event.rule.left.edge_keys:
        image = frozenset(match[node] for node in edge)
        if image not in world.edge_keys or (
            event.rule.left.edge_labels(edge, semantic=True)
            != world.edge_labels(image, semantic=True)
        ):
            raise LocalHistoryError(
                "The supplied event requires exact raw bond coordinates, not phase equivalence."
            )


@dataclass(frozen=True)
class LocalCoordinateHistory:
    """A certified trace's coordinate-dependence poset on persistent atom IDs.

    This is a sufficient trace class, not all possible chemistry or a maximal
    independence inference. Coordinates recreated with equal values remain
    writes because the source interface did not preserve them.
    """

    initial: LewisLabelledGraph
    records: tuple[LocalEventRecord, ...]
    bond_pairs: frozenset[frozenset[Hashable]]
    causal_pairs: tuple[tuple[str, str], ...]
    declared_precedence: tuple[tuple[str, str], ...] = ()

    @property
    def causal_graph(self) -> nx.DiGraph:
        """Return a fresh graph of directed semantic access dependencies."""
        graph = nx.DiGraph()
        graph.add_nodes_from(record.event.event_id for record in self.records)
        graph.add_edges_from(self.causal_pairs)
        return graph

    def replay(self) -> bool:
        """Reconstruct the retained trace, accesses, certificates and event order."""
        try:
            fresh = _construct(
                self.initial,
                tuple(record.event for record in self.records),
                self.declared_precedence,
            )
            return fresh == self
        except (ValueError, TypeError, KeyError):
            return False

    def _configuration_ids(self, completed):
        completed = frozenset(completed)
        graph = self.causal_graph
        if not completed <= set(graph) or any(
            second in completed and first not in completed
            for first, second in graph.edges
        ):
            raise LocalHistoryError(
                "A coordinate cut must name a known downward-closed event ideal."
            )
        return completed

    def configuration(self, completed: Iterable[str]) -> CoordinateCut:
        """Decode an ideal by last writers without choosing or replaying a schedule.

        The retained construction is revalidated first. Its cost is separate
        from the coordinate projection, which performs no schedule enumeration.

        :param completed: IDs of completed events forming an order ideal.
        :return: Anchored world and semantic-coordinate versions at that cut.
        :raises LocalHistoryError: If history replay or ideal validation fails.
        """
        if not self.replay():
            raise LocalHistoryError(
                "The retained coordinate history is not replayable."
            )
        completed = self._configuration_ids(completed)
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

    def execute(self, order: Iterable[str]) -> tuple[DPOApplication, ...]:
        """Certify actual primitive source execution in a supplied full extension.

        :param order: All event IDs once in a topological order.
        :return: Primitive DPO applications on their actual whole-world frames.
        :raises LocalHistoryError: If replay or the declared extension is invalid.
        """
        if not self.replay():
            raise LocalHistoryError(
                "The retained coordinate history is not replayable."
            )
        order = tuple(order)
        positions = {event: index for index, event in enumerate(order)}
        graph = self.causal_graph
        if (
            len(positions) != len(order)
            or set(order) != set(graph)
            or any(
                positions[first] >= positions[second] for first, second in graph.edges
            )
        ):
            raise LocalHistoryError("Execution requires a complete linear extension.")
        by_id = {record.event.event_id: record.event for record in self.records}
        world = self.initial
        applications = []
        for event_id in order:
            event = by_id[event_id]
            _require_raw_match(event, world)
            application = apply_dpo(event.rule, world, dict(event.match))
            if not application.certificate.replay().valid:
                raise LocalHistoryError(
                    "A reordered primitive certificate failed replay."
                )
            applications.append(application)
            world = application.result
        return tuple(applications)


def build_local_coordinate_history(
    initial: LewisLabelledGraph,
    events: Iterable[GroundedLocalEvent],
    *,
    precedence: Iterable[tuple[str, str]] = (),
) -> LocalCoordinateHistory:
    """Construct a partial-order history from an admitted grounded source trace.

    All rule atoms are preserved, endpoints/worlds total, guards absent, and
    coordinates raw. Read-only context may be shared. Nonlocal guards, open
    environment state, aromatic phase equivalence and atom allocation are not
    inferred. Aromatic rules conservatively read the complete raw world. This
    is a trace-specific sufficient independence construction.

    :param initial: Initial total world with persistent atom identities.
    :param events: One admitted sequence with explicit rule matches and IDs.
    :param precedence: Additional declared ordering constraints respected by that sequence.
    :return: Retained primitive certificates and their coordinate-dependence DAG.
    :raises LocalHistoryError: If the declared profile is unsupported or inconsistent.
    """
    return _construct(initial, tuple(events), tuple(precedence))
