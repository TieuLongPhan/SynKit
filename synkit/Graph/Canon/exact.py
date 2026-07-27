"""Exact native canonical labeling for finite coloured graphs.

The default implementation is intentionally small and unpruned.  It provides
the correctness oracle for an opt-in automorphism-pruned
individualization--refinement search.  Canonical identity is available only
after a complete search.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from time import perf_counter
from typing import Any, Hashable

import networkx as nx

from .exact_refinement import ExactRefinementMixin

ColorToken = tuple[Any, ...]
ColorSelector = str | Sequence[str] | Callable[[Mapping[str, Any]], Any] | None

_STABILIZER_WORK_LIMIT = 4096


class CanonicalSearchIncomplete(RuntimeError):
    """Raised when a bounded search cannot establish canonical identity."""


def _encode_color(value: Any) -> ColorToken:
    """Return an exact, totally orderable token for supported colour values."""
    if value is None:
        return ("none",)
    if isinstance(value, Enum):
        return (
            "enum",
            type(value).__module__,
            type(value).__qualname__,
            _encode_color(value.value),
        )
    if isinstance(value, bool):
        return ("bool", int(value))
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Mapping):
        items = sorted(
            ((_encode_color(key), _encode_color(item)) for key, item in value.items())
        )
        return ("mapping", *items)
    if isinstance(value, tuple):
        return ("tuple", *(_encode_color(item) for item in value))
    if isinstance(value, list):
        return ("list", *(_encode_color(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("set", *sorted(_encode_color(item) for item in value))
    raise TypeError(
        "Canonical colours require None, bool, number, string, bytes, enum, "
        f"mapping, or finite container values; received {type(value).__name__}."
    )


def _selector_value(
    attributes: Mapping[str, Any],
    selector: ColorSelector,
) -> Any:
    if selector is None:
        return None
    if isinstance(selector, str):
        return attributes.get(selector)
    if callable(selector):
        return selector(attributes)
    return tuple(attributes.get(key) for key in selector)


@dataclass(frozen=True)
class AutomorphismWitness:
    """One verified source-to-image automorphism."""

    pairs: tuple[tuple[Hashable, Hashable], ...]

    def as_dict(self) -> dict[Hashable, Hashable]:
        """Return the witness as a mutable convenience mapping."""
        return dict(self.pairs)


@dataclass(frozen=True)
class CanonicalSearchStatistics:
    """Observable search effort for one exact or bounded run."""

    visited_nodes: int
    leaves: int
    refinement_rounds: int
    elapsed_seconds: float


@dataclass(frozen=True)
class ExactCanonicalResult:
    """Complete canonical identity and symmetry evidence.

    ``canonical_key``, ``certificate_text``, and ``canonical_code`` are
    invariant under every colour-preserving relabeling and construction-order
    permutation. ``canonical_digest`` is only a compact index of that exact
    certificate.

    ``canonical_order`` and mappings derived from it retain original node
    handles. They are labeling witnesses, not relabeling-invariant choices
    among vertices in one non-trivial automorphism orbit. Compare such
    witnesses after transport and modulo :attr:`automorphisms`.
    """

    canonical_order: tuple[Hashable, ...]
    canonical_key: tuple[Any, ...]
    certificate_text: str
    canonical_code: str
    canonical_digest: str
    automorphisms: tuple[AutomorphismWitness, ...]
    orbits: tuple[frozenset[Hashable], ...]
    statistics: CanonicalSearchStatistics
    automorphisms_complete: bool = True
    complete: bool = True
    exact: bool = True

    @property
    def canonical_mapping(self) -> dict[Hashable, int]:
        """Map original nodes to consecutive one-based canonical labels."""
        return {
            node: position
            for position, node in enumerate(self.canonical_order, start=1)
        }

    def same_canonical_graph(self, other: object) -> bool:
        """Compare exact certificates, never their finite digests."""
        return (
            isinstance(other, ExactCanonicalResult)
            and self.canonical_key == other.canonical_key
        )


@dataclass(frozen=True)
class IncompleteCanonicalResult:
    """Diagnostic evidence from a search that did not prove canonicality."""

    reason: str
    statistics: CanonicalSearchStatistics
    complete: bool = False
    exact: bool = False

    def require_complete(self) -> ExactCanonicalResult:
        """Refuse promotion of bounded-search evidence to canonical identity."""
        raise CanonicalSearchIncomplete(
            f"Canonical search is incomplete: {self.reason}."
        )


CanonicalResult = ExactCanonicalResult | IncompleteCanonicalResult


@dataclass
class _SearchState:
    """Mutable state owned by one depth-first canonical search."""

    started: float
    timeout_seconds: float | None
    max_search_nodes: int | None
    max_depth: int | None
    visited_nodes: int = 0
    leaves: int = 0
    termination_reason: str | None = None
    best_key: tuple[Any, ...] | None = None
    best_order: tuple[Hashable, ...] | None = None
    equal_orders: list[tuple[Hashable, ...]] = field(default_factory=list)
    generators: list[dict[Hashable, Hashable]] = field(default_factory=list)
    generator_signatures: set[tuple[Hashable, ...]] = field(default_factory=set)
    stabilizer_cache: dict[
        tuple[int, tuple[Hashable, ...]],
        tuple[dict[Hashable, Hashable], ...],
    ] = field(default_factory=dict)


class ExactColoredGraphCanonicalizer(ExactRefinementMixin):
    """Complete individualization--refinement for finite coloured graphs.

    Graph structure and semantic colours are snapshotted during construction.
    Node identifiers are retained only as witness handles and never enter the
    canonical certificate. ``prune_automorphisms`` enables verified orbit
    pruning together with incremental refinement, adaptive stabilizer chains,
    nonuniform-component recursion, and trace-guided certificate pruning. The
    unpruned default remains the small exhaustive correctness oracle.
    """

    def __init__(
        self,
        graph: nx.Graph,
        *,
        node_color: ColorSelector = "color",
        edge_color: ColorSelector = "color",
        prune_automorphisms: bool = False,
        enumerate_automorphism_group: bool = True,
    ) -> None:
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            raise TypeError("Exact canonicalization requires a NetworkX graph.")
        if graph.is_multigraph():
            raise TypeError(
                "Encode parallel edges as coloured incidence vertices before "
                "exact canonicalization."
            )
        self._graph = graph.copy()
        self._directed = self._graph.is_directed()
        self._nodes = tuple(self._graph.nodes())
        self._node_colors = {
            node: _encode_color(_selector_value(self._graph.nodes[node], node_color))
            for node in self._nodes
        }
        self._edge_colors = {
            (left, right): _encode_color(_selector_value(attributes, edge_color))
            for left, right, attributes in self._graph.edges(data=True)
        }
        self._edge_lookup = dict(self._edge_colors)
        if not self._directed:
            self._edge_lookup.update(
                {
                    (right, left): token
                    for (left, right), token in self._edge_colors.items()
                }
            )
        outgoing: dict[Hashable, list[tuple[Hashable, ColorToken]]] = {
            node: [] for node in self._nodes
        }
        incoming: dict[Hashable, list[tuple[Hashable, ColorToken]]] = {
            node: [] for node in self._nodes
        }
        for (left, right), token in self._edge_colors.items():
            outgoing[left].append((right, token))
            incoming[right].append((left, token))
            if not self._directed and left != right:
                outgoing[right].append((left, token))
                incoming[left].append((right, token))
        self._outgoing = {
            node: tuple(neighbours) for node, neighbours in outgoing.items()
        }
        self._incoming = {
            node: tuple(neighbours) for node, neighbours in incoming.items()
        }
        self._refinement_rounds = 0
        self._complete_cache: ExactCanonicalResult | None = None
        self._prune_automorphisms = prune_automorphisms
        self._enumerate_automorphism_group = enumerate_automorphism_group

    @property
    def graph(self) -> nx.Graph:
        """Return a defensive copy of the snapshotted graph."""
        return self._graph.copy()

    def _edge_token(
        self,
        left: Hashable,
        right: Hashable,
    ) -> ColorToken | None:
        return self._edge_lookup.get((left, right))

    @staticmethod
    def _multiset(values: list[ColorToken]) -> tuple[Any, ...]:
        counts = Counter(values)
        return tuple(sorted(counts.items()))

    def _cell_signature(
        self,
        node: Hashable,
        partition: tuple[tuple[Hashable, ...], ...],
        cell_index: Mapping[Hashable, int] | None = None,
    ) -> tuple[Any, ...]:
        if cell_index is None:
            cell_index = {
                member: index for index, cell in enumerate(partition) for member in cell
            }
        outgoing: dict[int, list[ColorToken]] = {}
        for other, token in self._outgoing[node]:
            outgoing.setdefault(cell_index[other], []).append(token)
        # Empty cells sort before non-empty cells in the former dense
        # signature.  Omitting them and negating the cell index preserves
        # that lexicographic order without materializing one entry per cell.
        if self._directed:
            incoming: dict[int, list[ColorToken]] = {}
            for other, token in self._incoming[node]:
                incoming.setdefault(cell_index[other], []).append(token)
            per_cell = tuple(
                (
                    -index,
                    (
                        self._multiset(outgoing.get(index, [])),
                        self._multiset(incoming.get(index, [])),
                    ),
                )
                for index in sorted(outgoing.keys() | incoming.keys())
            )
        else:
            per_cell = tuple(
                (-index, self._multiset(tokens))
                for index, tokens in sorted(outgoing.items())
            )
        return (self._node_colors[node], per_cell)

    def _refine(
        self,
        partition: tuple[tuple[Hashable, ...], ...],
    ) -> tuple[tuple[Hashable, ...], ...]:
        current = partition
        while True:
            changed = False
            refined: list[tuple[Hashable, ...]] = []
            cell_index = {
                node: index for index, cell in enumerate(current) for node in cell
            }
            for cell in current:
                buckets: dict[tuple[Any, ...], list[Hashable]] = {}
                for node in cell:
                    signature = self._cell_signature(
                        node,
                        current,
                        cell_index,
                    )
                    buckets.setdefault(signature, []).append(node)
                if len(buckets) > 1:
                    changed = True
                refined.extend(
                    tuple(buckets[signature]) for signature in sorted(buckets)
                )
            self._refinement_rounds += 1
            current = tuple(refined)
            if not changed:
                return current

    def _initial_partition(self) -> tuple[tuple[Hashable, ...], ...]:
        buckets: dict[ColorToken, list[Hashable]] = {}
        for node in self._nodes:
            buckets.setdefault(self._node_colors[node], []).append(node)
        return tuple(tuple(buckets[color]) for color in sorted(buckets))

    def _canonical_key(
        self,
        order: tuple[Hashable, ...],
    ) -> tuple[Any, ...]:
        node_part = tuple(self._node_colors[node] for node in order)
        absent = ("absent",)
        size = len(order)
        positions = {node: index for index, node in enumerate(order)}
        if self._directed:
            adjacency = [absent] * (size * size)
            for (left, right), token in self._edge_colors.items():
                adjacency[positions[left] * size + positions[right]] = (
                    "edge",
                    token,
                )
        else:
            adjacency = [absent] * (size * (size + 1) // 2)
            for (left, right), token in self._edge_colors.items():
                left_position, right_position = positions[left], positions[right]
                if left_position > right_position:
                    left_position, right_position = right_position, left_position
                row_start = (
                    left_position * size
                    - left_position * (left_position - 1) // 2
                )
                adjacency[row_start + right_position - left_position] = (
                    "edge",
                    token,
                )
        return (
            ("directed", int(self._directed)),
            ("nodes", node_part),
            ("adjacency", tuple(adjacency)),
        )

    def _is_automorphism(
        self,
        mapping: Mapping[Hashable, Hashable],
    ) -> bool:
        if set(mapping) != set(self._nodes) or set(mapping.values()) != set(
            self._nodes
        ):
            return False
        if any(
            self._node_colors[node] != self._node_colors[mapping[node]]
            for node in self._nodes
        ):
            return False
        # A vertex bijection maps distinct source edges to distinct pairs.
        # Since source and target have the same edge count, preserving every
        # coloured edge also proves that non-edges are preserved.
        return all(
            token == self._edge_token(mapping[left], mapping[right])
            for (left, right), token in self._edge_colors.items()
        )

    def _orbits(
        self,
        witnesses: tuple[AutomorphismWitness, ...],
    ) -> tuple[frozenset[Hashable], ...]:
        parent = {node: node for node in self._nodes}

        def find(node: Hashable) -> Hashable:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: Hashable, right: Hashable) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for witness in witnesses:
            for left, right in witness.pairs:
                union(left, right)
        classes: dict[Hashable, set[Hashable]] = {}
        for node in self._nodes:
            classes.setdefault(find(node), set()).add(node)
        position = {node: index for index, node in enumerate(self._nodes)}
        return tuple(
            sorted(
                (frozenset(values) for values in classes.values()),
                key=lambda cell: min(position[node] for node in cell),
            )
        )

    @staticmethod
    def _stop_reason(state: _SearchState, depth: int) -> str | None:
        if (
            state.timeout_seconds is not None
            and perf_counter() - state.started >= state.timeout_seconds
        ):
            return "timeout"
        if (
            state.max_search_nodes is not None
            and state.visited_nodes >= state.max_search_nodes
        ):
            return "search_node_budget"
        if state.max_depth is not None and depth > state.max_depth:
            return "depth_budget"
        return None

    def _record_leaf(
        self,
        partition: tuple[tuple[Hashable, ...], ...],
        state: _SearchState,
    ) -> tuple[tuple[Any, ...], tuple[Hashable, ...]]:
        state.leaves += 1
        order = tuple(cell[0] for cell in partition)
        key = self._canonical_key(order)
        if state.best_key is None or key < state.best_key:
            state.best_key = key
            state.best_order = order
            state.equal_orders = [order]
        elif key == state.best_key:
            state.equal_orders.append(order)
        return key, order

    def _visit_partition(
        self,
        partition: tuple[tuple[Hashable, ...], ...],
        depth: int,
        state: _SearchState,
    ) -> None:
        if state.termination_reason is not None:
            return
        reason = self._stop_reason(state, depth)
        if reason is not None:
            state.termination_reason = reason
            return
        state.visited_nodes += 1
        refined = self._refine(partition)
        ambiguous = tuple(
            (index, cell) for index, cell in enumerate(refined) if len(cell) > 1
        )
        if not ambiguous:
            self._record_leaf(refined, state)
            return
        cell_index, target = min(
            ambiguous,
            key=lambda item: (len(item[1]), item[0]),
        )
        for chosen in target:
            remainder = tuple(node for node in target if node != chosen)
            child = list(refined)
            child[cell_index : cell_index + 1] = (
                [(chosen,), remainder] if remainder else [(chosen,)]
            )
            self._visit_partition(tuple(child), depth + 1, state)
            if state.termination_reason is not None:
                return

    def _register_generator(
        self,
        left_order: tuple[Hashable, ...],
        right_order: tuple[Hashable, ...],
        state: _SearchState,
    ) -> None:
        mapping = {
            left_order[index]: right_order[index] for index in range(len(left_order))
        }
        signature = tuple(mapping[node] for node in self._nodes)
        if signature in state.generator_signatures:
            return
        if not self._is_automorphism(mapping):
            raise RuntimeError(
                "Equal canonical subtrees did not induce an automorphism."
            )
        state.generator_signatures.add(signature)
        state.generators.append(mapping)

    def _identity_mapping(self) -> dict[Hashable, Hashable]:
        return {node: node for node in self._nodes}

    def _mapping_signature(
        self,
        mapping: Mapping[Hashable, Hashable],
    ) -> tuple[Hashable, ...]:
        return tuple(mapping[node] for node in self._nodes)

    def _compose_mappings(
        self,
        first: Mapping[Hashable, Hashable],
        second: Mapping[Hashable, Hashable],
    ) -> dict[Hashable, Hashable]:
        """Return ``second`` after ``first``."""
        return {node: second[first[node]] for node in self._nodes}

    def _inverse_mapping(
        self,
        mapping: Mapping[Hashable, Hashable],
    ) -> dict[Hashable, Hashable]:
        return {image: source for source, image in mapping.items()}

    def _deduplicate_mappings(
        self,
        mappings: Sequence[Mapping[Hashable, Hashable]],
        *,
        include_identity: bool = False,
    ) -> tuple[dict[Hashable, Hashable], ...]:
        identity_signature = tuple(self._nodes)
        unique: dict[tuple[Hashable, ...], dict[Hashable, Hashable]] = {}
        for mapping in mappings:
            signature = self._mapping_signature(mapping)
            if not include_identity and signature == identity_signature:
                continue
            unique.setdefault(signature, dict(mapping))
        return tuple(unique.values())

    def _point_stabilizer_generators(
        self,
        generators: Sequence[Mapping[Hashable, Hashable]],
        point: Hashable,
    ) -> tuple[dict[Hashable, Hashable], ...]:
        """Return Schreier generators for the subgroup fixing ``point``."""
        seeds = self._deduplicate_mappings(generators)
        if not seeds:
            return ()

        identity = self._identity_mapping()
        transversals = {point: identity}
        pending = [point]
        while pending:
            current = pending.pop()
            current_transversal = transversals[current]
            for generator in seeds:
                image = generator[current]
                if image in transversals:
                    continue
                transversals[image] = self._compose_mappings(
                    current_transversal,
                    generator,
                )
                pending.append(image)

        stabilizer_generators = []
        for current, current_transversal in transversals.items():
            for generator in seeds:
                image = generator[current]
                schreier = self._compose_mappings(
                    self._compose_mappings(
                        current_transversal,
                        generator,
                    ),
                    self._inverse_mapping(transversals[image]),
                )
                stabilizer_generators.append(schreier)
        return self._deduplicate_mappings(stabilizer_generators)

    def _stabilizer_generators(
        self,
        generators: Sequence[Mapping[Hashable, Hashable]],
        fixed: tuple[Hashable, ...],
    ) -> tuple[dict[Hashable, Hashable], ...]:
        """Return generators for the subgroup fixing every point in ``fixed``."""
        stabilizers = self._deduplicate_mappings(generators)
        for point in fixed:
            stabilizers = self._point_stabilizer_generators(stabilizers, point)
            if not stabilizers:
                break
        return stabilizers

    def _cached_stabilizer_generators(
        self,
        state: _SearchState,
        fixed: tuple[Hashable, ...],
    ) -> tuple[dict[Hashable, Hashable], ...]:
        cache_key = (len(state.generators), fixed)
        cached = state.stabilizer_cache.get(cache_key)
        if cached is None:
            cached = self._stabilizer_generators(state.generators, fixed)
            state.stabilizer_cache[cache_key] = cached
        return cached

    def _pruning_generators(
        self,
        state: _SearchState,
        fixed: tuple[Hashable, ...],
        cell_index: Mapping[Hashable, int],
    ) -> tuple[dict[Hashable, Hashable], ...]:
        """Choose exact stabilizer generators without excessive group overhead."""
        direct = tuple(
            mapping
            for mapping in state.generators
            if self._stabilizes_partition(mapping, cell_index)
        )
        estimated_work = (
            len(self._nodes) * max(1, len(state.generators)) * max(1, len(fixed))
        )
        # Schreier generators save search only while their construction remains
        # cheaper than direct filtering of the verified generators.
        if estimated_work > _STABILIZER_WORK_LIMIT:
            return direct
        subgroup = self._cached_stabilizer_generators(state, fixed)
        return tuple(
            mapping
            for mapping in subgroup
            if self._stabilizes_partition(mapping, cell_index)
        )

    @staticmethod
    def _stabilizes_partition(
        mapping: Mapping[Hashable, Hashable],
        cell_index: Mapping[Hashable, int],
    ) -> bool:
        return all(cell_index[node] == cell_index[mapping[node]] for node in cell_index)

    @staticmethod
    def _generator_orbit(
        node: Hashable,
        generators: Sequence[Mapping[Hashable, Hashable]],
    ) -> frozenset[Hashable]:
        orbit = {node}
        pending = [node]
        while pending:
            current = pending.pop()
            for mapping in generators:
                image = mapping[current]
                if image not in orbit:
                    orbit.add(image)
                    pending.append(image)
        return frozenset(orbit)

    def _visit_partition_pruned(
        self,
        partition: tuple[tuple[Hashable, ...], ...],
        depth: int,
        state: _SearchState,
        *,
        fixed: tuple[Hashable, ...] = (),
        changed_members: frozenset[Hashable] | None = None,
        active_component: frozenset[Hashable] | None = None,
        already_refined: bool = False,
    ) -> tuple[tuple[Any, ...], tuple[Hashable, ...]] | None:
        """Visit one subtree, pruning only branches joined by a verified orbit."""
        if state.termination_reason is not None:
            return None
        reason = self._stop_reason(state, depth)
        if reason is not None:
            state.termination_reason = reason
            return None
        state.visited_nodes += 1
        refined = self._refine_search_partition(
            partition,
            changed_members,
            already_refined,
        )
        ambiguous = tuple(
            (index, cell) for index, cell in enumerate(refined) if len(cell) > 1
        )
        if self._certificate_prunes(refined, ambiguous, state.best_key):
            return None
        if not ambiguous:
            return self._record_leaf(refined, state)

        cell_index, target, active_component = (
            self._select_component_target(
                refined,
                ambiguous,
                active_component,
            )
        )
        partition_cells = {
            node: index for index, cell in enumerate(refined) for node in cell
        }
        explored: list[tuple[Hashable, tuple[Any, ...], tuple[Hashable, ...]]] = []
        subtree_best: tuple[tuple[Any, ...], tuple[Hashable, ...]] | None = None
        ordered_children = self._ordered_search_children(
            refined,
            cell_index,
            target,
        )
        for _trace, _position, chosen, child_refined in ordered_children:
            stabilizers = self._pruning_generators(
                state,
                fixed,
                partition_cells,
            )
            if self._orbit_already_explored(
                chosen,
                explored,
                stabilizers,
            ):
                continue

            if child_refined is None:
                child_refined = self._individualized_child(
                    refined,
                    cell_index,
                    target,
                    chosen,
                )
            child_best = self._visit_partition_pruned(
                child_refined,
                depth + 1,
                state,
                fixed=(*fixed, chosen),
                active_component=active_component,
                already_refined=True,
            )
            if state.termination_reason is not None:
                return None
            if child_best is None:
                continue
            key, order = child_best
            self._register_equal_child_generators(
                explored,
                key,
                order,
                state,
            )
            explored.append((chosen, key, order))
            if subtree_best is None or key < subtree_best[0]:
                subtree_best = (key, order)
        return subtree_best

    def _verified_witnesses(
        self,
        best_order: tuple[Hashable, ...],
        equal_orders: list[tuple[Hashable, ...]],
    ) -> tuple[AutomorphismWitness, ...]:
        witnesses = []
        for order in equal_orders:
            mapping = {
                best_order[index]: order[index] for index in range(len(best_order))
            }
            if not self._is_automorphism(mapping):
                raise RuntimeError(
                    "Equal canonical leaves did not induce an automorphism."
                )
            witnesses.append(AutomorphismWitness(tuple(mapping.items())))
        return tuple(witnesses)

    def _generated_witnesses(
        self,
        best_order: tuple[Hashable, ...],
        equal_orders: list[tuple[Hashable, ...]],
        generators: list[dict[Hashable, Hashable]],
    ) -> tuple[AutomorphismWitness, ...]:
        """Enumerate the exact group generated by pruned-search witnesses."""
        seeds = list(generators)
        for order in equal_orders:
            seeds.append(
                {best_order[index]: order[index] for index in range(len(best_order))}
            )
        identity = tuple(self._nodes)
        generator_signatures = {
            tuple(mapping[node] for node in self._nodes) for mapping in seeds
        }
        known = {identity}
        pending = [identity]
        while pending:
            current = pending.pop()
            current_mapping = dict(zip(self._nodes, current))
            for generator_signature in generator_signatures:
                generator = dict(zip(self._nodes, generator_signature))
                composed = tuple(
                    generator[current_mapping[node]] for node in self._nodes
                )
                if composed not in known:
                    known.add(composed)
                    pending.append(composed)

        witnesses = []
        for signature in sorted(
            known,
            key=lambda item: tuple(repr(value) for value in item),
        ):
            mapping = dict(zip(self._nodes, signature))
            if not self._is_automorphism(mapping):
                raise RuntimeError(
                    "A generated symmetry witness was not an automorphism."
                )
            witnesses.append(AutomorphismWitness(tuple(mapping.items())))
        return tuple(witnesses)

    def _generator_witnesses(
        self,
        best_order: tuple[Hashable, ...],
        equal_orders: list[tuple[Hashable, ...]],
        generators: list[dict[Hashable, Hashable]],
    ) -> tuple[AutomorphismWitness, ...]:
        """Return verified generators without expanding the generated group."""
        mappings = list(generators)
        mappings.extend(
            {best_order[index]: order[index] for index in range(len(best_order))}
            for order in equal_orders
        )
        identity = {node: node for node in self._nodes}
        mappings.append(identity)
        unique = {
            tuple(mapping[node] for node in self._nodes): mapping
            for mapping in mappings
        }
        witnesses = []
        for signature in sorted(
            unique,
            key=lambda item: tuple(repr(value) for value in item),
        ):
            mapping = unique[signature]
            if not self._is_automorphism(mapping):
                raise RuntimeError("A symmetry generator was not an automorphism.")
            witnesses.append(AutomorphismWitness(tuple(mapping.items())))
        return tuple(witnesses)

    def search(
        self,
        *,
        timeout_seconds: float | None = None,
        max_search_nodes: int | None = None,
        max_depth: int | None = None,
    ) -> CanonicalResult:
        """Run a complete or explicitly bounded canonical search."""
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("Canonical timeout must be non-negative.")
        if max_search_nodes is not None and max_search_nodes < 1:
            raise ValueError("Canonical search-node budget must be positive.")
        if max_depth is not None and max_depth < 0:
            raise ValueError("Canonical search depth must be non-negative.")

        started = perf_counter()
        self._refinement_rounds = 0
        state = _SearchState(
            started,
            timeout_seconds,
            max_search_nodes,
            max_depth,
        )
        if self._prune_automorphisms:
            self._visit_partition_pruned(self._initial_partition(), 0, state)
        else:
            self._visit_partition(self._initial_partition(), 0, state)
        statistics = CanonicalSearchStatistics(
            visited_nodes=state.visited_nodes,
            leaves=state.leaves,
            refinement_rounds=self._refinement_rounds,
            elapsed_seconds=perf_counter() - started,
        )
        if state.termination_reason is not None:
            return IncompleteCanonicalResult(
                state.termination_reason,
                statistics,
            )
        if state.best_key is None or state.best_order is None:
            raise RuntimeError("Complete canonical search produced no leaf.")

        if self._prune_automorphisms and self._enumerate_automorphism_group:
            automorphisms = self._generated_witnesses(
                state.best_order,
                state.equal_orders,
                state.generators,
            )
        elif self._prune_automorphisms:
            automorphisms = self._generator_witnesses(
                state.best_order,
                state.equal_orders,
                state.generators,
            )
        else:
            automorphisms = self._verified_witnesses(
                state.best_order,
                state.equal_orders,
            )
        certificate_text = json.dumps(
            state.best_key,
            separators=(",", ":"),
        )
        result = ExactCanonicalResult(
            canonical_order=state.best_order,
            canonical_key=state.best_key,
            certificate_text=certificate_text,
            # The code is the injective serialization.  The digest is only a
            # compact index and must not be used as mathematical proof of
            # graph identity.
            canonical_code=certificate_text,
            canonical_digest=hashlib.sha256(
                certificate_text.encode("utf-8")
            ).hexdigest(),
            automorphisms=automorphisms,
            orbits=self._orbits(automorphisms),
            statistics=statistics,
            automorphisms_complete=(
                not self._prune_automorphisms or self._enumerate_automorphism_group
            ),
        )
        if timeout_seconds is None and max_search_nodes is None and max_depth is None:
            self._complete_cache = result
        return result

    def canonicalize(self) -> ExactCanonicalResult:
        """Return exact identity, running an unbounded complete search once."""
        if self._complete_cache is None:
            result = self.search()
            if not isinstance(result, ExactCanonicalResult):
                raise RuntimeError("Unbounded canonical search was incomplete.")
            self._complete_cache = result
        return self._complete_cache


__all__ = [
    "AutomorphismWitness",
    "CanonicalResult",
    "CanonicalSearchIncomplete",
    "CanonicalSearchStatistics",
    "ExactCanonicalResult",
    "ExactColoredGraphCanonicalizer",
    "IncompleteCanonicalResult",
]
