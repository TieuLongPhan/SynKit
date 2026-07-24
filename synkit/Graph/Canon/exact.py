"""Exact native canonical labeling for finite coloured graphs.

The implementation is intentionally small and unpruned.  It provides the
correctness oracle for later individualization--refinement optimizations:
partition refinement reduces the search tree, but every unresolved branch is
still visited.  Canonical identity is available only after a complete search.
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

ColorToken = tuple[Any, ...]
ColorSelector = str | Sequence[str] | Callable[[Mapping[str, Any]], Any] | None


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


class ExactColoredGraphCanonicalizer:
    """Complete individualization--refinement for finite coloured graphs.

    Graph structure and semantic colours are snapshotted during construction.
    Node identifiers are retained only as witness handles and never enter the
    canonical certificate.
    """

    def __init__(
        self,
        graph: nx.Graph,
        *,
        node_color: ColorSelector = "color",
        edge_color: ColorSelector = "color",
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
        adjacency = []
        if self._directed:
            pairs = (
                (left, right)
                for left in range(len(order))
                for right in range(len(order))
            )
        else:
            pairs = (
                (left, right)
                for left in range(len(order))
                for right in range(left, len(order))
            )
        for left, right in pairs:
            token = self._edge_token(order[left], order[right])
            adjacency.append(("absent",) if token is None else ("edge", token))
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
    ) -> None:
        state.leaves += 1
        order = tuple(cell[0] for cell in partition)
        key = self._canonical_key(order)
        if state.best_key is None or key < state.best_key:
            state.best_key = key
            state.best_order = order
            state.equal_orders = [order]
        elif key == state.best_key:
            state.equal_orders.append(order)

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
