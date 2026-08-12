"""Exact typed partial-overlap search for verified RBL fusion."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Hashable, Mapping, Sequence

import networkx as nx

from synkit.Graph.Fusion import (
    FusionInterface,
    FusionInterfaceError,
    FusionInterfaceIssueCode,
)


@dataclass(frozen=True)
class TypedOverlapLimits:
    """Explicit finite bounds for one typed-overlap enumeration."""

    max_states: int = 250_000
    max_overlaps: int = 50_000
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_states <= 0 or self.max_overlaps <= 0:
            raise ValueError("Typed-overlap limits must be positive.")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None.")

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "max_states": self.max_states,
            "max_overlaps": self.max_overlaps,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class TypedOverlapCertificate:
    """Replayable accounting for an admitted finite interface universe."""

    complete: bool
    termination: str
    states_explored: int
    overlaps_emitted: int
    anchored_pairs: tuple[tuple[Hashable, Hashable], ...]
    anchor_conflicts: tuple[tuple[Hashable, Hashable], ...]
    source_nodes: int
    target_nodes: int
    candidate_pairs: int
    limits: TypedOverlapLimits

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "termination": self.termination,
            "states_explored": self.states_explored,
            "overlaps_emitted": self.overlaps_emitted,
            "anchored_pairs": [
                [repr(left), repr(right)] for left, right in self.anchored_pairs
            ],
            "anchor_conflicts": [
                [repr(left), repr(right)] for left, right in self.anchor_conflicts
            ],
            "source_nodes": self.source_nodes,
            "target_nodes": self.target_nodes,
            "candidate_pairs": self.candidate_pairs,
            "limits": self.limits.to_dict(),
            "universe": (
                "all injective typed partial overlaps containing every "
                "compatible unique provenance anchor"
            ),
            "quotient": "none",
        }


@dataclass(frozen=True)
class TypedOverlapResult:
    mappings: tuple[Mapping[Hashable, Hashable], ...]
    certificate: TypedOverlapCertificate


def _endpoint_scalar(value: Any) -> Any:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0] if value[0] == value[1] else None
    return value


def _is_wildcard(value: Any, wildcard_element: Any) -> bool:
    scalar = (
        wildcard_element[0]
        if isinstance(wildcard_element, (tuple, list))
        else wildcard_element
    )
    return value in {wildcard_element, scalar}


def _mapping_key(mapping: Mapping[Hashable, Hashable]) -> tuple[Any, ...]:
    return tuple(sorted(((repr(left), repr(right)) for left, right in mapping.items())))


def _unique_provenance_pairs(
    forward: nx.Graph,
    backward: nx.Graph,
    source_nodes: Sequence[Hashable],
    target_nodes: Sequence[Hashable],
    *,
    provenance_key: str,
) -> tuple[tuple[Hashable, Hashable], ...]:
    left: dict[Any, list[Hashable]] = {}
    right: dict[Any, list[Hashable]] = {}
    for node in source_nodes:
        value = _endpoint_scalar(forward.nodes[node].get(provenance_key))
        if value not in {None, 0}:
            left.setdefault(value, []).append(node)
    for node in target_nodes:
        value = _endpoint_scalar(backward.nodes[node].get(provenance_key))
        if value not in {None, 0}:
            right.setdefault(value, []).append(node)
    return tuple(
        sorted(
            (
                (left[value][0], right[value][0])
                for value in left.keys() & right.keys()
                if len(left[value]) == len(right[value]) == 1
            ),
            key=lambda item: (repr(item[0]), repr(item[1])),
        )
    )


def enumerate_typed_overlaps(  # noqa: C901
    forward: nx.Graph,
    backward: nx.Graph,
    *,
    node_keys: Sequence[str],
    edge_keys: Sequence[str],
    element_key: str = "element",
    wildcard_element: Any = ("*", "*"),
    provenance_key: str = "atom_map",
    limits: TypedOverlapLimits | None = None,
) -> TypedOverlapResult:
    """Enumerate every admitted partial injection within explicit limits.

    Unique, compatible atom-map identities are treated as authoritative
    provenance anchors. Remaining nodes may be included or skipped; therefore
    smaller admissible interfaces are not hidden behind a maximum-cardinality
    objective. Typed wildcard leaves are part of this declared universe and
    are also completed by the exact port-assignment stage when needed.
    """
    active = limits or TypedOverlapLimits()
    source_nodes = tuple(forward.nodes)
    target_nodes = tuple(backward.nodes)

    def admitted(mapping: Mapping[Hashable, Hashable]) -> bool:
        if not mapping:
            return False
        try:
            FusionInterface.from_mapping(
                forward,
                backward,
                mapping,
                node_keys=node_keys,
                edge_keys=edge_keys,
                element_key=element_key,
                wildcard_element=wildcard_element,
            )
        except FusionInterfaceError:
            return False
        return True

    def extendable(mapping: Mapping[Hashable, Hashable]) -> bool:
        """Accept a partial arm when only a future owner incidence is absent."""
        if not mapping:
            return True
        try:
            FusionInterface.from_mapping(
                forward,
                backward,
                mapping,
                node_keys=node_keys,
                edge_keys=edge_keys,
                element_key=element_key,
                wildcard_element=wildcard_element,
            )
        except FusionInterfaceError as error:
            return bool(error.issues) and all(
                issue.code is FusionInterfaceIssueCode.OWNER_OUTSIDE_INTERFACE
                for issue in error.issues
            )
        return True

    def potentially_compatible(source: Hashable, target: Hashable) -> bool:
        if extendable({source: target}):
            return True
        source_wildcard = _is_wildcard(
            forward.nodes[source].get(element_key), wildcard_element
        )
        target_wildcard = _is_wildcard(
            backward.nodes[target].get(element_key), wildcard_element
        )
        return source_wildcard or target_wildcard

    domains: dict[Hashable, tuple[Hashable, ...]] = {}
    for source in source_nodes:
        domains[source] = tuple(
            target
            for target in target_nodes
            if potentially_compatible(source, target)
        )

    anchors: dict[Hashable, Hashable] = {}
    anchor_conflicts: list[tuple[Hashable, Hashable]] = []
    anchor_source_nodes = tuple(
        node
        for node in source_nodes
        if not _is_wildcard(
            forward.nodes[node].get(element_key), wildcard_element
        )
    )
    anchor_target_nodes = tuple(
        node
        for node in target_nodes
        if not _is_wildcard(
            backward.nodes[node].get(element_key), wildcard_element
        )
    )
    for source, target in _unique_provenance_pairs(
        forward,
        backward,
        anchor_source_nodes,
        anchor_target_nodes,
        provenance_key=provenance_key,
    ):
        proposed = {**anchors, source: target}
        if target not in anchors.values() and extendable(proposed):
            anchors[source] = target
        else:
            anchor_conflicts.append((source, target))

    optional_sources = tuple(
        sorted(
            (node for node in source_nodes if node not in anchors),
            key=lambda node: (
                _is_wildcard(
                    forward.nodes[node].get(element_key), wildcard_element
                ),
                len(domains[node]),
                repr(node),
            ),
        )
    )
    mappings: list[dict[Hashable, Hashable]] = []
    seen: set[tuple[Any, ...]] = set()
    states = 0
    complete = True
    termination = "exhausted"
    started = monotonic()

    def limit_reached() -> bool:
        nonlocal complete, termination
        if states >= active.max_states:
            complete = False
            termination = "state_limit"
            return True
        if len(mappings) >= active.max_overlaps:
            complete = False
            termination = "overlap_limit"
            return True
        if (
            active.timeout_seconds is not None
            and monotonic() - started >= active.timeout_seconds
        ):
            complete = False
            termination = "timeout"
            return True
        return False

    def visit(
        index: int,
        current: dict[Hashable, Hashable],
        used_targets: set[Hashable],
    ) -> None:
        nonlocal states
        if limit_reached():
            return
        states += 1
        if index == len(optional_sources):
            if current and admitted(current):
                key = _mapping_key(current)
                if key not in seen:
                    seen.add(key)
                    mappings.append(dict(current))
            return

        source = optional_sources[index]
        # Stronger overlaps are visited first but do not exclude the skip
        # branch, which is essential for all-partial-overlap completeness.
        for target in domains[source]:
            if target in used_targets:
                continue
            current[source] = target
            if extendable(current):
                used_targets.add(target)
                visit(index + 1, current, used_targets)
                used_targets.remove(target)
            del current[source]
            if limit_reached():
                return
        visit(index + 1, current, used_targets)

    visit(0, dict(anchors), set(anchors.values()))
    mappings.sort(key=lambda mapping: (-len(mapping), _mapping_key(mapping)))
    certificate = TypedOverlapCertificate(
        complete=complete,
        termination=termination,
        states_explored=states,
        overlaps_emitted=len(mappings),
        anchored_pairs=tuple(anchors.items()),
        anchor_conflicts=tuple(anchor_conflicts),
        source_nodes=len(source_nodes),
        target_nodes=len(target_nodes),
        candidate_pairs=sum(map(len, domains.values())),
        limits=active,
    )
    return TypedOverlapResult(tuple(mappings), certificate)


__all__ = [
    "TypedOverlapCertificate",
    "TypedOverlapLimits",
    "TypedOverlapResult",
    "enumerate_typed_overlaps",
]
