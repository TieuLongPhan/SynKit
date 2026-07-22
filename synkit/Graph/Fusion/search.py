"""Complete deterministic verified-fusion enumeration and exact deduplication."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from time import perf_counter
from typing import Any, Mapping, Sequence

import networkx as nx

from .candidate import (
    FusionCandidate,
    fusion_candidate_from_construction,
    fusion_candidates_exactly_equivalent,
)
from .constructor import FusionConstructionError, construct_pushout
from .identity import graph_identity_digest
from .interface import FusionInterface, FusionInterfaceError


@dataclass(frozen=True)
class FusionProposal:
    """One proposed mapping between a forward/backward graph pair."""

    forward_graph: nx.Graph = field(compare=False, repr=False)
    backward_graph: nx.Graph = field(compare=False, repr=False)
    mapping: tuple[tuple[Any, Any], ...]
    rsmi: str | None = None

    @classmethod
    def create(
        cls,
        forward_graph: nx.Graph,
        backward_graph: nx.Graph,
        mapping: Mapping[Any, Any],
        *,
        rsmi: str | None = None,
    ) -> "FusionProposal":
        return cls(
            forward_graph,
            backward_graph,
            tuple(
                sorted(mapping.items(), key=lambda item: (repr(item[0]), repr(item[1])))
            ),
            rsmi,
        )

    def deterministic_key(self) -> tuple[Any, ...]:
        return (
            graph_identity_digest(self.forward_graph),
            graph_identity_digest(self.backward_graph),
            tuple((repr(left), repr(right)) for left, right in self.mapping),
        )


@dataclass(frozen=True)
class FusionSearchCounts:
    proposed: int
    explored: int
    rejected: int
    valid: int
    deduplicated: int
    returned: int
    truncated: int

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class FusionSearchResult:
    candidates: tuple[FusionCandidate, ...]
    rejected: tuple[Mapping[str, Any], ...]
    counts: FusionSearchCounts
    complete: bool
    timings: Mapping[str, float]

    @property
    def fused_its(self) -> list[nx.Graph]:
        return [candidate.its.copy() for candidate in self.candidates]

    @property
    def fused_rsmis(self) -> list[str]:
        return [
            candidate.rsmi
            for candidate in self.candidates
            if candidate.rsmi is not None
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejected": [dict(item) for item in self.rejected],
            "counts": self.counts.to_dict(),
            "complete": self.complete,
            "timings": dict(self.timings),
        }


class VerifiedFusionSearch:
    """Construct and validate every proposal before deduplication and ranking."""

    def __init__(
        self,
        proposals: Sequence[FusionProposal],
        *,
        cap: int | None = None,
        interface_kwargs: Mapping[str, Any] | None = None,
        construction_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if cap is not None and cap < 1:
            raise ValueError("cap must be positive or None for complete search.")
        self.proposals = tuple(proposals)
        self.cap = cap
        self.interface_kwargs = dict(interface_kwargs or {})
        self.construction_kwargs = dict(construction_kwargs or {})

    def run(self) -> FusionSearchResult:
        total_start = perf_counter()
        discovery_start = perf_counter()
        ordered = tuple(
            sorted(self.proposals, key=lambda item: item.deterministic_key())
        )
        discovery_time = perf_counter() - discovery_start
        limit = len(ordered) if self.cap is None else min(self.cap, len(ordered))
        selected = ordered[:limit]
        rejected: list[Mapping[str, Any]] = []
        valid: list[FusionCandidate] = []
        interface_time = 0.0
        construction_time = 0.0
        proof_time = 0.0
        for proposal_index, proposal in enumerate(selected):
            stage_start = perf_counter()
            try:
                interface = FusionInterface.from_mapping(
                    proposal.forward_graph,
                    proposal.backward_graph,
                    dict(proposal.mapping),
                    **self.interface_kwargs,
                )
            except FusionInterfaceError as exc:
                interface_time += perf_counter() - stage_start
                rejected.append(
                    {
                        "proposal_index": proposal_index,
                        "issues": [issue.to_dict() for issue in exc.issues],
                    }
                )
                continue
            interface_time += perf_counter() - stage_start

            stage_start = perf_counter()
            try:
                construction = construct_pushout(
                    proposal.forward_graph,
                    proposal.backward_graph,
                    interface,
                    **self.construction_kwargs,
                )
            except FusionConstructionError as exc:
                construction_time += perf_counter() - stage_start
                rejected.append(
                    {
                        "proposal_index": proposal_index,
                        "issues": [issue.to_dict() for issue in exc.issues],
                    }
                )
                continue
            construction_time += perf_counter() - stage_start

            stage_start = perf_counter()
            valid.append(
                fusion_candidate_from_construction(
                    construction,
                    rsmi=proposal.rsmi,
                )
            )
            proof_time += perf_counter() - stage_start
        dedup_start = perf_counter()
        unique: list[FusionCandidate] = []
        buckets: dict[str, list[FusionCandidate]] = defaultdict(list)
        for candidate in valid:
            if any(
                fusion_candidates_exactly_equivalent(candidate, previous)
                for previous in buckets[candidate.canonical_signature]
            ):
                continue
            unique.append(candidate)
            buckets[candidate.canonical_signature].append(candidate)
        dedup_time = perf_counter() - dedup_start

        ranking_start = perf_counter()
        unique.sort(
            key=lambda candidate: (
                candidate.score,
                candidate.canonical_signature,
                candidate.proof_digest,
            )
        )
        ranking_time = perf_counter() - ranking_start
        serialization_start = perf_counter()
        for candidate in unique:
            json.dumps(candidate.to_dict(), sort_keys=True)
        serialization_time = perf_counter() - serialization_start
        truncated = len(ordered) - len(selected)
        counts = FusionSearchCounts(
            proposed=len(ordered),
            explored=len(selected),
            rejected=len(rejected),
            valid=len(valid),
            deduplicated=len(valid) - len(unique),
            returned=len(unique),
            truncated=truncated,
        )
        timings = {
            "discovery": discovery_time,
            "interface_validation": interface_time,
            "construction": construction_time,
            "endpoint_proof": proof_time,
            "deduplication": dedup_time,
            "ranking": ranking_time,
            "serialization": serialization_time,
            "total": perf_counter() - total_start,
        }
        return FusionSearchResult(
            tuple(unique),
            tuple(rejected),
            counts,
            truncated == 0,
            timings,
        )


__all__ = [
    "FusionProposal",
    "FusionSearchCounts",
    "FusionSearchResult",
    "VerifiedFusionSearch",
]
