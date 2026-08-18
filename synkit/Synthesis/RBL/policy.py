"""Explicit search, proof, and acceptance policies for RBL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SearchScope(str, Enum):
    """Maximum candidate-generation scope available to one RBL run."""

    FAST_PATHS_ONLY = "fast_paths_only"
    BOUNDED_FUSION = "bounded_fusion"
    FUSION = "fusion"


class TerminationPolicy(str, Enum):
    """Condition under which an RBL search terminates."""

    FIRST_VALID = "first_valid"
    EXHAUSTIVE = "exhaustive"


class OverlapScope(str, Enum):
    """Declared universe of fusion interfaces."""

    NONE = "none"
    MAXIMUM_COMMON_SUBGRAPHS = "maximum_common_subgraphs"
    ALL_TYPED = "all_typed_overlaps"


class ProofLevel(str, Enum):
    """Evidence required for an accepted fusion candidate."""

    NONE = "none"
    CONSTRUCTION = "construction"
    REPLAYABLE = "replayable"


class AcceptanceTask(str, Enum):
    """Chemical observation/conservation contract for accepted outputs."""

    COMPATIBILITY = "compatibility"
    STRICT_RECONSTRUCTION = "strict_reconstruction"


class SearchOutcomeStatus(str, Enum):
    """Four-valued outcome that never conflates limits with absence."""

    FOUND = "FOUND"
    PROVED_NONE = "PROVED_NONE"
    INCOMPLETE = "INCOMPLETE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class RBLSearchPolicy:
    """Orthogonal description of candidate, proof, and acceptance scope."""

    scope: SearchScope
    termination: TerminationPolicy
    overlap_scope: OverlapScope = OverlapScope.MAXIMUM_COMMON_SUBGRAPHS
    proof_level: ProofLevel = ProofLevel.CONSTRUCTION
    acceptance_task: AcceptanceTask = AcceptanceTask.COMPATIBILITY

    def __post_init__(self) -> None:
        if (
            self.scope in {SearchScope.FAST_PATHS_ONLY, SearchScope.BOUNDED_FUSION}
            and self.termination is TerminationPolicy.EXHAUSTIVE
        ):
            raise ValueError(
                f"{self.scope.name} supports FIRST_VALID only; bounded search "
                "is not a complete candidate enumerator."
            )

    @classmethod
    def from_mode(cls, mode: str) -> "RBLSearchPolicy":
        """Resolve a backward-compatible mode name to an explicit policy."""
        policies = {
            "fast_track": cls(
                SearchScope.FAST_PATHS_ONLY,
                TerminationPolicy.FIRST_VALID,
                OverlapScope.NONE,
                ProofLevel.NONE,
            ),
            "fast_fusion": cls(
                SearchScope.BOUNDED_FUSION,
                TerminationPolicy.FIRST_VALID,
                OverlapScope.MAXIMUM_COMMON_SUBGRAPHS,
                ProofLevel.CONSTRUCTION,
            ),
            "early_stop": cls(
                SearchScope.FUSION,
                TerminationPolicy.FIRST_VALID,
                OverlapScope.MAXIMUM_COMMON_SUBGRAPHS,
                ProofLevel.CONSTRUCTION,
            ),
            "full": cls(
                SearchScope.FUSION,
                TerminationPolicy.EXHAUSTIVE,
                OverlapScope.MAXIMUM_COMMON_SUBGRAPHS,
                ProofLevel.CONSTRUCTION,
            ),
            "verified": cls(
                SearchScope.FUSION,
                TerminationPolicy.EXHAUSTIVE,
                OverlapScope.ALL_TYPED,
                ProofLevel.REPLAYABLE,
                AcceptanceTask.STRICT_RECONSTRUCTION,
            ),
        }
        try:
            return policies[mode]
        except KeyError as exc:
            raise ValueError(
                f"Invalid mode {mode!r}. Choose from {tuple(policies)}."
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serialisable policy description."""
        return {
            "scope": self.scope.value,
            "termination": self.termination.value,
            "overlap_scope": self.overlap_scope.value,
            "proof_level": self.proof_level.value,
            "acceptance_task": self.acceptance_task.value,
        }


__all__ = [
    "AcceptanceTask",
    "OverlapScope",
    "ProofLevel",
    "RBLSearchPolicy",
    "SearchOutcomeStatus",
    "SearchScope",
    "TerminationPolicy",
]
