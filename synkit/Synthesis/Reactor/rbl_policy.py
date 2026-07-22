"""Explicit search-scope and termination policies for RBL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SearchScope(str, Enum):
    """Maximum candidate-generation scope available to one RBL run."""

    FAST_PATHS_ONLY = "fast_paths_only"
    FUSION = "fusion"


class TerminationPolicy(str, Enum):
    """Condition under which an RBL search terminates."""

    FIRST_VALID = "first_valid"
    EXHAUSTIVE = "exhaustive"


@dataclass(frozen=True)
class RBLSearchPolicy:
    """Orthogonal description of RBL search scope and termination."""

    scope: SearchScope
    termination: TerminationPolicy

    def __post_init__(self) -> None:
        if (
            self.scope is SearchScope.FAST_PATHS_ONLY
            and self.termination is TerminationPolicy.EXHAUSTIVE
        ):
            raise ValueError(
                "FAST_PATHS_ONLY supports FIRST_VALID only; fast paths are "
                "shortcuts rather than a complete candidate enumerator."
            )

    @classmethod
    def from_mode(cls, mode: str) -> "RBLSearchPolicy":
        """Resolve a backward-compatible mode name to an explicit policy."""
        policies = {
            "fast_track": cls(
                SearchScope.FAST_PATHS_ONLY,
                TerminationPolicy.FIRST_VALID,
            ),
            "early_stop": cls(
                SearchScope.FUSION,
                TerminationPolicy.FIRST_VALID,
            ),
            "full": cls(
                SearchScope.FUSION,
                TerminationPolicy.EXHAUSTIVE,
            ),
            "verified": cls(
                SearchScope.FUSION,
                TerminationPolicy.EXHAUSTIVE,
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
        }


__all__ = [
    "RBLSearchPolicy",
    "SearchScope",
    "TerminationPolicy",
]
