from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass
class DerivationState:
    """
    Lightweight builder state inspired by derivation-graph builders.

    This keeps the active chemical universe separate from the graph object so
    the orchestration layer does not need to treat NetworkX as its working
    memory for frontier expansion.
    """

    pool_keys: Set[str] = field(default_factory=set)
    frontier_keys: Set[str] = field(default_factory=set)
    step: int = 0

    def begin_step(self, step: int) -> None:
        self.step = int(step)

    def set_initial(self, *, pool_keys: Set[str], frontier_keys: Set[str]) -> None:
        self.pool_keys = set(pool_keys)
        self.frontier_keys = set(frontier_keys)
        self.step = 0

    def advance(self, next_frontier: Set[str]) -> None:
        self.frontier_keys = set(next_frontier)
        self.step += 1
