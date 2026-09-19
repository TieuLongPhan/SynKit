from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Protocol, Tuple

from .mixtures import (
    iter_mixtures_arity1,
    iter_mixtures_arity2,
    iter_mixtures_arityk,
)


class ConstructionStrategy(Protocol):
    def iter_mixtures(
        self,
        *,
        pool_keys: List[str],
        frontier_keys: List[str],
        arity: int,
        use_frontier: bool,
        allow_self_mixtures: bool,
        cap: int,
        max_components: int,
    ) -> Iterator[Tuple[str, ...]]: ...


@dataclass(frozen=True)
class FrontierStrategy:
    """
    Default strategy with the same semantics as the validated monolith.

    The purpose is not to change chemistry, only to separate the exploration
    strategy from the builder. This mirrors the derivation-graph idea of having
    an execution policy distinct from rule application.
    """

    def iter_mixtures(
        self,
        *,
        pool_keys: List[str],
        frontier_keys: List[str],
        arity: int,
        use_frontier: bool,
        allow_self_mixtures: bool,
        cap: int,
        max_components: int,
    ) -> Iterator[Tuple[str, ...]]:
        if arity < 1 or arity > max_components:
            return iter(())

        if arity == 1:
            return iter_mixtures_arity1(
                pool_keys,
                frontier_keys,
                use_frontier=use_frontier,
                cap=cap,
            )

        if arity == 2:
            return iter_mixtures_arity2(
                pool_keys,
                frontier_keys,
                use_frontier=use_frontier,
                allow_self_mixtures=allow_self_mixtures,
                cap=cap,
            )

        return iter_mixtures_arityk(
            pool_keys,
            frontier_keys,
            use_frontier=use_frontier,
            allow_self_mixtures=allow_self_mixtures,
            arity=arity,
            cap=cap,
        )
