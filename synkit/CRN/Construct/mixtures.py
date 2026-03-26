from __future__ import annotations

from itertools import combinations, combinations_with_replacement
from typing import Iterator, List, Set, Tuple


def iter_mixtures_arity1(
    pool_keys: List[str],
    frontier_keys: List[str],
    *,
    use_frontier: bool,
    cap: int,
) -> Iterator[Tuple[str, ...]]:
    pool_u = sorted(set(pool_keys))
    if not pool_u:
        return

    frontier_u = sorted(set(frontier_keys))
    if not use_frontier:
        frontier_u = pool_u

    n = 0
    for f in frontier_u:
        yield (f,)
        n += 1
        if n >= cap:
            return


def iter_mixtures_arity2(
    pool_keys: List[str],
    frontier_keys: List[str],
    *,
    use_frontier: bool,
    allow_self_mixtures: bool,
    cap: int,
) -> Iterator[Tuple[str, ...]]:
    pool_u = sorted(set(pool_keys))
    if not pool_u:
        return

    n = 0
    if not use_frontier:
        iterator = (
            combinations_with_replacement(pool_u, 2)
            if allow_self_mixtures
            else combinations(pool_u, 2)
        )
        for tup in iterator:
            yield tuple(sorted(tup))
            n += 1
            if n >= cap:
                return
        return

    frontier_u = sorted(set(frontier_keys))
    seen: Set[Tuple[str, ...]] = set()

    for f in frontier_u:
        candidates = pool_u if allow_self_mixtures else [x for x in pool_u if x != f]
        for x in candidates:
            tup = tuple(sorted((f, x)))
            if tup in seen:
                continue
            seen.add(tup)
            yield tup
            n += 1
            if n >= cap:
                return


def iter_mixtures_arityk(
    pool_keys: List[str],
    frontier_keys: List[str],
    *,
    use_frontier: bool,
    allow_self_mixtures: bool,
    arity: int,
    cap: int,
) -> Iterator[Tuple[str, ...]]:
    pool_u = sorted(set(pool_keys))
    if not pool_u:
        return

    n = 0
    if not use_frontier:
        iterator = (
            combinations_with_replacement(pool_u, arity)
            if allow_self_mixtures
            else combinations(pool_u, arity)
        )
        for tup in iterator:
            yield tuple(sorted(tup))
            n += 1
            if n >= cap:
                return
        return

    frontier_u = sorted(set(frontier_keys))
    seen: Set[Tuple[str, ...]] = set()

    for f in frontier_u:
        if allow_self_mixtures:
            rest_iter = combinations_with_replacement(pool_u, arity - 1)
        else:
            others = [x for x in pool_u if x != f]
            rest_iter = combinations(others, arity - 1)

        for rest in rest_iter:
            tup = tuple(sorted((f, *rest)))
            if tup in seen:
                continue
            seen.add(tup)
            yield tup
            n += 1
            if n >= cap:
                return
