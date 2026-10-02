"""Memory-bounded permutation-group helpers for exact mapper branching."""

from __future__ import annotations

_MAX_STABILIZER_GENERATORS = 512
_MAX_SCHREIER_WORK = 100_000


def _identity(size):
    return tuple(range(size))


def _compose(first, second):
    """Return ``second`` after ``first`` for image-tuple permutations."""
    return tuple(second[first[index]] for index in range(len(first)))


def _inverse(permutation):
    inverse = [0] * len(permutation)
    for source, image in enumerate(permutation):
        inverse[image] = source
    return tuple(inverse)


def _deduplicate_generators(generators):
    generators = tuple(tuple(permutation) for permutation in generators)
    if not generators:
        return ()
    identity = _identity(len(generators[0]))
    return tuple(dict.fromkeys(g for g in generators if g != identity))


def _point_stabilizer_data(
    generators,
    point,
    *,
    max_generators=_MAX_STABILIZER_GENERATORS,
    max_work=_MAX_SCHREIER_WORK,
):
    """Return stabilizer generators, orbit size, and budget-completion status."""
    seeds = _deduplicate_generators(generators)
    if not seeds:
        return (), 1, True
    size = len(seeds[0])
    identity = _identity(size)
    transversals = {point: identity}
    pending = [point]
    work = 0
    while pending:
        current = pending.pop()
        current_transversal = transversals[current]
        for generator in seeds:
            work += 1
            if work > max_work:
                return (), len(transversals), False
            image = generator[current]
            if image not in transversals:
                transversals[image] = _compose(current_transversal, generator)
                pending.append(image)

    stabilizers = {}
    for current, current_transversal in transversals.items():
        for generator in seeds:
            image = generator[current]
            schreier = _compose(
                _compose(current_transversal, generator),
                _inverse(transversals[image]),
            )
            if schreier == identity:
                continue
            stabilizers.setdefault(schreier, None)
            if len(stabilizers) > max_generators:
                return (), len(transversals), False
    return tuple(stabilizers), len(transversals), True


def point_stabilizer_generators_checked(
    generators,
    point,
    *,
    max_generators=_MAX_STABILIZER_GENERATORS,
    max_work=_MAX_SCHREIER_WORK,
):
    """Return ``(generators, complete)`` for the subgroup fixing ``point``.

    ``complete=False`` means the bounded Schreier construction deliberately
    disabled deeper symmetry pruning.  Callers that only need safe pruning may
    ignore the flag; callers deriving orbit multiplicities must retain it.
    """
    stabilizers, _, complete = _point_stabilizer_data(
        generators,
        point,
        max_generators=max_generators,
        max_work=max_work,
    )
    return stabilizers, complete


def point_stabilizer_generators(
    generators,
    point,
    *,
    max_generators=_MAX_STABILIZER_GENERATORS,
    max_work=_MAX_SCHREIER_WORK,
):
    """Return Schreier generators fixing ``point``, or identity on budget exit.

    Returning no generators on a work-budget exit is conservative: subsequent
    levels perform less symmetry pruning but cannot lose a mapping.
    """
    stabilizers, _ = point_stabilizer_generators_checked(
        generators,
        point,
        max_generators=max_generators,
        max_work=max_work,
    )
    return stabilizers


def permutation_group_order(
    generators,
    *,
    max_generators=_MAX_STABILIZER_GENERATORS,
    max_work=_MAX_SCHREIER_WORK,
):
    """Return the generated subgroup order without enumerating its elements.

    Orbit--stabilizer is applied along a deterministic point chain.  ``None``
    is returned on a Schreier work-budget exit, so callers never promote a
    partial stabilizer calculation to an exact labeled-solution count.
    """
    stabilizers = _deduplicate_generators(generators)
    if not stabilizers:
        return 1
    size = len(stabilizers[0])
    order = 1
    for point in range(size):
        stabilizers, orbit_size, complete = _point_stabilizer_data(
            stabilizers,
            point,
            max_generators=max_generators,
            max_work=max_work,
        )
        if not complete:
            return None
        order *= orbit_size
        if not stabilizers:
            break
    return order


def orbital_candidate_witnesses(candidates, generators):
    """Map candidates to ``None`` representatives or exact pruning witnesses."""
    candidates = tuple(sorted(candidates))
    seeds = _deduplicate_generators(generators)
    if not seeds:
        return {candidate: None for candidate in candidates}

    size = len(seeds[0])
    identity = _identity(size)
    candidate_set = set(candidates)
    seen = set()
    witnesses = {}
    for representative in candidates:
        if representative in seen:
            continue
        transversals = {representative: identity}
        pending = [representative]
        while pending:
            current = pending.pop()
            current_transversal = transversals[current]
            for generator in seeds:
                image = generator[current]
                if image in transversals:
                    continue
                transversals[image] = _compose(current_transversal, generator)
                pending.append(image)
        orbit = candidate_set.intersection(transversals)
        seen.update(orbit)
        witnesses[representative] = None
        for candidate in orbit:
            if candidate != representative:
                # The inverse transporter maps the rejected candidate to the
                # least representative while fixing the existing prefix.
                witnesses[candidate] = _inverse(transversals[candidate])
    return witnesses


def largest_cyclic_subgroup(permutations, *, max_order=256):
    """Return the largest fully enumerated cyclic subgroup found safely."""
    permutations = tuple(tuple(permutation) for permutation in permutations)
    if not permutations:
        return ()
    identity = _identity(len(permutations[0]))
    best = (identity,)
    for generator in permutations:
        powers = [identity]
        current = identity
        while len(powers) <= max_order:
            current = _compose(current, generator)
            if current == identity:
                if len(powers) > len(best):
                    best = tuple(powers)
                break
            if current in powers:
                break
            powers.append(current)
    return best


__all__ = [
    "largest_cyclic_subgroup",
    "orbital_candidate_witnesses",
    "permutation_group_order",
    "point_stabilizer_generators",
    "point_stabilizer_generators_checked",
]
