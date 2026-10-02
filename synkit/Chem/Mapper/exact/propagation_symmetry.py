"""Budgeted verified cyclic-subgroup selection for Synister-CP."""

from .propagation_limits import PropagationDeadline, check_deadline


def bounded_cyclic_subgroup(permutations, *, max_order, deadline):
    """Retain a fully closed subgroup; interruption never returns partial powers."""
    if not permutations:
        return ()
    identity = tuple(range(len(permutations[0])))
    best = (identity,)
    for generator in permutations:
        check_deadline(deadline)
        powers, seen, current = [identity], {identity}, identity
        while len(powers) <= max_order:
            check_deadline(deadline)
            current = tuple(generator[image] for image in current)
            if current == identity:
                if len(powers) > len(best):
                    best = tuple(powers)
                break
            if current in seen:
                break
            seen.add(current)
            powers.append(current)
    return best


def bounded_generated_subgroup(permutations, *, max_order, deadline):
    """Close verified generators, retaining only fully completed subgroups."""
    if not permutations:
        return ()
    identity = tuple(range(len(permutations[0])))
    subgroup = {identity}
    try:
        for generator in permutations:
            check_deadline(deadline)
            if generator in subgroup:
                continue
            inverse = [0] * len(generator)
            for index, image in enumerate(generator):
                inverse[image] = index
            inverse = tuple(inverse)
            closure = set(subgroup)
            pending = list(subgroup)
            exceeded = False
            factors = (*subgroup, generator, inverse)
            while pending and not exceeded:
                check_deadline(deadline)
                current = pending.pop()
                for factor in factors:
                    product = tuple(
                        current[factor[index]] for index in range(len(identity))
                    )
                    if product in closure:
                        continue
                    if len(closure) >= max_order:
                        exceeded = True
                        break
                    closure.add(product)
                    pending.append(product)
            if not exceeded:
                subgroup = closure
    except PropagationDeadline:
        pass
    return tuple(sorted(subgroup))
