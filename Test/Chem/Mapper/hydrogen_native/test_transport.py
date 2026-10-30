import math
from collections import Counter
from itertools import permutations

import pytest

from synkit.Chem.Mapper._native.hydrogen import enumerate_hydrogen_transfers
from synkit.Chem.Mapper._native.runtime import CancellationToken


def oracle(rows, columns, reservoir):
    n = len(rows)
    left = [i for i, count in enumerate(rows) for _ in range(count)]
    right = [i for i, count in enumerate(columns) for _ in range(count)]
    result = {}
    for mapping in permutations(range(len(right))):
        cells = [0] * (n * n)
        cost = 0
        for source, image in enumerate(mapping):
            i, j = left[source], right[image]
            cells[i * n + j] += 1
            if i != j:
                cost += int(i != reservoir) + int(j != reservoir)
        result.setdefault(cost, Counter())[tuple(cells)] += 1
    return result


def cells(plan):
    n = len(plan.preserved)
    result = [0] * (n * n)
    for i, count in enumerate(plan.preserved):
        result[i * n + i] = count
    for i, j, count in plan.transfers:
        result[i * n + j] = count
    return tuple(result)


@pytest.mark.parametrize(
    "rows,columns,reservoir",
    [
        ([2, 1, 0], [0, 1, 2], None),
        ([1, 1], [1, 1], None),
        ([2, 2, 0, 0], [0, 0, 1, 3], 3),
        ([0, 0, 1, 3], [2, 2, 0, 0], 3),
        ([2, 0, 1], [0, 2, 1], 2),
        ([0, 0, 0], [0, 0, 0], 2),
    ],
)
def test_complete_shells_and_exact_multiplicities(
    symmetry_library, rows, columns, reservoir
):
    expected = oracle(rows, columns, reservoir)
    minimum = min(expected)
    for CD, mode in [
        ("minimal", "exact"),
        *[(cost, "exact") for cost in range(max(expected) + 2)],
        *[(cost, "at_most") for cost in range(max(expected) + 1)],
    ]:
        result = enumerate_hydrogen_transfers(
            rows,
            columns,
            library_path=symmetry_library,
            reservoir=reservoir,
            CD=CD,
            CD_mode=mode,
        )
        selected = (
            [minimum]
            if CD == "minimal"
            else (
                [cost for cost in expected if cost == CD]
                if mode == "exact"
                else [cost for cost in expected if cost <= CD]
            )
        )
        target = Counter()
        for cost in selected:
            target.update(expected[cost])
        actual = {cells(plan): plan.labeled_mapping_count for plan in result.plans}
        assert result.complete and result.minimum_distance == minimum
        assert actual == dict(target)
        assert result.labeled_mapping_count == sum(target.values())


def test_arbitrary_precision_multiplicity(symmetry_library):
    result = enumerate_hydrogen_transfers([30], [30], library_path=symmetry_library)
    assert result.complete and result.labeled_mapping_count == math.factorial(30)
    result = enumerate_hydrogen_transfers([2000], [2000], library_path=symmetry_library)
    assert result.complete and result.labeled_mapping_count == math.factorial(2000)


def test_shell_lattice_and_interruption(symmetry_library):
    for query in (0.25, 0.5, 1.25):
        result = enumerate_hydrogen_transfers(
            [1, 1], [1, 1], library_path=symmetry_library, CD=query
        )
        assert result.complete and not result.plans
    for kwargs, reason in [
        ({"time_limit_seconds": 0}, "time_limit"),
        ({"max_nodes": 1}, "max_states"),
        ({"max_plans": 1}, "plan_limit"),
        ({"plan_callback": lambda _: True}, "callback"),
    ]:
        result = enumerate_hydrogen_transfers(
            [2, 0, 1], [0, 2, 1], library_path=symmetry_library, reservoir=2, **kwargs
        )
        assert not result.complete and result.truncation_reason == reason
        assert result.minimum_distance == 4
    with CancellationToken(symmetry_library) as token:
        token.cancel()
        result = enumerate_hydrogen_transfers(
            [1], [1], library_path=symmetry_library, cancel_token=token
        )
        assert not result.complete and result.truncation_reason == "cancelled"

    def fail(_):
        raise LookupError("H plan callback")

    with pytest.raises(LookupError, match="H plan callback"):
        enumerate_hydrogen_transfers(
            [1], [1], library_path=symmetry_library, plan_callback=fail
        )
