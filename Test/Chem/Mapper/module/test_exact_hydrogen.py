import itertools

import pytest

from synkit.Chem.Mapper.exact.hydrogen import (
    enumerate_minimal_hydrogen_transfers,
)


def _weak_compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _weak_compositions(total - first, parts - 1):
            yield (first, *rest)


def test_hydrogen_flow_counts_match_all_small_labeled_bijections():
    for total in range(5):
        compositions = tuple(_weak_compositions(total, 3))
        for reactant, product in itertools.product(compositions, repeat=2):
            result = enumerate_minimal_hydrogen_transfers(
                reactant,
                product,
                [0, 1, 2],
                collect_plans=False,
            )
            reactant_parents = tuple(
                parent for parent, count in enumerate(reactant) for _ in range(count)
            )
            product_parents = tuple(
                parent for parent, count in enumerate(product) for _ in range(count)
            )
            distances = [
                2
                * sum(
                    source != product_parents[image]
                    for source, image in zip(reactant_parents, permutation)
                )
                for permutation in itertools.permutations(range(total))
            ]
            minimum = min(distances, default=0)
            assert result.minimum_distance == minimum
            assert result.labeled_mapping_count == sum(
                distance == minimum for distance in distances
            )


def test_single_minimal_hydrogen_transfer_and_labeled_count():
    result = enumerate_minimal_hydrogen_transfers(
        [3, 1],
        [2, 2],
        [0, 1],
    )

    assert result.complete is True
    assert result.minimum_distance == 2
    assert result.observed_plan_count == 1
    assert result.labeled_mapping_count == 12
    assert result.plans[0].preserved == (2, 1)
    assert result.plans[0].transfers == ((0, 1, 1),)
    assert result.plans[0].labeled_mapping_count == 12


def test_multiple_parent_flows_are_enumerated_without_hydrogen_nodes():
    result = enumerate_minimal_hydrogen_transfers(
        [2, 2, 0, 0],
        [0, 0, 2, 2],
        [0, 1, 2, 3],
    )

    assert result.complete is True
    assert result.minimum_distance == 8
    assert result.observed_plan_count == 3
    assert {plan.transfers for plan in result.plans} == {
        ((0, 2, 2), (1, 3, 2)),
        ((0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1)),
        ((0, 3, 2), (1, 2, 2)),
    }


def test_unchanged_hydrogens_have_one_flow_and_local_factorial_multiplicity():
    result = enumerate_minimal_hydrogen_transfers(
        [3, 2],
        [3, 2],
        [0, 1],
    )

    assert result.minimum_distance == 0
    assert result.observed_plan_count == 1
    assert result.plans[0].transfers == ()
    assert result.labeled_mapping_count == 12


def test_hydrogen_plan_cap_and_inventory_validation():
    capped = enumerate_minimal_hydrogen_transfers(
        [2, 2, 0, 0],
        [0, 0, 2, 2],
        [0, 1, 2, 3],
        max_plans=2,
    )
    assert capped.complete is False
    assert capped.observed_plan_count == 2

    with pytest.raises(ValueError, match="inventories differ"):
        enumerate_minimal_hydrogen_transfers([1], [0], [0])
