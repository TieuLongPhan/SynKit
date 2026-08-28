import itertools

import pytest

from synkit.Chem.Mapper.exact.hydrogen import (
    enumerate_minimal_hydrogen_transfers,
    summarize_minimal_hydrogen_lifts,
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


def test_closed_form_summary_matches_all_small_flow_enumerations():
    for total in range(5):
        compositions = tuple(_weak_compositions(total, 3))
        for reactant, product in itertools.product(compositions, repeat=2):
            enumerated = enumerate_minimal_hydrogen_transfers(
                reactant,
                product,
                [0, 1, 2],
                collect_plans=False,
            )
            summary = summarize_minimal_hydrogen_lifts(
                reactant,
                product,
                [0, 1, 2],
            )
            assert summary.minimum_distance == enumerated.minimum_distance
            assert summary.transferred_hydrogen_count * 2 == summary.minimum_distance
            assert summary.labeled_mapping_count == enumerated.labeled_mapping_count


def test_heavy_only_optimum_need_not_minimize_full_distance_after_h_lifting():
    # Identical heavy paths have zero-CD identity and reversal maps.  With
    # r=(3,0,0) and p=(0,3,0), each has hydrogen distance 6.  The non-isomorphic
    # heavy permutation below costs 2 but preserves every H parent, so its full
    # additive distance is 2 rather than 6.
    edges = {(0, 1), (1, 2)}

    def heavy_distance(mapping):
        transported = {
            tuple(sorted((mapping[left], mapping[right]))) for left, right in edges
        }
        return len(edges.symmetric_difference(transported))

    scores = {}
    heavy_scores = {}
    for mapping in itertools.permutations(range(3)):
        heavy_scores[mapping] = heavy_distance(mapping)
        hydrogen = summarize_minimal_hydrogen_lifts(
            [3, 0, 0],
            [0, 3, 0],
            mapping,
        )
        scores[mapping] = heavy_scores[mapping] + hydrogen.minimum_distance

    assert min(heavy_scores.values()) == 0
    assert {
        scores[mapping] for mapping, score in heavy_scores.items() if score == 0
    } == {6}
    assert heavy_scores[(1, 0, 2)] == 2
    assert scores[(1, 0, 2)] == min(scores.values()) == 2


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
