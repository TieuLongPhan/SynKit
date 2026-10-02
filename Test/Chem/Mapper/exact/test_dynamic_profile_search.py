"""Independent checks of prefix-conditioned bounds and dynamic branching."""
from itertools import permutations

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.exact.distance_bounds import atom_profile_costs
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph

def graph(matrix, labels):
    return LabeledGraph(
        {i: {j: float(v) for j, v in enumerate(row) if i != j and v}
         for i, row in enumerate(matrix)}, labels,
    )

@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("weights", [(0, 1, 1.5), (-1, 0, 0.5), (0, 0.1, 0.7)])
def test_dynamic_shells_equal_exhaustive_typed_assignments(directed, weights):
    rng = np.random.default_rng(217)
    labels = [6, 6, 6, 8, 8, 8]
    for _ in range(5):
        a, b = [rng.choice(weights, size=(6, 6)).astype(float) for _ in range(2)]
        if not directed:
            a = np.triu(a, 1)
            a += a.T
            b = np.triu(b, 1)
            b += b.T
        np.fill_diagonal(a, 0)
        np.fill_diagonal(b, 0)
        feasible = [p for p in permutations(range(6))
                    if all(labels[i] == labels[j] for i, j in enumerate(p))]
        values = {p: 0.5 * np.abs(a - b[np.ix_(p, p)]).sum() for p in feasible}
        pair = (graph(a, labels), graph(b, labels))
        shells = sorted(set(values.values()))
        for target in (shells[0], shells[len(shells) // 2], shells[-1]):
            result = enumerate_distance_mappings(
                pair, CD=float(target), binary=False, compute_minimum_cost=False,
                symmetry_pruning=True, max_bijections=None, expand_symmetry=True,
            )
            expected = {p for p, cost in values.items() if abs(cost - target) <= 1e-9}
            assert result.complete
            assert set(map(tuple, result.mappings)) == expected
        fixed = {0: 0}
        best = min(v for p, v in values.items() if p[0] == 0)
        streamed = []
        result = enumerate_distance_mappings(
            pair, CD="minimal", binary=False, max_bijections=None,
            fixed_mapping=fixed, collect_mappings=False,
            mapping_callback=lambda p, cost: streamed.append(tuple(p)),
        )
        assert result.complete
        assert result.minimum_cost == pytest.approx(best)
        assert set(streamed) == {p for p, v in values.items()
                                 if p[0] == 0 and abs(v - best) <= 1e-9}

def test_histogram_profiles_equal_sorted_matching_for_signed_weights():
    rng = np.random.default_rng(812)
    labels = [6] * 9 + [8] * 5
    a, b = [rng.choice([-2, -0.5, 0, 1, 1.5, 3], size=(14, 14))
            for _ in range(2)]
    actual = atom_profile_costs(a, b, labels, labels)
    for i in range(14):
        for j in range(14):
            if labels[i] != labels[j]:
                assert np.isinf(actual[i, j])
                continue
            expected = sum(
                np.abs(np.sort(a[i, np.array(labels) == e])
                       - np.sort(b[j, np.array(labels) == e])).sum()
                for e in set(labels)
            ) / 2
            assert actual[i, j] == expected
