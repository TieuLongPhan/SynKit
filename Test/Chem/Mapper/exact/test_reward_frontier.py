"""Literal all-bijection oracles for the sparse reward representation."""

import itertools
import time

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.reward_frontier import RewardFrontierSpectrum
from synkit.Chem.Mapper.exact.propagation_limits import PropagationDeadline
from synkit.Chem.Mapper.exact.propagation import (
    PropagationConfig,
    enumerate_synister_cp_mappings,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def literal(a, b, rows, columns, unary, allowed):
    result = {}
    for permutation in itertools.permutations(range(len(rows))):
        if not all(allowed[i, image] for i, image in enumerate(permutation)):
            continue
        cost = sum(int(unary[i, image]) for i, image in enumerate(permutation))
        cost += sum(
            abs(
                int(a[rows[i], rows[k]])
                - int(b[columns[permutation[i]], columns[permutation[k]]])
            )
            for i in range(len(rows))
            for k in range(i + 1, len(rows))
        )
        result.setdefault(cost, set()).add(tuple(columns[j] for j in permutation))
    return result


@pytest.mark.parametrize("seed", range(24))
def test_signed_weighted_subset_all_shells(seed):
    rng = np.random.default_rng(seed)
    n = 2 + seed % 5
    arrays = []
    for _ in range(2):
        matrix = np.triu(rng.integers(-4, 5, size=(n + 1, n + 1)), 1)
        arrays.append(matrix + matrix.T)
    a, b = arrays
    rows = tuple(int(i) for i in rng.permutation(n + 1)[:n])
    columns = tuple(int(i) for i in rng.permutation(n + 1)[:n])
    unary = rng.integers(0, 7, size=(n, n))
    allowed = rng.random((n, n)) > 0.25
    np.fill_diagonal(allowed, True)
    expected = literal(a, b, rows, columns, unary, allowed)
    cap = max(expected) // 2 if seed % 2 else max(expected)
    diagram = RewardFrontierSpectrum(a, b, rows, columns, unary, allowed, max_cost=cap)
    assert diagram.prepare()
    selected = {cost: mappings for cost, mappings in expected.items() if cost <= cap}
    assert diagram.reachable_costs() == frozenset(selected)
    assert {
        cost: set(diagram.mappings_at(cost)) for cost in diagram.reachable_costs()
    } == selected
    before = diagram.states
    assert set(diagram.mappings_between(0, cap)) == {
        (mapping, cost) for cost, mappings in selected.items() for mapping in mappings
    }
    assert diagram.states == before


def test_product_only_edges_and_forgotten_images():
    n = 7
    a = np.zeros((n, n), dtype=np.int64)
    b = a.copy()
    b[0, 1] = b[1, 0] = -3
    b[3, 4] = b[4, 3] = 2
    diagram = RewardFrontierSpectrum(
        a, b, range(n), range(n), a, np.ones_like(a, dtype=bool), max_cost=5
    )
    assert diagram.prepare()
    assert diagram.max_frontier == 0
    assert diagram.states == 2**n
    assert diagram.reachable_costs() == frozenset({5})
    assert len(set(diagram.mappings_at(5))) == 5040


@pytest.mark.parametrize(
    "options", [{"max_states": 2}, {"max_reward": 1}, {"max_seconds": 1e-12}]
)
def test_budget_failure_is_fallback(options):
    n = 4
    a = np.ones((n, n), dtype=np.int64) - np.eye(n, dtype=np.int64)
    diagram = RewardFrontierSpectrum(
        a, a, range(n), range(n), a, np.ones_like(a, dtype=bool), max_cost=50, **options
    )
    assert not diagram.prepare()
    assert diagram.reachable_costs() == frozenset()
    assert list(diagram.mappings_at(0)) == []


def test_expired_global_deadline_propagates():
    zero = np.zeros((2, 2), dtype=np.int64)
    diagram = RewardFrontierSpectrum(
        zero,
        zero,
        range(2),
        range(2),
        zero,
        np.ones_like(zero, dtype=bool),
        max_cost=0,
        deadline=time.perf_counter() - 1,
    )
    with pytest.raises(PropagationDeadline):
        diagram.prepare()


def test_invalid_domain_and_symmetry_queries():
    zero = np.zeros((3, 3), dtype=np.int64)
    allowed = np.ones_like(zero, dtype=bool)
    allowed[0] = False
    diagram = RewardFrontierSpectrum(
        zero, zero, range(3), range(3), zero, allowed, max_cost=0
    )
    assert diagram.prepare()
    assert diagram.reachable_costs() == frozenset()
    group = tuple(itertools.permutations(range(3)))
    full = RewardFrontierSpectrum(
        zero, zero, range(3), range(3), zero, np.ones_like(zero, dtype=bool), max_cost=0
    )
    assert set(full.mappings_at(0, symmetry_group=group)) == {(0, 1, 2)}
    assert full.orbits_pruned > 0


@pytest.mark.parametrize("fixed", [None, {0: 0}])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("early", [False, True])
def test_pabs_frontier_matches_existing_minimum_and_selected_shells(
    fixed, expand, early
):
    matrices = [
        np.asarray([[0, 1, 0, 0], [1, 0, -1, 0], [0, -1, 0, 1], [0, 0, 1, 0]]),
        np.asarray([[0, 2, 0, 0], [2, 0, 1, 0], [0, 1, 0, 2], [0, 0, 2, 0]]),
    ]
    graphs = [
        LabeledGraph(
            {i: {j: float(m[i, j]) for j in range(4) if m[i, j]} for i in range(4)},
            [6] * 4,
        )
        for m in matrices
    ]
    for target in ("minimal", 3, 5, 7):
        options = dict(
            CD=target,
            binary=False,
            max_bijections=None,
            fixed_mapping=fixed,
            symmetry_pruning=True,
            expand_symmetry=expand,
        )
        baseline = enumerate_synister_cp_mappings(graphs, **options)
        frontier = enumerate_synister_cp_mappings(
            graphs,
            config=PropagationConfig(
                suffix_spectrum=True,
                suffix_spectrum_orbit_pruning=early,
                suffix_spectrum_representation="reward_frontier",
                suffix_spectrum_max_seconds_per_call=0.1,
            ),
            **options,
        )
        assert baseline.complete and frontier.complete
        assert baseline.minimum_cost == frontier.minimum_cost
        assert set(baseline.mappings) == set(frontier.mappings)
        assert baseline.distances == frontier.distances or sorted(
            baseline.distances
        ) == sorted(frontier.distances)


def test_cost_cap_can_certify_empty_support_before_state_cap():
    a = 10 * (np.ones((4, 4), dtype=np.int64) - np.eye(4, dtype=np.int64))
    zero = np.zeros_like(a)
    diagram = RewardFrontierSpectrum(
        a,
        zero,
        range(4),
        range(4),
        zero,
        np.ones_like(a, dtype=bool),
        max_cost=0,
        max_states=5,
    )
    assert diagram.prepare()
    assert diagram.reachable_costs() == frozenset()


def test_frontier_symmetry_expansion_preserves_mapping_cap_status():
    graph = LabeledGraph({i: {} for i in range(4)}, [6] * 4)
    result = enumerate_synister_cp_mappings(
        [graph, graph],
        CD=0,
        binary=False,
        max_bijections=None,
        max_mappings=3,
        compute_minimum_cost=False,
        symmetry_pruning=True,
        expand_symmetry=True,
        config=PropagationConfig(
            suffix_spectrum=True,
            suffix_spectrum_orbit_pruning=True,
            suffix_spectrum_representation="reward_frontier",
            suffix_spectrum_max_seconds_per_call=0.1,
        ),
    )
    assert not result.complete
    assert result.truncation_reason == "mapping_limit"
    assert len(result.mappings) == len(set(result.mappings)) == 3
    assert (
        result.backend_statistics["search"]["suffix_spectrum_reward_frontier_calls"] > 0
    )
