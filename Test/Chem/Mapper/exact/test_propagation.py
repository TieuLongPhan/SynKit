"""Independent tiny-instance oracles for the opt-in propagation engine."""

import itertools
import random

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.assignment_domains import AssignmentDomains
from synkit.Chem.Mapper.exact.incremental_assignment import solve_assignment
from synkit.Chem.Mapper.exact.propagation import enumerate_synister_cp_mappings
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.slap.lap import chemical_distance


@pytest.mark.parametrize("size", range(1, 7))
def test_weighted_graph_shells_against_all_permutations(size):
    rng = random.Random(2700 + size)
    labels = [6] * (size // 2) + [8] * (size - size // 2)
    graphs = []
    for _ in range(2):
        adjacency = {i: {} for i in range(size)}
        for i in range(size):
            for j in range(i):
                weight = rng.choice([0, 0.5, 1, 1.5])
                if weight:
                    adjacency[i][j] = adjacency[j][i] = weight
        graphs.append(LabeledGraph(adjacency, labels))
    costs = {
        mapping: chemical_distance(graphs, mapping, binary=False)
        for mapping in itertools.permutations(range(size))
        if all(labels[i] == labels[j] for i, j in enumerate(mapping))
    }
    for target in ["minimal", *sorted(set(costs.values()))]:
        expected_cost = min(costs.values()) if target == "minimal" else target
        expected = {m for m, cost in costs.items() if cost == expected_cost}
        result = enumerate_synister_cp_mappings(
            graphs,
            CD=target,
            binary=False,
            max_bijections=None,
            symmetry_pruning=True,
            expand_symmetry=True,
        )
        assert result.complete
        assert set(map(tuple, result.mappings)) == expected
        assert len(result.mappings) == len(expected)


def test_hall_filter_removes_only_unsupported_edges_and_restores():
    masks = [3, 3, 7, 12]
    domains = AssignmentDomains(masks)
    token = domains.checkpoint()
    feasible, _ = domains.propagate(list(range(4)), 15)
    assert feasible
    supported = [0] * 4
    for matching in itertools.permutations(range(4)):
        if all(masks[i] & (1 << j) for i, j in enumerate(matching)):
            for i, j in enumerate(matching):
                supported[i] |= 1 << j
    assert domains.masks == supported
    domains.restore(token)
    assert domains.masks == masks
    assert not AssignmentDomains([3, 3, 3]).propagate([0, 1, 2], 7)[0]


def test_integer_assignment_repair_matches_brute_force_with_changed_costs():
    rng = np.random.default_rng(19)
    parent = None
    for size in range(6, 0, -1):
        costs = rng.integers(0, 40, (size, size))
        allowed = rng.random((size, size)) > 0.35
        np.fill_diagonal(allowed, True)
        rows = tuple(range(size))
        state = solve_assignment(costs, allowed, rows, rows, parent=parent)
        expected = min(
            sum(int(costs[i, j]) for i, j in enumerate(p))
            for p in itertools.permutations(rows)
            if all(allowed[i, j] for i, j in enumerate(p))
        )
        assert state.lower_bound == expected
        parent = state


def test_fixed_constraints_streaming_and_output_cap_are_honest():
    graphs = [LabeledGraph({i: {} for i in range(4)}, [6] * 4)] * 2
    streamed = []
    result = enumerate_synister_cp_mappings(
        graphs,
        fixed_mapping={0: 2},
        collect_mappings=False,
        mapping_callback=lambda mapping, cost: streamed.append(tuple(mapping)),
        symmetry_pruning=True,
        expand_symmetry=True,
        max_mappings=6,
    )
    assert result.complete
    assert len(streamed) == 6
    assert all(m[0] == 2 for m in streamed)
    capped = enumerate_synister_cp_mappings(graphs, max_mappings=2)
    assert not capped.complete
    assert len(capped.mappings) == 2
    timed = enumerate_synister_cp_mappings(graphs, time_limit_seconds=0)
    assert not timed.complete
    assert timed.truncation_reason == "time_limit"


@pytest.mark.parametrize(
    "options",
    [
        {"config": False},
        {"max_symmetry_automorphisms": None},
        {"symmetry_max_search_nodes": None},
    ],
)
def test_invalid_options_do_not_silently_change_search(options):
    graphs = [LabeledGraph({0: {}}, [6])] * 2
    with pytest.raises((TypeError, ValueError)):
        enumerate_synister_cp_mappings(graphs, **options)


def test_generic_nonlattice_graph_uses_explicit_legacy_fallback():
    a = LabeledGraph({0: {1: 0.3}, 1: {0: 0.3}, 2: {}}, [6] * 3)
    b = LabeledGraph({0: {}, 1: {2: 0.3}, 2: {1: 0.3}}, [6] * 3)
    result = enumerate_synister_cp_mappings([a, b], binary=False)
    assert result.complete
    assert result.minimum_cost == 0
    assert result.backend == "synister_cp_legacy_fallback"
    assert len(result.mappings) == 2


def test_signed_half_integer_objective_is_exact():
    a = LabeledGraph({0: {1: -0.5}, 1: {0: -0.5, 2: 1}, 2: {1: 1}}, [6] * 3)
    b = LabeledGraph({0: {2: -1}, 1: {2: 0.5}, 2: {0: -1, 1: 0.5}}, [6] * 3)
    costs = {
        m: chemical_distance([a, b], m, binary=False)
        for m in itertools.permutations(range(3))
    }
    for target in sorted(set(costs.values())):
        result = enumerate_synister_cp_mappings([a, b], CD=target, binary=False)
        assert result.complete
        assert set(map(tuple, result.mappings)) == {
            m for m, c in costs.items() if c == target
        }
