"""Exhaustive bound, configuration, restoration and deadline controls."""

from itertools import permutations, product

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.propagation import (
    PropagationConfig,
    enumerate_synister_cp_mappings,
)
from synkit.Chem.Mapper.exact.propagation_limits import PropagationDeadline
from synkit.Chem.Mapper.exact.residual_typed_bonds import ResidualTypedBonds
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.slap.lap import chemical_distance


def test_typed_signed_mass_bounds_cover_all_residual_permutations_and_restore():
    rng = np.random.default_rng(22)
    labels = [6, 6, 8, 8, 6]
    matrices = []
    for _ in range(2):
        a = np.triu(rng.integers(-3, 4, (5, 5)) * 2, 1)
        matrices.append(a + a.T)
    a, b = matrices
    rows, columns = tuple(range(5)), tuple(range(5))
    bonds = ResidualTypedBonds(a, b, labels, labels, rows, columns)
    original = (
        bonds.reactant.copy(),
        bonds.product.copy(),
        bonds.upper_bound(),
        bonds.transport_lower_bound(),
    )
    trail = []
    while rows:
        costs = [
            int(np.abs(a[np.ix_(rows, rows)] - b[np.ix_(p, p)]).sum() // 2)
            for p in permutations(columns)
            if all(labels[i] == labels[j] for i, j in zip(rows, p))
        ]
        assert bonds.lower_bound() <= min(costs)
        transport = bonds.transport_lower_bound()
        if transport is not None:
            assert bonds.lower_bound() <= transport <= min(costs)
        assert bonds.upper_bound() >= max(costs)
        row = rows[0]
        image = next(j for j in columns if labels[j] == labels[row])
        rows, columns = rows[1:], tuple(j for j in columns if j != image)
        trail.append(bonds.remove(row, image, rows, columns))
    assert bonds.lower_bound() == bonds.upper_bound() == 0
    for token in reversed(trail):
        bonds.restore(token)
    assert np.array_equal(bonds.reactant, original[0])
    assert np.array_equal(bonds.product, original[1])
    assert bonds.upper_bound() == original[2]
    assert bonds.transport_lower_bound() == original[3]


def test_transport_bound_dominates_signed_mass_and_large_alphabets_fall_back():
    # Two typed cross-pairs with weights (0, 4) versus (2, 2): equal signed
    # mass, while sorted L1 transport is four quarter units.
    a = np.zeros((3, 3), dtype=np.int64)
    b = np.zeros((3, 3), dtype=np.int64)
    a[0, 1] = a[1, 0] = 4
    b[0, 2] = b[2, 0] = 2
    b[1, 2] = b[2, 1] = 2
    labels = [6, 6, 6]
    # One type bucket has lists (0, 0, 4) and (0, 2, 2).
    left = ResidualTypedBonds(a, b, labels, labels, range(3), range(3))
    assert left.transport_lower_bound() >= left.lower_bound()
    assert left.transport_lower_bound() <= min(
        sum(abs(a[i, k] - b[p[i], p[k]]) for i, k in ((0, 1), (0, 2), (1, 2)))
        for p in permutations(range(3))
        if all(labels[i] == labels[p[i]] for i in range(3))
    )

    n = 7
    dense_a = np.zeros((n, n), dtype=np.int64)
    dense_b = np.zeros((n, n), dtype=np.int64)
    values = iter(range(1, n * (n - 1) // 2 + 1))
    for i, k in product(range(n), repeat=2):
        if i < k:
            dense_a[i, k] = dense_a[k, i] = next(values)
    values = iter(range(40, 40 + n * (n - 1) // 2))
    for i, k in product(range(n), repeat=2):
        if i < k:
            dense_b[i, k] = dense_b[k, i] = next(values)
    fallback = ResidualTypedBonds(
        dense_a, dense_b, [6] * n, [6] * n, range(n), range(n)
    )
    assert not fallback.histogram_enabled
    assert fallback.transport_lower_bound() is None


def test_transport_histogram_and_forced_batch_keep_literal_sets_for_both_modes():
    labels = [6, 6, 6, 8]
    a = LabeledGraph(
        {0: {1: -0.5}, 1: {0: -0.5, 2: 1}, 2: {1: 1, 3: 1.5}, 3: {2: 1.5}},
        labels,
    )
    b = LabeledGraph(
        {0: {2: 0.5}, 1: {3: 1}, 2: {0: 0.5, 3: -1}, 3: {1: 1, 2: -1}},
        labels,
    )
    costs = {
        p: chemical_distance([a, b], p, binary=False)
        for p in permutations(range(4))
        if all(labels[i] == labels[j] for i, j in enumerate(p))
    }
    minimum = min(costs.values())
    for transport in (False, True):
        for batch in (False, True):
            config = PropagationConfig(
                typed_transport_bounds=transport,
                batch_forced_assignments=batch,
            )
            for target in ("minimal", *sorted(set(costs.values()))):
                expected_cost = minimum if target == "minimal" else target
                expected = {p for p, value in costs.items() if value == expected_cost}
                result = enumerate_synister_cp_mappings(
                    [a, b],
                    CD=target,
                    binary=False,
                    config=config,
                    symmetry_pruning=True,
                    expand_symmetry=True,
                )
                assert result.complete
                assert set(map(tuple, result.mappings)) == expected
                assert len(result.mappings) == len(expected)


def test_bounded_typed_seed_swaps_only_improve_the_feasible_incumbent():
    labels = [6] * 4
    reactant = LabeledGraph(
        {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 1}, 3: {2: 1}}, labels
    )
    product = LabeledGraph(
        {0: {1: 1, 2: 1, 3: 1}, 1: {0: 1}, 2: {0: 1}, 3: {0: 1}}, labels
    )
    result = enumerate_synister_cp_mappings(
        [reactant, product],
        CD="minimal",
        initial_mapping=[0, 1, 2, 3],
        binary=False,
        config=PropagationConfig(seed_local_search=True),
    )
    search = result.backend_statistics["search"]
    assert result.complete and result.minimum_cost == 2
    assert search["seed_swaps"] > 0
    assert search["seed_improvement_quarters"] >= 8
    assert search["seed_improvement_seconds"] <= 0.1


def test_domain_aware_star_assignments_preserve_minimum_and_specific_shells():
    labels = [6] * 4
    reactant = LabeledGraph(
        {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 1}, 3: {2: 1}}, labels
    )
    product = LabeledGraph(
        {0: {1: 1, 2: 1, 3: 1}, 1: {0: 1}, 2: {0: 1}, 3: {0: 1}}, labels
    )
    costs = {
        mapping: chemical_distance([reactant, product], mapping, binary=False)
        for mapping in permutations(range(4))
    }
    for enabled in (False, True):
        config = PropagationConfig(
            star_assignment_bounds=enabled,
            star_bound_residual_limit=4,
            star_bound_max_anchors=16,
            star_bound_seconds=0.1,
        )
        for target in ("minimal", *sorted(set(costs.values()))):
            expected_cost = min(costs.values()) if target == "minimal" else target
            expected = {
                mapping for mapping, cost in costs.items() if cost == expected_cost
            }
            result = enumerate_synister_cp_mappings(
                [reactant, product], CD=target, binary=False, config=config
            )
            assert result.complete
            assert set(map(tuple, result.mappings)) == expected
            assert len(result.mappings) == len(expected)
        stats = result.backend_statistics["search"]
        if enabled:
            assert stats["star_inner_assignments"] > 0
            assert stats["star_improved_entries"] > 0
            assert stats["star_bound_seconds"] <= 0.1
        else:
            assert stats["star_inner_assignments"] == 0


def test_pairwise_edge_bound_preserves_minimum_and_specific_shells():
    labels = [6, 6, 6, 6, 8]
    reactant = LabeledGraph(
        {0: {1: 1, 2: -0.5}, 1: {0: 1, 2: 1}, 2: {0: -0.5, 1: 1, 3: 1}, 3: {2: 1}},
        labels,
    )
    product = LabeledGraph(
        {0: {1: 0.5, 3: 1}, 1: {0: 0.5, 2: 1}, 2: {1: 1}, 3: {0: 1}},
        labels,
    )
    costs = {
        mapping: chemical_distance([reactant, product], mapping, binary=False)
        for mapping in permutations(range(5))
        if labels == [labels[j] for j in mapping]
    }
    targets = ["minimal", *sorted(set(costs.values()))]
    enabled_stats = []
    for enabled in (False, True):
        config = PropagationConfig(
            pairwise_edge_bounds=enabled,
            pairwise_bound_residual_limit=5,
            pairwise_bound_max_anchors=32,
            pairwise_bound_seconds=0.1,
        )
        for target in targets:
            optimum = min(costs.values()) if target == "minimal" else target
            expected = {mapping for mapping, cost in costs.items() if cost == optimum}
            result = enumerate_synister_cp_mappings(
                [reactant, product], CD=target, binary=False, config=config
            )
            assert result.complete
            assert set(map(tuple, result.mappings)) == expected
            assert len(result.mappings) == len(expected)
            if enabled:
                enabled_stats.append(result.backend_statistics["search"])
    assert sum(s["pairwise_bound_anchors"] for s in enabled_stats) > 0
    assert all(s["pairwise_bound_seconds"] <= 0.1 for s in enabled_stats)


def test_all_256_configuration_combinations_keep_exact_fixed_weighted_shells():
    labels = [6, 6, 6, 8]
    a = LabeledGraph(
        {0: {1: -0.5}, 1: {0: -0.5, 2: 1}, 2: {1: 1, 3: 1.5}, 3: {2: 1.5}}, labels
    )
    b = LabeledGraph(
        {0: {2: 0.5}, 1: {3: 1}, 2: {0: 0.5, 3: -1}, 3: {1: 1, 2: -1}}, labels
    )
    costs = {
        p: chemical_distance([a, b], p, binary=False)
        for p in permutations(range(4))
        if p[0] == 1 and all(labels[i] == labels[j] for i, j in enumerate(p))
    }
    flags = [
        "domain_propagation",
        "adaptive_bounds",
        "incremental_assignments",
        "cache_propagation",
        "block_assignments",
        "typed_mass_bounds",
        "column_cost_bounds",
        "cycle_cost_bounds",
    ]
    for values in product([False, True], repeat=len(flags)):
        config = PropagationConfig(**dict(zip(flags, values)))
        for target in ["minimal", *sorted(set(costs.values()))]:
            expected = min(costs.values()) if target == "minimal" else target
            result = enumerate_synister_cp_mappings(
                [a, b],
                CD=target,
                binary=False,
                fixed_mapping={0: 1},
                config=config,
                symmetry_pruning=True,
                expand_symmetry=True,
            )
            assert result.complete
            assert set(map(tuple, result.mappings)) == {
                p for p, c in costs.items() if c == expected
            }


def test_pagerank_branch_order_changes_no_minimum_or_shell_members():
    labels = [6, 6, 6, 8]
    a = LabeledGraph(
        {0: {1: 1}, 1: {0: 1, 2: -0.5}, 2: {1: -0.5}, 3: {}}, labels
    )
    b = LabeledGraph(
        {0: {2: 1}, 1: {3: -0.5}, 2: {0: 1, 3: -0.5}, 3: {1: -0.5, 2: -0.5}},
        labels,
    )
    costs = {
        mapping: chemical_distance([a, b], mapping, binary=False)
        for mapping in permutations(range(4))
        if all(labels[i] == labels[j] for i, j in enumerate(mapping))
    }
    expected = {mapping for mapping, cost in costs.items() if cost == min(costs.values())}
    for branch_order in ("default", "pagerank", "impact", "contention"):
        result = enumerate_synister_cp_mappings(
            [a, b],
            binary=False,
            config=PropagationConfig(branch_order=branch_order),
        )
        assert result.complete and set(map(tuple, result.mappings)) == expected


def test_root_assignment_interruption_does_not_claim_infeasibility(monkeypatch):
    import synkit.Chem.Mapper.exact.incremental_assignment as module

    def interrupt(*args, **kwargs):
        raise PropagationDeadline("simulated root deadline")

    monkeypatch.setattr(module, "_augment", interrupt)
    graphs = [LabeledGraph({0: {}, 1: {}}, [6, 6])] * 2
    result = enumerate_synister_cp_mappings(graphs)
    assert not result.complete and result.minimum_cost is None
    assert result.truncation_reason == "time_limit"
    assert result.backend_statistics["search"]["interrupted_phase"] == "preprocessing"


def test_expired_budget_does_not_start_symmetry_discovery(monkeypatch):
    import synkit.Chem.Mapper.exact.propagation as module

    monkeypatch.setattr(
        module,
        "bounded_automorphism_permutations",
        lambda *a, **k: pytest.fail("budget already expired"),
    )
    graphs = [LabeledGraph({0: {}}, [6])] * 2
    result = enumerate_synister_cp_mappings(
        graphs, symmetry_pruning=True, time_limit_seconds=0
    )
    assert result.truncation_reason == "time_limit"


def test_unbounded_symmetry_timeout_uses_node_cap_when_search_is_unbounded():
    graph = LabeledGraph(
        {0: {1: 1}, 1: {0: 1}, 2: {}},
        [6, 6, 6],
    )
    result = enumerate_synister_cp_mappings(
        [graph, graph],
        symmetry_pruning=True,
        expand_symmetry=True,
        symmetry_timeout_seconds=None,
        symmetry_max_search_nodes=100,
        max_symmetry_automorphisms=16,
    )
    assert result.backend == "synister_cp"
    assert result.complete


def test_cached_branch_neighbor_counts_match_every_literal_prefix(monkeypatch):
    from synkit.Chem.Mapper.exact.propagation_search import PropagatedSearch

    original = PropagatedSearch.select_row
    inspected = []

    def checked(self, rows):
        assigned = np.flatnonzero(np.asarray(self.mapping) >= 0)
        literal = np.count_nonzero(self.a[:, assigned], axis=1)
        assert np.array_equal(self.assigned_neighbors, literal)
        inspected.append(tuple(self.mapping))
        return original(self, rows)

    monkeypatch.setattr(PropagatedSearch, "select_row", checked)
    a = LabeledGraph(
        {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 1}, 3: {2: 1, 4: 1}, 4: {3: 1}},
        [6] * 5,
    )
    b = LabeledGraph(
        {
            0: {1: 1, 4: 1},
            1: {0: 1, 2: 1},
            2: {1: 1, 3: 1},
            3: {2: 1, 4: 1},
            4: {0: 1, 3: 1},
        },
        [6] * 5,
    )
    expected = {
        p
        for p in permutations(range(5))
        if p[0] == 0 and chemical_distance([a, b], p, binary=False) == 3
    }
    result = enumerate_synister_cp_mappings(
        [a, b], CD=3, binary=False, fixed_mapping={0: 0}
    )
    assert result.complete and set(map(tuple, result.mappings)) == expected
    # Forced singleton batching compresses deterministic prefixes, so fewer
    # branching calls are expected while every observed counter remains exact.
    assert inspected
