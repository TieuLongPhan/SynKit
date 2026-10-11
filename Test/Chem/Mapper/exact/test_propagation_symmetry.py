"""Controls for bounded closure of verified product automorphisms."""

from itertools import permutations

from synkit.Chem.Mapper.exact.propagation import enumerate_synister_cp_mappings
from synkit.Chem.Mapper.exact.propagation_symmetry import bounded_generated_subgroup
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.slap.lap import chemical_distance


def test_independent_swaps_generate_complete_noncyclic_product_subgroup():
    identity = (0, 1, 2, 3)
    left = (1, 0, 2, 3)
    right = (0, 1, 3, 2)
    combined = (1, 0, 3, 2)
    group = bounded_generated_subgroup(
        (identity, left, right), max_order=4, deadline=None
    )
    assert set(group) == {identity, left, right, combined}


def test_noncommuting_generators_close_under_the_entire_generated_group():
    left = (1, 0, 2)
    right = (0, 2, 1)
    group = bounded_generated_subgroup((left, right), max_order=6, deadline=None)
    assert set(group) == set(permutations(range(3)))


def test_overflow_keeps_the_last_fully_closed_subgroup():
    identity = (0, 1, 2, 3)
    left = (1, 0, 2, 3)
    right = (0, 1, 3, 2)
    group = bounded_generated_subgroup((left, right), max_order=3, deadline=None)
    assert set(group) == {identity, left}
    assert len(group) == 2


def test_expired_deadline_never_returns_a_partial_subgroup():
    group = bounded_generated_subgroup(((1, 0),), max_order=2, deadline=0)
    assert group == ((0, 1),)


def test_product_group_expansion_preserves_all_maps_and_fixed_subspaces():
    graph = LabeledGraph({i: {} for i in range(4)}, [6] * 4)
    whole_space = enumerate_synister_cp_mappings(
        [graph, graph],
        symmetry_pruning=True,
        expand_symmetry=True,
        max_symmetry_automorphisms=8,
        time_limit_seconds=3,
    )
    assert whole_space.complete
    assert set(map(tuple, whole_space.mappings)) == set(permutations(range(4)))
    constrained = enumerate_synister_cp_mappings(
        [graph, graph],
        fixed_mapping={0: 1},
        symmetry_pruning=True,
        expand_symmetry=True,
        max_symmetry_automorphisms=8,
        time_limit_seconds=3,
    )
    assert constrained.complete
    assert len(constrained.mappings) == 6
    assert all(mapping[0] == 1 for mapping in constrained.mappings)


def test_two_sided_automorphism_search_preserves_full_indexed_shells():
    reactant = LabeledGraph(
        {0: {1: 1}, 1: {0: 1, 2: 1}, 2: {1: 1, 3: 1}, 3: {2: 1}}, [6] * 4
    )
    product = LabeledGraph(
        {0: {1: 1, 2: 1, 3: 1}, 1: {0: 1}, 2: {0: 1}, 3: {0: 1}}, [6] * 4
    )
    graphs = [reactant, product]
    all_costs = {
        mapping: chemical_distance(graphs, mapping, binary=False)
        for mapping in permutations(range(4))
    }
    for fixed in ({}, {0: 0}):
        costs = {
            mapping: cost
            for mapping, cost in all_costs.items()
            if all(mapping[i] == j for i, j in fixed.items())
        }
        for target in ("minimal", *sorted(set(costs.values()))):
            optimum = min(costs.values()) if target == "minimal" else target
            expected = {mapping for mapping, cost in costs.items() if cost == optimum}
            result = enumerate_synister_cp_mappings(
                graphs,
                CD=target,
                binary=False,
                fixed_mapping=fixed,
                symmetry_pruning=True,
                reactant_symmetry_pruning=True,
                expand_symmetry=True,
                max_symmetry_automorphisms=24,
            )
            assert result.complete
            assert set(map(tuple, result.mappings)) == expected
            assert len(result.mappings) == len(expected)
            if not fixed:
                stats = result.backend_statistics["search"]
                assert stats["reactant_symmetry_group_order"] == 2
                assert stats["reactant_symmetry_pruned"] > 0
