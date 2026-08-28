import itertools

import pytest

from synkit.Chem.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.exact.edit_support import (
    binary_edit_budget,
    enumerate_binary_edit_support_mappings,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.slap.lap import chemical_distance


def _graph(size, edges, labels=None):
    adjacency = {atom: {} for atom in range(size)}
    for left, right in edges:
        adjacency[left][right] = 1
        adjacency[right][left] = 1
    return LabeledGraph(adjacency, labels or [6] * size)


def test_binary_budget_encodes_broken_formed_edges_and_parity():
    lgp = smiles2lgp("CCC>>CCC", add_Hs=False)

    exact = binary_edit_budget(lgp, 2)
    assert exact.feasible is True
    assert exact.broken_edge_count == 1
    assert exact.formed_edge_count == 1
    assert exact.support_pair_count == 4

    odd = binary_edit_budget(lgp, 1)
    assert odd.feasible is False
    assert "parity" in odd.rejection_reason
    fractional = binary_edit_budget(lgp, 1.5)
    assert fractional.feasible is False
    assert fractional.target == 1.5
    assert "integral" in fractional.rejection_reason


def test_edit_support_matches_assignment_shells_for_propane():
    lgp = smiles2lgp("CCC>>CCC", add_Hs=False)
    for target in (0, 2, 4):
        assignment = enumerate_distance_mappings(
            lgp,
            CD=target,
            binary=True,
            max_bijections=None,
            compute_minimum_cost=False,
        )
        edit_support = enumerate_binary_edit_support_mappings(lgp, target)
        assert edit_support.complete is True
        assert {tuple(mapping) for mapping in edit_support.mappings} == {
            tuple(mapping) for mapping in assignment.mappings
        }


def test_edit_support_matches_brute_force_on_coloured_small_graphs():
    labels = [6, 6, 6, 8, 8]
    reactant = _graph(5, ((0, 1), (1, 2), (2, 3), (2, 4)), labels)
    product = _graph(5, ((0, 2), (1, 2), (1, 3), (1, 4)), labels)
    lgp = [reactant, product]
    compatible = [
        permutation
        for permutation in itertools.permutations(range(5))
        if all(labels[atom] == labels[image] for atom, image in enumerate(permutation))
    ]
    brute = {
        permutation: chemical_distance(lgp, permutation, binary=True)
        for permutation in compatible
    }

    for target in sorted(set(brute.values())):
        result = enumerate_binary_edit_support_mappings(
            lgp,
            target,
            max_support_pairs=None,
        )
        assert result.complete is True
        assert {tuple(mapping) for mapping in result.mappings} == {
            permutation for permutation, cost in brute.items() if cost == target
        }


def test_edit_support_exhaustively_matches_all_three_node_graph_pairs():
    possible_edges = tuple(itertools.combinations(range(3), 2))
    graphs = [
        _graph(
            3,
            [edge for index, edge in enumerate(possible_edges) if mask & (1 << index)],
        )
        for mask in range(1 << len(possible_edges))
    ]
    permutations = tuple(itertools.permutations(range(3)))

    for reactant, product in itertools.product(graphs, repeat=2):
        lgp = [reactant, product]
        brute_shells = {}
        for permutation in permutations:
            cost = chemical_distance(lgp, permutation, binary=True)
            brute_shells.setdefault(cost, set()).add(permutation)
        for target, expected in brute_shells.items():
            result = enumerate_binary_edit_support_mappings(
                lgp,
                target,
                max_support_pairs=None,
            )
            assert result.complete is True
            assert {tuple(mapping) for mapping in result.mappings} == expected


def test_edit_support_handles_relabelled_cycle_symmetry():
    permutation = [0, 2, 4, 1, 3]
    reactant_edges = [(atom, (atom + 1) % 5) for atom in range(5)]
    product_edges = [
        (permutation[left], permutation[right]) for left, right in reactant_edges
    ]
    lgp = [_graph(5, reactant_edges), _graph(5, product_edges)]

    result = enumerate_binary_edit_support_mappings(lgp, 0)

    assert result.complete is True
    assert result.selected_mapping_count == 10
    assert permutation in result.mappings


def test_edit_support_limits_and_streaming_are_explicit():
    lgp = smiles2lgp("CCC>>CCC", add_Hs=False)
    refused = enumerate_binary_edit_support_mappings(
        lgp,
        2,
        max_support_pairs=3,
    )
    assert refused.complete is False
    assert refused.truncation_reason == "support_pair_limit"

    streamed = []
    result = enumerate_binary_edit_support_mappings(
        lgp,
        2,
        collect_mappings=False,
        mapping_callback=lambda mapping, cost: streamed.append((mapping, cost)),
    )
    assert result.complete is True
    assert result.mappings == []
    assert result.selected_mapping_count == len(streamed) == 4
    assert {cost for _, cost in streamed} == {2}

    limited = enumerate_binary_edit_support_mappings(lgp, 2, max_mappings=1)
    assert limited.complete is False
    assert limited.truncation_reason == "mapping_limit"

    timed = enumerate_binary_edit_support_mappings(lgp, 2, time_limit_seconds=0)
    assert timed.complete is False
    assert timed.truncation_reason == "time_limit"


def test_edit_support_rejects_directed_inputs():
    directed = LabeledGraph({0: {1: 1}, 1: {}}, [6, 6])
    with pytest.raises(ValueError, match="undirected"):
        enumerate_binary_edit_support_mappings([directed, directed.copy()], 0)


def test_edit_support_uses_binary_bond_presence_for_weighted_inputs():
    weighted = LabeledGraph({0: {1: 2}, 1: {0: 2}}, [6, 6])
    result = enumerate_binary_edit_support_mappings([weighted, weighted.copy()], 0)
    assert result.complete is True
    assert {tuple(mapping) for mapping in result.mappings} == {(0, 1), (1, 0)}
