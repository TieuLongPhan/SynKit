from synkit.Chem.Mapper.chem.smiles import smiles2lgp
from synkit.Chem.Mapper.graph.automorphism import (
    bounded_automorphism_permutations,
    node_orbits,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def test_automorphism_node_orbits_cover_all_nodes():
    graph = smiles2lgp("CC>>CC", add_Hs=False)[0]

    covered = set().union(*node_orbits(graph))

    assert covered == set(range(len(graph.labels)))


def test_symmetry_node_properties_refine_safe_subgroup():
    graph = LabeledGraph({0: {}, 1: {}}, [6, 6])
    graph.set_prop("atomic numbers", [6, 6])
    graph.set_prop("hcounts", [4, 0])
    graph.set_prop("charges", [0, 0])

    plain, _ = bounded_automorphism_permutations(graph, limit=8)
    hydrogen_refined, _ = bounded_automorphism_permutations(
        graph,
        limit=8,
        node_properties=("hcounts", "charges"),
    )

    assert len(plain) == 2
    assert hydrogen_refined == ((0, 1),)


def test_bounded_search_recovers_verified_twins_after_timeout():
    from synkit.Chem.Mapper.exact.symmetry import permutation_group_order
    from synkit.Chem.Mapper.graph.automorphism import (
        _is_exact_automorphism,
        _objective_graph_data,
    )

    graph = smiles2lgp("CC(C)C>>CC(C)C", add_Hs=False)[0]
    permutations, complete = bounded_automorphism_permutations(
        graph, timeout_seconds=0
    )
    matrix, elements = _objective_graph_data(graph)
    assert complete is False
    assert permutation_group_order(permutations[1:]) == 6
    assert all(_is_exact_automorphism(p, matrix, elements) for p in permutations)
    limited, complete = bounded_automorphism_permutations(
        graph, timeout_seconds=0, limit=2
    )
    assert len(limited) == 2
    assert not complete


def test_twin_fallback_respects_incoming_weights_and_node_properties():
    from synkit.Chem.Mapper.graph.automorphism import _verified_twin_permutations

    # Equal outgoing rows alone do not establish directed-graph symmetry.
    directed = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 2.0, 0.0))
    assert _verified_twin_permutations(directed, (6, 6, 8), 8) == ((0, 1, 2),)
    graph = LabeledGraph({0: {}, 1: {}, 2: {}}, [6, 6, 6])
    graph.set_prop("hcounts", [4, 4, 0])
    permutations, complete = bounded_automorphism_permutations(
        graph, timeout_seconds=0, node_properties=("hcounts",)
    )
    assert permutations == ((0, 1, 2), (1, 0, 2))
    assert not complete
    graph.props["hcounts"][1] = 1
    permutations, _ = bounded_automorphism_permutations(
        graph, timeout_seconds=0, node_properties=("hcounts",)
    )
    assert permutations == ((0, 1, 2),)


def test_twin_fallback_preserves_exhaustive_mappings_and_certificate():
    from synkit.Chem.Mapper.exact.distance import (
        enumerate_distance_mappings,
        verify_distance_enumeration_certificate,
    )

    pair = smiles2lgp("CC(C)C>>CC(C)C", add_Hs=False)
    baseline = enumerate_distance_mappings(pair, CD=0, binary=False)
    reduced = enumerate_distance_mappings(
        pair, CD=0, binary=False, symmetry_pruning=True,
        symmetry_timeout_seconds=0, certify=True,
    )
    assert reduced.complete
    assert reduced.symmetry_group_order == 6
    assert reduced.selected_labeled_mapping_count == len(baseline.mappings)
    assert len(reduced.mappings) == 1
    verify_distance_enumeration_certificate(pair, reduced.certificate)
    expanded = enumerate_distance_mappings(
        pair, CD=0, binary=False, symmetry_pruning=True,
        symmetry_timeout_seconds=0, expand_symmetry=True,
    )
    assert set(map(tuple, expanded.mappings)) == set(map(tuple, baseline.mappings))


def test_component_symmetry_matches_independent_graph_isomorphisms():
    import networkx as nx

    from synkit.Chem.Mapper.exact.symmetry import permutation_group_order
    from synkit.Chem.Mapper.graph.automorphism import (
        _is_exact_automorphism,
        _objective_graph_data,
        to_nx,
    )

    # Equal components have internal symmetries and can also exchange places.
    graph = smiles2lgp("C1CC1.C1CC1>>C1CC1.C1CC1", add_Hs=False)[0]
    permutations, complete = bounded_automorphism_permutations(
        graph, limit=64, timeout_seconds=1
    )
    nx_graph = to_nx(graph)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        nx_graph, nx_graph,
        node_match=lambda a, b: a["element"] == b["element"],
        edge_match=lambda a, b: a["order"] == b["order"],
    )
    expected = sum(1 for _ in matcher.isomorphisms_iter())
    matrix, elements = _objective_graph_data(graph)
    assert complete
    assert expected == 72
    assert permutation_group_order(permutations[1:]) == expected
    assert all(_is_exact_automorphism(p, matrix, elements) for p in permutations)
    limited, complete = bounded_automorphism_permutations(
        graph, limit=2, timeout_seconds=1
    )
    assert len(limited) <= 2
    assert not complete


def test_component_symmetry_preserves_fixed_subspace_and_weight_colors():
    from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
    from synkit.Chem.Mapper.exact.symmetry import permutation_group_order

    pair = smiles2lgp("CC.CC>>CC.CC", add_Hs=False)
    plain = enumerate_distance_mappings(pair, CD=0, fixed_mapping={0: 0})
    reduced = enumerate_distance_mappings(
        pair, CD=0, fixed_mapping={0: 0}, symmetry_pruning=True
    )
    assert reduced.complete and reduced.symmetry_quotient_complete
    assert reduced.selected_labeled_mapping_count == len(plain.mappings) == 2
    graph = LabeledGraph(
        {0: {1: 1}, 1: {0: 1}, 2: {3: 2}, 3: {2: 2}}, [6] * 4
    )
    weighted, _ = bounded_automorphism_permutations(graph, binary=False)
    binary, _ = bounded_automorphism_permutations(graph, binary=True)
    assert permutation_group_order(weighted[1:]) == 4
    assert permutation_group_order(binary[1:]) == 8
