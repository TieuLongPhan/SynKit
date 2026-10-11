import itertools
from collections import Counter


from synkit.Chem.Mapper import (
    GlobalShellConfig,
    analyze_reference_blinded_global_shell,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.slap.lap import chemical_distance


def _graph(size, edges, labels=None):
    adjacency = {atom: {} for atom in range(size)}
    for left, right in edges:
        adjacency[left][right] = 1
        adjacency[right][left] = 1
    return LabeledGraph(adjacency, labels or [6] * size)


def _cycle_graph(size, permutation=None):
    permutation = list(range(size)) if permutation is None else permutation
    edges = [
        (permutation[atom], permutation[(atom + 1) % size]) for atom in range(size)
    ]
    return _graph(size, edges)


def _config(**kwargs):
    return GlobalShellConfig(
        binary=True,
        max_bijections=None,
        max_mappings=None,
        time_limit_seconds=5,
        **kwargs,
    )


def test_seed_repair_lowers_a_verified_feasible_cost_without_a_reference(monkeypatch):
    from synkit.Chem.Mapper import analysis

    lgp = [_graph(3, ((0, 1),)), _graph(3, ((1, 2),))]
    monkeypatch.setattr(analysis, "recover_mapping", lambda graphs: [0, 1, 2])
    original, old = analysis._reference_free_slap_seed(lgp, False)
    repaired, new = analysis._reference_free_slap_seed(lgp, False, repair=True)
    assert old["cost"] == 2
    assert new["initial_cost"] == old["cost"]
    assert new["cost"] == chemical_distance(lgp, repaired, binary=False) == 0
    assert sorted(repaired) == sorted(original) == [0, 1, 2]
    assert new["repair"]["improved"]


def test_seed_repair_failure_retains_the_original_feasible_mapping(monkeypatch):
    from synkit.Chem.Mapper import analysis
    from synkit.Chem.Mapper.exact import enumerate as enumeration

    lgp = [_graph(3, ((0, 1),)), _graph(3, ((1, 2),))]
    monkeypatch.setattr(analysis, "recover_mapping", lambda graphs: [0, 1, 2])

    def failed_repair(*args):
        raise RuntimeError("repair unavailable")

    monkeypatch.setattr(enumeration, "_greedy_local_swap_descent", failed_repair)
    mapping, statistics = analysis._reference_free_slap_seed(lgp, False, repair=True)
    assert mapping == [0, 1, 2]
    assert statistics["available"]
    assert statistics["cost"] == 2
    assert statistics["repair"]["error_type"] == "RuntimeError"


def test_reference_cd_search_is_invariant_to_equal_cost_reference_choice():
    graph = _graph(3, ((0, 1), (1, 2)))
    lgp = [graph, graph.copy()]
    identity = analyze_reference_blinded_global_shell(
        lgp,
        [0, 1, 2],
        target_mode="reference_cd",
        config=_config(),
    )
    reversal = analyze_reference_blinded_global_shell(
        lgp,
        [2, 1, 0],
        target_mode="reference_cd",
        config=_config(),
    )

    assert identity.reference_cd == reversal.reference_cd == 0
    assert identity.complete is reversal.complete is True
    assert identity.reference_class_observed is True
    assert reversal.reference_class_observed is True
    assert identity.reaction_center == reversal.reaction_center
    assert identity.representative_solution_count == 1
    assert identity.labeled_solution_count == 2


def test_reaction_center_properties_always_refine_symmetry_pruning():
    reactant = _graph(3, ((0, 1), (1, 2)))
    product = reactant.copy()
    reactant.set_prop("hcounts", [1, 0, 0])
    product.set_prop("hcounts", [1, 0, 0])

    result = analyze_reference_blinded_global_shell(
        [reactant, product],
        [0, 1, 2],
        target_mode="minimal",
        config=_config(
            reaction_center_properties=("hcounts",),
            symmetry_node_properties=(),
        ),
    )

    # Reversing the path preserves the bond objective but swaps unequal H
    # counts.  It must remain a separate representative for ITS analysis.
    assert result.complete is True
    assert result.representative_solution_count == 2
    assert result.labeled_solution_count == 2


def test_minimal_search_reveals_suboptimal_reference_after_completion():
    permutation = [0, 2, 4, 1, 3]
    lgp = [_cycle_graph(5), _cycle_graph(5, permutation)]

    result = analyze_reference_blinded_global_shell(
        lgp,
        list(range(5)),
        target_mode="minimal",
        config=_config(),
    )

    assert result.complete is True
    assert result.minimum_cost == 0
    assert result.reference_cd > result.minimum_cost
    assert result.reference_gap_from_minimum == result.reference_cd
    assert result.reference_class_observed is False
    assert result.reference_is_global_minimum_proven is False
    assert result.labeled_solution_count == 10


def test_reaction_center_counts_match_direct_labeled_shell():
    graph = _graph(3, ((0, 1), (1, 2)))
    lgp = [graph, graph.copy()]
    reference = [0, 2, 1]
    target = chemical_distance(lgp, reference, binary=True)
    expected = [
        permutation
        for permutation in itertools.permutations(range(3))
        if chemical_distance(lgp, permutation, binary=True) == target
    ]
    bond_counts = Counter()
    for mapping in expected:
        for left, right in itertools.combinations(range(3), 2):
            before = int(right in graph.graph[left])
            after = int(mapping[right] in graph.graph[mapping[left]])
            if before != after:
                bond_counts[(left, right)] += 1

    result = analyze_reference_blinded_global_shell(
        lgp,
        reference,
        target_mode="reference_cd",
        config=_config(symmetry_pruning=False),
    )

    observed = {
        (left, right): count
        for left, right, count in result.reaction_center.bond_change_counts
    }
    assert result.complete is True
    assert result.backend == "binary_edit_support"
    assert result.representative_solution_count == len(expected)
    assert observed == bond_counts
    assert set(result.reaction_center.bond_intersection) == {
        pair for pair, count in bond_counts.items() if count == len(expected)
    }


def test_zero_time_limit_is_explicitly_incomplete():
    graph = _graph(4, ((0, 1), (1, 2), (2, 3)))
    result = analyze_reference_blinded_global_shell(
        [graph, graph.copy()],
        list(range(4)),
        target_mode="minimal",
        config=GlobalShellConfig(
            binary=True,
            max_bijections=None,
            time_limit_seconds=0,
        ),
    )

    assert result.complete is False
    assert result.status == "timeout"
    assert result.representative_solution_count == 0
    assert result.reference_is_global_minimum_proven is False


def test_internal_transport_fingerprint_preserves_relabeling_and_unary_properties():
    import numpy as np
    from synkit.Chem.Mapper.analysis import _transport_sha256

    product = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]], dtype=float)
    props = {"charges": ([0, 0, 0], [0, 1, 2])}
    mapping = [2, 0, 1]
    permutation = [1, 2, 0]
    inverse = np.argsort(permutation)
    relabeled_props = {
        "charges": ([0, 0, 0], [props["charges"][1][i] for i in permutation])
    }
    original = _transport_sha256(product, props, mapping)
    assert original == _transport_sha256(
        product[np.ix_(permutation, permutation)], relabeled_props, inverse[mapping]
    )
    assert original != _transport_sha256(
        product, {"charges": ([0, 0, 0], [0, 1, 3])}, mapping
    )


def test_vectorized_reaction_center_counts_match_scalar_tolerance_checks():
    import math
    import numpy as np
    from synkit.Chem.Mapper.analysis import _BlindShellObserver

    a = np.array([[0, 0.1, 0, 1], [0.1, 0, 0.5, 0], [0, 0.5, 0, 1.5], [1, 0, 1.5, 0]])
    b = a.copy()
    b[0, 1] = b[1, 0] = 0.1 + 5e-10
    props = {"charges": ([0, 0, 0, 0], [0, 1, 0, 0])}
    for tolerance in (0, 1e-9, 0.5):
        observer = _BlindShellObserver(
            a,
            b,
            [6] * 4,
            props,
            GlobalShellConfig(tolerance=tolerance, structure_analysis=False),
        )
        bonds, atoms = Counter(), Counter()
        for mapping in itertools.permutations(range(4)):
            observer.observe(mapping, 0)
            for i in range(4):
                if props["charges"][0][i] != props["charges"][1][mapping[i]]:
                    atoms[i] += 1
                for j in range(i + 1, 4):
                    if not math.isclose(
                        float(a[i, j]),
                        float(b[mapping[i], mapping[j]]),
                        abs_tol=tolerance,
                        rel_tol=0,
                    ):
                        bonds[(i, j)] += 1
        assert observer.bond_counts == bonds
        assert observer.atom_counts == atoms
