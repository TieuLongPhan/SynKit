"""Independent mapping-set checks for native minimum and bounded-CD adapters."""

from itertools import permutations

import networkx as nx
import numpy as np
import pytest

from Test.Chem.Mapper._helpers import graph
from synkit.Chem.Mapper.exact.native.search import enumerate_cpp_mappings


def _oracle(a, b):
    return {
        mapping: float(np.abs(a - b[np.ix_(mapping, mapping)]).sum() / 2)
        for mapping in permutations(range(len(a)))
    }


@pytest.mark.parametrize("seed", range(8))
def test_minimum_and_cost_ranges_match_raw_permutations(native_library, seed):
    rng = np.random.default_rng(61025 + seed)
    matrices = []
    for _ in range(2):
        upper = np.triu(rng.choice([-0.5, 0.0, 0.5, 1.5], (5, 5)), 1)
        matrices.append(upper + upper.T)
    a, b = matrices
    pair = graph(a), graph(b)
    expected = _oracle(a, b)
    best = min(expected.values())
    result = enumerate_cpp_mappings(
        pair, library_path=native_library, binary=False
    )
    assert result.complete and result.minimum_cost == best
    assert {tuple(m) for m in result.mappings} == {
        m for m, cost in expected.items() if cost == best
    }
    assert result.distances == [best] * len(result.mappings)
    assert result.backend_statistics["minimum_shell_probes"] == 0
    assert result.backend_statistics["minimum_optimization"]["complete"]

    costs = sorted(set(expected.values()))
    target = costs[len(costs) // 2]
    for mode, threshold in (("exact", target), ("at_most", target + 0.1)):
        selected = {
            m: cost
            for m, cost in expected.items()
            if (cost == threshold if mode == "exact" else cost <= threshold)
        }
        found = enumerate_cpp_mappings(
            pair,
            library_path=native_library,
            binary=False,
            CD=threshold,
            CD_mode=mode,
            compute_minimum_cost=False,
        )
        assert found.complete and found.minimum_cost is None
        assert dict(zip(map(tuple, found.mappings), found.distances)) == selected
        assert found.selected_labeled_mapping_count == len(selected)


def test_bounded_callbacks_report_actual_cost_without_collecting(native_library):
    a = nx.to_numpy_array(nx.path_graph(4))
    b = nx.to_numpy_array(nx.star_graph(3))
    expected = _oracle(a, b)
    received = {}
    result = enumerate_cpp_mappings(
        (graph(a), graph(b)),
        library_path=native_library,
        CD=max(expected.values()),
        CD_mode="at_most",
        binary=False,
        compute_minimum_cost=False,
        collect_mappings=False,
        mapping_callback=lambda mapping, cost: received.__setitem__(tuple(mapping), cost),
    )
    assert result.complete and received == expected
    assert len(set(received.values())) > 1
    assert result.mappings == result.distances == []
    assert result.cost is None
    assert result.selected_mapping_count == len(expected)


@pytest.mark.parametrize("kind", ["cycle", "star", "path"])
def test_symmetry_candidates_cover_every_exact_class(native_library, kind):
    a = nx.to_numpy_array(
        {"cycle": nx.cycle_graph(5), "star": nx.star_graph(4), "path": nx.path_graph(5)}[kind]
    )
    b = nx.to_numpy_array(nx.path_graph(5))
    oracle = _oracle(a, b)
    automorphisms = [
        m for m in oracle if np.array_equal(a, a[np.ix_(m, m)])
    ]

    def key(mapping):
        transported = b[np.ix_(mapping, mapping)]
        return min(transported[np.ix_(m, m)].tobytes() for m in automorphisms)

    for mode, target in (("exact", min(oracle.values())), ("at_most", max(oracle.values()))):
        result = enumerate_cpp_mappings(
            (graph(a), graph(b)),
            library_path=native_library,
            CD=target,
            CD_mode=mode,
            symmetry=True,
            node_properties=(),
            compute_minimum_cost=False,
        )
        selected = [m for m, c in oracle.items() if c == target or mode == "at_most"]
        assert result.complete
        assert {key(m) for m in result.mappings} == {key(m) for m in selected}
        assert result.selected_mapping_count < len(selected)
        assert result.selected_labeled_mapping_count is None
        assert "modulo_verified_side_symmetry" in result.scope
        assert result.symmetry_search_complete


def test_minimum_and_enumeration_completion_are_independent(native_library):
    a = nx.to_numpy_array(nx.path_graph(4))
    pair = graph(a), graph(a)
    interrupted = enumerate_cpp_mappings(
        pair, library_path=native_library, time_limit_seconds=0
    )
    assert not interrupted.complete and interrupted.minimum_cost is None
    assert interrupted.selected_mapping_count == 0
    limited = enumerate_cpp_mappings(
        pair, library_path=native_library, max_mappings=1
    )
    assert limited.minimum_cost == 0 and not limited.complete
    assert limited.truncation_reason == "mapping_limit"


def test_partial_verified_groups_only_weaken_pruning(native_library, monkeypatch):
    import synkit.Chem.Mapper.exact.native.candidates as candidates

    def incomplete_group(*args, **kwargs):
        return [tuple(range(4)), (3, 2, 1, 0)], False

    monkeypatch.setattr(candidates, "bounded_automorphism_permutations", incomplete_group)
    a = nx.to_numpy_array(nx.path_graph(4))
    pair = graph(a), graph(a)
    with pytest.raises(ValueError, match="complete side groups"):
        candidates.prepare_native_candidates(
            pair, 0, library_path=native_library, node_properties=()
        )
    result = enumerate_cpp_mappings(
        pair, library_path=native_library, symmetry=True
    )
    assert result.complete and result.minimum_cost == 0
    assert len(result.mappings) == 1
    assert not result.symmetry_search_complete
    assert not result.symmetry_quotient_complete


@pytest.mark.parametrize("mode,target", [("invalid", 2), ("at_most", "minimal")])
def test_invalid_cost_mode_rejected(native_library, mode, target):
    pair = graph(np.zeros((1, 1))), graph(np.zeros((1, 1)))
    with pytest.raises(ValueError, match="CD_mode"):
        enumerate_cpp_mappings(pair, library_path=native_library, CD=target, CD_mode=mode)


def test_finite_oversized_targets_do_not_overflow(native_library):
    pair = graph(np.zeros((2, 2))), graph(np.zeros((2, 2)))
    exact = enumerate_cpp_mappings(
        pair, library_path=native_library, CD=1e308, compute_minimum_cost=False
    )
    bounded = enumerate_cpp_mappings(
        pair,
        library_path=native_library,
        CD=1e308,
        CD_mode="at_most",
        compute_minimum_cost=False,
    )
    assert exact.complete and exact.mappings == []
    assert bounded.complete and len(bounded.mappings) == 2
