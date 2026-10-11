"""Output and interruption contracts for free product-group expansion."""

from itertools import permutations

import pytest

from synkit.Chem.Mapper.exact.propagation import enumerate_synister_cp_mappings
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


@pytest.mark.parametrize("size", [0, 1])
def test_empty_and_singleton_emission(size):
    graph = LabeledGraph({i: {} for i in range(size)}, [6] * size)
    result = enumerate_synister_cp_mappings(
        [graph, graph], symmetry_pruning=True, expand_symmetry=True, max_mappings=1
    )
    assert result.complete
    assert result.mappings == [tuple(range(size))]
    assert result.selected_mapping_count == 1


@pytest.mark.parametrize("cap", [1, 2, 5, 6, 7, None])
@pytest.mark.parametrize("fixed", [{}, {0: 1}])
@pytest.mark.parametrize(
    "symmetry,expand", [(False, False), (True, False), (True, True)]
)
def test_product_emission_caps_streaming_and_fixed_subspaces(
    cap, fixed, symmetry, expand
):
    graph = LabeledGraph({i: {} for i in range(3)}, [6] * 3)
    expected = {
        p for p in permutations(range(3)) if all(p[i] == j for i, j in fixed.items())
    }
    streamed = []
    result = enumerate_synister_cp_mappings(
        [graph, graph],
        fixed_mapping=fixed,
        symmetry_pruning=symmetry,
        expand_symmetry=expand,
        max_symmetry_automorphisms=6,
        max_mappings=cap,
        mapping_callback=lambda mapping, cost: streamed.append((mapping, cost)),
    )
    total = 1 if symmetry and not expand else len(expected)
    selected = total if cap is None else min(total, cap)
    assert len(streamed) == len(result.mappings) == selected
    assert result.selected_mapping_count == selected
    assert len(set(result.mappings)) == selected
    assert set(result.mappings) <= expected
    assert streamed == [(mapping, 0) for mapping in result.mappings]
    assert result.complete == (cap is None or cap >= total)
    assert result.truncation_reason == (None if result.complete else "mapping_limit")
    if result.complete and not (symmetry and not expand):
        assert set(result.mappings) == expected


def test_deadline_during_product_expansion_stops_before_next_callback(monkeypatch):
    import synkit.Chem.Mapper.exact.propagation_search as search_module

    clock = [0.0]
    monkeypatch.setattr(search_module.time, "perf_counter", lambda: clock[0])
    graph = LabeledGraph({i: {} for i in range(3)}, [6] * 3)
    streamed = []

    def callback(mapping, cost):
        streamed.append(mapping)
        clock[0] = 2.0

    result = enumerate_synister_cp_mappings(
        [graph, graph],
        symmetry_pruning=True,
        expand_symmetry=True,
        time_limit_seconds=1,
        mapping_callback=callback,
    )
    assert not result.complete
    assert result.truncation_reason == "time_limit"
    assert result.selected_mapping_count == 1
    assert result.mappings == streamed


def test_two_sided_nonfree_action_still_deduplicates_at_exact_cap():
    graph = LabeledGraph({i: {} for i in range(3)}, [6] * 3)
    result = enumerate_synister_cp_mappings(
        [graph, graph],
        symmetry_pruning=True,
        reactant_symmetry_pruning=True,
        expand_symmetry=True,
        max_symmetry_automorphisms=6,
        max_mappings=6,
    )
    # S3 x S3 acts nonfreely: 36 group pairs produce only six bijections.
    assert result.complete
    assert set(result.mappings) == set(permutations(range(3)))
    assert len(result.mappings) == result.selected_mapping_count == 6
