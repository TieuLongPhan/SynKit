"""Tests for exact several-CD enumeration with one shared suffix diagram."""

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.exact.multi_shell import enumerate_pabs_shells
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def _graphs():
    a = np.asarray(
        [[0, 1, 0, 0], [1, 0, 2, 0], [0, 2, 0, 1], [0, 0, 1, 0]],
        dtype=np.int64,
    )
    b = np.asarray(
        [[0, 2, 0, 0], [2, 0, 1, 0], [0, 1, 0, 2], [0, 0, 2, 0]],
        dtype=np.int64,
    )

    def graph(matrix):
        return LabeledGraph(
            {
                i: {j: float(matrix[i, j]) for j in range(len(matrix)) if matrix[i, j]}
                for i in range(len(matrix))
            },
            [6] * len(matrix),
        )

    return [graph(a), graph(b)]


@pytest.mark.parametrize("fixed_mapping", [None, {0: 0}])
def test_multi_shell_spectrum_matches_independent_exact_queries(fixed_mapping):
    graphs = _graphs()
    targets = (0, 1, 2, 3, 4, 5)
    actual = enumerate_pabs_shells(
        graphs,
        targets,
        binary=False,
        fixed_mapping=fixed_mapping,
        max_states=100_000,
    )
    for target in targets:
        reference = enumerate_distance_mappings(
            graphs,
            CD=target,
            binary=False,
            fixed_mapping=fixed_mapping,
            compute_minimum_cost=False,
            max_bijections=None,
        )
        assert actual[target].complete and reference.complete
        assert {tuple(mapping) for mapping in actual[target].mappings} == {
            tuple(mapping) for mapping in reference.mappings
        }
        assert actual[target].distances == reference.distances
        assert actual[target].backend == "pabs_suffix_spectrum"
        assert actual[target].backend_statistics["shared_spectrum_prepared"] is True
    states = {
        result.backend_statistics["spectrum_states"] for result in actual.values()
    }
    assert len(states) == 1


@pytest.mark.parametrize(
    "caps",
    [{"max_states": 1}, {"max_seconds_per_prepare": 1e-12}],
)
def test_multi_shell_cap_falls_back_without_losing_shells(caps):
    graphs = _graphs()
    actual = enumerate_pabs_shells(
        graphs, (2, 3), binary=False, max_bijections=None, **caps
    )
    for target in (2, 3):
        reference = enumerate_distance_mappings(
            graphs,
            CD=target,
            binary=False,
            compute_minimum_cost=False,
            max_bijections=None,
        )
        assert actual[target].complete and reference.complete
        assert {tuple(mapping) for mapping in actual[target].mappings} == {
            tuple(mapping) for mapping in reference.mappings
        }
        assert actual[target].backend_statistics["multi_shell_fallback_reason"] == (
            "spectrum_resource_cap"
        )


def test_multi_shell_output_cap_reports_incomplete_shell_only():
    results = enumerate_pabs_shells(
        _graphs(),
        (0, 3),
        binary=False,
        max_bijections=None,
        max_mappings=1,
    )
    assert results[0].complete
    assert results[3].complete is False
    assert results[3].truncation_reason == "mapping_limit"
    assert len(results[3].mappings) == 1
