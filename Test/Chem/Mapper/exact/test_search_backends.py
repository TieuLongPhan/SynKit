"""Compare both public backends against an independent permutation oracle."""

from itertools import permutations

import numpy as np
import pytest

from Test.Chem.Mapper._helpers import graph
from synkit.Chem.Mapper import enumerate_pabs_mappings
from synkit.Chem.Mapper.slap.lap import chemical_distance
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("seed", range(4))
def test_backends_match_all_weighted_shells(native_library, binary, seed):
    rng = np.random.default_rng(seed)
    endpoints = []
    for _ in range(2):
        a = np.triu(rng.choice([-0.5, 0, 0.5, 1.5], (4, 4)), 1)
        endpoints.append(graph(a + a.T))
    shells = {}
    for mapping in permutations(range(4)):
        cost = chemical_distance(endpoints, mapping, binary)
        shells.setdefault(cost, set()).add(mapping)
    for target in ["minimal", *sorted(shells), max(shells) + 0.5, 0.25]:
        expected = (
            shells[min(shells)] if target == "minimal" else shells.get(target, set())
        )
        for backend in ("python", "cpp"):
            options = {"library_path": native_library} if backend == "cpp" else {}
            result = enumerate_pabs_mappings(
                endpoints,
                backend=backend,
                CD=target,
                binary=binary,
                compute_minimum_cost=False,
                **options,
            )
            assert result.complete
            assert {tuple(mapping) for mapping in result.mappings} == expected
            assert result.selected_mapping_count == len(expected)
            if target == "minimal":
                assert result.minimum_cost == min(shells)


def test_cpp_stream_cap_and_deadline(native_library):
    lgp = [graph(np.zeros((4, 4)))] * 2
    streamed = []
    result = enumerate_pabs_mappings(
        lgp,
        backend="cpp",
        library_path=native_library,
        collect_mappings=False,
        max_mappings=3,
        mapping_callback=lambda mapping, cost: streamed.append(mapping),
    )
    assert result.minimum_cost == 0
    assert not result.complete and result.truncation_reason == "mapping_limit"
    assert result.mappings == [] and len(streamed) == result.selected_mapping_count == 3
    result = enumerate_pabs_mappings(
        lgp,
        backend="cpp",
        library_path=native_library,
        time_limit_seconds=0,
    )
    assert not result.complete and result.minimum_cost is None
    assert result.backend == "pabs_cpp"


def test_backend_selection_is_explicit(native_library):
    lgp = [graph(np.zeros((2, 2)))] * 2
    with pytest.raises(ValueError, match="backend must"):
        enumerate_pabs_mappings(lgp, backend="auto")
    with pytest.raises(ValueError, match="requires library_path"):
        enumerate_pabs_mappings(lgp, backend="cpp")
    with pytest.raises(TypeError, match="fixed_mapping"):
        enumerate_pabs_mappings(
            lgp, backend="cpp", library_path=native_library, fixed_mapping={0: 0}
        )
    with pytest.raises(OSError):
        enumerate_pabs_mappings(lgp, backend="cpp", library_path="/missing/kernel.so")


def test_cpp_typed_inventory_and_near_lattice_target(native_library):
    lgp = [
        LabeledGraph({0: {1: 1}, 1: {0: 1}, 2: {}}, labels)
        for labels in ([6, 6, 8], [6, 8, 6])
    ]
    expected = {
        mapping
        for mapping in permutations(range(3))
        if all(lgp[0].labels[i] == lgp[1].labels[j] for i, j in enumerate(mapping))
        and chemical_distance(lgp, mapping, False) == 2
    }
    result = enumerate_pabs_mappings(
        lgp,
        backend="cpp",
        library_path=native_library,
        CD=2 + 1e-10,
        binary=False,
        compute_minimum_cost=False,
    )
    assert {tuple(mapping) for mapping in result.mappings} == expected
    assert result.cost == 2 and result.distances == [2] * len(expected)
    assert result.total_bijections == 2
