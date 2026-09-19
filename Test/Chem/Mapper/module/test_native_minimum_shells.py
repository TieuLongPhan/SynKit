"""Independent exhaustive checks of native lower-shell minimum certificates."""

import time
from itertools import permutations

import networkx as nx
import numpy as np
import pytest

from Test.Chem.Mapper.module.test_two_sided_symmetry import graph, native_library  # noqa: F401
from synkit.Chem.Mapper import native_analysis as analysis
from synkit.Chem.Mapper.slap.lap import chemical_distance


@pytest.mark.parametrize("seed", range(16))
def test_native_minimum_matches_exhaustive_weighted_permutations(native_library, seed):
    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(2):
        matrix = np.triu(rng.choice([0.0, 0.5, 1.0, 1.5], (5, 5)), 1)
        matrices.append(matrix + matrix.T)
    a, b = matrices
    lgp = graph(a), graph(b)
    labels = [6] * 5
    expected = min(chemical_distance(lgp, p, binary=False) for p in permutations(range(5)))
    witness, probe = analysis._attain_native_profile_bound(
        lgp, a, b, labels, labels, library_path=native_library,
        node_properties=(), deadline=time.perf_counter() + 10,
    )
    if witness is None:
        witness, proof = analysis._prove_native_seed_minimum(
            lgp, a, b, labels, labels, tuple(range(5)), probe,
            library_path=native_library, node_properties=(),
            deadline=time.perf_counter() + 10,
        )
        assert witness is not None and proof["complete"]
        assert proof["minimum_cost"] == expected
        assert all(s["target"] < expected or s.get("reason") == "minimum_witness"
                   for s in proof["shells"])
    assert chemical_distance(lgp, witness, binary=False) == expected


def _unattained(native_library):
    a = nx.to_numpy_array(nx.cycle_graph(6))
    b = nx.to_numpy_array(nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3)))
    lgp = graph(a), graph(b)
    labels = [6] * 6
    witness, probe = analysis._attain_native_profile_bound(
        lgp, a, b, labels, labels, library_path=native_library,
        node_properties=(), deadline=time.perf_counter() + 10,
    )
    assert witness is None and probe["probe_enumeration_complete"]
    return lgp, a, b, labels, probe


def test_integer_lattice_skips_only_impossible_shells(native_library):
    lgp, a, b, labels, probe = _unattained(native_library)
    witness, proof = analysis._prove_native_seed_minimum(
        lgp, a, b, labels, labels, tuple(range(6)), probe,
        library_path=native_library, node_properties=(), deadline=time.perf_counter() + 10,
    )
    expected = min(chemical_distance(lgp, p, binary=False) for p in permutations(range(6)))
    assert proof["complete"] and proof["lattice_spacing"] == 2
    assert proof["minimum_cost"] == expected == chemical_distance(lgp, witness, binary=False)
    assert all(s["target"] % 2 == 0 for s in proof["shells"])


def test_interrupted_lower_shell_never_certifies_seed(native_library, monkeypatch):
    from synkit.Chem.Mapper.exact import native_candidates

    lgp, a, b, labels, probe = _unattained(native_library)
    monkeypatch.setattr(native_candidates, "enumerate_native_candidates", lambda *a, **k: {
        "visited_nodes": 1, "complete": False, "reason": "time_limit",
    })
    witness, proof = analysis._prove_native_seed_minimum(
        lgp, a, b, labels, labels, tuple(range(6)), probe,
        library_path=native_library, node_properties=(), deadline=time.perf_counter() + 10,
    )
    assert witness is None and not proof["complete"]


def test_expired_deadline_and_incomplete_probe_fail_closed(native_library):
    lgp, a, b, labels, probe = _unattained(native_library)
    for current_probe, deadline in (
        (probe, time.perf_counter() - 1),
        (dict(probe, probe_enumeration_complete=False), time.perf_counter() + 10),
    ):
        witness, proof = analysis._prove_native_seed_minimum(
            lgp, a, b, labels, labels, tuple(range(6)), current_probe,
            library_path=native_library, node_properties=(), deadline=deadline,
        )
        assert witness is None
        assert proof is None or not proof["complete"]
