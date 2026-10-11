"""Native assignment, typed-orbit, and exactness regression contracts."""

from collections import Counter
from itertools import permutations

import networkx as nx
import numpy as np
import pytest

from Test.Chem.Mapper._helpers import graph
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


@pytest.mark.parametrize("cycle", [False, True])
def test_native_assignment_certificate_preserves_large_finite_costs(
    native_library, cycle
):
    """The advertised 64 by 1e6 domain exceeds the old INF/2 threshold."""
    import ctypes

    n, inf = 64, 100000000
    function = ctypes.CDLL(str(native_library)).synkit_assignment_certificate
    pointer = ctypes.POINTER(ctypes.c_int32)
    function.argtypes = [ctypes.c_int, pointer, pointer, pointer, pointer, pointer]
    function.restype = ctypes.c_int
    matrix = np.full((n, n), inf if cycle else 1000000, dtype=np.int32)
    expected = np.full((n, n), inf if cycle else n * 1000000, dtype=np.int32)
    if cycle:
        # Exactly two perfect matchings: the diagonal and the full directed
        # cycle. Every forced cycle edge requires a 63-million return path.
        for i in range(n):
            matrix[i, i] = expected[i, i] = 0
            matrix[i, (i + 1) % n] = 1000000
            expected[i, (i + 1) % n] = n * 1000000
    match, u, v = [np.empty(n, dtype=np.int32) for _ in range(3)]
    forced = np.empty((n, n), dtype=np.int32)
    observed = function(
        n, *(x.ctypes.data_as(pointer) for x in (matrix, match, u, v, forced))
    )
    assert observed == (0 if cycle else n * 1000000)
    assert int(matrix[np.arange(n), match].sum()) == observed
    assert int(u.sum() + v.sum()) == observed
    assert np.all(u[:, None] + v[None, :] <= matrix)
    assert np.array_equal(forced, expected)


@pytest.mark.parametrize("seed", range(12))
def test_native_signed_typed_orbits_against_raw_matrix_oracle(
    native_library, monkeypatch, seed
):
    """Independent orbit keys use only explicit permutations and raw matrices."""
    from synkit.Chem.Mapper.analysis import GlobalShellConfig, _BlindShellObserver
    from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates
    from synkit.Chem.Mapper.exact.orbit_aggregation import OrbitAccumulator

    rng = np.random.default_rng(20260909 + seed)
    n = 6
    families = [
        nx.cycle_graph(n),
        nx.star_graph(n - 1),
        nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3)),
        nx.disjoint_union_all([nx.path_graph(2)] * 3),
    ]
    a = nx.to_numpy_array(families[seed % 4]) * (-0.5 if seed % 2 else 1.5)
    if seed % 3:
        b = nx.to_numpy_array(families[(seed + 1) % 4]) * 0.5
    else:
        b = np.triu(rng.choice([-1.0, 0.0, 0.5, 1.5], size=(n, n)), 1)
        b += b.T
    at = [6] * n if seed % 2 else [6, 6, 6, 8, 8, 8]
    bt = [int(value) for value in rng.permutation(at)]
    perms = list(permutations(range(n)))
    compatible = [m for m in perms if all(at[i] == bt[m[i]] for i in range(n))]
    rg = [
        r
        for r in perms
        if all(at[i] == at[r[i]] for i in range(n))
        and np.array_equal(a, a[np.ix_(r, r)])
    ]
    pg = [
        p
        for p in perms
        if all(bt[i] == bt[p[i]] for i in range(n))
        and np.array_equal(b, b[np.ix_(p, p)])
    ]
    pair = (LabeledGraph(graph(a).graph, at), LabeledGraph(graph(b).graph, bt))
    costs, keys, raw_bonds = {}, {}, {}
    for m in compatible:
        transported = b[np.ix_(m, m)]
        costs[m] = float(np.abs(a - transported).sum() / 2)
        keys[m] = min(
            np.asarray(transported[np.ix_(r, r)] * 2, dtype=np.int16).tobytes()
            for r in rg
        )
        raw_bonds[m] = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if a[i, j] != transported[i, j]
        ]
    shells = sorted(set(costs.values()))
    for target in sorted({shells[0], shells[len(shells) // 2], shells[-1]}):
        selected = [m for m in compatible if costs[m] == target]
        expected = Counter(keys[m] for m in selected)
        enumerated = enumerate_native_candidates(
            pair,
            target,
            library_path=native_library,
            node_properties=(),
            initial_mapping=compatible[int(rng.integers(len(compatible)))],
            max_mappings=None,
            time_limit_seconds=10,
        )
        assert enumerated["complete"]
        candidates = enumerated["mappings"]
        assert all(costs[m] == target for m in candidates)
        assert {keys[m] for m in candidates} == set(expected)
        outcomes = []
        for enabled in ("0", "1"):
            monkeypatch.setenv("SYNKIT_NATIVE_PATTERN_CACHE", enabled)
            observer = _BlindShellObserver(
                a,
                b,
                at,
                {},
                GlobalShellConfig(
                    symmetry_node_properties=(),
                    reaction_center_properties=(),
                    structure_timeout_seconds=2,
                ),
            )
            accumulator = OrbitAccumulator(
                observer, rg[1:], len(rg), len(pg), library_path=native_library
            )
            # All labeled mappings and repeated mappings stress the cache;
            # candidate pruning has been checked independently above.
            for m in selected + selected[:3]:
                accumulator.observe(m, target)
            accumulator.finish()
            assert len(accumulator.seen) == len(expected)
            assert observer.count * len(pg) == len(selected)
            assert sorted(v[0] * len(pg) for v in accumulator.seen.values()) == sorted(
                expected.values()
            )
            expected_bonds = Counter(pair for m in selected for pair in raw_bonds[m])
            assert {
                k: v * len(pg) for k, v in observer.bond_counts.items() if v
            } == dict(expected_bonds)
            outcomes.append(
                (
                    dict(observer.structure.its_counts),
                    dict(observer.structure.template_counts),
                    dict(observer.bond_counts),
                )
            )
        assert outcomes[0] == outcomes[1]


def test_pattern_cache_requires_verified_generators_and_fresh_state(native_library):
    import ctypes

    lib = ctypes.CDLL(str(native_library))
    pointer = ctypes.POINTER(ctypes.c_int32)
    lib.synkit_pattern_cache_create.argtypes = [
        ctypes.c_int,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.synkit_pattern_cache_create.restype = ctypes.c_void_p
    lib.synkit_pattern_cache_lookup.argtypes = [ctypes.c_void_p, pointer, pointer]
    lib.synkit_pattern_cache_remember.argtypes = [ctypes.c_void_p]
    lib.synkit_pattern_cache_destroy.argtypes = [ctypes.c_void_p]
    lib.synkit_pattern_cache_destroy.restype = None
    nodes = np.zeros(4, dtype=np.int32)
    edges = np.asarray(nx.to_numpy_array(nx.path_graph(4)), dtype=np.int32)
    generators = np.array(
        [
            [3, 2, 1, 0],  # valid reversal
            [1, 0, 2, 3],  # bijective, but not a baseline automorphism
            [-1, 1, 2, 3],  # invalid image
            [0, 0, 2, 3],  # non-bijection
        ],
        dtype=np.int32,
    )

    def ptr(x):
        return x.ctypes.data_as(pointer)

    for trial in range(2):
        handle = lib.synkit_pattern_cache_create(
            4, ptr(nodes), ptr(edges), ptr(generators), len(generators), 64
        )
        assert handle
        try:
            changed = nodes.copy()
            changed[0] = 7
            # A second fresh handle must not inherit the first handle's hits.
            assert (
                lib.synkit_pattern_cache_lookup(handle, ptr(changed), ptr(edges)) == 0
            )
            assert lib.synkit_pattern_cache_remember(handle) == 0
            for atom, expected in ((0, 1), (3, 1), (1, 0), (2, 0)):
                changed[:] = 0
                changed[atom] = 7
                assert (
                    lib.synkit_pattern_cache_lookup(handle, ptr(changed), ptr(edges))
                    == expected
                )
        finally:
            lib.synkit_pattern_cache_destroy(handle)


def test_pattern_cache_capacity_forgets_without_false_hits(native_library):
    import ctypes

    lib = ctypes.CDLL(str(native_library))
    pointer = ctypes.POINTER(ctypes.c_int32)
    lib.synkit_pattern_cache_create.argtypes = [
        ctypes.c_int,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.synkit_pattern_cache_create.restype = ctypes.c_void_p
    lib.synkit_pattern_cache_lookup.argtypes = [ctypes.c_void_p, pointer, pointer]
    lib.synkit_pattern_cache_remember.argtypes = [ctypes.c_void_p]
    lib.synkit_pattern_cache_destroy.argtypes = [ctypes.c_void_p]
    lib.synkit_pattern_cache_destroy.restype = None
    n = 256
    nodes = np.zeros(n, dtype=np.int32)
    baseline = np.zeros((n, n), dtype=np.int32)
    generators = np.empty((0, n), dtype=np.int32)
    edges = np.ones((n, n), dtype=np.int32)
    np.fill_diagonal(edges, 0)

    def ptr(x):
        return x.ctypes.data_as(pointer)

    handle = lib.synkit_pattern_cache_create(
        n, ptr(nodes), ptr(baseline), ptr(generators), 0, 0
    )
    assert handle
    try:
        changed = nodes.copy()
        # Large exact deviation vectors exceed the payload cap after 255 entries.
        for value in range(1, 261):
            changed[:] = value
            assert (
                lib.synkit_pattern_cache_lookup(handle, ptr(changed), ptr(edges)) == 0
            )
            assert lib.synkit_pattern_cache_remember(handle) == 0
        assert lib.synkit_pattern_cache_lookup(handle, ptr(changed), ptr(edges)) == 1
        changed[:] = 1
        assert lib.synkit_pattern_cache_lookup(handle, ptr(changed), ptr(edges)) == 0
        # A miss does not insert anything until remember is called.
        assert lib.synkit_pattern_cache_lookup(handle, ptr(changed), ptr(edges)) == 0
    finally:
        lib.synkit_pattern_cache_destroy(handle)


@pytest.mark.parametrize(
    "left,right",
    [
        ([True, 1], [True, 1]),
        ([0.0, 0], [0.0, 0]),
        ([6, 6], [6.0, 6.0]),
    ],
)
def test_native_rejects_atom_equality_inconsistent_with_typed_colors(
    native_library, left, right
):
    from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates

    pair = (
        LabeledGraph({0: {}, 1: {}}, left),
        LabeledGraph({0: {}, 1: {}}, right),
    )
    with pytest.raises(ValueError, match="typed"):
        enumerate_native_candidates(
            pair, 0, library_path=native_library, node_properties=()
        )


@pytest.mark.parametrize("seed", range(8))
def test_native_certificate_only_equals_python(seed, native_library):
    from synkit.Chem.Mapper.spectrum import _exact_code, _NativeExactCodeBackend

    rng = np.random.default_rng(seed)
    g = nx.gnp_random_graph(8, 0.3, seed=seed)
    for node in g:
        g.nodes[node]["color"] = ("atom", int(rng.integers(2)))
    for a, b in g.edges:
        g.edges[a, b]["color"] = (1.0, float(rng.integers(2)))
    expected, reason = _exact_code(g, timeout_seconds=5, max_search_nodes=100000)
    actual, native_reason = _NativeExactCodeBackend(native_library).code(
        g, timeout_seconds=5, max_search_nodes=100000
    )
    assert reason is None and native_reason is None
    assert actual == expected


def test_exact_reference_witnesses_do_not_depend_on_hash(monkeypatch):
    import synkit.Chem.Mapper.analysis as analysis

    monkeypatch.setattr(analysis, "_mapping_sha256", lambda _: b"collision")
    monkeypatch.setattr(analysis, "_transport_sha256", lambda *args: b"collision")
    a = np.zeros((3, 3))
    b = nx.to_numpy_array(nx.path_graph(3))
    observer = analysis._BlindShellObserver(
        a, b, [6] * 3, {}, analysis.GlobalShellConfig()
    )
    observer.observe((0, 1, 2), 2)
    assert analysis._mapping_key((0, 2, 1)) not in observer.mapping_hashes
    assert analysis._transport_key(b, {}, (0, 2, 1)) not in observer.transport_hashes


@pytest.mark.parametrize("seed", range(4))
def test_native_minimum_matches_raw_permutation_oracle(seed, native_library):
    from synkit.Chem.Mapper.analysis import GlobalShellConfig
    from synkit.Chem.Mapper.native_analysis import (
        analyze_reference_blinded_native_shell,
    )

    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(2):
        a = np.triu(rng.choice([-0.5, 0.0, 1.0, 1.5], size=(4, 4)), 1)
        matrices.append(a + a.T)
    a, b = matrices
    costs = {
        p: float(np.abs(a - b[np.ix_(p, p)]).sum() / 2) for p in permutations(range(4))
    }
    # Deliberately give the worst-cost reference. It cannot establish a minimum.
    reference = max(costs, key=costs.get)
    result = analyze_reference_blinded_native_shell(
        (graph(a), graph(b)),
        reference,
        target_mode="minimal",
        library_path=native_library,
        workers=1,
        config=GlobalShellConfig(
            time_limit_seconds=10,
            use_slap_seed=False,
            symmetry_node_properties=(),
            reaction_center_properties=(),
        ),
    )
    minimum = min(costs.values())
    assert result.complete and result.structure.complete
    assert result.minimum_cost == minimum
    assert result.target == "minimal"
    assert result.labeled_solution_count == sum(c == minimum for c in costs.values())
    assert result.reference_is_global_minimum_proven == (costs[reference] == minimum)


def test_native_minimum_deadline_does_not_publish_incumbent(native_library):
    from synkit.Chem.Mapper.analysis import GlobalShellConfig
    from synkit.Chem.Mapper.native_analysis import (
        analyze_reference_blinded_native_shell,
    )

    a = nx.to_numpy_array(nx.path_graph(5))
    b = nx.to_numpy_array(nx.star_graph(4))
    with pytest.raises(TimeoutError, match="minimum proof incomplete"):
        analyze_reference_blinded_native_shell(
            (graph(a), graph(b)),
            tuple(range(5)),
            target_mode="minimal",
            library_path=native_library,
            workers=1,
            config=GlobalShellConfig(
                time_limit_seconds=0,
                use_slap_seed=False,
                symmetry_node_properties=(),
                reaction_center_properties=(),
            ),
        )


def test_bound_probe_certifies_witness_without_heuristic(native_library, monkeypatch):
    import synkit.Chem.Mapper.native_analysis as analysis
    from synkit.Chem.Mapper.analysis import GlobalShellConfig

    a = nx.to_numpy_array(nx.path_graph(4))

    def forbidden(*args, **kwargs):
        raise AssertionError("attained lower bound must not need a heuristic")

    monkeypatch.setattr(analysis, "_reference_free_slap_seed", forbidden)
    result = analysis.analyze_reference_blinded_native_shell(
        (graph(a), graph(a)),
        (0, 1, 2, 3),
        target_mode="minimal",
        library_path=native_library,
        workers=1,
        config=GlobalShellConfig(
            time_limit_seconds=10,
            symmetry_node_properties=(),
            reaction_center_properties=(),
        ),
    )
    assert result.complete and result.minimum_cost == 0
    assert (
        result.backend_statistics["minimum_proof"]["method"]
        == "attained_profile_assignment_bound"
    )


def test_unattained_profile_bound_does_not_claim_minimum(native_library):
    from synkit.Chem.Mapper.native_analysis import (
        analyze_reference_blinded_native_shell,
    )
    from synkit.Chem.Mapper.analysis import GlobalShellConfig

    a = nx.to_numpy_array(nx.cycle_graph(6))
    b = nx.to_numpy_array(nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3)))
    # Equal degree profiles give bound zero, but these graphs are not isomorphic.
    result = analyze_reference_blinded_native_shell(
        (graph(a), graph(b)),
        tuple(range(6)),
        target_mode="minimal",
        library_path=native_library,
        workers=1,
        config=GlobalShellConfig(
            time_limit_seconds=10,
            use_slap_seed=False,
            symmetry_node_properties=(),
            reaction_center_properties=(),
        ),
    )
    assert result.complete and result.minimum_cost > 0
    assert not result.backend_statistics["minimum_probe"]["attained"]
    assert (
        result.backend_statistics["minimum_proof"]["method"]
        == "assignment_optimization"
    )


def test_immutable_json_payload_preserves_public_json_and_default_mutability(
    native_library,
):
    import json
    from synkit.Chem.Mapper.analysis import GlobalShellConfig
    from synkit.Chem.Mapper.native_analysis import (
        analyze_reference_blinded_native_shell,
    )

    a = nx.to_numpy_array(nx.path_graph(4))
    result = analyze_reference_blinded_native_shell(
        (graph(a), graph(a)),
        tuple(range(4)),
        target_mode="minimal",
        library_path=native_library,
        workers=1,
        config=GlobalShellConfig(
            time_limit_seconds=10,
            use_slap_seed=False,
            symmetry_node_properties=(),
            reaction_center_properties=(),
        ),
    )
    normal = result.as_dict()
    immutable = result.as_dict(copy_sequences=False)
    assert json.dumps(normal) == json.dumps(immutable)
    assert isinstance(normal["structure"]["its_class_counts"], list)
    assert isinstance(normal["structure"]["its_class_counts"][0], list)
    normal["structure"]["its_class_counts"][0][1] = -123
    assert result.structure.its_class_counts[0][1] > 0
    assert (
        immutable["structure"]["its_class_counts"] is result.structure.its_class_counts
    )


def test_stale_assignment_hints_repair_duals_and_forced_costs(native_library):
    """Arbitrary stale hints must agree with exhaustive constrained optima."""
    import ctypes

    inf = 100000000
    library = ctypes.CDLL(str(native_library))
    function = library.synkit_assignment_seed_certificate
    pointer = ctypes.POINTER(ctypes.c_int32)
    function.argtypes = [ctypes.c_int] + [pointer] * 7
    function.restype = ctypes.c_int
    rng = np.random.default_rng(20260909)
    cases = [
        np.zeros((7, 7), dtype=np.int32),
        np.array([[0, inf, inf], [0, inf, inf], [inf, 0, 0]], dtype=np.int32),
        np.full((3, 3), inf, dtype=np.int32),
        np.array([[1000000, inf], [999999, 1000000]], dtype=np.int32),
    ]
    for n in range(1, 8):
        for _ in range(10):
            cost = rng.integers(0, 32, size=(n, n), dtype=np.int32)
            cost[rng.random((n, n)) < 0.35] = inf
            cases.append(cost)
    for cost in cases:
        n = len(cost)
        expected = np.full((n, n), inf, dtype=np.int32)
        for mapping in permutations(range(n)):
            values = cost[np.arange(n), mapping]
            if np.any(values == inf):
                continue
            total = int(values.sum())
            for row, col in enumerate(mapping):
                expected[row, col] = min(expected[row, col], total)
        for hint_kind in range(3):
            seed_v = rng.integers(-1000000, 1000001, n, dtype=np.int32)
            seed_match = rng.integers(-1, n, n, dtype=np.int32)
            if hint_kind == 1:
                seed_v[:] = np.iinfo(np.int32).max
                seed_match[:] = 0  # Duplicate, possibly forbidden edges.
            elif hint_kind == 2:
                seed_v[:] = np.iinfo(np.int32).min
                seed_match[:] = np.iinfo(np.int32).max
            match, u, v = [np.empty(n, dtype=np.int32) for _ in range(3)]
            forced = np.empty((n, n), dtype=np.int32)
            observed = function(
                n,
                *(
                    array.ctypes.data_as(pointer)
                    for array in (cost, seed_v, seed_match, match, u, v, forced)
                ),
            )
            assert observed == int(expected.min())
            if observed == inf:
                continue
            assert sorted(match.tolist()) == list(range(n))
            assert int(cost[np.arange(n), match].sum()) == observed
            assert int(u.sum() + v.sum()) == observed
            assert np.all(u[:, None] + v[None, :] <= cost)
            assert np.array_equal(forced, expected)


@pytest.mark.parametrize("exchange_components", [False, True])
def test_local_twin_factors_preserve_mixed_automorphism_groups(
    native_library, exchange_components
):
    from synkit.Chem.Mapper.exact.native_canonical import native_canonical_code
    from synkit.Graph.Canon.exact import ExactColoredGraphCanonicalizer

    if exchange_components:
        # Exchanging stars joins DIFFERENT twin classes in a support component.
        g = nx.disjoint_union(nx.star_graph(2), nx.star_graph(2))
        expected_order = 8
    else:
        # One S3 twin factor and one D3 factor requiring the general route.
        g = nx.disjoint_union(nx.star_graph(3), nx.cycle_graph(3))
        expected_order = 36
    expected = ExactColoredGraphCanonicalizer(g).canonicalize()
    assert expected.complete
    actual, order = native_canonical_code(g, library_path=native_library)
    assert actual == expected.canonical_code
    assert order == expected_order
    renamed = nx.relabel_nodes(g, {i: 100 - 3 * i for i in g})
    other, other_order = native_canonical_code(renamed, library_path=native_library)
    assert (other, other_order) == (actual, order)


def test_template_encoding_cache_binds_tokens_and_preserves_published_bytes(
    native_library,
):
    from synkit.Chem.Mapper.analysis import GlobalShellConfig, _BlindShellObserver
    from synkit.Chem.Mapper.exact.native_canonical import pack_canonical_key
    from synkit.Chem.Mapper.exact.native_its import NativeITSCanonicalizer
    from synkit.Graph.Canon.exact import _encode_color

    a = nx.to_numpy_array(nx.path_graph(2))
    observer = _BlindShellObserver(a, a, [6, 6], {}, GlobalShellConfig())
    canon = NativeITSCanonicalizer(observer, (), native_library)
    colors = np.zeros(2, dtype=np.int32)
    order = np.arange(2, dtype=np.int32)
    matrix = np.array([[0, 1], [1, 0]], dtype=np.int32)
    saved = []
    for i in range(300):
        token = _encode_color(("different exact token", i))
        expected = pack_canonical_key(((token, token), ((0, 1, canon.edge_tokens[1]),)))
        actual = canon.packed_key(colors, (token,), order, matrix, False)
        assert actual.parts == expected
        assert canon.packed_key(colors, (token,), order, matrix, False) == actual
        assert len(canon.template_packing_palettes) <= 256
        saved.append((actual, expected))
    # Later writes to shared scratch and cache evictions cannot alter keys.
    assert all(actual.parts == expected for actual, expected in saved)
