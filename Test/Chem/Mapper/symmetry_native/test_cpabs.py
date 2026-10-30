"""Independent complete permutation/class oracle for the native quotient tree."""

import ctypes as ct
from itertools import permutations

import networkx as nx
import numpy as np
import pytest

from synkit.Chem.Mapper._native.symmetry import cpabs_profile

INT = ct.POINTER(ct.c_int32)
L = ct.POINTER(ct.c_int64)
CALLBACK = ct.CFUNCTYPE(ct.c_int, INT, ct.c_int64, ct.c_void_p)


class Problem(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_int32),
        ("n", ct.c_int32),
        ("a2", L),
        ("b2", L),
    ] + [
        (name, INT)
        for name in (
            "types_r",
            "types_p",
            "node_colors_r",
            "node_colors_p",
            "edge_colors_r",
            "edge_colors_p",
        )
    ]


class Request(ct.Structure):
    _fields_ = [
        (name, ct.c_int32)
        for name in (
            "abi_version",
            "backend",
            "query",
            "symmetry",
            "classify",
            "workers",
        )
    ]
    _fields_ += [("target2", ct.c_int64), ("seconds", ct.c_double)]
    _fields_ += [
        (name, ct.c_int64)
        for name in (
            "max_states",
            "max_candidates",
            "max_classes",
            "max_frontier",
            "memory_limit_bytes",
        )
    ]
    _fields_ += [
        ("slap_max_fingerprints", ct.c_int32),
        ("slap_stop_flags", INT),
        ("cancel_token", ct.c_void_p),
    ]


class Statistics(ct.Structure):
    _fields_ = [
        (name, ct.c_int32)
        for name in (
            "abi_version",
            "backend",
            "status",
            "reason",
            "minimum_proven",
            "search_complete",
            "classification_complete",
            "hydrogen_complete",
            "heuristic_finished",
        )
    ]
    _fields_ += [
        (name, ct.c_int64)
        for name in (
            "minimum2",
            "incumbent2",
            "lower_bound2",
            "nodes",
            "leaves",
            "candidate_count",
            "class_count",
            "labeled_count",
            "peak_frontier",
            "peak_frontier_bytes",
            "peak_memory_bytes",
        )
    ]
    _fields_ += [
        (name, ct.c_double)
        for name in (
            "model_seconds",
            "optimization_seconds",
            "enumeration_seconds",
            "classification_seconds",
            "callback_seconds",
            "elapsed_seconds",
        )
    ]


def run(
    library_path,
    a,
    b,
    *,
    query=0,
    target=0,
    symmetry=True,
    types=None,
    colors=None,
    edge_colors=None,
    **limits,
):
    n = len(a)
    a, b = (np.ascontiguousarray(x, dtype=np.int64) for x in (a, b))
    types = types or ([0] * n, [0] * n)
    colors = colors or ([0] * n, [0] * n)
    edge_colors = edge_colors or (np.zeros((n, n)), np.zeros((n, n)))
    ints = [np.ascontiguousarray(x, dtype=np.int32) for x in (*types, *colors, *edge_colors)]
    p = Problem(
        1,
        n,
        a.ctypes.data_as(L),
        b.ctypes.data_as(L),
        *(x.ctypes.data_as(INT) for x in ints),
    )
    request = Request(
        1,
        6,
        query,
        int(symmetry),
        0,
        1,
        target,
        limits.pop("seconds", -1),
        limits.pop("max_states", -1),
        limits.pop("max_candidates", -1),
        limits.pop("max_classes", -1),
        limits.pop("max_frontier", -1),
        limits.pop("memory_limit_bytes", -1),
        100,
        None,
        None,
    )
    assert not limits
    output = []

    @CALLBACK
    def receive(mapping, cost, _):
        output.append((tuple(mapping[i] for i in range(n)), cost))
        return 0

    library = ct.CDLL(str(library_path))
    function = library.symmetry_test_cpabs
    function.argtypes = [
        ct.POINTER(Problem),
        ct.POINTER(Request),
        CALLBACK,
        ct.c_void_p,
        ct.POINTER(Statistics),
    ]
    function.restype = ct.c_int
    statistics = Statistics()
    assert function(ct.byref(p), ct.byref(request), receive, None, ct.byref(statistics)) == 0
    return output, statistics


def score(a, b, mapping):
    return sum(abs(int(a[i, j]) - int(b[mapping[i], mapping[j]])) for i in range(len(a)) for j in range(i + 1, len(a)))


def test_profile_seed_closes_isomorphic_weighted_graph_at_root(symmetry_library):
    # Degree ordering plus eight greedy two-swap passes gets stuck at cost12
    # on this relabeling. Typed sorted-neighbor profiles supply a zero-cost
    # feasible map, whose optimality follows from the nonnegative root bound.
    a = np.array(
        [
            [0, 2, 2, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 0, 2, 0, 0],
            [2, 2, 0, 0, 2, 0, 0, 2],
            [0, 0, 0, 0, 0, 2, 0, 4],
            [0, 0, 2, 0, 0, 0, 2, 4],
            [0, 2, 0, 2, 0, 0, 2, 0],
            [0, 0, 0, 0, 2, 2, 0, 0],
            [0, 0, 2, 4, 4, 0, 0, 0],
        ],
        dtype=int,
    )
    order = [4, 7, 3, 2, 1, 0, 6, 5]
    b = a[np.ix_(order, order)]
    output, stats = run(symmetry_library, a, b, query=3, max_states=1)
    assert output == []
    assert stats.minimum_proven and stats.search_complete and stats.minimum2 == 0
    assert stats.lower_bound2 == stats.minimum2
    assert stats.nodes == 1
    assert cpabs_profile(symmetry_library)["seed_cost2"] == 0


def test_interrupted_search_preserves_valid_nonzero_root_bound(symmetry_library):
    a = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]], dtype=int)
    b = np.zeros((3, 3), dtype=int)
    _, stats = run(symmetry_library, a, b, query=1, target=4, symmetry=False, max_states=1)
    assert not stats.search_complete and not stats.minimum_proven
    assert stats.lower_bound2 == min(score(a, b, f) for f in permutations(range(3)))
    assert stats.lower_bound2 == 4
    assert cpabs_profile(symmetry_library)["seed_cost2"] == -1


def test_shared_seed_domain_does_not_narrow_cpabs_weight_support(symmetry_library):
    # These weights exceed the common PABS seed helper's admission. The old
    # cPABS seed is retained, and the complete problem remains supported.
    a = np.array([[0, 2048], [2048, 0]], dtype=int)
    b = np.zeros((2, 2), dtype=int)
    output, stats = run(symmetry_library, a, b, query=3)
    assert output == [] and stats.search_complete and stats.minimum_proven
    assert stats.minimum2 == 2048 and stats.status == 0
    assert cpabs_profile(symmetry_library)["seed_cost2"] == 2048


def test_shared_seed_domain_does_not_narrow_cpabs_level_support(symmetry_library):
    a = np.zeros((7, 7), dtype=int)
    a[np.triu_indices(7, 1)] = np.arange(1, 22)
    a += a.T
    b = np.zeros_like(a)
    output, stats = run(symmetry_library, a, b, query=3)
    assert output == [] and stats.search_complete and stats.minimum_proven
    assert stats.minimum2 == 231 and stats.status == 0
    assert cpabs_profile(symmetry_library)["seed_cost2"] == 231


def test_shared_seed_domain_does_not_narrow_cpabs_mass_support(symmetry_library):
    a = (np.ones((32, 32), dtype=int) - np.eye(32, dtype=int)) * 1024
    b = np.zeros_like(a)
    expected = 32 * 31 // 2 * 1024
    assert expected > 500000
    output, stats = run(symmetry_library, a, b, query=3)
    assert output == [] and stats.search_complete and stats.minimum_proven
    assert stats.minimum2 == expected and stats.status == 0
    assert cpabs_profile(symmetry_library)["seed_cost2"] == expected


def its(a, b, mapping, colors=None, edge_colors=None):
    n = len(a)
    colors = colors or ([0] * n, [0] * n)
    edge_colors = edge_colors or (np.zeros((n, n)), np.zeros((n, n)))
    graph = nx.Graph()
    for i in range(n):
        graph.add_node(i, color=(int(colors[0][i]), int(colors[1][mapping[i]])))
    for i in range(n):
        for j in range(i + 1, n):
            token = (
                int(a[i, j]),
                int(b[mapping[i], mapping[j]]),
                int(edge_colors[0][i, j]),
                int(edge_colors[1][mapping[i], mapping[j]]),
            )
            if any(token):
                graph.add_edge(i, j, color=token)
    return graph


def class_representatives(graphs):
    def equal(x, y):
        return x["color"] == y["color"]

    representatives = []
    for graph in graphs:
        if not any(nx.is_isomorphic(graph, old, node_match=equal, edge_match=equal) for old in representatives):
            representatives.append(graph)
    return representatives


def compare_classes(actual, expected):
    def equal(x, y):
        return x["color"] == y["color"]

    actual = class_representatives(actual)
    expected = class_representatives(expected)
    assert len(actual) == len(expected)
    assert all(
        any(nx.is_isomorphic(graph, old, node_match=equal, edge_match=equal) for old in actual) for graph in expected
    )


@pytest.mark.parametrize("query", [0, 1, 2])
@pytest.mark.parametrize("symmetry", [False, True])
def test_complete_weighted_tiny_class_sets(symmetry_library, query, symmetry):
    rng = np.random.default_rng(20261025)
    for n in (2, 3, 4, 5):
        for repeat in range(4):
            endpoints = []
            for _ in range(2):
                matrix = np.triu(rng.integers(0, 4, size=(n, n)), 1)
                endpoints.append(matrix + matrix.T)
            a, b = endpoints
            all_maps = list(permutations(range(n)))
            costs = [score(a, b, mapping) for mapping in all_maps]
            target = sorted(set(costs))[len(set(costs)) // 2]
            selected = [
                m
                for m, cost in zip(all_maps, costs)
                if (cost == min(costs) if query == 0 else cost == target if query == 1 else cost <= target)
            ]
            actual, stats = run(symmetry_library, a, b, query=query, target=target, symmetry=symmetry)
            assert stats.search_complete and stats.status == 0
            assert all(cost == score(a, b, mapping) for mapping, cost in actual)
            if query == 0:
                assert stats.minimum_proven and stats.minimum2 == min(costs)
                assert stats.lower_bound2 == stats.minimum2
            if not symmetry:
                assert {m for m, _ in actual} == set(selected)
            compare_classes([its(a, b, m) for m, _ in actual], [its(a, b, m) for m in selected])


@pytest.mark.parametrize("kind", ["twins", "cycle", "components"])
def test_coupled_symmetries_and_repeated_components(symmetry_library, kind):
    n = 4
    a = np.zeros((n, n), dtype=int)
    edges = {
        "twins": [],
        "cycle": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "components": [(0, 1), (2, 3)],
    }[kind]
    for i, j in edges:
        a[i, j] = a[j, i] = 2
    output, stats = run(symmetry_library, a, a)
    assert stats.search_complete and stats.minimum_proven and len(output) == 1
    profile = cpabs_profile(symmetry_library)
    assert profile["subgroup_order"] == (576 if kind == "twins" else 64)
    assert profile["canonical_calls"] > 0
    # Earlier exact prefix/twin rejection may avoid canonical-parent rejects;
    # correctness depends on the complete coupled orbit set, not this count.
    assert not profile["identity_fallback_r"] and not profile["identity_fallback_p"]


def test_large_twin_factor_remains_complete(symmetry_library):
    a = np.zeros((5, 5), dtype=int)
    output, stats = run(symmetry_library, a, a)
    profile = cpabs_profile(symmetry_library)
    assert stats.search_complete and len(output) == 1
    assert not profile["identity_fallback_r"] and not profile["identity_fallback_p"]
    assert profile["subgroup_order"] == 14400


def test_large_terminal_flower_has_implicit_factorial_symmetry(symmetry_library):
    n = 14
    a = np.zeros((n, n), dtype=int)
    a[0, 1:] = a[1:, 0] = 2
    types = ([0] + [1] * (n - 1), [0] + [1] * (n - 1))
    output, stats = run(symmetry_library, a, a, types=types, max_states=100)
    assert stats.search_complete and stats.minimum_proven and stats.minimum2 == 0
    assert len(output) == 1
    profile = cpabs_profile(symmetry_library)
    assert profile["twin_blocks_r"] == profile["twin_blocks_p"] == 2
    assert profile["quotient_order_r"] == profile["quotient_order_p"] == 1
    assert profile["subgroup_order"] == -1  # exact order exceeds int64
    assert profile["bound_calls"] > 0 and profile["seed_cost2"] == 0
    assert not profile["identity_fallback_r"] and not profile["identity_fallback_p"]


def test_flower_colors_preserve_distinct_flow_classes(symmetry_library):
    a = np.zeros((5, 5), dtype=int)
    a[0, 1:] = a[1:, 0] = 2
    types = ([0, 1, 1, 1, 1], [0, 1, 1, 1, 1])
    colors = ([0, 1, 1, 2, 2], [0, 1, 1, 2, 2])
    output, stats = run(symmetry_library, a, a, types=types, colors=colors)
    assert stats.search_complete and len(output) == 3
    expected = [(0, *m) for m in permutations(range(1, 5))]
    compare_classes(
        [its(a, a, m, colors) for m, _ in output],
        [its(a, a, m, colors) for m in expected],
    )


def test_realistic_explicit_h_molecule_keeps_symmetry(symmetry_library):
    from synkit.Chem.Mapper._native.symmetry import map_reaction_cpabs

    result = map_reaction_cpabs(
        "CCCC>>CCCC",
        library_path=symmetry_library,
        time_limit_seconds=15,
        max_mappings=None,
    )
    assert result.complete and result.minimum_cost == 0 and len(result.reactions) == 1
    profile = cpabs_profile(symmetry_library)
    assert profile["subgroup_order"] == 82944
    assert profile["twin_blocks_r"] < 15
    assert not profile["identity_fallback_r"] and not profile["identity_fallback_p"]


def test_capped_quotient_discovery_keeps_closed_nontrivial_subgroup(symmetry_library):
    # Seven repeated three-vertex paths have an S7 component factor. The two
    # leaves of each path also supply independent implicit S2 twin factors.
    a = np.zeros((21, 21), dtype=int)
    for component in range(7):
        center = 3 * component
        for leaf in (center + 1, center + 2):
            a[center, leaf] = a[leaf, center] = 2
    output, stats = run(symmetry_library, a, a, query=3, max_states=10)
    assert not output and stats.search_complete and stats.minimum2 == 0
    profile = cpabs_profile(symmetry_library)
    assert profile["identity_fallback_r"] and profile["identity_fallback_p"]
    assert profile["quotient_order_r"] > 1 and profile["quotient_order_p"] > 1
    assert profile["subgroup_order"] > 128**2


def test_attribute_symmetries_and_endpoint_order(symmetry_library):
    a = np.array([[0, 2, 0, 0], [2, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 0]])
    b = a.copy()
    colors = ([0, 0, 1, 1], [1, 0, 0, 1])
    er, ep = np.zeros((4, 4), dtype=int), np.zeros((4, 4), dtype=int)
    er[0, 1] = er[1, 0] = 1
    ep[0, 1] = ep[1, 0] = 2
    expected = [its(a, b, m, colors, (er, ep)) for m in permutations(range(4))]
    output, stats = run(symmetry_library, a, b, query=2, target=100, colors=colors, edge_colors=(er, ep))
    assert stats.search_complete
    compare_classes([its(a, b, m, colors, (er, ep)) for m, _ in output], expected)
    r, p = np.array([2, 0, 3, 1]), np.array([1, 3, 0, 2])
    new_colors = ([colors[0][i] for i in r], [colors[1][i] for i in p])
    new_edges = er[np.ix_(r, r)], ep[np.ix_(p, p)]
    ar, bp = a[np.ix_(r, r)], b[np.ix_(p, p)]
    reordered, stats = run(
        symmetry_library,
        ar,
        bp,
        query=2,
        target=100,
        colors=new_colors,
        edge_colors=new_edges,
    )
    assert stats.search_complete
    compare_classes([its(ar, bp, m, new_colors, new_edges) for m, _ in reordered], expected)


@pytest.mark.parametrize(
    "limit,reason",
    [
        ({"seconds": 0}, 1),
        ({"max_states": 1}, 2),
        ({"max_candidates": 1}, 3),
        ({"max_frontier": 1}, 6),
    ],
)
def test_limits_preserve_incomplete_status(symmetry_library, limit, reason):
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    _, stats = run(symmetry_library, a, a, query=2, target=100, symmetry=False, **limit)
    assert not stats.search_complete and stats.status == 1 and stats.reason == reason


def test_proof_only_and_hard_compatibility(symmetry_library):
    a = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]])
    b = a[::-1, ::-1].copy()
    types = ([0, 1, 0], [0, 1, 0])
    expected = [
        (m, score(a, b, m)) for m in permutations(range(3)) if all(types[0][i] == types[1][m[i]] for i in range(3))
    ]
    output, stats = run(symmetry_library, a, b, query=3, types=types)
    assert not output and stats.minimum_proven and stats.search_complete
    assert stats.minimum2 == min(cost for _, cost in expected)
