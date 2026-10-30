"""Full PABS capacity; large padding does not substitute for independent scoring."""

import ctypes as ct
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.native.canonical import (
    pack_canonical_key,
    unpack_canonical_key,
)

IP = ct.POINTER(ct.c_int32)
LP = ct.POINTER(ct.c_int64)
COST_CB = ct.CFUNCTYPE(ct.c_int, IP, ct.c_int, ct.c_void_p)
CB = ct.CFUNCTYPE(ct.c_int, IP, ct.c_void_p)
FRONTIER_CB = ct.CFUNCTYPE(ct.c_int, IP, ct.c_int, ct.c_void_p)


class Prepared:
    def __init__(self, n, *, free_four=False):
        self.n = n
        self.a = np.zeros((n, n), dtype=np.int32)
        self.b = self.a.copy()
        if free_four:
            self.types = np.r_[np.zeros(4, dtype=np.int32), np.arange(1, n - 3, dtype=np.int32)]
            self.a[:4, :4] = [[0, 1, 3, 0], [1, 0, 2, 1], [3, 2, 0, 0], [0, 1, 0, 0]]
            self.b[:4, :4] = [[0, 2, 0, 1], [2, 0, 1, 0], [0, 1, 0, 3], [1, 0, 3, 0]]
            self.a[0, n - 1] = self.a[n - 1, 0] = 2
            self.b[1, n - 1] = self.b[n - 1, 1] = 2
            self.a[2, n - 2] = self.a[n - 2, 2] = 3
            self.b[0, n - 2] = self.b[n - 2, 0] = 1
        else:
            self.types = np.arange(n, dtype=np.int32)
            self.a[0, n - 1] = self.a[n - 1, 0] = 3
            self.b[0, n - 1] = self.b[n - 1, 0] = 1
            self.a[n - 2, n - 1] = self.a[n - 1, n - 2] = 2
        self.order = np.arange(n, dtype=np.int32)
        self.pred = np.zeros((n, n), dtype=np.int32)
        self.empty = np.empty(0, dtype=np.int32)
        self.nt = int(self.types.max()) + 1
        self.nl = len(np.unique(np.r_[self.a.ravel(), self.b.ravel()])) - 1

    @property
    def arrays(self):
        return [self.a, self.b, self.types, self.types, self.order, self.pred, self.empty]

    def args(self):
        return [array.ctypes.data_as(IP) for array in self.arrays] + [0, self.empty.ctypes.data_as(IP), 0]

    def score4(self, mapping):
        """Direct unordered-pair objective, independent of native bounds/seed."""
        mapped = self.b[np.ix_(mapping, mapping)]
        return int(np.abs(self.a.astype(np.int64) - mapped.astype(np.int64)).sum())

    def oracle(self):
        if self.nt == self.n:
            identity = tuple(range(self.n))
            return {identity: self.score4(identity)}
        return {
            tuple(permutation) + tuple(range(4, self.n)): self.score4(tuple(permutation) + tuple(range(4, self.n)))
            for permutation in itertools.permutations(range(4))
        }


def _range(library, problem, lower, upper, *, seconds=5, cap=-1, stop=False):
    raw = ct.CDLL(str(library))
    function = raw.synkit_distance_range
    function.argtypes = (
        [ct.c_int] * 5 + [ct.c_double, ct.c_int64] + [IP] * 7 + [ct.c_int, IP, ct.c_int, COST_CB, ct.c_void_p, LP]
    )
    function.restype = ct.c_int
    found = {}

    @COST_CB
    def receive(mapping, cost4, _):
        found[tuple(mapping[i] for i in range(problem.n))] = cost4
        return int(stop)

    stats = (ct.c_int64 * 6)()
    result = function(
        problem.n, problem.nt, problem.nl, lower, upper, seconds, cap, *problem.args(), receive, None, stats
    )
    return result, found, list(stats)


def _minimum(library, problem, *, seconds=5):
    raw = ct.CDLL(str(library))
    function = raw.synkit_distance_minimum
    function.argtypes = [ct.c_int] * 3 + [ct.c_double] + [IP] * 7 + [ct.c_int, IP, ct.c_int] + [IP] * 3 + [LP]
    function.restype = ct.c_int
    witness = np.empty(problem.n, dtype=np.int32)
    value = ct.c_int()
    stats = (ct.c_int64 * 6)()
    status = function(
        problem.n,
        problem.nt,
        problem.nl,
        seconds,
        *problem.args(),
        problem.order.ctypes.data_as(IP),
        witness.ctypes.data_as(IP),
        ct.byref(value),
        stats,
    )
    return status, tuple(map(int, witness)), value.value, list(stats)


@pytest.mark.parametrize("n", [4, 256, 257, 281, 512])
def test_large_unique_typed_minimum_and_exact_range(native_library, n):
    problem = Prepared(n)
    expected = problem.oracle()
    cost = next(iter(expected.values()))
    status, witness, value, _ = _minimum(native_library, problem)
    assert status == 0 and witness in expected and value == expected[witness]
    for lower, upper, selected in [(cost, cost, expected), (0, cost, expected), (0, cost - 1, {})]:
        status, found, _ = _range(native_library, problem, lower, upper)
        assert status == 0 and found == selected
    assert ct.CDLL(str(native_library)).synkit_distance_abi() == 3


@pytest.mark.parametrize("n", [8, 257, 281, 512])
def test_padded_large_search_matches_every_small_unrestricted_completion(native_library, n):
    problem = Prepared(n, free_four=True)
    expected = problem.oracle()
    optimum = min(expected.values())
    status, witness, cost4, _ = _minimum(native_library, problem)
    assert status == 0 and cost4 == optimum and expected[witness] == optimum
    threshold = sorted(set(expected.values()))[len(set(expected.values())) // 2]
    for lower, upper in [(optimum, optimum), (threshold, threshold), (0, threshold)]:
        status, found, _ = _range(native_library, problem, lower, upper)
        assert status == 0
        assert found == {mapping: value for mapping, value in expected.items() if lower <= value <= upper}


@pytest.mark.parametrize("n", [257, 512])
def test_large_interruptions_and_candidate_caps_are_not_completion(native_library, n):
    problem = Prepared(n, free_four=True)
    expected = problem.oracle()
    maximum = max(expected.values())
    status, found, _ = _range(native_library, problem, 0, maximum, seconds=0)
    assert status == 1 and not found
    status, witness, cost4, _ = _minimum(native_library, problem, seconds=0)
    assert status == 1 and cost4 == expected[witness]
    status, found, _ = _range(native_library, problem, 0, maximum, cap=1)
    assert status == 2 and len(found) == 1
    assert all(expected[mapping] == value for mapping, value in found.items())
    status, found, _ = _range(native_library, problem, 0, maximum, stop=True)
    assert status == 3 and len(found) == 1


def test_legacy_candidate_and_both_frontier_entries_support_281(native_library):
    raw = ct.CDLL(str(native_library))
    problem = Prepared(281)
    expected = problem.oracle()
    cost4 = next(iter(expected.values()))
    found, frontier = [], []

    @CB
    def receive(mapping, _):
        found.append(tuple(mapping[i] for i in range(problem.n)))
        return 0

    @FRONTIER_CB
    def emit(mapping, count, _):
        frontier.append(tuple(mapping[i] for i in range(count)))
        return 0

    prefix_common = (
        [ct.c_int] * 4
        + [ct.c_double, ct.c_int64]
        + [IP] * 7
        + [ct.c_int, IP, ct.c_int]
        + [ct.c_int] * 2
        + [CB, ct.c_void_p, LP]
    )
    arguments = [problem.n, problem.nt, problem.nl, cost4, 5.0, -1, *problem.args(), 0, 1, receive, None]
    function = raw.synkit_distance_candidates
    function.argtypes = prefix_common
    stats = (ct.c_int64 * 6)()
    assert function(*arguments, stats) == 0
    assert found == list(expected)
    found.clear()
    function = raw.synkit_distance_frontier
    function.argtypes = prefix_common + [IP, ct.c_int, ct.c_int64, FRONTIER_CB]
    assert function(*arguments, stats, problem.empty.ctypes.data_as(IP), 0, 1, emit) == 0
    assert not found and frontier == list(expected)
    values = np.asarray(frontier[0], dtype=np.int32)
    frontier.clear()
    assert function(*arguments, stats, values.ctypes.data_as(IP), problem.n, 2, emit) == 0
    assert found == list(expected) and not frontier
    found.clear()
    function = raw.synkit_distance_frontier_batch
    function.argtypes = prefix_common + [IP, IP, ct.c_int, ct.c_int64, FRONTIER_CB]
    offsets = np.asarray([0, problem.n], dtype=np.int32)
    assert function(*arguments, stats, values.ctypes.data_as(IP), offsets.ctypes.data_as(IP), 1, 2, emit) == 0
    assert found == list(expected) and not frontier


def test_513_vertices_rejected_without_claiming_solution(native_library):
    problem = Prepared(513)
    assert _range(native_library, problem, 0, 20)[0] == -2
    assert _minimum(native_library, problem)[0] == -2


@pytest.mark.parametrize("n", [256, 257, 281, 512])
def test_skc1_uint16_large_node_and_edge_roundtrip(n):
    nodes = tuple(("node", i % 3) for i in range(n))
    edges = ((0, n - 1, ("edge", 3)), (256 if n > 256 else 255, n - 1, ("edge", 1)))
    key = nodes, tuple(sorted(edges))
    packed = pack_canonical_key(key)
    assert packed[:4] == b"SKC1"
    assert unpack_canonical_key(packed) == key


def test_packed_writer_rejects_513_but_does_not_change_format():
    with pytest.raises(ValueError, match="512"):
        pack_canonical_key((tuple(("node", 0) for _ in range(513)), ()))


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("g++") is None, reason="Linux C++ sanitizer qualification")
def test_512_forced_array_and_deep_search_under_address_and_undefined_sanitizers(tmp_path):
    source = Path(__file__).resolve().parents[4] / "synkit/Chem/Mapper/solvers/exact/native/native_distance.cpp"
    assert source.is_file()
    driver = tmp_path / "capacity.cpp"
    executable = tmp_path / "capacity"
    probe = subprocess.run(
        [
            "g++",
            "-x",
            "c++",
            "-",
            "-std=c++17",
            "-fsanitize=address,undefined",
            "-o",
            str(tmp_path / "sanitizer-probe"),
        ],
        input="int main() { return 0; }",
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode:
        pytest.skip("AddressSanitizer/UBSan toolchain runtime unavailable: " + probe.stderr.strip().splitlines()[-1])
    driver.write_text('#include "' + str(source) + '"\n' + r"""
struct CapacitySeen {int n;int free;int count=0;};
static int receive_capacity(const int* map,int cost4,void* data) {
  auto &seen=*static_cast<CapacitySeen*>(data);
  if(cost4)std::abort();
  std::vector<char> used(seen.n,0);
  for(int i=0;i<seen.n;++i) {
    if(map[i]<0||map[i]>=seen.n||used[map[i]]++)std::abort();
    if((!seen.free||i>=4)&&map[i]!=i)std::abort();
  }
  ++seen.count;return 0;
}
int main() {
  constexpr int n=512;
  std::vector<int> a(n*n,0),b(n*n,0),types(n),order(n),pred(n*n,0);
  std::iota(order.begin(),order.end(),0);
  for(int free: {0,1}) {
    for(int i=0;i<n;++i)types[i]=free?(i<4?0:i-3):i;
    CapacitySeen seen{n,free,0};int64_t stats[6]={};
    int status=synkit_distance_range(n,free?n-3:n,0,0,0,10.,-1,a.data(),b.data(),
       types.data(),types.data(),order.data(),pred.data(),nullptr,0,nullptr,0,
       receive_capacity,&seen,stats);
    if(status||seen.count!=(free?24:1))return 2;
  }
}
""")
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            str(driver),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("n", [257, 281, 512])
def test_common_pabs_large_capacity_and_exact_costs(n):
    library = os.environ.get("SYNKIT_CORE_LIBRARY")
    if not library:
        pytest.skip("Set SYNKIT_CORE_LIBRARY to qualify the integrated core")
    from synkit.Chem.Mapper._native.core import enumerate_core_mappings
    from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph

    problem = Prepared(n)
    labels = list(map(int, problem.types))
    graphs = [
        LabeledGraph(
            {i: {j: int(value) / 2 for j, value in enumerate(row) if value} for i, row in enumerate(matrix)}, labels
        )
        for matrix in (problem.a, problem.b)
    ]
    expected_cd = problem.score4(tuple(range(n))) / 4
    minimum = enumerate_core_mappings(
        graphs, library_path=library, backend="pabs", symmetry=False, time_limit_seconds=5
    )
    assert minimum.complete and minimum.minimum_cost == expected_cd
    assert minimum.distances == [expected_cd]
    for target, mode, expected in [
        (expected_cd, "exact", [expected_cd]),
        (expected_cd + 0.5, "at_most", [expected_cd]),
        (expected_cd - 0.5, "at_most", []),
    ]:
        result = enumerate_core_mappings(
            graphs, library_path=library, backend="pabs", symmetry=False, CD=target, CD_mode=mode, time_limit_seconds=5
        )
        assert result.complete and result.distances == expected


def test_integrated_class_512_legacy_final_counts_and_update_delivery():
    library = os.environ.get("SYNKIT_CORE_LIBRARY")
    if not library:
        pytest.skip("Set SYNKIT_CORE_LIBRARY to qualify the integrated class pipeline")
    from synkit.Chem.Mapper._native.classification import NativeITSBatchClassifier

    n = 512
    matrix = np.zeros((n, n), dtype=np.float64)
    classifier = NativeITSBatchClassifier(matrix, matrix, [6] * n, {}, library_path=library)
    identity = list(range(n))
    swapped = identity.copy()
    swapped[256], swapped[511] = swapped[511], swapped[256]
    keys = []
    for stream in (False, True):
        events = []
        result = classifier.classify(
            [identity, swapped, identity],
            costs2=[0, 0, 0],
            time_limit_seconds=5,
            class_callback=lambda record: events.append(record),
            stream=stream,
        )
        assert result.complete and result.candidates == 3
        assert len(result.classes) == len(events) == 1
        assert result.classes[0].observed_mapping_count == 3
        assert events[0].observed_mapping_count == (1 if stream else 3)
        key = result.classes[0].key.parts
        assert len(unpack_canonical_key(key)[0]) == n
        keys.append(key)
    assert keys[0] == keys[1]
