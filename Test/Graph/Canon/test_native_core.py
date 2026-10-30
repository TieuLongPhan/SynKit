"""Independent partitions and resource contracts of the opt-in native graph core."""

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import permutations
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import networkx as nx
import pytest

from synkit.Graph.Canon._native import (
    CancellationToken,
    build_native_canon,
    bundled_native_canon,
    canonicalize_native_graph,
)
from synkit.Graph.Canon._native.runtime import Request, load_library
from synkit.Graph.Canon.exact import CanonicalSearchIncomplete, ExactColoredGraphCanonicalizer


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    try:
        return bundled_native_canon()
    except FileNotFoundError:
        pass
    return build_native_canon(tmp_path_factory.mktemp("native-canon"))


def brute(graph):
    nodes = tuple(graph)
    return len(nodes), min(tuple(int(graph.has_edge(u, v)) for u in p for v in p)
                           for p in permutations(nodes))


def decode(result):
    code = result.require_complete().canonical_code
    offset = code.index(b"\0") + 1
    palette_size = struct.unpack_from("<Q", code, offset)[0]
    palette = json.loads(code[offset + 8:offset + 8 + palette_size])
    payload = code[offset + 8 + palette_size:]
    magic, n, m = struct.unpack_from("<III", payload)
    assert magic == 0x31304753
    assert len(payload) == 12 + 4 * n + 8 * m
    colors = struct.unpack_from(f"<{n}I", payload, 12)
    edges = [struct.unpack_from("<II", payload, 12 + 4 * n + 8 * i) for i in range(m)]
    return palette, colors, edges


def test_partitions_match_independent_brute_force_and_reference(library):
    forward, reverse, reference_codes = {}, {}, {}
    for graph in nx.graph_atlas_g():
        if len(graph) > 5:
            continue
        result = canonicalize_native_graph(graph, library_path=library)
        expected = brute(graph)
        assert forward.setdefault(result.canonical_code, expected) == expected
        assert reverse.setdefault(expected, result.canonical_code) == result.canonical_code
        _, colors, edges = decode(result)
        order = result.canonical_order
        assert len(colors) == len(graph)
        assert {frozenset((order[i], order[j])) for i, j in edges} == {frozenset(e) for e in graph.edges}
        for renamed in (nx.relabel_nodes(graph, {v: f"node:{11 - v}" for v in graph}),):
            native = canonicalize_native_graph(renamed, library_path=library)
            assert result.same_canonical_graph(native)
        # Compare equivalence relations, never bytes across distinct conventions.
        reference = ExactColoredGraphCanonicalizer(graph, edge_color=None).canonicalize()
        assert reference_codes.setdefault(reference.canonical_code, result.canonical_code) == result.canonical_code


def test_exact_palette_and_insertion_order_are_semantic(library):
    graph = nx.cycle_graph(5)
    nx.set_node_attributes(graph, {0: 1, 1: "1", 2: True, 3: None, 4: ("x", 2)}, "color")
    expected = canonicalize_native_graph(graph, library_path=library)
    renamed = nx.Graph()
    renamed.add_nodes_from((f"v:{v}", data) for v, data in reversed(list(graph.nodes(data=True))))
    renamed.add_edges_from((f"v:{u}", f"v:{v}") for u, v in reversed(list(graph.edges)))
    assert expected.same_canonical_graph(canonicalize_native_graph(renamed, library_path=library))
    renamed.nodes["v:0"]["color"] = "1"
    assert not expected.same_canonical_graph(canonicalize_native_graph(renamed, library_path=library))


def test_persisted_native_convention_has_frozen_bytes(library):
    # Fix the palette encoding, refinement order and sparse wire convention.
    # A change to these bytes needs an explicit namespace migration decision.
    expected = bytes.fromhex(
        "73796e6b69742e6e61746976652d7370617273652d63616e6f6e6963616c2d67726170682f3100"
        "0a000000000000005b5b226e6f6e65225d5d534730310300000002000000"
        "00000000000000000000000000000000020000000100000002000000"
    )
    result = canonicalize_native_graph(nx.path_graph(3), library_path=library)
    assert result.require_complete().canonical_code == expected


def test_comparison_requires_full_bytes_and_shared_convention(library, monkeypatch):
    path = canonicalize_native_graph(nx.path_graph(3), library_path=library)
    triangle = canonicalize_native_graph(nx.complete_graph(3), library_path=library)
    monkeypatch.setattr(type(path), "canonical_digest", property(lambda self: "collision"))
    assert path.canonical_digest == triangle.canonical_digest
    assert not path.same_canonical_graph(triangle)
    with pytest.raises(TypeError, match="conventions"):
        path.same_canonical_graph(replace(path, schema="foreign/1"))


@pytest.mark.parametrize("options,reason", [
    ({"timeout_seconds": 0}, "time_limit"),
    ({"max_search_nodes": 1}, "node_limit"),
    ({"max_depth": 0}, "depth_limit"),
    ({"max_workspace_bytes": 1}, "workspace_limit"),
    ({"max_output_bytes": 1}, "output_limit"),
])
def test_stopped_requests_expose_no_key_or_order(library, options, reason):
    result = canonicalize_native_graph(nx.cycle_graph(9), library_path=library, **options)
    assert not result.complete and result.reason == reason
    assert result.canonical_code is None and result.canonical_digest is None
    assert not result.canonical_order
    with pytest.raises(CanonicalSearchIncomplete):
        result.same_canonical_graph(result)


def test_workspace_and_output_accounting(library):
    graph = nx.complete_graph(80)
    result = canonicalize_native_graph(graph, library_path=library, max_workspace_bytes=4000)
    assert result.reason == "workspace_limit"
    assert result.statistics["peak_workspace_bytes"] <= 4000
    result = canonicalize_native_graph(nx.path_graph(10), library_path=library)
    assert result.complete
    full_output = len(result.canonical_code) + 4 * 10
    same = canonicalize_native_graph(nx.path_graph(10), library_path=library, max_output_bytes=full_output)
    assert result.same_canonical_graph(same)
    stopped = canonicalize_native_graph(nx.path_graph(10), library_path=library, max_output_bytes=full_output - 1)
    assert stopped.reason == "output_limit"


def test_graphs_above_mapper_capacity_and_sparse_payload(library):
    graph = nx.path_graph(1200)
    nx.set_node_attributes(graph, {v: v for v in graph}, "color")
    result = canonicalize_native_graph(graph, library_path=library, timeout_seconds=5)
    assert result.complete
    _, colors, edges = decode(result)
    assert len(colors) == 1200 and len(edges) == 1199
    # The graph payload stores only colors and present edges, not N squared absences.
    payload_bytes = 12 + 4 * len(colors) + 8 * len(edges)
    assert payload_bytes < 15000


@pytest.mark.parametrize("graph,factors,vertices", [
    (nx.star_graph(12), 1, 12),  # false twins
    (nx.complete_graph(12), 1, 12),  # adjacent true twins
    (nx.complete_multipartite_graph(3, 4, 2), 3, 9),
    (nx.cycle_graph(4), 2, 4),
    (nx.cycle_graph(6), 0, 0),  # coupled reflection orbits are not twin factors
])
def test_implicit_twin_factors_are_exact_colored_transpositions(library, graph, factors, vertices):
    result = canonicalize_native_graph(graph, library_path=library)
    assert result.complete
    assert result.statistics["implicit_twin_factors"] == factors
    assert result.statistics["twin_vertices"] == vertices
    renamed = nx.relabel_nodes(graph, {v: f"atom:{100 - v}" for v in graph})
    assert result.same_canonical_graph(canonicalize_native_graph(renamed, library_path=library))


def test_colors_split_twin_factors_and_coupled_orbits_remain_distinct(library):
    star = nx.star_graph(7)
    star.nodes[1]["color"] = "distinguished"
    result = canonicalize_native_graph(star, library_path=library)
    assert result.statistics["twin_vertices"] == 6
    adjacent, separated = nx.cycle_graph(6), nx.cycle_graph(6)
    for graph, blue in ((adjacent, 1), (separated, 2)):
        graph.nodes[0]["color"] = "red"
        graph.nodes[blue]["color"] = "blue"
    first = canonicalize_native_graph(adjacent, library_path=library)
    second = canonicalize_native_graph(separated, library_path=library)
    assert not first.same_canonical_graph(second)
    assert not nx.is_isomorphic(adjacent, separated, node_match=nx.algorithms.isomorphism.categorical_node_match(
        "color", None))


def test_cancellation_and_independent_parallel_calls(library):
    with CancellationToken(library) as token:
        with token.borrow(token.library):
            with pytest.raises(RuntimeError, match="in use"):
                token.close()
        token.cancel()
        result = canonicalize_native_graph(nx.cycle_graph(9), library_path=library, cancel_token=token)
        assert result.reason == "cancelled" and result.canonical_code is None
    with pytest.raises(RuntimeError, match="closed"):
        token.cancel()
    with CancellationToken(library) as token, ThreadPoolExecutor(2) as pool:
        running = pool.submit(canonicalize_native_graph, nx.cycle_graph(1500),
                              library_path=library, timeout_seconds=5, cancel_token=token)
        deadline = time.monotonic() + 3
        while not token._active and not running.done() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert token._active, "Search never entered the cancellable native call"
        token.cancel()
        assert running.result(timeout=3).reason == "cancelled"
        independent = pool.submit(canonicalize_native_graph, nx.cycle_graph(5), library_path=library)
        assert independent.result().complete


@pytest.mark.parametrize("graph", [nx.DiGraph([(0, 1)]), nx.MultiGraph([(0, 1)])])
def test_unsupported_graph_kinds_are_not_silently_flattened(library, graph):
    with pytest.raises(TypeError):
        canonicalize_native_graph(graph, library_path=library)


def test_invalid_inputs_and_abi_are_explicit(library):
    with pytest.raises(NotImplementedError, match="edge colors"):
        canonicalize_native_graph(nx.path_graph(3), library_path=library, edge_color="color")
    graph = nx.Graph([(0, 0)])
    with pytest.raises(ValueError, match="self loops"):
        canonicalize_native_graph(graph, library_path=library)
    lib = load_library(str(library))
    request, handle = Request(), ct.c_void_p()
    request.abi_version = 999
    assert lib.synkit_canon_search(ct.byref(request), ct.byref(handle)) == -1
    assert not handle.value
    assert b"ABI" in lib.synkit_canon_error()


@pytest.mark.parametrize("offsets,neighbors,colors", [
    ([0, 1, 2], [1, 1], [0, 0]),  # self loop and asymmetric incidence
    ([0, 2, 4], [1, 1, 0, 0], [0, 0]),  # duplicate edges
    ([0, 0, 2], [0, 0], [0, 0]),  # missing reverse edge
    ([0, -1, 2], [1, 0], [0, 0]),  # invalid CSR offset
    ([0, 1, 2], [1, 0], [-1, 0]),  # invalid color rank
])
def test_raw_abi_rejects_malformed_csr(library, offsets, neighbors, colors):
    arrays = ((ct.c_int64 * 3)(*offsets), (ct.c_int32 * len(neighbors))(*neighbors),
              (ct.c_int32 * 2)(*colors))
    request = Request(1, ct.sizeof(Request), 2, len(neighbors), *arrays, -1, -1, -1, -1, 100, None)
    handle = ct.c_void_p()
    lib = load_library(str(library))
    assert lib.synkit_canon_search(ct.byref(request), ct.byref(handle)) == -1
    assert not handle.value and lib.synkit_canon_error()


def test_source_build_is_explicit_immutable_and_verified(tmp_path):
    library = build_native_canon(tmp_path)
    assert build_native_canon(library.parent) == library
    manifest = json.loads(library.with_suffix(".json").read_text())
    assert manifest["configuration"]["abi"] == 1
    assert manifest["configuration"]["sources"]
    # A damaged copied build must not be reused or repaired silently.
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        damaged = Path(directory) / library.name
        shutil.copy2(library, damaged)
        shutil.copy2(library.with_suffix(".json"), damaged.with_suffix(".json"))
        damaged.with_suffix(".json").write_text('{}')
        with pytest.raises((KeyError, RuntimeError)):
            build_native_canon(directory)


def test_certificate_determinism_across_hash_seeds(library):
    script = '''
import networkx as nx
from synkit.Graph.Canon._native import canonicalize_native_graph
g=nx.Graph()
g.add_nodes_from({"a", "b", "c", "d"})
g.add_edges_from({("a","b"),("b","c"),("c","d"),("d","a")})
print(canonicalize_native_graph(g,library_path=LIBRARY).canonical_code.hex())
'''.replace("LIBRARY", repr(str(library)))
    values = [
        subprocess.check_output([sys.executable, "-c", script],
                                env=dict(os.environ, PYTHONHASHSEED=seed), text=True)
        for seed in ("1", "777")
    ]
    assert values[0] == values[1]
