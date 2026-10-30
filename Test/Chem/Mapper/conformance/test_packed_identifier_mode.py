"""Packed identifiers change presentation only, never exact class equality.

All parity comparisons use unlimited execution or deterministic state/count
caps. Equal wall-time prefixes are intentionally not compared: identifier cost
is precisely the behavior under investigation.
"""

import base64
import hashlib
import os
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from synkit.Chem.Mapper._native.classification import NativeITSBatchClassifier
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from Test.Chem.Mapper.conformance.oracle import (
    class_representatives,
    exactly_equivalent,
    independent_cost2,
)
from Test.Chem.Mapper.conformance.test_implicit_heavy_oracle import (
    REACTIONS,
    _classes,
    _endpoints,
    _full_h_costs_by_heavy_map,
    _its,
    _same,
    _scores,
)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    override = os.environ.get("SYNKIT_CORE_LIBRARY")
    if override:
        return Path(override)
    from synkit.Chem.Mapper._native.build import build_mapper_native

    return build_mapper_native(tmp_path_factory.mktemp("packed-identifiers"))


def _fixture(kind="weighted"):
    n = 0 if kind == "empty" else 5 if kind == "twins" else 4
    a2 = np.zeros((n, n), dtype=np.int64)
    b2 = a2.copy()
    attrs = [()] * n
    properties = {}
    if kind in {"weighted", "escaped"}:
        for i, j, value in [(0, 1, 2), (1, 2, 1), (2, 3, 4)]:
            a2[i, j] = a2[j, i] = value
        for i, j, value in [(0, 1, 1), (0, 2, 4), (0, 3, 2)]:
            b2[i, j] = b2[j, i] = value
        values = ["R'\\", "é", "", '"quote"'] if kind == "escaped" else [0, 0, 1, 1]
        properties = {"fixture": (values, list(reversed(values)))}
        attrs = [(value,) for value in values]
    return a2, b2, properties, attrs


def _classifier(library, mode, kind="weighted"):
    a2, b2, props, _ = _fixture(kind)
    return NativeITSBatchClassifier(
        a2 / 2,
        b2 / 2,
        ["C"] * len(a2),
        props,
        library_path=library,
        identifier_mode=mode,
    )


def _signature(result):
    return [
        (
            record.key.parts,
            record.representative,
            record.cost2,
            record.observed_mapping_count,
        )
        for record in result.classes
    ]


def _assert_packed_hashes(result):
    for record in result.classes:
        assert record.key.parts[:4] == b"SKC1"
        assert record.identifier == hashlib.sha256(record.key.parts).hexdigest()


@pytest.mark.parametrize("kind", ["empty", "twins", "weighted", "escaped"])
@pytest.mark.parametrize("stream", [False, True])
def test_batch_modes_have_identical_actual_bytes_representatives_and_counts(
    library, kind, stream
):
    a2, b2, _properties, attrs = _fixture(kind)
    # First encounter, multiplicity and input order matter, not sorted labels.
    maps = list(reversed(list(permutations(range(len(a2)))))) * 3
    costs = [independent_cost2(a2, b2, mapping) for mapping in maps]
    results, callbacks = [], []
    for mode in ("legacy", "packed"):
        events = []
        result = _classifier(library, mode, kind).classify(
            maps,
            costs2=costs,
            stream=stream,
            class_callback=lambda record, events=events: events.append(
                (
                    record.key.parts,
                    record.representative,
                    record.cost2,
                    record.observed_mapping_count,
                )
            ),
        )
        assert result.complete and result.reason is None
        assert result.candidates == len(maps)
        assert sum(record.observed_mapping_count for record in result.classes) == len(
            maps
        )
        assert len(events) == len(result.classes)
        results.append(result)
        callbacks.append(events)
    legacy, packed = results
    assert _signature(legacy) == _signature(packed)
    assert callbacks[0] == callbacks[1]
    assert legacy.canonical_nodes == packed.canonical_nodes
    _assert_packed_hashes(packed)
    expected = class_representatives(a2, b2, set(maps), attrs, list(reversed(attrs)))
    actual = class_representatives(
        a2, b2, [r.representative for r in packed.classes], attrs, list(reversed(attrs))
    )
    assert len(actual) == len(expected) == len(packed.classes)
    assert all(any(exactly_equivalent(a, b) for _, b in actual) for _, a in expected)
    multiplicities = Counter(maps)
    assert sum(multiplicities.values()) == sum(
        record.observed_mapping_count for record in packed.classes
    )
    # The mode is observable while the equality certificate remains identical.
    assert any(
        a.identifier != b.identifier for a, b in zip(legacy.classes, packed.classes)
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "limit,reason",
    [
        ({"max_classes": 2}, "class_limit"),
        ({"max_canonical_nodes": 3}, "canonical_node_limit"),
        ({"class_callback": lambda _: True}, "callback_stop"),
    ],
)
def test_batch_deterministic_caps_and_callback_stop_are_mode_invariant(
    library, stream, limit, reason
):
    mappings = list(permutations(range(4))) * 3
    results = [
        _classifier(library, mode).classify(mappings, stream=stream, **limit)
        for mode in ("legacy", "packed")
    ]
    assert all(not result.complete and result.reason == reason for result in results)
    assert _signature(results[0]) == _signature(results[1])
    assert results[0].candidates == results[1].candidates
    assert results[0].canonical_nodes == results[1].canonical_nodes
    _assert_packed_hashes(results[1])


@pytest.mark.parametrize("stream", [False, True])
def test_batch_precancellation_and_callback_errors_are_mode_invariant(library, stream):
    from synkit.Chem.Mapper._native.runtime import CancellationToken

    for mode in ("legacy", "packed"):
        classifier = _classifier(library, mode)
        with CancellationToken(library) as token:
            token.cancel()
            result = classifier.classify(
                permutations(range(4)), cancel_token=token, stream=stream
            )
        assert not result.complete and result.reason == "cancelled"
        assert result.candidates == 0 and not result.classes

        def fail(_):
            raise LookupError("packed-mode callback fixture")

        with pytest.raises(LookupError, match="packed-mode callback fixture"):
            classifier.classify(
                permutations(range(4)), class_callback=fail, stream=stream
            )


def _graphs():
    a2, b2, props, _ = _fixture()
    result = []
    for side, matrix in enumerate((a2, b2)):
        graph = LabeledGraph(
            {
                i: {j: int(value) / 2 for j, value in enumerate(row) if value}
                for i, row in enumerate(matrix)
            },
            ["C"] * len(matrix),
        )
        graph.props["fixture"] = props["fixture"][side]
        result.append(graph)
    return result


def _prepared_search(backend, caps):
    from synkit.Chem.Mapper._native.runtime import BACKENDS, INT, LONG, Problem, Request

    a2, b2, props, _ = _fixture()
    zero_nodes = np.zeros(4, dtype=np.int32)
    zero_edges = np.zeros((4, 4), dtype=np.int32)
    colors = [np.asarray(side, dtype=np.int32) for side in props["fixture"]]
    problem = Problem(
        1,
        4,
        a2.ctypes.data_as(LONG),
        b2.ctypes.data_as(LONG),
        *[
            array.ctypes.data_as(INT)
            for array in (zero_nodes, zero_nodes, *colors, zero_edges, zero_edges)
        ],
    )
    request = Request()
    request.abi_version = request.workers = 1
    request.backend = BACKENDS[backend]
    request.query = 2  # at most: include every score in this finite fixture
    request.target2 = 14
    request.seconds = -1
    for name in (
        "max_states",
        "max_candidates",
        "max_classes",
        "max_frontier",
        "memory_limit_bytes",
    ):
        setattr(request, name, -1)
    for name, value in caps.items():
        if name != "max_canonical_nodes":
            setattr(
                request, "max_candidates" if name == "max_mappings" else name, value
            )
    request.slap_max_fingerprints = 10000
    # Explicitly retain every ctypes buffer for the complete synchronous call.
    return problem, request, (a2, b2, zero_nodes, zero_edges, colors)


@pytest.mark.parametrize("backend", ["pabs", "kbb"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "cap",
    [
        {},
        {"max_states": 3},
        {"max_mappings": 3},
        {"max_classes": 2},
        {"max_canonical_nodes": 3},
    ],
)
def test_classified_solver_same_exact_bytes_scores_caps_and_domains(
    library, backend, stream, cap
):
    outcomes = []
    for mode in ("legacy", "packed"):
        classifier = _classifier(library, mode)
        problem, request, _keepalive = _prepared_search(backend, cap)
        allowed = None
        if backend == "kbb":
            allowed = np.ones((4, 4), dtype=np.uint8)
            allowed[0, 0] = 0
        callbacks = []
        result, statistics = classifier.solve_prepared(
            problem,
            request,
            allowed_domains=allowed,
            stream=stream,
            max_canonical_nodes=cap.get("max_canonical_nodes"),
            class_callback=lambda record, callbacks=callbacks: callbacks.append(
                (
                    record.key.parts,
                    record.representative,
                    record.cost2,
                    record.observed_mapping_count,
                )
            ),
        )
        if backend == "pabs" and "max_states" in cap:
            # Legacy PABS has no state-cap contract. Preserve its explicit
            # unsupported result; KBB below exercises actual capped search.
            assert statistics.status == 2 and statistics.reason == 8
            assert not result.classes and result.candidates == 0
        elif not cap:
            assert statistics.status != 2  # An unsupported no-op is not parity.
            assert result.complete and statistics.search_complete
            assert result.candidates == (24 if backend == "pabs" else 18)
        else:
            assert statistics.status != 2
            assert not (result.complete and statistics.search_complete)
        records = result.classes
        if mode == "packed":
            for record in records:
                assert record.identifier == hashlib.sha256(record.key.parts).hexdigest()
        a2, b2, _, _ = _fixture()
        assert all(
            record.cost2 == independent_cost2(a2, b2, record.representative)
            for record in records
        )
        if allowed is not None:
            assert all(record.representative[0] != 0 for record in records)
        outcomes.append(
            (
                result.complete,
                result.reason,
                result.candidates,
                result.canonical_nodes,
                [
                    (
                        record.key.parts,
                        record.representative,
                        record.cost2,
                        record.observed_mapping_count,
                    )
                    for record in records
                ],
                callbacks,
                {
                    name: getattr(statistics, name)
                    for name, _ in statistics._fields_
                    if isinstance(getattr(statistics, name), int)
                },
            )
        )
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("stream", [False, True])
def test_raw_ex_duplicate_palette_tokens_still_merge_by_actual_bytes(library, stream):
    import ctypes as ct

    from synkit.Chem.Mapper._native.classification import (
        _CALLBACK,
        _INT,
        _LONG,
        _load,
        _Problem,
        _Statistics,
    )

    # Deliberate raw-palette aliases: distinct canonical integer color keys
    # have exactly the same complete SKC1 bytes. Exact bytes remain the key.
    node = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
    zero = np.zeros((2, 2), dtype=np.int32)
    edgepair = np.asarray([0], dtype=np.int32)
    palette = (ct.c_char_p * 2)(b'"same"', b'"same"')
    problem = _Problem(
        2,
        node.ctypes.data_as(_INT),
        zero.ctypes.data_as(_INT),
        zero.ctypes.data_as(_INT),
        1,
        1,
        edgepair.ctypes.data_as(_INT),
        2,
        palette,
    )
    maps = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int32)
    costs = np.zeros(4, dtype=np.int64)
    native = _load(str(library.resolve()))
    outputs = []
    for packed_mode in (False, True):
        events = []

        @_CALLBACK
        def receive(mapping, cost2, count, packed, size, identifier, _, events=events):
            actual_bytes = ct.string_at(packed, size)
            name = identifier.decode("ascii")
            events.append((actual_bytes, tuple(mapping[:2]), cost2, count, name))
            return 0

        statistics = _Statistics()
        status = native.synkit_mapper_classify_batch_ex(
            ct.byref(problem),
            4,
            maps.ctypes.data_as(_INT),
            costs.ctypes.data_as(_LONG),
            -1,
            -1,
            -1,
            None,
            receive,
            None,
            ct.byref(statistics),
            int(stream),
            int(packed_mode),
        )
        assert status == 0 and statistics.complete
        assert statistics.classes == 1 and statistics.candidates == 4
        assert [event[-2] for event in events] == ([1, 4] if stream else [4])
        assert len({event[0] for event in events}) == 1
        if packed_mode:
            assert all(
                event[-1] == hashlib.sha256(event[0]).hexdigest() for event in events
            )
        outputs.append([event[:-1] for event in events])
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("mode", ["legacy", "packed"])
def test_classified_solver_precancellation_is_explicit(library, mode):
    from synkit.Chem.Mapper._native.core import enumerate_core_mappings
    from synkit.Chem.Mapper._native.runtime import CancellationToken

    with CancellationToken(library) as token:
        token.cancel()
        result = enumerate_core_mappings(
            _graphs(),
            library_path=library,
            backend="pabs",
            cancel_token=token,
            _native_classifier=_classifier(library, mode),
            _native_class_registry={},
        )
    assert not result.complete and result.truncation_reason == "cancelled"
    assert not result.mappings


@pytest.mark.parametrize("reaction", REACTIONS)
@pytest.mark.parametrize("query", ["minimum", "exact", "at_most"])
def test_implicit_packed_and_legacy_same_full_oracle_scores_and_nx_classes(
    library, reaction, query
):
    from synkit.Chem.Mapper._native.implicit import map_reaction_implicit

    endpoints = _endpoints(reaction)
    costs = _scores(endpoints)
    assert {
        mapping: min(values)
        for mapping, values in _full_h_costs_by_heavy_map(endpoints).items()
    } == costs
    threshold2 = min(costs.values()) if query == "minimum" else max(costs.values())
    selected = {
        mapping
        for mapping, value in costs.items()
        if (value <= threshold2 if query == "at_most" else value == threshold2)
    }
    options = {} if query == "minimum" else {"CD": threshold2 / 2, "CD_mode": query}
    results = [
        map_reaction_implicit(
            reaction,
            library_path=library,
            identifier_mode=mode,
            **options,
        )
        for mode in ("legacy", "packed")
    ]
    assert all(result.complete for result in results)
    if query == "minimum":
        assert all(result.minimum_cost == threshold2 / 2 for result in results)
    left, right = results
    assert [
        (
            record.canonical_key_base64,
            record.mapping,
            record.distance,
            record.mapped_reaction,
        )
        for record in left.reactions
    ] == [
        (
            record.canonical_key_base64,
            record.mapping,
            record.distance,
            record.mapped_reaction,
        )
        for record in right.reactions
    ]
    expected = _classes(endpoints, selected)
    actual = []
    n = len(endpoints[0].types)
    for record in right.reactions:
        packed = base64.b64decode(record.canonical_key_base64, validate=True)
        assert record.class_id == hashlib.sha256(packed).hexdigest()
        heavy_map = tuple(record.mapping[:n])
        assert heavy_map in selected
        assert record.distance * 2 == costs[heavy_map]
        actual.append(_its(endpoints, heavy_map))
    assert len(actual) == len(expected)
    assert all(any(_same(a, b) for b in actual) for a in expected)
    assert all(any(_same(a, b) for a in expected) for b in actual)


@pytest.mark.parametrize(
    "reaction,expected",
    [
        ("CO>>C", "ea9ee4a4f2739738056b2aea43789b3450e70a3687431803faa0c1c8ea704e3a"),
        (
            "F/C=C/F>>F/C=C\\F",
            "8892462fcd7c4c4f3f020dbb025bece8a19907d12e6d8875954feb297719aaea",
        ),
    ],
)
def test_existing_public_frontend_default_legacy_ids_remain_frozen(
    library, reaction, expected
):
    from synkit.Chem.Mapper._native.core import map_reaction_core

    result = map_reaction_core(
        reaction,
        library_path=library,
        backend="pabs",
        hydrogen="heavy",
        objective="bond",
        balance="dummy",
    )
    assert result.complete and [record.class_id for record in result.reactions] == [
        expected
    ]


@pytest.mark.parametrize("bad", ["", "sha256", "PACKED"])
def test_unknown_identifier_mode_is_rejected(library, bad):
    with pytest.raises(ValueError, match="identifier_mode"):
        _classifier(library, bad)


def test_historical_library_explicitly_rejects_opt_in_packed_identifiers():
    baseline = Path(
        os.environ.get(
            "SYNKIT_CLASSIFICATION_BASELINE_LIBRARY",
            "",
        )
    )
    if not baseline.is_file():
        pytest.skip(
            "Historical ABI1 library not available; current legacy fixtures still run"
        )
    # Older libraries continue to serve the default legacy interface, and
    # must never silently interpret a requested packed ID as a legacy ID.
    assert _classifier(baseline, "legacy").classify(permutations(range(4))).complete
    with pytest.raises(RuntimeError, match="packed"):
        _classifier(baseline, "packed")
