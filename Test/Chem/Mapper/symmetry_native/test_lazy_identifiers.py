"""Exact byte/identifier/count compatibility when duplicate hashes are deferred."""

import ctypes as ct
import os
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from synkit.Chem.Mapper._native.classification import (
    _CALLBACK,
    NativeITSBatchClassifier,
    _load,
    _Problem,
    _Statistics,
)
from synkit.Chem.Mapper.exact.native.canonical import (
    compact_canonical_code,
    compact_code_identifier,
    unpack_canonical_key,
)
from synkit.Chem.Mapper.reporting.spectrum import (
    _attributed_its_graph,
    _code_identifier,
    _exact_code,
)

BASELINE = Path(os.environ.get("SYNKIT_CLASSIFICATION_BASELINE_LIBRARY", ""))


@pytest.fixture(scope="module")
def lazy_library(mapper_core_library):
    return mapper_core_library


def _libraries(candidate):
    # Release/CI must exercise the candidate even without this optional local
    # historical artifact. Only the cross-library comparison is conditional.
    return (BASELINE, candidate) if BASELINE.is_file() else (candidate,)


def _reference_classes(a, b, properties, mappings):
    multiplicities = Counter(mappings)
    expected = {}
    for mapping, count in multiplicities.items():
        images = np.asarray(mapping, dtype=int)
        candidate = _attributed_its_graph(
            a, b[np.ix_(images, images)], ["C"] * len(a), properties, mapping
        )
        code, reason = _exact_code(
            candidate, timeout_seconds=5, max_search_nodes=100000
        )
        assert reason is None
        identifier = _code_identifier(code)
        value = int(np.abs(a - b[np.ix_(images, images)]).sum())
        if identifier in expected:
            expected[identifier][0] += count
        else:
            expected[identifier] = [count, tuple(mapping), value, code]
    return expected


def _signature(result):
    return [
        (
            record.key.parts,
            record.identifier,
            record.representative,
            record.cost2,
            record.observed_mapping_count,
        )
        for record in result.classes
    ]


@pytest.mark.parametrize("kind", ["empty", "twins", "several", "escaped"])
@pytest.mark.parametrize("stream", [False, True])
def test_repeated_full_sets_preserve_exact_bytes_legacy_ids_and_counts(
    lazy_library, kind, stream
):
    n = 0 if kind == "empty" else 5 if kind == "twins" else 4
    a = np.zeros((n, n))
    b = a.copy()
    properties = {}
    if kind in ("several", "escaped"):
        for i, j, weight in [(0, 1, 1), (1, 2, 0.5), (2, 3, 2)]:
            a[i, j] = a[j, i] = weight
        for i, j, weight in [(0, 1, 0.5), (0, 2, 2), (0, 3, 1)]:
            b[i, j] = b[j, i] = weight
        values = ["R'\\", "é", "", '"quote"'] if kind == "escaped" else [0, 0, 1, 1]
        properties = {"fixture": (values, list(reversed(values)))}
    mappings = list(permutations(range(n))) * 4
    costs = [int(np.abs(a - b[np.ix_(mapping, mapping)]).sum()) for mapping in mappings]
    expected = _reference_classes(a, b, properties, mappings)
    results, events = [], []
    for library in _libraries(lazy_library):
        seen = []
        classifier = NativeITSBatchClassifier(
            a, b, ["C"] * n, properties, library_path=library
        )
        result = classifier.classify(
            mappings,
            costs2=costs,
            time_limit_seconds=5,
            stream=stream,
            class_callback=lambda record, seen=seen: seen.append(
                (record.key.parts, record.identifier, record.observed_mapping_count)
            ),
        )
        assert result.complete and result.candidates == len(mappings)
        assert sum(record.observed_mapping_count for record in result.classes) == len(
            mappings
        )
        assert {record.identifier for record in result.classes} == set(expected)
        for record in result.classes:
            count, representative, value, code = expected[record.identifier]
            assert record.observed_mapping_count == count
            assert record.representative == representative and record.cost2 == value
            assert (
                compact_canonical_code(unpack_canonical_key(record.key.parts)) == code
            )
            # Plain unpacked tuples bypass the identifier cached in the record.
            assert (
                compact_code_identifier(unpack_canonical_key(record.key.parts))
                == record.identifier
            )
        results.append(result)
        events.append(seen)
        assert len(seen) == len(result.classes)
        assert [count for _, _, count in seen] == (
            [1] * len(result.classes)
            if stream
            else [record.observed_mapping_count for record in result.classes]
        )
    if len(results) == 2:
        assert _signature(results[0]) == _signature(results[1])
        assert results[0].canonical_nodes == results[1].canonical_nodes
        assert events[0] == events[1]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "limit",
    [
        {"max_classes": 1},
        {"max_canonical_nodes": 3},
        {"time_limit_seconds": 0},
        {"class_callback": lambda _: True},
    ],
)
def test_caps_and_callback_completion_contract_matches_frozen(
    lazy_library, stream, limit
):
    a = np.zeros((4, 4))
    b = a.copy()
    b[0, 1] = b[1, 0] = 1
    properties = {"charge": ([0, 1, 2, 3], [3, 2, 1, 0])}
    mappings = list(permutations(range(4))) * 3
    results = []
    for library in _libraries(lazy_library):
        classifier = NativeITSBatchClassifier(
            a, b, ["C"] * 4, properties, library_path=library
        )
        result = classifier.classify(mappings, stream=stream, **limit)
        assert not result.complete
        if "max_classes" in limit:
            assert result.reason == "class_limit" and result.candidates == 1
            assert len(result.classes) == 1
        elif "max_canonical_nodes" in limit:
            assert (
                result.reason == "canonical_node_limit" and result.canonical_nodes == 3
            )
            assert result.candidates == len(result.classes) == 3
        elif "time_limit_seconds" in limit:
            assert result.reason == "time_limit" and result.candidates == 0
            assert not result.classes
        else:
            assert result.reason == "callback_stop" and len(result.classes) == 1
            assert result.candidates == (1 if stream else len(mappings))
        results.append(result)
    if len(results) == 2:
        assert results[0].reason == results[1].reason
        assert results[0].candidates == results[1].candidates
        assert _signature(results[0]) == _signature(results[1])


@pytest.mark.parametrize("stream", [False, True])
def test_duplicate_raw_palette_tokens_still_merge_by_packed_bytes(lazy_library, stream):
    # Raw API palettes need not be injective. Integer canonical.best_key differs
    # for these mappings, whereas their complete SKC1 bytes are identical.
    # Dedup must therefore remain keyed by the exact packed bytes.
    node = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
    zero = np.zeros((2, 2), dtype=np.int32)
    edgepair = np.asarray([0], dtype=np.int32)
    palette = (ct.c_char_p * 2)(b'"same"', b'"same"')
    pointer = ct.POINTER(ct.c_int32)
    problem = _Problem(
        2,
        node.ctypes.data_as(pointer),
        zero.ctypes.data_as(pointer),
        zero.ctypes.data_as(pointer),
        1,
        1,
        edgepair.ctypes.data_as(pointer),
        2,
        palette,
    )
    mappings = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int32)
    costs = np.zeros(4, dtype=np.int64)
    outputs = []
    for path in _libraries(lazy_library):
        library = _load(str(path))
        events = []

        @_CALLBACK
        def receive(mapping, cost, count, packed, size, identifier, _, events=events):
            events.append((ct.string_at(packed, size), identifier.decode(), count))
            return 0

        statistics = _Statistics()
        function = (
            library.synkit_mapper_classify_batch_updates_with_cancel
            if stream
            else library.synkit_mapper_classify_batch_with_cancel
        )
        status = function(
            ct.byref(problem),
            4,
            mappings.ctypes.data_as(pointer),
            costs.ctypes.data_as(ct.POINTER(ct.c_int64)),
            -1.0,
            -1,
            -1,
            None,
            receive,
            None,
            ct.byref(statistics),
        )
        assert (
            status == 0
            and statistics.complete
            and statistics.classes == 1
            and statistics.candidates == 4
        )
        assert [event[2] for event in events] == ([1, 4] if stream else [4])
        assert len({packed for packed, _, _ in events}) == 1
        for _, identifier, _ in events:
            assert len(identifier) == 64 and int(identifier, 16) >= 0
        outputs.append(events)
    if len(outputs) == 2:
        assert outputs[0] == outputs[1]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("when", ["prior", "discovery"])
def test_cancellation_and_post_callback_poll_match_frozen(lazy_library, stream, when):
    from synkit.Chem.Mapper._native.runtime import CancellationToken

    a = np.zeros((4, 4))
    props = {"charge": ([0, 1, 2, 3], [0, 1, 2, 3])}
    mappings = list(permutations(range(4)))
    results = []
    for path in _libraries(lazy_library):
        classifier = NativeITSBatchClassifier(a, a, ["C"] * 4, props, library_path=path)
        with CancellationToken(path) as token:
            if when == "prior":
                token.cancel()

            def cancel_on_discovery(_):
                token.cancel()

            result = classifier.classify(
                mappings,
                stream=stream,
                cancel_token=token,
                class_callback=cancel_on_discovery,
            )
        assert not result.complete and result.reason == "cancelled"
        if when == "prior":
            assert result.candidates == 0 and not result.classes
        else:
            assert result.candidates == (1 if stream else len(mappings))
            assert len(result.classes) == 1
            assert result.classes[0].observed_mapping_count == 1
        results.append(result)
    if len(results) == 2:
        assert results[0].candidates == results[1].candidates
        assert _signature(results[0]) == _signature(results[1])
