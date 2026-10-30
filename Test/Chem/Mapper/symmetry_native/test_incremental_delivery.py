"""Exact discovery survives interruption without changing legacy counts."""

import ctypes as ct
import time
from itertools import permutations

import numpy as np

from synkit.Chem.Mapper._native.classification import (
    _CALLBACK,
    _INT,
    _LONG,
    NativeITSBatchClassifier,
    _Statistics,
)
from Test.Chem.Mapper.symmetry_native.test_pipeline import prepared


def classifier(library, n=4):
    a = np.zeros((n, n))
    return NativeITSBatchClassifier(a, a, ["C"] * n, {}, library_path=library)


def test_legacy_final_counts_and_incremental_updates_are_distinct_apis(
    symmetry_library,
):
    model = classifier(symmetry_library)
    maps = np.ascontiguousarray(list(permutations(range(4))), dtype=np.int32)
    costs = np.zeros(len(maps), dtype=np.int64)
    outputs = []
    for name in (
        "synkit_mapper_classify_batch_with_cancel",
        "synkit_mapper_classify_batch_updates_with_cancel",
    ):
        events = []

        @_CALLBACK
        def receive(mapping, cost2, count, packed, size, identifier, _, output=events):
            output.append((count, ct.string_at(packed, size)))
            return 0

        statistics = _Statistics()
        status = getattr(model.library, name)(
            ct.byref(model.problem),
            len(maps),
            maps.ctypes.data_as(_INT),
            costs.ctypes.data_as(_LONG),
            -1,
            -1,
            -1,
            None,
            receive,
            None,
            ct.byref(statistics),
        )
        assert status == 0 and statistics.complete and statistics.candidates == 24
        outputs.append(events)
    assert [count for count, _ in outputs[0]] == [24]
    assert [count for count, _ in outputs[1]] == [1, 24]
    assert len({key for events in outputs for _, key in events}) == 1


def test_python_merges_counts_without_duplicate_discovery_callbacks(symmetry_library):
    model = classifier(symmetry_library)
    callbacks = []
    result = model.classify(
        permutations(range(4)), stream=True, class_callback=callbacks.append
    )
    assert result.complete and result.candidates == 24
    assert len(result.classes) == len(callbacks) == 1
    assert callbacks[0].observed_mapping_count == 1
    assert result.classes[0].observed_mapping_count == 24
    legacy = model.classify(permutations(range(4)))
    assert result.classes == legacy.classes


def test_discovery_callback_stops_before_unvisited_invalid_mapping(symmetry_library):
    model = classifier(symmetry_library)
    result = model.classify(
        [[0, 1, 2, 3], [0, 0, 2, 3]],
        stream=True,
        class_callback=lambda _: True,
    )
    assert not result.complete and result.reason == "callback_stop"
    assert result.candidates == 1 and len(result.classes) == 1
    witness = model.classify([[0, 1, 2, 3]])
    assert result.classes[0].key.parts == witness.classes[0].key.parts


def test_callback_wall_time_keeps_published_class_and_marks_timeout(symmetry_library):
    model = classifier(symmetry_library)

    def expensive_consumer(_):
        time.sleep(0.35)

    result = model.classify(
        permutations(range(4)),
        stream=True,
        time_limit_seconds=0.3,
        class_callback=expensive_consumer,
    )
    assert not result.complete and result.reason == "time_limit"
    assert len(result.classes) == result.candidates == 1
    assert result.elapsed_seconds >= 0.35
    assert result.classes[0].observed_mapping_count == 1


def test_search_pipeline_streams_exact_keys_and_final_counts(symmetry_library):
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    b = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    model = NativeITSBatchClassifier(a, b, ["C"] * 3, {}, library_path=symmetry_library)
    p, request, keepalive = prepared(a, b)
    callbacks = []
    stream, statistics = model.solve_prepared(
        p, request, stream=True, class_callback=callbacks.append
    )
    batch, original = model.solve_prepared(p, request)
    assert keepalive and statistics.search_complete and original.search_complete
    assert stream.complete and statistics.classification_complete
    assert len(stream.classes) == len(callbacks) == 2
    assert sum(record.observed_mapping_count for record in stream.classes) == 6
    assert stream.classes == batch.classes
