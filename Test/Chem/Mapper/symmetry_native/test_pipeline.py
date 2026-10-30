import numpy as np
import pytest

from synkit.Chem.Mapper._native.classification import NativeITSBatchClassifier
from synkit.Chem.Mapper._native.runtime import (
    INT,
    LONG,
    CancellationToken,
    Problem,
    Request,
)


def prepared(a, b, *, query=2, target=100, symmetry=False, **limits):
    n = len(a)
    matrices = [np.ascontiguousarray(x * 2, dtype=np.int64) for x in (a, b)]
    colors = [np.zeros(n, dtype=np.int32) for _ in range(4)]
    edges = [np.zeros((n, n), dtype=np.int32) for _ in range(2)]
    keepalive = matrices + colors + edges
    problem = Problem(
        1,
        n,
        *(x.ctypes.data_as(LONG) for x in matrices),
        *(x.ctypes.data_as(INT) for x in colors + edges),
    )
    request = Request()
    request.abi_version, request.backend, request.query = 1, 6, query
    request.workers, request.symmetry = 1, int(symmetry)
    request.target2 = target
    request.seconds = limits.pop("seconds", -1)
    for name in (
        "max_states",
        "max_candidates",
        "max_classes",
        "max_frontier",
        "memory_limit_bytes",
    ):
        setattr(request, name, limits.pop(name, -1))
    assert not limits
    return problem, request, keepalive


def test_native_search_to_class_batch(symmetry_library):
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    b = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    classifier = NativeITSBatchClassifier(
        a, b, ["C"] * 3, {}, library_path=symmetry_library
    )
    problem, request, keepalive = prepared(a, b)
    callbacks = []
    result, statistics = classifier.solve_prepared(
        problem, request, class_callback=lambda record: callbacks.append(record)
    )
    assert keepalive
    assert (
        statistics.search_complete
        and statistics.classification_complete
        and result.complete
    )
    assert (
        statistics.candidate_count == 6 and len(callbacks) == len(result.classes) == 2
    )
    assert (
        statistics.class_count == 2
        and sum(x.observed_mapping_count for x in callbacks) == 6
    )
    assert (
        statistics.classification_seconds >= 0 and statistics.enumeration_seconds >= 0
    )
    for record in callbacks:
        direct = classifier.classify([record.representative], [record.cost2])
        assert direct.classes[0].key == record.key
    assert not request.classify  # caller-owned request is not mutated


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"max_canonical_nodes": 1}, "canonical_node_limit"),
        ({"max_classes": 1}, "class_limit"),
    ],
)
def test_class_limits_keep_search_completion_separate(symmetry_library, kwargs, reason):
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    b = np.zeros((3, 3))
    classifier = NativeITSBatchClassifier(
        a,
        b,
        ["C"] * 3,
        {"charge": ([0, 1, 2], [0, 1, 2])},
        library_path=symmetry_library,
    )
    problem, request, keepalive = prepared(
        a, b, max_classes=kwargs.get("max_classes", -1)
    )
    result, statistics = classifier.solve_prepared(
        problem, request, max_canonical_nodes=kwargs.get("max_canonical_nodes")
    )
    assert keepalive and statistics.search_complete
    assert (
        not result.complete
        and not statistics.classification_complete
        and result.reason == reason
    )


def test_pipeline_cancellation_and_callback_exception(symmetry_library):
    a = np.zeros((4, 4))
    classifier = NativeITSBatchClassifier(
        a, a, ["H"] * 4, {}, library_path=symmetry_library
    )
    with CancellationToken(symmetry_library) as token:
        token.cancel()
        result = classifier.classify([range(4)], cancel_token=token)
        assert not result.complete and result.reason == "cancelled"
        problem, request, keepalive = prepared(a, a)
        request.cancel_token = token.pointer
        result, statistics = classifier.solve_prepared(problem, request)
        assert (
            keepalive
            and not statistics.search_complete
            and not statistics.classification_complete
        )

    def fail(_):
        raise LookupError("native class callback")

    problem, request, keepalive = prepared(a, a)
    with pytest.raises(LookupError, match="native class callback"):
        classifier.solve_prepared(problem, request, class_callback=fail)
    assert keepalive


def test_classified_slap_preserves_core_failure_message(tmp_path_factory):
    """The inherited SLAP failure happens before class processing has an error."""
    import os
    import shutil

    from synkit.Chem.Mapper._native.build import build_mapper_native

    library = os.environ.get("SYNKIT_MAPPER_LIBRARY")
    if library is None:
        if shutil.which("g++") is None:
            pytest.skip("native SLAP requires a C++17 compiler")
        library = build_mapper_native(tmp_path_factory.getbasetemp() / "mapper-native")
    # Same feasible fixture as slap/test_native_slap.py's inherited
    # stop-on-multiple failure; the heuristic reaches no full bijection.
    a2 = np.array(
        [
            [0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 4, 1],
            [0, 0, 3, 4, 0, 0],
            [2, 3, 0, 1, 0, 0],
        ]
    )
    b2 = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 3],
            [0, 0, 0, 0, 3, 0],
        ]
    )
    classifier = NativeITSBatchClassifier(
        a2 / 2, b2 / 2, ["C"] * 6, {}, library_path=library
    )
    # A stale classification error must not replace a fresh search failure.
    with pytest.raises(RuntimeError, match="bijection"):
        classifier.classify([[0, 0, 1, 2, 3, 4]])
    problem, request, keepalive = prepared(a2 / 2, b2 / 2, query=0)
    request.backend = 0
    request.slap_max_fingerprints = 10000
    flags = np.ones(6, dtype=np.int32)
    request.slap_stop_flags = flags.ctypes.data_as(INT)
    with pytest.raises(RuntimeError, match="no full bijection"):
        classifier.solve_prepared(problem, request)
    assert keepalive and flags.all()


def test_class_pipeline_forwards_backend_frontier_memory_cap(symmetry_library):
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    classifier = NativeITSBatchClassifier(
        a, a, ["C"] * 3, {}, library_path=symmetry_library
    )
    for cap, complete in [(64 * 1024 * 1024, True), (1, False)]:
        problem, request, keepalive = prepared(a, a, query=0, memory_limit_bytes=cap)
        request.backend = 5  # actual native A*, whose cap counts frontier storage
        result, statistics = classifier.solve_prepared(problem, request)
        assert keepalive and bool(statistics.search_complete) == complete
        if complete:
            assert result.complete and statistics.classification_complete
        else:
            assert statistics.reason == 7  # memory limit, not unsupported
