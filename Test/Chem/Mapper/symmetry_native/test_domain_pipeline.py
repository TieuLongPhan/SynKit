"""Class batching must preserve the native kernel's exact restricted space."""

from itertools import permutations

import numpy as np
import pytest

from synkit.Chem.Mapper._native.classification import NativeITSBatchClassifier
from synkit.Chem.Mapper._native.runtime import BACKENDS, CancellationToken
from Test.Chem.Mapper.symmetry_native.test_pipeline import prepared


def problem_and_classifier(library, query, **limits):
    a = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]], dtype=float)
    classifier = NativeITSBatchClassifier(
        a, a, ["C"] * 3, {"tag": ([0, 1, 2], [0, 1, 2])}, library_path=library
    )
    problem, request, keepalive = prepared(a, a, query=query, target=8, **limits)
    request.backend = BACKENDS["kbb"]
    return a, classifier, problem, request, keepalive


@pytest.mark.parametrize("query", [0, 1, 2])
@pytest.mark.parametrize(
    "mask",
    [
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        [[1, 1, 0], [1, 1, 0], [0, 0, 1]],
    ],
)
def test_complete_native_domain_class_pipeline(symmetry_library, query, mask):
    a, classifier, problem, request, keepalive = problem_and_classifier(
        symmetry_library, query
    )
    allowed = np.asarray(mask, dtype=bool)
    before = allowed.copy()
    costs = {}
    for mapping in permutations(range(3)):
        if all(allowed[i, j] for i, j in enumerate(mapping)):
            costs[mapping] = sum(
                abs(a[i, j] - a[mapping[i], mapping[j]])
                for i in range(3)
                for j in range(i)
            )
    target = min(costs.values()) if query == 0 else 4
    selected = {
        m
        for m, cost in costs.items()
        if cost == target or (query == 2 and cost <= target)
    }
    callbacks = []
    result, statistics = classifier.solve_prepared(
        problem,
        request,
        allowed_domains=allowed,
        class_callback=lambda record: callbacks.append(record),
    )
    assert (
        keepalive
        and result.complete
        and statistics.search_complete
        and statistics.classification_complete
    )
    assert np.array_equal(allowed, before) and not request.classify
    assert {record.representative for record in result.classes} == selected
    assert statistics.candidate_count == len(selected) == len(callbacks)
    assert all(
        all(allowed[i, j] for i, j in enumerate(record.representative))
        for record in result.classes
    )
    if query == 0:
        assert statistics.minimum_proven and statistics.minimum2 == 2 * min(
            costs.values()
        )
        if not allowed[0, 0]:
            assert statistics.minimum2 > 0  # the full-space minimum0 is forbidden


@pytest.mark.parametrize("query", [0, 2])
def test_empty_domain_shell_is_complete_and_empty(symmetry_library, query):
    _, classifier, problem, request, keepalive = problem_and_classifier(
        symmetry_library, query
    )
    result, statistics = classifier.solve_prepared(
        problem, request, allowed_domains=np.zeros((3, 3), dtype=bool)
    )
    assert (
        keepalive
        and result.complete
        and statistics.search_complete
        and not result.classes
    )
    assert statistics.candidate_count == 0


@pytest.mark.parametrize(
    "mask", [[[True]], np.full((3, 3), 2), np.full((3, 3), np.nan)]
)
def test_domain_mask_validation(symmetry_library, mask):
    _, classifier, problem, request, keepalive = problem_and_classifier(
        symmetry_library, 2
    )
    with pytest.raises(ValueError, match="zero/one mask"):
        classifier.solve_prepared(problem, request, allowed_domains=mask)
    assert keepalive


def test_domain_caps_cancellation_and_callback_errors(symmetry_library):
    mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    _, classifier, problem, request, keepalive = problem_and_classifier(
        symmetry_library, 2, max_classes=1
    )
    result, statistics = classifier.solve_prepared(
        problem, request, allowed_domains=mask
    )
    assert (
        keepalive
        and statistics.search_complete
        and not statistics.classification_complete
    )
    assert not result.complete and result.reason == "class_limit"
    with CancellationToken(symmetry_library) as token:
        token.cancel()
        request.cancel_token = token.pointer
        result, statistics = classifier.solve_prepared(
            problem, request, allowed_domains=mask
        )
        assert not statistics.search_complete and not statistics.classification_complete
        assert not result.complete and result.reason == "cancelled"
    request.cancel_token = None

    def fail(_):
        raise LookupError("domain class callback")

    with pytest.raises(LookupError, match="domain class callback"):
        classifier.solve_prepared(
            problem, request, allowed_domains=mask, class_callback=fail
        )
