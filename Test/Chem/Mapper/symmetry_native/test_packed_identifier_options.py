"""Optional identifier policy never changes exact certificate equality."""

import ctypes as ct
import hashlib
import os
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from synkit.Chem.Mapper._native.classification import (
    _CALLBACK,
    NativeITSBatchClassifier,
    _Statistics,
)
from synkit.Chem.Mapper._native.core import enumerate_core_mappings
from synkit.Chem.Mapper._native.implicit import (
    make_implicit_classifier,
    map_reaction_implicit,
    prepare_implicit_reaction,
)


@pytest.fixture(scope="module")
def library(mapper_core_library):
    return mapper_core_library


def _classifier(library, mode="legacy", *, n=4):
    a = np.zeros((n, n))
    b = a.copy()
    if n:
        for i in range(n - 1):
            a[i, i + 1] = a[i + 1, i] = 1
        b[0, -1] = b[-1, 0] = 1.5
    return NativeITSBatchClassifier(
        a,
        b,
        ["C"] * n,
        {"escaped": (["quote'\\\x00"] * n, ["line\n"] * n)},
        library_path=library,
        identifier_mode=mode,
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("n", [0, 4])
def test_same_keys_representatives_counts_and_packed_sha(library, stream, n):
    mappings = list(permutations(range(n))) * 2
    old = _classifier(library, n=n).classify(mappings, stream=stream)
    new = _classifier(library, "packed", n=n).classify(mappings, stream=stream)
    assert old.complete and new.complete
    assert old.candidates == new.candidates == len(mappings)
    assert old.canonical_nodes == new.canonical_nodes
    assert [
        (r.key.parts, r.representative, r.cost2, r.observed_mapping_count)
        for r in old.classes
    ] == [
        (r.key.parts, r.representative, r.cost2, r.observed_mapping_count)
        for r in new.classes
    ]
    assert all(
        r.identifier == hashlib.sha256(r.key.parts).hexdigest() for r in new.classes
    )


@pytest.mark.parametrize("updates", [0, 1])
def test_ex_legacy_option_matches_legacy_entrypoint(library, updates):
    classifier = _classifier(library)
    mappings = np.asarray(list(permutations(range(4))) * 2, dtype=np.int32)
    costs = np.zeros(len(mappings), dtype=np.int64)
    actual = []

    @_CALLBACK
    def receive(mapping, cost, observed, packed, size, identifier, _):
        actual.append((ct.string_at(packed, size), identifier.decode(), observed))
        return 0

    stats = _Statistics()
    status = classifier.library.synkit_mapper_classify_batch_ex(
        ct.byref(classifier.problem),
        len(mappings),
        mappings.ctypes.data_as(ct.POINTER(ct.c_int32)),
        costs.ctypes.data_as(ct.POINTER(ct.c_int64)),
        -1,
        -1,
        -1,
        None,
        receive,
        None,
        ct.byref(stats),
        updates,
        0,
    )
    assert status == 0 and stats.complete
    expected = classifier.classify(mappings, stream=bool(updates))
    assert expected.complete
    # Update callbacks can publish a lower count first; final upserts are exact.
    merged = {key: (identifier, observed) for key, identifier, observed in actual}
    assert merged == {
        r.key.parts: (r.identifier, r.observed_mapping_count) for r in expected.classes
    }


@pytest.mark.parametrize("updates,packed", [(-1, 0), (2, 0), (0, -1), (0, 2)])
def test_invalid_c_option_flags_are_rejected(library, updates, packed):
    classifier = _classifier(library)
    status = classifier.library.synkit_mapper_classify_batch_ex(
        None,
        0,
        None,
        None,
        -1,
        -1,
        -1,
        None,
        _CALLBACK(),
        None,
        ct.byref(_Statistics()),
        updates,
        packed,
    )
    assert status == -1
    assert b"flags" in classifier.library.synkit_mapper_classification_error()


def test_old_library_explicitly_rejects_packed_mode():
    baseline = Path(
        os.environ.get(
            "SYNKIT_CLASSIFICATION_BASELINE_LIBRARY",
            "",
        )
    )
    if not baseline.is_file():
        pytest.skip("Optional historical library absent")
    with pytest.raises(RuntimeError, match="does not support packed"):
        _classifier(baseline, "packed")
    assert _classifier(baseline).identifier_mode == "legacy"


@pytest.mark.parametrize("mode", ["wrong", "PACKED", None, 0])
def test_python_rejects_invalid_identifier_mode(library, mode):
    with pytest.raises(ValueError, match="identifier_mode"):
        _classifier(library, mode)


def test_implicit_factory_propagates_mode_and_keys_are_unchanged(library):
    reaction = "CCO>>COC"
    old = map_reaction_implicit(reaction, library_path=library)
    new = map_reaction_implicit(
        reaction, library_path=library, identifier_mode="packed"
    )
    assert old.complete and new.complete
    assert old.minimum_cost == new.minimum_cost == 4
    assert [r.canonical_key_base64 for r in old.reactions] == [
        r.canonical_key_base64 for r in new.reactions
    ]
    assert old.search_result.backend_statistics["identifier_mode"] == "legacy"
    assert new.search_result.backend_statistics["identifier_mode"] == "packed"
    import base64

    assert all(
        r.class_id
        == hashlib.sha256(base64.b64decode(r.canonical_key_base64)).hexdigest()
        for r in new.reactions
    )


def test_explicit_pipeline_retains_restricted_domain_mask(library):
    model = prepare_implicit_reaction("CCC>>CCC")
    n = len(model.graphs[0].labels)
    outputs = []
    for mode in ("legacy", "packed"):
        registry = {}
        classifier = make_implicit_classifier(
            model, library_path=library, identifier_mode=mode
        )
        result = enumerate_core_mappings(
            model.graphs,
            library_path=library,
            backend="kbb",
            allowed_domains=np.eye(n, dtype=np.uint8),
            collect_mappings=False,
            _native_classifier=classifier,
            _native_class_registry=registry,
        )
        assert result.complete and result.minimum_cost == 0
        assert result.backend_statistics["domain_restricted"]
        assert list(registry) == [tuple(range(n))]
        outputs.append(next(iter(registry.values())))
    assert outputs[0].key.parts == outputs[1].key.parts
    assert outputs[1].identifier == hashlib.sha256(outputs[1].key.parts).hexdigest()
