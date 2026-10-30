"""Exact production palette preparation versus the uncached constructor oracle."""

from enum import IntEnum
from types import SimpleNamespace

import numpy as np
import pytest

from synkit.Chem.Mapper._native import classification
from Test.Chem.Mapper.symmetry_native import _palette_reference


@pytest.fixture
def factories(monkeypatch):
    # Constructor equivalence requires no native execution or optional engines.
    monkeypatch.setattr(classification, "_load", lambda _: object())
    monkeypatch.setattr(_palette_reference, "_load", lambda _: object())
    return (
        _palette_reference.UncachedReferenceClassifier,
        classification.NativeITSBatchClassifier,
    )


def assert_equal(a, b):
    assert a.n == b.n
    assert list(a.palette) == list(b.palette)
    assert np.array_equal(a.node_colors, b.node_colors)
    assert np.array_equal(a.edge_colors, b.edge_colors)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(a.edge_matrices, b.edge_matrices)
    )
    for name in ("n", "edge_states_r", "edge_states_p", "palette_size"):
        assert getattr(a.problem, name) == getattr(b.problem, name)


class State(IntEnum):
    FIRST = 1


class SameReprDict(dict):
    def __repr__(self):
        return "identical_repr"


@pytest.mark.parametrize("n", [0, 1, 4, 16])
def test_exact_types_signed_zero_and_custom_edge_mapping_repr(factories, n):
    a = np.zeros((n, n))
    b = np.zeros((n, n))
    attrs = np.full((n, n), "", dtype=object)
    states = [
        True,
        1,
        1.0,
        "1",
        State.FIRST,
        SameReprDict(x=1),
        SameReprDict(x=2),
        (False, -0.0),
    ]
    for i in range(n):
        for j in range(i):
            a[i, j] = a[j, i] = float((i + j) % 3)
            b[i, j] = b[j, i] = -0.0 if (i + j) % 2 else 0.5
            attrs[i, j] = attrs[j, i] = states[(i + j) % len(states)]
    values = (states * ((n + len(states) - 1) // len(states)))[:n]
    props = {
        "literal": (values, list(reversed(values))),
        "dummy": ([False] * n, [True] * n),
    }
    args = (a, b, [(6, 0)] * n, props)
    kwargs = {"library_path": "unused", "edge_properties": {"typed": (attrs, attrs)}}
    assert_equal(*(factory(*args, **kwargs) for factory in factories))


@pytest.mark.parametrize("seed", range(12))
def test_random_discrete_exact_palette_equivalence(factories, seed):
    rng = np.random.default_rng(seed)
    n = 1 + seed
    matrices = []
    edge_attrs = []
    for _ in range(2):
        upper = np.triu(rng.integers(0, 5, (n, n)), 1) / 2
        matrices.append(upper + upper.T)
        values = np.full((n, n), "", dtype=object)
        for i in range(n):
            for j in range(i):
                values[i, j] = values[j, i] = ["", "STEREOE", "STEREOZ"][
                    (i + j + seed) % 3
                ]
        edge_attrs.append(values)
    properties = {
        "charge": (
            [int(x) for x in rng.integers(-1, 2, n)],
            [int(x) for x in rng.integers(-1, 2, n)],
        ),
        "dummy": (
            [bool(x) for x in rng.integers(0, 2, n)],
            [bool(x) for x in rng.integers(0, 2, n)],
        ),
        "stereo": (["R", "", "S"] * n, ["S", "R", ""] * n),
    }
    properties["stereo"] = tuple(values[:n] for values in properties["stereo"])
    args = (*matrices, [(6 if i % 2 else 8, i % 3) for i in range(n)], properties)
    kwargs = {"library_path": "unused", "edge_properties": {"stereo": edge_attrs}}
    assert_equal(*(factory(*args, **kwargs) for factory in factories))


@pytest.mark.parametrize(
    "case",
    [
        "size",
        "shape",
        "negative",
        "nan",
        "infinite",
        "asymmetric",
        "self_loop",
        "atom_property",
        "edge_shape",
        "edge_asymmetric",
    ],
)
def test_original_validation_and_error_messages_preserved(factories, case):
    a, b, n, props, edges = np.zeros((2, 2)), np.zeros((2, 2)), 2, {}, {}
    if case == "size":
        n = 513
        a = b = np.zeros((n, n))
    elif case == "shape":
        a = np.zeros((1, 1))
    elif case in {"negative", "nan", "infinite"}:
        a[0, 1] = a[1, 0] = {
            "negative": -1,
            "nan": float("nan"),
            "infinite": float("inf"),
        }[case]
    elif case == "asymmetric":
        a[0, 1] = 1
    elif case == "self_loop":
        a[0, 0] = 1
    elif case == "atom_property":
        props = {"charge": ([0], [0, 0])}
    elif case == "edge_shape":
        edges = {"stereo": (np.zeros((1, 1)), np.zeros((2, 2)))}
    else:
        attrs = np.full((2, 2), "", dtype=object)
        attrs[0, 1] = "STEREOE"
        edges = {"stereo": (attrs, attrs)}
    errors = []
    for factory in factories:
        with pytest.raises(ValueError) as error:
            factory(
                a, b, ["C"] * n, props, library_path="unused", edge_properties=edges
            )
        errors.append(str(error.value))
    if case in {"size", "shape"}:
        assert all("requires square endpoints" in message for message in errors)
    else:
        assert errors[0] == errors[1]


def test_legacy_library_capacity_remains_256(factories):
    native = factories[1]
    n = 256
    result = native(
        np.zeros((n, n)), np.zeros((n, n)), ["C"] * n, {}, library_path="unused"
    )
    assert result.n == 256
    with pytest.raises(ValueError, match="at most 256"):
        native(
            np.zeros((257, 257)),
            np.zeros((257, 257)),
            ["C"] * 257,
            {},
            library_path="unused",
        )


@pytest.mark.parametrize("n", [257, 512])
def test_large_constructor_requires_advertised_capacity(factories, monkeypatch, n):
    native = factories[1]
    library = SimpleNamespace(synkit_mapper_classification_max_vertices=lambda: 512)
    monkeypatch.setattr(classification, "_load", lambda _: library)
    result = native(
        np.zeros((n, n)), np.zeros((n, n)), ["C"] * n, {}, library_path="unused"
    )
    assert result.n == n and result.library is library


def test_small_invalid_input_does_not_load_library(factories, monkeypatch):
    def unexpected(_):
        raise AssertionError("Small input validation should precede native load")

    monkeypatch.setattr(classification, "_load", unexpected)
    with pytest.raises(ValueError, match="symmetric"):
        factories[1](
            np.array([[0, 1], [0, 0]]),
            np.zeros((2, 2)),
            ["C"] * 2,
            {},
            library_path="unused",
        )
