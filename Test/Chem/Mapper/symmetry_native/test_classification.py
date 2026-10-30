from itertools import permutations

import networkx as nx
import numpy as np
import pytest

from synkit.Chem.Mapper._native.classification import NativeITSBatchClassifier
from synkit.Chem.Mapper.exact.native.canonical import compact_canonical_code
from synkit.Chem.Mapper.reporting.spectrum import (
    _attributed_its_graph,
    _code_identifier,
    _exact_code,
)


def graph(a, b, elements, properties, mapping, stereo=None):
    m = np.asarray(mapping, dtype=int)
    g = _attributed_its_graph(a, b[np.ix_(m, m)], elements, properties, mapping)
    if stereo is not None:
        for i, j, attrs in g.edges(data=True):
            attrs["color"] += (stereo[0][i, j], stereo[1][m[i], m[j]])
    return g


def independent_classes(graphs):
    representatives = []

    def equal(left, right):
        return left["color"] == right["color"]

    for candidate in graphs:
        if not any(nx.is_isomorphic(candidate, old, node_match=equal, edge_match=equal) for old in representatives):
            representatives.append(candidate)
    return representatives


@pytest.mark.parametrize("attribute", ["charge", "radical", "hcount", "isotope", "dummy", "stereo"])
def test_exact_attributes_and_legacy_identifiers(symmetry_library, attribute):
    a = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
    b = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
    values = {
        "charge": [0, 1, -1, 0],
        "radical": [0, 1, 0, 0],
        "hcount": [0, 1, 2, 0],
        "isotope": [0, 13, 0, 0],
        "dummy": [False, False, True, False],
        "stereo": ["", "R", "S", ""],
    }[attribute]
    props = {attribute: (values, list(reversed(values)))}
    elements = ["C"] * 4
    mappings = list(permutations(range(4)))
    native = NativeITSBatchClassifier(a, b, elements, props, library_path=symmetry_library)
    result = native.classify(mappings, [4] * len(mappings))
    assert result.complete and result.candidates == 24
    expected = independent_classes([graph(a, b, elements, props, m) for m in mappings])
    assert len(result.classes) == len(expected)
    assert sum(record.observed_mapping_count for record in result.classes) == 24
    for record in result.classes:
        code, reason = _exact_code(
            graph(a, b, elements, props, record.representative),
            timeout_seconds=5,
            max_search_nodes=100000,
        )
        assert reason is None
        assert compact_canonical_code(tuple(record.key)) == code
        assert record.identifier == _code_identifier(code)


def test_ez_and_endpoint_renumbering(symmetry_library):
    a = np.array([[0, 1, 0, 0], [1, 0, 2, 0], [0, 2, 0, 1], [0, 0, 1, 0]], dtype=float)
    b = a.copy()
    sr = np.full((4, 4), "STEREONONE", dtype=object)
    sp = sr.copy()
    sr[1, 2] = sr[2, 1] = "STEREOE"
    sp[1, 2] = sp[2, 1] = "STEREOZ"
    props = {"dummy": ([False] * 4, [False] * 4)}
    mappings = list(permutations(range(4)))

    def classify(left, right, properties, stereo):
        return NativeITSBatchClassifier(
            left,
            right,
            [("C", 13)] * 4,
            properties,
            edge_properties={"stereo": stereo},
            library_path=symmetry_library,
        ).classify(mappings)

    result = classify(a, b, props, (sr, sp))
    expected = independent_classes([graph(a, b, [("C", 13)] * 4, props, m, (sr, sp)) for m in mappings])
    assert result.complete and len(result.classes) == len(expected)
    for record in result.classes:
        code, reason = _exact_code(
            graph(a, b, [("C", 13)] * 4, props, record.representative, (sr, sp)),
            timeout_seconds=5,
            max_search_nodes=100000,
        )
        assert reason is None
        assert record.identifier == _code_identifier(code)
    r, p = np.array([2, 0, 3, 1]), np.array([1, 3, 0, 2])
    changed = classify(a[np.ix_(r, r)], b[np.ix_(p, p)], props, (sr[np.ix_(r, r)], sp[np.ix_(p, p)]))
    assert {x.key for x in result.classes} == {x.key for x in changed.classes}


def test_twins_disconnected_and_empty(symmetry_library):
    for n, a in [
        (0, np.zeros((0, 0))),
        (5, np.zeros((5, 5))),
        (4, np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])),
    ]:
        result = NativeITSBatchClassifier(a, a, ["H"] * n, {}, library_path=symmetry_library).classify(
            permutations(range(n))
        )
        expected = independent_classes([graph(a, a, ["H"] * n, {}, m) for m in permutations(range(n))])
        assert result.complete and len(result.classes) == len(expected)


def test_interruption_and_callback_exception(symmetry_library):
    a = np.zeros((4, 4))
    b = a.copy()
    b[0, 1] = b[1, 0] = 1
    classifier = NativeITSBatchClassifier(
        a,
        b,
        ["C"] * 4,
        {"charge": ([0, 1, 2, 3], [0, 1, 2, 3])},
        library_path=symmetry_library,
    )
    mappings = list(permutations(range(4)))
    for kwargs, reason in [
        ({"time_limit_seconds": 0}, "time_limit"),
        ({"max_canonical_nodes": 1}, "canonical_node_limit"),
        ({"max_classes": 1}, "class_limit"),
        ({"class_callback": lambda _: True}, "callback_stop"),
    ]:
        result = classifier.classify(mappings, **kwargs)
        assert not result.complete and result.reason == reason

    def fail(_):
        raise LookupError("callback fixture")

    with pytest.raises(LookupError, match="callback fixture"):
        classifier.classify(mappings, class_callback=fail)
    with pytest.raises(RuntimeError, match="bijection"):
        classifier.classify([[0, 0, 1, 2]])


@pytest.mark.parametrize(
    "reaction",
    [
        "[NH4+].[OH-]>>N.O",
        "[13CH3]CO>>[13CH3]CO",
        "CO>>C",
        "[CH3].[OH]>>CO",
        "F[C@H](Cl)Br>>F[C@@H](Cl)Br",
        r"F/C=C/F>>F/C=C\F",
    ],
)
def test_existing_chemistry_frontend_contract(symmetry_library, reaction):
    from synkit.Chem.Mapper.reaction_search import _graphs, _parse
    from synkit.Chem.Mapper.slap.lap import _adjacency_and_elements

    mols, _ = _parse(reaction, "heavy")
    endpoints = _graphs(mols, "dummy", "bond")
    a, elements = _adjacency_and_elements(endpoints[0], False)
    b, other_elements = _adjacency_and_elements(endpoints[1], False)
    n = len(elements)
    properties = {
        name: (endpoints[0].props[name], endpoints[1].props[name])
        for name in ("charge", "radical", "hcount", "stereo", "dummy")
    }
    stereo = []
    for mol in mols:
        matrix = np.full((n, n), "", dtype=object)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            matrix[i, j] = matrix[j, i] = str(bond.GetStereo())
        stereo.append(matrix)
    mappings = [m for m in permutations(range(n)) if all(elements[i] == other_elements[m[i]] for i in range(n))]
    classifier = NativeITSBatchClassifier.from_endpoint_graphs(mols, endpoints, library_path=symmetry_library)
    result = classifier.classify(mappings)
    assert result.complete
    expected = independent_classes([graph(a, b, elements, properties, m, stereo) for m in mappings])
    assert len(result.classes) == len(expected)
    for record in result.classes:
        code, reason = _exact_code(
            graph(a, b, elements, properties, record.representative, stereo),
            timeout_seconds=5,
            max_search_nodes=100000,
        )
        assert reason is None and record.identifier == _code_identifier(code)


def test_exact_edge_types_and_signed_zero(symmetry_library):
    a = np.array([[0, -0.0, 1], [-0.0, 0, 1], [1, 1, 0]], dtype=float)
    b = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    left = np.array([[0, True, 1], [True, 0, False], [1, False, 0]], dtype=object)
    right = np.full((3, 3), "", dtype=object)
    classifier = NativeITSBatchClassifier(
        a,
        b,
        ["C"] * 3,
        {},
        edge_properties={"edge_type": (left, right)},
        library_path=symmetry_library,
    )
    result = classifier.classify(permutations(range(3)))
    assert result.complete
    for record in result.classes:
        code, reason = _exact_code(
            graph(a, b, ["C"] * 3, {}, record.representative, (left, right)),
            timeout_seconds=5,
            max_search_nodes=100000,
        )
        assert reason is None and record.identifier == _code_identifier(code)
