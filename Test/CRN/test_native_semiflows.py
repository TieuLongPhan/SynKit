"""Native CRN enumeration: independent small-cone oracle and exact witnesses."""

from itertools import combinations
import numpy as np
import pytest
from synkit.CRN import (
    build_native,
    enumerate_semiflows,
    enumerate_structural_sets,
    conserved_moieties,
    PetriAnalyzer,
    SemiflowSearchLimit,
)
from synkit.CRN.Petrinet.minimal_semiflows import minimal_semiflows
from synkit.CRN.Props.stoich import _exact_null_space
from Experiment.CRN.benchmarks.networks import BENCHMARK_NETWORKS
from Experiment.CRN.benchmarks.scaling import generate_network


def oracle(s, kind):
    matrix = s if kind == "p" else s.T
    n = matrix.shape[0]
    out = []
    for size in range(1, n + 1):
        for support in combinations(range(n), size):
            if any(set(np.flatnonzero(row)) < set(support) for row in out):
                continue
            basis = _exact_null_space(matrix[list(support), :].T.tolist(), size)
            if len(basis) == 1 and all(v > 0 for v in basis[0]):
                row = [0] * n
                for i, v in zip(support, basis[0]):
                    row[i] = v
                out.append(row)
    return sorted(out, key=lambda row: (sum(v != 0 for v in row), row))


@pytest.mark.parametrize("kind", ["p", "t"])
def test_native_reference_networks(exact_library, kind):
    for entry in BENCHMARK_NETWORKS:
        s = entry.build().to_stoichiometric_matrices()["S"]
        expected = minimal_semiflows(s, kind=kind, backend="python")
        actual = enumerate_semiflows(s, kind=kind, library_path=exact_library)
        assert actual.complete, (entry.name, actual.termination_reason)
        assert actual.value["flows"] == expected, entry.name


@pytest.mark.parametrize("kind", ["p", "t"])
def test_native_against_independent_support_oracle(exact_library, kind):
    rng = np.random.default_rng(1969)
    for _ in range(30):
        s = rng.integers(-2, 3, size=(int(rng.integers(1, 6)), int(rng.integers(1, 6))))
        expected = oracle(s, kind)
        assert minimal_semiflows(s, kind=kind) == expected
        assert (
            enumerate_semiflows(s, kind=kind, library_path=exact_library).value["flows"]
            == expected
        )


def test_large_exact_coefficients(exact_library):
    k = 10**12
    s = np.array([[-k, 0], [1, -k], [0, 1]], dtype=np.int64)
    result = enumerate_semiflows(s, library_path=exact_library)
    assert result.complete and result.value["flows"] == [[1, k, k * k]]
    assert minimal_semiflows(s, backend="cpp", library_path=exact_library) == [
        [1, k, k * k]
    ]
    n = 2**53 + 1
    s = np.array([[n], [-(n + 1)]], dtype=np.int64)
    assert enumerate_semiflows(s, library_path=exact_library).value["flows"] == [
        [n + 1, n]
    ]


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"max_rows": 1}, "row_limit"),
        ({"max_candidates": 1}, "candidate_limit"),
        ({"max_coefficient_bits": 1}, "coefficient_limit"),
        ({"time_limit": 1e-12}, "time_limit"),
    ],
)
def test_limits_never_return_intermediate_tableaux(exact_library, options, reason):
    s = np.array([[-2, 0], [1, -2], [0, 1]])
    result = enumerate_semiflows(s, library_path=exact_library, **options)
    assert not result.complete and result.termination_reason == reason
    assert result.value["flows"] == []
    with pytest.raises(SemiflowSearchLimit) as exc:
        minimal_semiflows(s, backend="cpp", library_path=exact_library, **options)
    assert exc.value.result.termination_reason == reason


def test_result_limit_and_validation(exact_library):
    s = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    result = enumerate_semiflows(s, library_path=exact_library, max_results=1)
    assert not result.complete and result.termination_reason == "result_limit"
    for bad in [np.array([[float("nan")]]), np.array([[1.2]])]:
        with pytest.raises(ValueError):
            enumerate_semiflows(bad, library_path=exact_library)
    with pytest.raises(OverflowError):
        enumerate_semiflows(
            np.array([[2**63]], dtype=np.uint64), library_path=exact_library
        )


def test_public_pool_api_and_complete_native_structural_search(exact_library):
    c = generate_network("reversible_chain", 66)
    assert conserved_moieties(c, library_path=exact_library) == [[1] * 67]
    for kind in ["siphon", "trap"]:
        a = enumerate_structural_sets(
            c, kind=kind, backend="cpp", library_path=exact_library
        )
        b = enumerate_structural_sets(c, kind=kind)
        assert a.complete and a.value["sets"] == b.value["sets"]
        assert a.value["enumeration_backend"] == "cpp"
        limited = enumerate_structural_sets(
            c, kind=kind, backend="cpp", library_path=exact_library, max_nodes=1
        )
        assert not limited.complete and limited.termination_reason == "node_limit"


def test_analyzer_uses_native_backend_and_reuses_results(exact_library):
    from synkit.CRN import SynCRN

    c = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
    a = PetriAnalyzer(c, library_path=exact_library).compute_all()
    assert a.persistence_ok is True
    assert a.as_dict() == PetriAnalyzer(c).compute_all().as_dict()


def test_native_structural_enumeration_against_exhaustive_subsets(exact_library):
    from synkit.CRN import PetriNet

    rng = np.random.default_rng(2026)
    for _ in range(200):
        net = PetriNet()
        for i in range(5):
            net.add_place(str(i))
        for t in range(6):
            pre = {str(i): 1 for i in range(5) if rng.random() < 0.35}
            post = {str(i): 1 for i in range(5) if rng.random() < 0.35}
            net.add_transition(str(t), pre=pre, post=post)
        for kind in ["siphon", "trap"]:
            minimal = []
            for size in range(1, 6):
                for group in combinations(net.place_order, size):
                    group = set(group)
                    if any(s <= group for s in minimal):
                        continue
                    closed = True
                    for transition in net.transitions.values():
                        trigger, required = (
                            (transition.post, transition.pre)
                            if kind == "siphon"
                            else (transition.pre, transition.post)
                        )
                        if group.intersection(trigger) and not group.intersection(
                            required
                        ):
                            closed = False
                            break
                    if closed:
                        minimal.append(group)
            result = enumerate_structural_sets(
                net, kind=kind, backend="cpp", library_path=exact_library
            )
            assert result.complete
            assert {frozenset(s) for s in result.value["sets"]} == {
                frozenset(s) for s in minimal
            }


def test_native_structural_deadlines_and_result_limits(exact_library):
    from synkit.CRN import SynCRN

    c = SynCRN.from_reaction_strings(["A>>B", "B>>A", "C>>D", "D>>C"])
    r = enumerate_structural_sets(c, library_path=exact_library, max_results=1)
    assert not r.complete and r.termination_reason == "result_limit"
    r = enumerate_structural_sets(c, library_path=exact_library, time_limit=1e-12)
    assert not r.complete and r.termination_reason == "time_limit"
    for value in [float("nan"), float("inf"), 0]:
        with pytest.raises(ValueError):
            enumerate_structural_sets(c, library_path=exact_library, time_limit=value)


def test_closure_only_build_declares_missing_exact_backend(tmp_path):
    library = build_native(tmp_path, with_semiflows=False)
    from synkit.CRN import SynCRN

    assert enumerate_structural_sets(
        SynCRN.from_reaction_strings(["A>>B"]), library_path=library
    ).complete
    with pytest.raises(RuntimeError, match="no exact semiflow backend"):
        enumerate_semiflows(np.array([[-1], [1]]), library_path=library)


@pytest.mark.parametrize("size,expected", [(50, 331), (100, 272), (200, 47), (400, 87)])
def test_complete_random_family_siphons(exact_library, size, expected):
    from synkit.CRN.Petrinet.net import PetriNet
    from synkit.CRN.Petrinet.siphon_search import _incidence, _max_closed_within

    model = generate_network("random_sparse", size, seed=0)
    result = enumerate_structural_sets(
        model,
        library_path=exact_library,
        max_nodes=None,
        max_results=None,
        time_limit=None,
    )
    assert result.complete and result.termination_reason == "complete"
    groups = {frozenset(group) for group in result.value["sets"]}
    assert len(groups) == expected
    net = PetriNet.from_syncrn(model)
    pre, post = _incidence(net)
    for group in groups:
        assert group and all(
            not (group & post[t]) or group & pre[t] for t in net.transition_order
        )
        for place in group:
            assert not _max_closed_within(
                set(group) - {place},
                trigger=post,
                required=pre,
                transition_order=net.transition_order,
            )


def test_positive_kernel_exact_primal_and_dual_certificates():
    from fractions import Fraction
    from synkit.CRN import SynCRN
    from synkit.CRN.native_analysis import positive_kernel_certificate

    model = SynCRN.from_reaction_strings(["A>>B"])
    s = model.to_stoichiometric_matrices()["S"]
    primal = positive_kernel_certificate(model, kind="p", include_problem=True)
    assert primal["shape"] == list(s.T.shape)
    assert primal["problem_matrix"] == s.T.tolist()
    assert primal["holds"] is True and primal["exact"]
    x = primal["primal_witness"]
    assert all(v > 0 for v in x)
    assert all(sum(int(c) * v for c, v in zip(row, x)) == 0 for row in s.T)
    dual = positive_kernel_certificate(model, kind="t", include_problem=True)
    assert dual["shape"] == list(s.shape)
    assert dual["problem_matrix"] == s.tolist()
    assert dual["holds"] is False and dual["exact"]
    y = list(map(Fraction, dual["dual_witness"]))
    image = [
        sum(int(s[i, j]) * y[i] for i in range(s.shape[0])) for j in range(s.shape[1])
    ]
    assert all(v >= 0 for v in image) and any(v > 0 for v in image)
    assert image == list(map(Fraction, dual["dual_image"]))


def test_floating_infeasibility_is_not_an_exact_certificate():
    from types import SimpleNamespace
    from unittest.mock import patch
    from synkit.CRN import SynCRN
    from synkit.CRN.native_analysis import positive_kernel_certificate

    model = SynCRN.from_reaction_strings(["A>>B"])
    answers = [
        SimpleNamespace(success=False, status=2),
        SimpleNamespace(success=True, status=0, x=np.zeros(2)),
    ]
    with patch("scipy.optimize.linprog", side_effect=answers):
        result = positive_kernel_certificate(model, kind="t")
    assert result["holds"] is None and result["exact"] is False
