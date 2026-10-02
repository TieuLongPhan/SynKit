from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from synkit.Chem.Mapper.exact import assignment_certificate as ac


def test_certified_assignments_against_independent_permutation_minima():
    random = np.random.default_rng(20260920)
    for n in range(1, 8):
        for _ in range(12):
            integer = random.integers(-40, 41, size=(n, n))
            costs = integer / 4
            certificate = ac.certified_lattice_assignment(costs)
            expected = min(sum(int(integer[i, p]) for i, p in enumerate(mapping))
                           for mapping in permutations(range(n)))
            assert certificate.scaled_objective == expected
            assert ac.verify_assignment_certificate(costs, certificate)
            assert sum(certificate.row_potentials)+sum(certificate.column_potentials) == expected


def test_empty_tied_and_boundary_matrices():
    for costs in (np.empty((0, 0)), np.zeros((8, 8)),
                  np.array([[2**38, -2**38], [-2**38, 2**38]], dtype=float)):
        certificate = ac.certified_lattice_assignment(costs)
        assert ac.verify_assignment_certificate(costs, certificate)


@pytest.mark.parametrize("costs", [
    np.zeros((2, 3)), np.zeros((513, 513)), [[float("nan")]],
    [[float("inf")]], [[.1]], [[2**39]],
])
def test_unsupported_domain_is_rejected(costs):
    with pytest.raises(ValueError):
        ac.certified_lattice_assignment(costs)


def test_tampered_certificates_fail():
    costs = np.array([[0., 2.], [1., 0.]])
    certificate = ac.certified_lattice_assignment(costs)
    for tampered in (
        replace(certificate, permutation=(0, 0)),
        replace(certificate, scaled_objective=1),
        replace(certificate, row_potentials=(100, -100)),
        replace(certificate, row_potentials=(True, 0)),
        replace(certificate, column_potentials=(0,)),
        replace(certificate, permutation=None),
        replace(certificate, row_potentials=(2**100, -2**100)),
    ):
        assert not ac.verify_assignment_certificate(costs, tampered)


def test_incorrect_floating_solver_answer_is_not_trusted(monkeypatch):
    monkeypatch.setattr(ac, "linear_sum_assignment", lambda costs: (np.arange(2), np.arange(2)))
    with pytest.raises(ArithmeticError, match="primal-dual"):
        ac.certified_lattice_assignment([[10., 0.], [0., 10.]])


def test_search_bounds_use_certified_lap_and_refuse_bad_candidates(monkeypatch):
    from synkit.Chem.Mapper.exact.distance_bounds import (
        blocked_assignment_extreme, assignment_edge_lower_bounds)
    costs = np.array([[10., 0.], [0., 10.]])
    assert blocked_assignment_extreme(costs, range(2), range(2), [6, 6], [6, 6]) == 0
    assert blocked_assignment_extreme(costs, range(2), range(2), [6, 6], [6, 6], maximize=True) == 20
    assert np.array_equal(assignment_edge_lower_bounds(costs, [6, 6], [6, 6]), [[20, 0], [0, 20]])
    monkeypatch.setattr(ac, "linear_sum_assignment", lambda costs: (np.arange(2), np.arange(2)))
    with pytest.raises(ArithmeticError, match="primal-dual"):
        blocked_assignment_extreme(costs, range(2), range(2), [6, 6], [6, 6])
    with pytest.raises(ArithmeticError, match="primal-dual"):
        assignment_edge_lower_bounds(costs, [6, 6], [6, 6])


def test_generic_nonlattice_assignment_keeps_its_existing_scope():
    from synkit.Chem.Mapper.exact.distance_bounds import blocked_assignment_extreme
    costs = np.array([[.1, .9], [.8, .2]])
    assert blocked_assignment_extreme(costs, range(2), range(2), [6, 6], [6, 6]) == pytest.approx(.3)
