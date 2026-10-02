import numpy as np
import pytest

from synkit.Chem.Mapper.identifiability import Endpoint
from synkit.Chem.Mapper.numerical_scope import validate_study_domain
from synkit.Chem.Mapper.exact.distance_bounds import (
    ResidualBondMass, atom_profile_costs, assignment_edge_lower_bounds,
    conditioned_profile_assignment_lower_bound,
)


def test_domain_boundary_and_unsupported_weights():
    valid = Endpoint((6,)*512, (0,)*512, (4,)*512, ())
    assert validate_study_domain(valid, valid)["scaled_working_envelope"] < 2**53
    large = Endpoint((6,)*513, (0,)*513, (4,)*513, ())
    with pytest.raises(ValueError):
        validate_study_domain(large, large)
    with pytest.raises(ValueError):
        Endpoint((6, 6), (0, 0), (0, 0), ((0, 1, 10**16),))


def test_worker_enforces_numerical_domain_before_search():
    from Experiment.Synister.worker import perform
    endpoint = ".".join(["C"]*513)
    with pytest.raises(ValueError, match="1–512"):
        perform({"stage": "exact", "reaction": endpoint+">>"+endpoint, "search_seconds": 1})
    result = perform({"stage": "exact", "reaction": "CO>>CO", "search_seconds": 1})
    assert result["status"] == "complete"
    assert result["numerical_domain"]["heavy_atoms"] == 2


def test_dense_bounded_profile_and_residual_arithmetic_is_dyadic():
    rng = np.random.default_rng(19512)
    n = 64
    a = rng.choice([0, 1, 1.5, 2, 3], (n, n))
    b = rng.choice([0, 1, 1.5, 2, 3], (n, n))
    a = np.triu(a, 1); a += a.T
    b = np.triu(b, 1); b += b.T
    colors = [6]*n
    profile = atom_profile_costs(a, b, colors, colors)
    # Independent doubled-integer sorted L1 profiles.
    aa, bb = np.sort((2*a).astype(np.int64), axis=1), np.sort((2*b).astype(np.int64), axis=1)
    exact = np.abs(aa[:, None, :]-bb[None, :, :]).sum(axis=2)
    assert np.array_equal(profile*4, exact)
    edges = assignment_edge_lower_bounds(profile, colors, colors)
    assert np.equal(edges*4, np.rint(edges*4)).all()
    residual = ResidualBondMass(a, b, list(range(n)))
    initial = residual.interval(0)
    changes = [(i, residual.remove(i)) for i in range(n)]
    assert residual.interval(n) == (0, 0)
    for i, change in reversed(changes):
        residual.restore(i, change)
    assert residual.interval(0) == initial


def test_maximum_domain_quarter_profiles_and_forced_edges():
    n = 512
    a = np.full((n, n), 3.0)
    b = np.full((n, n), 1.5)
    np.fill_diagonal(a, 0)
    np.fill_diagonal(b, 0)
    colors = [6] * n
    profiles = atom_profile_costs(a, b, colors, colors)
    # Every row has one zero; every compatible assignment has the same cost.
    assert np.equal(profiles, 3 * (n-1) / 4).all()
    forced = assignment_edge_lower_bounds(profiles, colors, colors)
    assert np.equal(forced, 3 * n * (n-1) / 4).all()
    assert np.max(np.abs(forced))*4 < 128*n*n <= 2**25
    residual = ResidualBondMass(a, b, list(range(n)))
    saved = [residual.remove(i) for i in range(n)]
    assert residual.interval(n) == (0, 0)
    for i in reversed(range(n)):
        residual.restore(i, saved[i])
    assert residual.interval(0) == (3*n*(n-1)/4, 9*n*(n-1)/4)


def test_conditioned_cross_profile_bounds_against_literal_remaining_maps():
    from itertools import permutations
    rng = np.random.default_rng(20512)
    for _ in range(12):
        n = 6
        a = np.triu(rng.choice([0, 1, 1.5, 2, 3], (n, n)), 1)
        b = np.triu(rng.choice([0, 1, 1.5, 2, 3], (n, n)), 1)
        a += a.T
        b += b.T
        fixed = {0: 2, 1: 0}
        rows, columns = [2, 3, 4, 5], [1, 3, 4, 5]
        cross = sum(np.abs(a[:, i, None]-b[:, p][None, :]) for i, p in fixed.items())
        lower, costs = conditioned_profile_assignment_lower_bound(
            a, b, cross, rows, columns, [6]*n, [6]*n, return_costs=True)
        committed = abs(a[0, 1]-b[2, 0])
        doubled_a, doubled_b = (2*a).astype(int), (2*b).astype(int)
        for images in permutations(columns):
            mapping = [2, 0, *images]
            actual = sum(abs(int(doubled_a[i, j])-int(doubled_b[mapping[i], mapping[j]]))
                         for i in range(n) for j in range(i+1, n)) / 2
            assert committed + lower <= actual
        assert np.equal(4*costs, np.rint(4*costs)).all()
