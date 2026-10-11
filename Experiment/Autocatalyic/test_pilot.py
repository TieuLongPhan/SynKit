"""Regression checks for scientific invariants rather than plotting code."""
from dataclasses import replace

import numpy as np
import pytest

from analysis import (parent_criterion, parent_odd_matrix, productive_cores,
                      racemic_parent_states, stability, toy_racemate)
from models import (MODULES, ParentParameters, control_a, control_b, parent,
                    parent_supports)
from reference_check import check_reference
from run import syncrn_roundtrip


@pytest.mark.parametrize("mechanism", ["constructive", "destructive"])
def test_mass_action_jacobian_mirror_and_residue_balance(mechanism):
    p = ParentParameters(mechanism=mechanism, u=.03, kdim=.02, kr=.01)
    model = parent(p)
    rng = np.random.default_rng(1729)
    x = rng.uniform(0, .2, 7)
    numerical = np.column_stack([
        model.rhs(0, x.astype(complex)+1e-25j*np.eye(7)[i]).imag/1e-25
        for i in range(7)])
    np.testing.assert_allclose(model.jacobian(0, x), numerical, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(model.rhs(0, x[model.mirror]),
                               model.rhs(0, x)[model.mirror], atol=1e-14)
    weight = np.array([1, 1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(weight @ model.net, np.zeros(len(model.channels)))
    assert weight @ model.rhs(0, x) == pytest.approx(p.phi*(p.total_feed-weight @ x))
    assert np.isfinite(model.jacobian(0, np.zeros(7))).all()


@pytest.mark.parametrize("mechanism", ["constructive", "destructive"])
def test_explicit_catalyst_and_syncrn_roundtrip(mechanism):
    model = parent(ParentParameters(mechanism=mechanism))
    j = next(i for i, r in enumerate(model.channels) if r.id == "matched_L_f")
    ll = model.species.index("LL")
    assert model.minus[ll, j] == model.plus[ll, j] == 1
    assert model.net[ll, j] == 0
    low = np.ones(7)*.1
    high = low.copy()
    high[ll] *= 2
    assert model.flux(high)[j] == pytest.approx(2*model.flux(low)[j])
    result = syncrn_roundtrip(model)
    assert result["incidence_preserved"]
    assert result["rate_metadata_preserved"]


def test_productivity_positive_negative_and_reverse_support_controls():
    positive = productive_cores(control_a())
    negative = productive_cores(control_b())
    reverse = productive_cores(control_b(1e-6))
    assert len(positive["cores"]) == 2
    assert negative["all_nonempty_support_pairs"] == 225
    assert not negative["cores"]
    assert len(reverse["cores"]) == 1
    core = reverse["cores"][0]
    assert len(core["species"]) == len(core["channels"]) == 4
    witness = np.asarray(core["flux_witness"])
    assert np.min(np.asarray(core["net_stoichiometry"]) @ witness) >= 1-1e-8


def test_toy_sector_eigenvalues_and_racemization_threshold():
    x = toy_racemate()
    np.testing.assert_allclose(x, np.ones(4), atol=1e-13)
    info = stability(control_b(), x)
    assert info["alpha_odd"] == pytest.approx(-6+np.sqrt(37))
    assert info["alpha_even"] == pytest.approx(-6+np.sqrt(13))
    assert stability(control_b(kr=.49), x)["alpha_odd"] > 0
    assert abs(stability(control_b(kr=.5), x)["alpha_odd"]) < 1e-12
    assert stability(control_b(kr=.51), x)["alpha_odd"] < 0
    a = control_a()
    assert stability(a, np.repeat((1+np.sqrt(5))/2, 2))["alpha_odd"] == pytest.approx(-np.sqrt(5))


@pytest.mark.parametrize("mechanism", ["constructive", "destructive"])
def test_parent_quartic_sectors_and_exact_criterion_all_supports(mechanism):
    p = ParentParameters(mechanism=mechanism, u=.002, kdim=.015, kr=.012)
    supports = list(parent_supports())
    assert len(supports) == 16
    for modules in supports:
        model = parent(p, modules)
        for x in racemic_parent_states(p, modules):
            info = stability(model, x)
            assert info["residual_inf"] < 1e-9
            assert info["cross_block_norm"] < 1e-12
            assert info["mirror_commutator_norm"] < 1e-12
            odd = model.q_odd.T @ model.jacobian(0, x) @ model.q_odd
            block = parent_odd_matrix(p, x, modules)
            np.testing.assert_allclose(odd[:2, :2], block, atol=1e-12)
            margin = parent_criterion(p, x, modules)["margin"]
            assert np.linalg.det(block) == pytest.approx(-margin, abs=1e-12)
            assert (info["alpha_odd"] > 1e-8) == (margin > 1e-8)


def test_reverse_rate_constraints_and_exact_zero_channel_policy():
    p = ParentParameters(u=.02, kdim=.03)
    model = parent(p)
    rates = {r.id: r.rate for r in model.channels}
    for r in model.channels:
        if r.id.endswith("_f") and r.module in ("background", "matched", "mismatched"):
            assert rates[r.reverse]/r.rate == pytest.approx(p.ab[1]/p.ab[0])
        if r.id.endswith("_f") and r.module.endswith("ligation"):
            assert rates[r.reverse]/r.rate == pytest.approx(p.kdim)
    limit = parent(replace(p, u=0, kdim=0))
    assert not any(r.module == "background" for r in limit.channels)
    assert not any(r.module.endswith("ligation") and r.id.endswith("_b") for r in limit.channels)


def test_destructive_parent_zero_reverse_limit_remains_chirally_unstable():
    p = ParentParameters(mechanism="destructive", k1=10, k2=80, kmon=0)
    model = parent(p)
    assert not productive_cores(model)["cores"]
    states = racemic_parent_states(p)
    assert len(states) == 1
    info = stability(model, states[0])
    assert info["alpha_even"] < 0 < info["alpha_odd"]
    assert info["alpha_odd"] == pytest.approx(.02432949813216667, abs=1e-12)
    assert parent_criterion(p, states[0])["margin"] > 0


def test_independent_compiled_archive_equations_and_selected_trajectories():
    assert check_reference()["passed"]
