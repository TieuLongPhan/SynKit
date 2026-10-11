"""Regression tests for the sector comparison and the producer closure."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import (productive_cores, racemic_parent_states,  # noqa: E402
                      spectral_abscissa, toy_racemate)
from models import (MODULES, ParentParameters, control_a, control_b,  # noqa: E402
                    parent, parent_supports)
from sign_structure import (POINTS, analyze_state, pair_blocks,  # noqa: E402
                            producer_closure, sector_matrices)


def test_sector_comparison_on_random_cooperative_blocks():
    rng = np.random.default_rng(1729)
    for _ in range(2000):
        n0, n1 = rng.integers(0, 4), rng.integers(1, 5)
        a = rng.uniform(0, 1, (n1, n1))*(rng.random((n1, n1)) < 0.6)
        np.fill_diagonal(a, -rng.uniform(0, 3, n1))
        b = rng.uniform(0, 1, (n1, n1))*(rng.random((n1, n1)) < 0.6)
        e = rng.uniform(0, 1, (n0, n0))
        np.fill_diagonal(e, -rng.uniform(0, 3, n0))
        blocks = {"A": a, "B": b, "E": e, "C": rng.uniform(0, 1, (n0, n1)),
                  "G": rng.uniform(0, 1, (n1, n0))}
        even, odd = sector_matrices(blocks)
        pair_sum = spectral_abscissa(a+b)
        assert spectral_abscissa(odd) <= pair_sum+1e-10
        assert pair_sum <= spectral_abscissa(even)+1e-10


def test_one_negative_cross_entry_can_reverse_the_order():
    a = np.array([[-1.0, 0.0], [1.0, -1.0]])
    b = np.array([[0.0, -2.0], [0.0, 0.0]])
    even, odd = sector_matrices({"A": a, "B": b, "E": np.zeros((0, 0)),
                                 "C": np.zeros((0, 2)), "G": np.zeros((2, 0))})
    assert spectral_abscissa(even) < 0 < spectral_abscissa(odd)


@pytest.mark.parametrize("mechanism", ["constructive", "destructive"])
def test_pair_blocks_reproduce_the_orthonormal_sectors(mechanism):
    p = POINTS[mechanism]
    model = parent(p)
    for x in racemic_parent_states(p):
        blocks, _ = pair_blocks(model, x)
        even, odd = sector_matrices(blocks)
        j = model.jacobian(0, x)
        for ours, q in ((even, model.q_even), (odd, model.q_odd)):
            want = np.sort_complex(np.linalg.eigvals(q.T @ j @ q))
            assert np.allclose(np.sort_complex(np.linalg.eigvals(ours)), want, atol=1e-9)


def test_symbolic_odd_block_and_determinant():
    sp = pytest.importorskip("sympy")
    (P, L, D, LL, LD, DL, DD, u, k1, k2, a1, a2, a, b, kd, phi, kr, p, m, h, t) = sp.symbols(
        "P L D LL LD DL DD u k1 k2 a1 a2 a b kd phi kr p m h t")
    Pin, Lin = sp.symbols("Pin Lin")
    RL = (u+k1*LL+k2*DD)*(a*P-b*L)
    RD = (u+k1*DD+k2*LL)*(a*P-b*D)
    HL, HD = a1*(L**2-kd*LL), a1*(D**2-kd*DD)
    TLD, TDL = a2*(L*D-kd*LD), a2*(L*D-kd*DL)
    f = sp.Matrix([phi*(Pin-P)-RL-RD,
                   phi*(Lin-L)+RL-2*HL-TLD-TDL+kr*(D-L),
                   phi*(Lin-D)+RD-2*HD-TLD-TDL+kr*(L-D),
                   HL-phi*LL, TLD-phi*LD, TDL-phi*DL, HD-phi*DD])
    x = sp.Matrix([P, L, D, LL, LD, DL, DD])
    jac = f.jacobian(x).subs({P: p, L: m, D: m, LL: h, DD: h, LD: t, DL: t})
    left, right = [1, 3, 4], [2, 6, 5]
    odd = (jac.extract(left, left)-jac.extract(left, right)).applyfunc(sp.expand)
    K, F = u+(k1+k2)*h, a*p-b*m
    want = sp.Matrix([[-phi-b*K-4*a1*m-2*kr, (k1-k2)*F+2*a1*kd, 0],
                      [2*a1*m, -phi-a1*kd, 0],
                      [0, 0, -phi-a2*kd]])
    # Heteroligation contributes 2*a2*kd to the first row of the passive column
    # only through LD+DL, which is mirror even.
    assert (odd-want).applyfunc(sp.expand) == sp.zeros(3, 3)
    det = sp.expand(want.extract([0, 1], [0, 1]).det())
    claim = sp.expand((phi+a1*kd)*(phi+b*K+2*kr)+4*a1*m*phi-2*a1*m*(k1-k2)*F)
    assert sp.expand(det-claim) == 0


@pytest.mark.parametrize("mechanism", ["constructive", "destructive"])
def test_empty_closure_matches_absence_of_cores(mechanism):
    p = POINTS[mechanism]
    all_cores = productive_cores(parent(p))["cores"]
    for modules in parent_supports():
        species, _ = producer_closure(parent(p, modules))
        count = sum(set(c["modules"]) <= set(modules) for c in all_cores)
        assert bool(species) == bool(count)


def test_closure_of_controls_and_zero_reverse_limit():
    assert producer_closure(control_a())[0] == ["L", "D"]
    assert producer_closure(control_b(0.0)) == ([], [])
    assert producer_closure(control_b(1e-6))[0] == ["L", "D", "U", "V"]
    limit = replace(POINTS["destructive"], kmon=0.0)
    assert producer_closure(parent(limit)) == ([], [])


def test_negative_coupling_classes_of_the_minimal_supports():
    con, des = POINTS["constructive"], POINTS["destructive"]
    cases = ((con, ("matched", "homoligation"), {"shared"}),
             (des, ("mismatched", "homoligation"), {"cross"}))
    for p, modules, expected in cases:
        model = parent(p, modules)
        states = [x for x in racemic_parent_states(p, modules) if x[1] > 1e-9]
        records = [analyze_state(model, x) for x in states]
        record = next(r for r in records if r["alpha_even"] < -1e-8)
        found = {k for k, v in record["negative_couplings"].items() if v}
        assert found == expected and record["alpha_odd"] > 0
    toy = analyze_state(control_b(), toy_racemate())
    assert [k for k, v in toy["negative_couplings"].items() if v] == ["cross"]
    assert ParentParameters().mechanism == "constructive" and len(MODULES) == 4


def test_stored_tool_comparison_agrees_with_the_enumeration():
    """Checks the recorded Autogatito and BiRNe runs; the tools are not rerun."""
    import json
    path = Path(__file__).resolve().parent/"results"/"backends.json"
    if not path.exists():
        pytest.skip("backends.py has not been run")
    networks = json.loads(path.read_text())["networks"]
    assert len(networks) == 8
    for record in networks.values():
        assert record["autogatito_equals_lp"] and record["birne_equals_lp"]
    toy = networks["control_b"]["birne"]
    assert toy["autocatalytic_count"] == 0 and toy["nonautocatalytic_count"] == 1
    assert sorted(toy["unstable_cores"][0]["reactions"]) == [
        "destruction_D", "destruction_L", "ligation_D", "ligation_L"]
    assert networks["destructive_zero_reverse"]["birne"]["autocatalytic_count"] == 0


def test_cross_inhibition_reverses_the_first_inequality():
    """With A Metzler and B <= 0 the odd block dominates the pair-sum block."""
    rng = np.random.default_rng(2718)
    for _ in range(2000):
        n = rng.integers(1, 5)
        a = rng.uniform(0, 1, (n, n))*(rng.random((n, n)) < 0.7)
        np.fill_diagonal(a, -rng.uniform(0, 3, n))
        b = -rng.uniform(0, 1, (n, n))*(rng.random((n, n)) < 0.7)
        odd = a-b
        assert spectral_abscissa(a+b) <= spectral_abscissa(odd)+1e-10
        # Semipositivity of the Metzler odd block decides its stability.
        values, vectors = np.linalg.eig(odd+10*np.eye(n))
        w = np.abs(vectors[:, np.argmax(values.real)].real)
        if np.min(w) > 1e-6 and abs(spectral_abscissa(odd)) > 1e-6:
            assert (np.min(odd @ w) > 0) == (spectral_abscissa(odd) > 0)


def test_cross_inhibition_class_contains_control_b_and_destructive_minimum():
    toy_blocks, _ = pair_blocks(control_b(), toy_racemate())
    des = POINTS["destructive"]
    modules = ("mismatched", "homoligation")
    x = next(x for x in racemic_parent_states(des, modules) if x[1] > 1e-9)
    parent_blocks, _ = pair_blocks(parent(des, modules), x)
    for blocks in (toy_blocks, parent_blocks):
        a, b = blocks["A"], blocks["B"]
        assert np.all(b <= 1e-12) and np.all(a-np.diag(np.diag(a)) >= -1e-12)
        assert spectral_abscissa(a+b) <= spectral_abscissa(a-b)
    w = np.array([1.0, 1.9])
    assert np.all((toy_blocks["A"]-toy_blocks["B"]) @ w > 0)


def test_rate_domain_never_breaks_the_required_modules():
    import json
    path = Path(__file__).resolve().parent/"results"/"parameter_domain.json"
    if not path.exists():
        pytest.skip("parameter_domain.py has not been run")
    records = json.loads(path.read_text())["records"]
    import re
    stored = json.loads(path.read_text())
    for group in (stored["records"], stored["extended_records"]):
        for mechanism, needed in (("constructive", "1.1."), ("destructive", ".11.")):
            record = group[mechanism]
            assert record["unstable_supports_missing_required_modules"] == 0
            assert set(record["nonlinear_outcomes"]) <= {"confirmed", "slow"}
            assert all(re.fullmatch(needed, key) for key in record["unstable_by_support"])


def _stored(name):
    import json
    path = Path(__file__).resolve().parent/"results"/name
    if not path.exists():
        pytest.skip(f"{name} has not been generated")
    return json.loads(path.read_text())


def test_balance_reduction_and_instability_window():
    from theory_checks import balance_checks
    record = balance_checks(samples=40, seed=3)
    assert record["largest_spectrum_difference"] < 1e-9
    assert record["window_cases"] > 50
    assert record["window_agreements"] == record["window_cases"]


def test_pitchfork_is_supercritical_and_predicts_the_branch():
    for record in _stored("theory_checks.json")["pitchfork"].values():
        assert record["supercritical"]
        near = record["branch"][-1]
        assert abs(near["predicted_L_minus_D"]/near["integrated_L_minus_D"]-1) < 0.01


def test_rates_are_cycle_consistent_and_closure_is_one_sided():
    data = _stored("theory_checks.json")
    for record in data["thermodynamics"].values():
        assert record["wegscheider_residual"] < 1e-12
    closure = data["closure"]
    assert closure["empty_core"] == 0 and closure["nonempty_no_core"] > 0
    assert closure["model_counterexample"]["cores"] == 0
    assert closure["model_counterexample"]["closure_species"]


def test_literature_models_fall_into_the_classes():
    data = _stored("literature_models.json")
    assert data["Autocatalysis with recycling"]["classes"] == ["shared"]
    assert data["Kondepudi--Nelson"]["classes"] == ["cross"]
    assert set(data["Frank, flow reactor"]["classes"]) == {"cross", "shared"}
    for record in data.values():
        assert record["alpha_even"] < 0 < record["alpha_odd"]


def test_archived_sweep_is_reproduced_to_printed_precision():
    for record in _stored("archive_sweep.json").values():
        assert record["largest_concentration_difference"] < 6e-7
        assert record["largest_ee_difference"] < 6e-7
