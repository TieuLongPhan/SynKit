"""Pitchfork at the racemization threshold and the even-sector window.

The pitchfork coefficient is computed on the center manifold of the odd
block and compared with integrated branch amplitudes. The balance reduction
of the even block and the closed-form instability window are compared with
the numerical sector spectra on random rate sets.

    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalytic/Homochiral/bifurcation.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis import (integrate_endpoint, parent_criterion, positive_racemic_state,
                      racemic_parent_states, stability)
from couplings import POINTS, pair_blocks, sector_matrices
from models import ParentParameters, parent, parent_supports

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
LATEX = RESULTS/"latex"
WEIGHTS = np.array([1.0, 2.0, 2.0])  # residues in L, LL, LD


def reduced_even(blocks, weights=WEIGHTS):
    """Even block on the balance manifold for one mirror-fixed species."""
    g = blocks["G"][:, 0]
    return blocks["A"]+blocks["B"]-2*np.outer(g, weights)


def even_determinant(p, x, modules):
    """Closed form for supports without heteroligation."""
    modules = frozenset(modules)
    k1 = p.k1 if "matched" in modules else 0
    k2 = p.k2 if "mismatched" in modules else 0
    a1 = p.a1 if "homoligation" in modules else 0
    a, b = p.ab
    m, K, F = x[1], p.u+(k1+k2)*x[3], a*x[0]-b*x[1]
    gamma = p.phi+a1*p.kdim
    return gamma*(p.phi+(2*a+b)*K)+4*a1*m*p.phi-2*a1*m*((k1+k2)*F-4*a*K)


def balance_checks(samples=400, seed=11):
    rng = np.random.default_rng(seed)
    worst_spectrum, window_agreements, window_cases = 0.0, 0, 0
    for _ in range(samples):
        values = {k: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) for k, (lo, hi) in {
            "k1": (0.1, 100), "k2": (0.1, 100), "a1": (0.1, 10), "a2": (0.4, 40),
            "phi": (0.01, 1), "kmon": (1e-4, 1e-1), "u": (1e-4, 1e-1),
            "kdim": (1e-4, 1), "kr": (1e-4, 1)}.items()}
        p = ParentParameters(mechanism=str(rng.choice(["constructive", "destructive"])), **values)
        for modules in parent_supports():
            if "homoligation" not in modules:
                continue
            model = parent(p, modules)
            for x in racemic_parent_states(p, modules):
                if x[1] <= 1e-9:
                    continue
                blocks, _ = pair_blocks(model, x)
                even, _ = sector_matrices(blocks)
                want = np.sort_complex(np.linalg.eigvals(even))
                got = np.sort_complex(np.r_[-p.phi, np.linalg.eigvals(reduced_even(blocks))])
                worst_spectrum = max(worst_spectrum, float(np.max(np.abs(want-got))))
                if "heteroligation" in modules:
                    continue
                info = stability(model, x)
                if min(abs(info["alpha_even"]), abs(info["alpha_odd"])) < 1e-7:
                    continue
                window_cases += 1
                predicted = (parent_criterion(p, x, modules)["margin"] > 0
                             and even_determinant(p, x, modules) > 0)
                observed = info["alpha_even"] < 0 < info["alpha_odd"]
                window_agreements += predicted == observed
    return {"samples": samples, "largest_spectrum_difference": worst_spectrum,
            "window_cases": window_cases, "window_agreements": window_agreements}


def second_form(model, x, u, v):
    """Exact second differential for networks with at most bimolecular steps."""
    f = lambda y: model.rhs(0, y)
    return f(x+u+v)-f(x+u)-f(x+v)+f(x)


def pitchfork(mechanism):
    base = POINTS[mechanism]
    x = positive_racemic_state(base)
    kr_crit = parent_criterion(base, x)["kr_critical"]
    model = parent(replace(base, kr=kr_crit))
    jac = model.jacobian(0, x)
    qe, qo = model.q_even, model.q_odd
    values, right = np.linalg.eig(qo.T @ jac @ qo)
    i = int(np.argmin(np.abs(values)))
    q = qo @ right[:, i].real
    q /= q[model.species.index("L")]-q[model.species.index("D")]  # L-D = amplitude
    left_values, left = np.linalg.eig((qo.T @ jac @ qo).T)
    p = qo @ left[:, int(np.argmin(np.abs(left_values)))].real
    p /= p @ q
    quadratic = second_form(model, x, q, q)
    h = -0.5*qe @ np.linalg.solve(qe.T @ jac @ qe, qe.T @ quadratic)
    coefficient = float(p @ second_form(model, x, q, h))
    slope = float(p @ (parent(replace(base, kr=kr_crit+1)).jacobian(0, x)-jac) @ q)
    rows = []
    for fraction in (0.9, 0.99):
        kr = fraction*kr_crit
        live = parent(replace(base, kr=kr))
        growth = stability(live, x)["alpha_odd"]
        predicted = np.sqrt(-growth/coefficient) if coefficient < 0 else float("nan")
        seed = x+1e-3*q*np.min(x[x > 0])
        endpoint, *_ = integrate_endpoint(live, seed, horizon=max(4000., 40/growth))
        state = np.asarray(endpoint["state"])
        rows.append({"fraction": fraction, "growth_rate": growth,
                     "predicted_L_minus_D": float(predicted),
                     "integrated_L_minus_D": float(abs(state[1]-state[2]))})
    return {"mechanism": mechanism, "kr_critical": kr_crit, "cubic_coefficient": coefficient,
            "growth_slope_in_kr": slope, "supercritical": coefficient < 0, "branch": rows}


def main():
    data = {"balance": balance_checks(), "pitchfork": {m: pitchfork(m) for m in POINTS}}
    (RESULTS/"bifurcation.json").write_text(
        json.dumps(data, indent=2, default=lambda o: o.item())+"\n")
    con, des = data["pitchfork"]["constructive"], data["pitchfork"]["destructive"]
    values = {
        "BalanceCases": data["balance"]["window_cases"],
        "BalanceAgree": data["balance"]["window_agreements"],
        "PitchCon": f"{con['cubic_coefficient']:.3f}",
        "PitchDes": f"{des['cubic_coefficient']:.3f}",
        "PitchConPred": f"{con['branch'][1]['predicted_L_minus_D']:.4f}",
        "PitchConObs": f"{con['branch'][1]['integrated_L_minus_D']:.4f}",
        "PitchDesPred": f"{des['branch'][1]['predicted_L_minus_D']:.4f}",
        "PitchDesObs": f"{des['branch'][1]['integrated_L_minus_D']:.4f}",
    }
    LATEX.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by Experiment/Autocatalytic/Homochiral/bifurcation.py; do not edit."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items()]
    (LATEX/"bifurcation.tex").write_text("\n".join(lines)+"\n")
    print(json.dumps(values, indent=2, default=lambda o: o.item()))


if __name__ == "__main__":
    main()
