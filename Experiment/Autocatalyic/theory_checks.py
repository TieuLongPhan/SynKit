"""Numerical checks behind four statements of the manuscript.

1. Balance reduction: with one mirror-fixed species and a balance law the
   even spectrum is ``{-phi}`` together with that of ``A+B-2 g w^T``, and for
   supports without heteroligation the even block has a closed-form
   determinant. Together with the odd criterion this gives an instability
   window.
2. Pitchfork coefficient at the racemization threshold, with a comparison of
   the predicted and the integrated branch amplitude.
3. Thermodynamics: cycle (Wegscheider) consistency of the internal rates and
   the conversion affinity that a chiral instability requires.
4. The producer closure against exhaustive core enumeration on random
   networks, and one counterexample inside the model family.

Run from the repository root::

    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalyic/theory_checks.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from analysis import (integrate_endpoint, parent_criterion, positive_racemic_state,
                      productive_cores, racemic_parent_states, spectral_abscissa, stability)
from models import MODULES, Channel, Network, ParentParameters, parent, parent_supports
from sign_structure import POINTS, pair_blocks, producer_closure, sector_matrices

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
PAPER = HERE.parents[1]/"paper"/"autocatalytic"
WEIGHTS = np.array([1.0, 2.0, 2.0])  # residues in L, LL, LD


# ---------------------------------------------------------------- balance
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


# -------------------------------------------------------------- pitchfork
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


# ---------------------------------------------------------- thermodynamics
def wegscheider_residual(p):
    """Distance of the log equilibrium constants from a potential difference."""
    model = parent(p)
    ids = {r.id: j for j, r in enumerate(model.channels)}
    rows, logs = [], []
    for r in model.channels:
        if r.id.endswith("_f") and r.reverse in ids:
            rows.append(model.net[:, ids[r.id]])
            logs.append(np.log(r.rate/model.channels[ids[r.reverse]].rate))
    rows, logs = np.array(rows, float), np.array(logs)
    potential, *_ = np.linalg.lstsq(rows, logs, rcond=None)
    return float(np.max(np.abs(rows @ potential-logs))), len(logs)


def affinity(p, x):
    a, b = p.ab
    return float(np.log(a*x[0]/(b*x[1])))


def largest_unstable(base, field, upper):
    """Largest value of one coefficient with a clean chiral instability.

    The search follows a logarithmic grid upward and bisects the first
    failure. Loss of the active racemate counts as failure and is reported.
    """
    def state(value):
        p = replace(base, **{field: value})
        try:
            x = positive_racemic_state(p)
        except ValueError:
            return "no active racemate"
        return "unstable" if parent_criterion(p, x)["margin"] > 0 else "odd sector stable"
    grid = np.geomspace(max(getattr(base, field), 1e-6), upper, 400)
    last = None
    for low, high in zip(grid[:-1], grid[1:]):
        if state(low) != "unstable":
            break
        last = low
        if state(high) != "unstable":
            for _ in range(60):
                mid = np.sqrt(low*high)
                low, high = (mid, high) if state(mid) == "unstable" else (low, mid)
            return {"value": float(low), "ends_by": state(high)}
    return {"value": None if last is None else float(grid[-1]), "ends_by": "not reached"}


def thermodynamics():
    out = {}
    for mechanism, base in POINTS.items():
        reversible = replace(base, kdim=0.01)
        residual, pairs = wegscheider_residual(reversible)
        x = positive_racemic_state(base)
        ratio = "k1" if mechanism == "constructive" else "k2"
        fixed = base.k2 if mechanism == "constructive" else base.k1

        def margin(r):
            p = replace(base, **{ratio: r*fixed})
            return parent_criterion(p, positive_racemic_state(p))["margin"]
        low = 1.4 if mechanism == "constructive" else 2.0
        threshold = brentq(margin, low, 12.0, xtol=1e-12)
        at_threshold = replace(base, **{ratio: threshold*fixed})
        out[mechanism] = {
            "wegscheider_residual": residual, "reversible_pairs": pairs,
            "affinity_reference": affinity(base, x),
            "threshold_ratio": float(threshold),
            "affinity_threshold": affinity(at_threshold, positive_racemic_state(at_threshold)),
            "largest_unstable_kdim": largest_unstable(base, "kdim", 5.0),
            "largest_unstable_kmon": largest_unstable(base, "kmon", 0.9)}
    return out


# ---------------------------------------------------------------- closure
def random_network(rng, species=4, reactions=5):
    names = [f"X{i}" for i in range(species)]
    channels = []
    while len(channels) < reactions:
        left = rng.choice(species, size=rng.integers(1, 3), replace=True)
        right = rng.choice(species, size=rng.integers(1, 3), replace=True)
        reactants = {names[i]: int(np.sum(left == i)) for i in set(left)}
        products = {names[i]: int(np.sum(right == i)) for i in set(right)}
        if reactants == products:
            continue
        channels.append(Channel(f"r{len(channels)}", reactants, products, 1.0, "random", "1"))

    class Plain(Network):
        def __init__(self):
            self.name, self.species, self.internal = "random", tuple(names), tuple(names)
            self.channels = tuple(channels)
            self.minus = np.zeros((species, reactions), int)
            self.plus = np.zeros_like(self.minus)
            for j, r in enumerate(channels):
                for s, n in r.reactants.items():
                    self.minus[names.index(s), j] = n
                for s, n in r.products.items():
                    self.plus[names.index(s), j] = n
            self.net = self.plus-self.minus
    return Plain()


def closure_study(samples=3000, seed=5):
    rng = np.random.default_rng(seed)
    table = {"empty_no_core": 0, "nonempty_no_core": 0, "nonempty_core": 0, "empty_core": 0}
    for _ in range(samples):
        model = random_network(rng)
        empty = not producer_closure(model)[0]
        core = bool(productive_cores(model)["cores"])
        table[("empty" if empty else "nonempty")+("_core" if core else "_no_core")] += 1
    hetero = parent(replace(POINTS["constructive"], kdim=0.01), ("heteroligation",))
    counterexample = {"support": "0001 with reverse ligation",
                      "closure_species": producer_closure(hetero)[0],
                      "cores": len(productive_cores(hetero)["cores"])}
    return {"samples": samples, "species": 4, "reactions": 5, **table,
            "model_counterexample": counterexample}


def main():
    data = {"balance": balance_checks(),
            "pitchfork": {m: pitchfork(m) for m in POINTS},
            "thermodynamics": thermodynamics(),
            "closure": closure_study()}
    (RESULTS/"theory_checks.json").write_text(json.dumps(data, indent=2, default=lambda o: o.item())+"\n")
    print(json.dumps(data, indent=2, default=lambda o: o.item()))
    con, des = data["pitchfork"]["constructive"], data["pitchfork"]["destructive"]
    tc, td = data["thermodynamics"]["constructive"], data["thermodynamics"]["destructive"]
    closure = data["closure"]
    coreless = closure["empty_no_core"]+closure["nonempty_no_core"]
    values = {
        "BalanceCases": data["balance"]["window_cases"],
        "BalanceAgree": data["balance"]["window_agreements"],
        "PitchCon": f"{con['cubic_coefficient']:.3f}",
        "PitchDes": f"{des['cubic_coefficient']:.3f}",
        "PitchConPred": f"{con['branch'][1]['predicted_L_minus_D']:.4f}",
        "PitchConObs": f"{con['branch'][1]['integrated_L_minus_D']:.4f}",
        "PitchDesPred": f"{des['branch'][1]['predicted_L_minus_D']:.4f}",
        "PitchDesObs": f"{des['branch'][1]['integrated_L_minus_D']:.4f}",
        "AffConRef": f"{tc['affinity_reference']:.2f}",
        "AffDesRef": f"{td['affinity_reference']:.2f}",
        "AffConThr": f"{tc['affinity_threshold']:.2f}",
        "AffDesThr": f"{td['affinity_threshold']:.2f}",
        "KdimConMax": f"{tc['largest_unstable_kdim']['value']:.3f}",
        "KdimDesMax": f"{td['largest_unstable_kdim']['value']:.4f}",
        "KmonConMax": f"{tc['largest_unstable_kmon']['value']:.3f}",
        "KmonDesMax": f"{td['largest_unstable_kmon']['value']:.4f}",
        "ClosureSamples": closure["samples"],
        "ClosureCoreless": coreless,
        "ClosureCaught": closure["empty_no_core"],
        "ClosureMissed": closure["nonempty_no_core"],
        "ClosureWrong": closure["empty_core"],
    }
    lines = ["% Generated by Experiment/Autocatalyic/theory_checks.py; do not edit."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items()]
    (PAPER/"theory_results.tex").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
