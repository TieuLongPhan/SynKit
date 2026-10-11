"""Symmetry blocks, complete small-support productivity, and steady states."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, linprog, root

from models import MODULES, ParentParameters, parent


def spectral_abscissa(matrix):
    return float(np.max(np.linalg.eigvals(matrix).real))


def stability(model, x):
    j = model.jacobian(0, x)
    qe, qo = model.q_even, model.q_odd
    je, jo = qe.T @ j @ qe, qo.T @ j @ qo
    return {
        "residual_inf": float(np.max(np.abs(model.rhs(0, x)))),
        "alpha_even": spectral_abscissa(je), "alpha_odd": spectral_abscissa(jo),
        "alpha_full": spectral_abscissa(j),
        "eigenvalues_even": [[float(v.real), float(v.imag)] for v in np.linalg.eigvals(je)],
        "eigenvalues_odd": [[float(v.real), float(v.imag)] for v in np.linalg.eigvals(jo)],
        "mirror_distance": float(np.max(np.abs(x-x[model.mirror]))),
        "cross_block_norm": float(np.linalg.norm(qe.T @ j @ qo, ord=np.inf)),
        "mirror_commutator_norm": float(np.linalg.norm(j[:, model.mirror]-j[model.mirror, :], ord=np.inf)),
        "state": np.asarray(x).tolist(), "min_concentration": float(np.min(x)),
    }


def productive_cores(model):
    """Enumerate ALL row/column supports, pruning supersets of found cores.

    Food/waste rows and open-reactor input/loss are explicitly excluded.
    The feasibility problem Sv >= 1, v >= 1 is equivalent by rescaling
    to strictly positive v with Sv >> 0. All coefficients here are integers.
    HiGHS is a numerical certificate; returned witnesses are checked directly.
    """
    indices = [model.species.index(s) for s in model.internal]
    nm, nr = len(indices), len(model.channels)
    pairs = []
    for nrows in range(1, nm+1):
        for rows in combinations(indices, nrows):
            for ncols in range(1, nr+1):
                for cols in combinations(range(nr), ncols):
                    pairs.append((nrows+ncols, rows, cols))
    pairs.sort(key=lambda t: t[0])
    cores, lp_calls, well_formed, skipped = [], 0, 0, 0
    for _, rows, cols in pairs:
        rset, cset = frozenset(rows), frozenset(cols)
        if any(c["row_set"] <= rset and c["col_set"] <= cset for c in cores):
            skipped += 1
            continue
        ix = np.ix_(rows, cols)
        sm, sp, s = model.minus[ix], model.plus[ix], model.net[ix]
        if not (np.all(sm.sum(axis=0) > 0) and np.all(sp.sum(axis=0) > 0)):
            continue
        well_formed += 1
        if np.any(np.max(s, axis=1) <= 0):
            continue
        lp_calls += 1
        result = linprog(np.ones(len(cols)), A_ub=-s, b_ub=-np.ones(len(rows)),
                         bounds=[(1, None)]*len(cols), method="highs")
        if result.status == 2:
            continue
        if not result.success:
            raise RuntimeError(f"LP not resolved: {result.message}")
        if np.min(result.x) < 1-1e-7 or np.min(s @ result.x) < 1-1e-7:
            raise RuntimeError("Invalid productivity witness")
        cores.append({"row_set": rset, "col_set": cset,
                      "species": [model.species[i] for i in rows],
                      "channels": [model.channels[i].id for i in cols],
                      "modules": sorted({model.channels[i].module for i in cols}),
                      "net_stoichiometry": s.tolist(),
                      "flux_witness": result.x.tolist(),
                      "production_witness": (s @ result.x).tolist()})
    for c in cores:
        del c["row_set"], c["col_set"]
    return {"cores": cores, "all_nonempty_support_pairs": len(pairs),
            "well_formed_candidates_after_core_pruning": well_formed,
            "lp_calls": lp_calls, "core_supersets_pruned": skipped,
            "complete_over_declared_rows_and_channels": True,
            "boundary_policy": list(model.internal),
            "method": "exhaustive supports; HiGHS feasibility and direct witness validation"}


def racemic_parent_states(p, modules=MODULES):
    """Enumerate physical real roots of the exact racemic quartic.

    The precursor is eliminated using residue balance at a stationary CSTR.
    Missing dimers remain in the state space and have zero steady concentration.
    """
    if p.phi <= 0:
        raise ValueError("Quartic reduction requires positive dilution")
    modules = frozenset(modules)
    a, b = p.ab
    h = p.a1/(p.phi+p.a1*p.kdim) if "homoligation" in modules else 0
    t = p.a2/(p.phi+p.a2*p.kdim) if "heteroligation" in modules else 0
    d = h+t
    ksum = ((p.k1 if "matched" in modules else 0)
            + (p.k2 if "mismatched" in modules else 0))
    coefficients = np.array([
        p.phi*p.inlet[1]+p.u*a*p.total_feed,
        -p.u*(2*a+b)-p.phi,
        ksum*h*a*p.total_feed-4*p.u*a*d-2*p.phi*d,
        -ksum*h*(2*a+b), -4*ksum*h*a*d,
    ])
    raw = np.polynomial.polynomial.polyroots(np.trim_zeros(coefficients, "b"))
    model = parent(p, modules)
    states = []
    for r in raw:
        if abs(r.imag) > 1e-9 or r.real < -1e-10:
            continue
        m = max(0., float(r.real))
        precursor = p.total_feed-2*m-4*d*m*m
        if precursor < -1e-9:
            continue
        x = np.array([max(0., precursor), m, m, h*m*m, t*m*m, t*m*m, h*m*m])
        if np.max(np.abs(model.rhs(0, x))) > 1e-9:
            raise RuntimeError("Quartic root does not satisfy the full ODE")
        if not any(np.max(np.abs(x-y)) < 1e-8 for y in states):
            states.append(x)
    return sorted(states, key=lambda x: x[1])


def positive_racemic_state(p, modules=MODULES):
    states = [x for x in racemic_parent_states(p, modules) if x[1] > 1e-9]
    stable_even = [x for x in states if stability(parent(p, modules), x)["alpha_even"] < -1e-8]
    if len(stable_even) != 1:
        raise ValueError(f"Expected one positive even-stable racemate, found {len(stable_even)}")
    return stable_even[0]


def parent_odd_matrix(p, x, modules=MODULES):
    """Analytic (L-D, LL-DD) block; the LD-DL mode is passive."""
    modules = frozenset(modules)
    k1 = p.k1 if "matched" in modules else 0
    k2 = p.k2 if "mismatched" in modules else 0
    a1 = p.a1 if "homoligation" in modules else 0
    a, b = p.ab
    precursor, m, dimer = x[0], x[1], x[3]
    k = p.u+(k1+k2)*dimer
    force = a*precursor-b*m
    return np.array([[-p.phi-b*k-4*a1*m-2*p.kr,
                      (k1-k2)*force+2*a1*p.kdim],
                     [2*a1*m, -p.phi-a1*p.kdim]])


def parent_criterion(p, x, modules=MODULES):
    modules = frozenset(modules)
    k1 = p.k1 if "matched" in modules else 0
    k2 = p.k2 if "mismatched" in modules else 0
    a1 = p.a1 if "homoligation" in modules else 0
    a, b = p.ab
    m, k = x[1], p.u+(k1+k2)*x[3]
    driving = 2*a1*m*(k1-k2)*(a*x[0]-b*m)
    damping = (p.phi+a1*p.kdim)*(p.phi+b*k+2*p.kr)+4*a1*m*p.phi
    return {"driving": float(driving), "damping": float(damping),
            "margin": float(driving-damping),
            "kr_critical": float((driving-(p.phi+a1*p.kdim)*(p.phi+b*k)
                                   -4*a1*m*p.phi)/(2*(p.phi+a1*p.kdim)))}


def metrics(model, x):
    index = {s: i for i, s in enumerate(model.species)}
    l, d = x[index["L"]], x[index["D"]]
    u = x[index.get("LL", index.get("U"))] if "LL" in index or "U" in index else 0.
    v = x[index.get("DD", index.get("V"))] if "DD" in index or "V" in index else 0.
    hetero = sum(x[index[s]] for s in ("LD", "DL") if s in index)
    mass = l+d+2*(u+v+hetero)
    return {"ee_monomer": float((l-d)/(l+d)) if l+d > 1e-12 else None,
            "monomer_concentration": float(l+d),
            "ee_homodimer": float((u-v)/(u+v)) if u+v > 1e-12 else None,
            "ee_residues": float((l-d+2*(u-v))/mass) if mass > 1e-12 else None,
            "peptide_residue_outflow": float(2*sum(model.loss[index[s]]*x[index[s]]
                for s in ("LL", "LD", "DL", "DD", "U", "V") if s in index)),
            "model_residue_inventory": float(mass)}


def integrate_endpoint(model, initial, horizon=4000., sample=251,
                       rtol=2e-9, atol=2e-12):
    times = np.r_[0., np.geomspace(1e-3, horizon, sample-1)]
    sol = solve_ivp(model.rhs, (0, horizon), initial, method="BDF", jac=model.jacobian,
                    t_eval=times, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    if np.min(sol.y) < -1e-9:
        raise RuntimeError("Trajectory left the nonnegative orthant")
    endpoint = sol.y[:, -1]
    refined = root(lambda x: model.rhs(0, x), endpoint,
                   jac=lambda x: model.jacobian(0, x), tol=1e-11)
    residual = float(np.max(np.abs(model.rhs(0, refined.x))))
    if residual > 1e-9 or np.min(refined.x) < -1e-9:
        raise RuntimeError(f"Endpoint refinement failed: {refined.message}")
    # Reject jumping from a transient to a different stationary branch.
    distance = float(np.max(np.abs(refined.x-endpoint)))
    if distance > 1e-5:
        raise RuntimeError(f"Trajectory has not reached a stationary branch: {distance}")
    j = model.jacobian(0, refined.x)
    record = {"state": refined.x.tolist(), "residual_inf": residual,
              "alpha_full": spectral_abscissa(j),
              "eigenvalues_full": [[float(v.real), float(v.imag)] for v in np.linalg.eigvals(j)],
              "min_concentration": float(np.min(refined.x)),
              "trajectory_min_concentration": float(np.min(sol.y)),
              "refinement_distance": distance, "solver_success": sol.success,
              "root_success": bool(refined.success), "root_message": str(refined.message),
              "horizon": horizon, "metrics": metrics(model, refined.x)}
    return record, sol.t, sol.y


def paired_trajectories(model, racemate, delta=1e-4, horizon=4000.):
    jodd = model.q_odd.T @ model.jacobian(0, racemate) @ model.q_odd
    values, vectors = np.linalg.eig(jodd)
    direction = model.q_odd @ vectors[:, np.argmax(values.real)].real
    direction /= np.max(np.abs(direction))
    if direction[model.species.index("L")] < 0:
        direction *= -1
    active = np.abs(direction) > 1e-10
    if np.any(racemate[active] <= 0):
        raise ValueError("Perturbation requires positive active concentrations")
    amplitude = delta * np.min(racemate[active]/np.abs(direction[active]))
    initial = racemate+amplitude*direction
    plus, times, yplus = integrate_endpoint(model, initial, horizon)
    minus, times2, yminus = integrate_endpoint(model, initial[model.mirror], horizon)
    if not np.array_equal(times, times2):
        raise RuntimeError("Mismatched output times")
    mirror_error = float(np.max(np.abs(yplus[model.mirror]-yminus)))
    result = {"perturbation_relative_amplitude": delta,
              "perturbation_absolute_amplitude": float(amplitude),
              "initial_plus": initial.tolist(), "plus": plus, "minus": minus,
              "paired_trajectory_mirror_error": mirror_error,
              "mirror_endpoint_error": float(np.max(np.abs(
                  np.asarray(plus["state"])[model.mirror]-minus["state"]))) }
    return result, times, yplus, yminus


def toy_racemate(eps=0.):
    m = brentq(lambda m: m+(2-eps)*m*m+6*m**3-9, 0, 9, xtol=1e-14)
    return np.array([m, m, m*m, m*m])
