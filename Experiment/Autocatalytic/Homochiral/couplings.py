"""Sign structure of racemic Jacobians and a producer-closure core screen.

It covers two statements of the manuscript and writes the support table.

1. In mirror-pair coordinates a racemic Jacobian has the block form
   ``[[E, C, C], [G, A, B], [G, B, A]]``. The odd block is ``A-B`` and the even
   block is ``[[E, sqrt(2) C], [sqrt(2) G, A+B]]``. When every off-diagonal
   entry of the even block is nonnegative and ``B`` is entrywise nonnegative,
   the odd spectral abscissa cannot exceed the even one. Each support is
   therefore labeled by the negative couplings it contains.
2. Repeatedly deleting species without a producing channel and channels
   without an internal reactant and product leaves a closure that contains
   every productive subsystem. An empty closure excludes productive cores.
3. On random rate sets, the closed-form route criterion of the model agrees
   with the sign of alpha(A+B) at every chirally unstable racemate.

Run from the repository root::

    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalytic/Homochiral/couplings.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis import productive_cores, racemic_parent_states, spectral_abscissa, toy_racemate
from models import (MODULES, ParentParameters, control_a, control_b, parent,
                    parent_supports)

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
# LaTeX fragments with generated numbers; main.tex carries a copy of them.
PAPER = HERE/"results"/"latex"
PAPER.mkdir(parents=True, exist_ok=True)
TOL = 1e-12
ROUTE_SEED = 2718
ROUTE_DRAWS = 500  # per mechanism and rate domain
POINTS = {"constructive": ParentParameters("constructive", k1=3.0, k2=0.75),
          "destructive": ParentParameters("destructive", k1=10.0, k2=80.0)}


def pair_blocks(model, x):
    """Return (E, C, G, A, B) and the index lists (fixed, left, right)."""
    mirror = model.mirror
    fixed = [i for i in range(len(mirror)) if mirror[i] == i]
    left = [i for i in range(len(mirror)) if i < mirror[i]]
    right = [int(mirror[i]) for i in left]
    j = model.jacobian(0, np.asarray(x, dtype=float))
    blocks = {"E": j[np.ix_(fixed, fixed)], "C": j[np.ix_(fixed, left)],
              "G": j[np.ix_(left, fixed)], "A": j[np.ix_(left, left)],
              "B": j[np.ix_(left, right)]}
    # Mirror equivariance fixes the remaining blocks.
    for got, want in ((j[np.ix_(fixed, right)], blocks["C"]),
                      (j[np.ix_(right, fixed)], blocks["G"]),
                      (j[np.ix_(right, right)], blocks["A"]),
                      (j[np.ix_(right, left)], blocks["B"])):
        if np.max(np.abs(got-want), initial=0.) > 1e-10:
            raise ValueError("State or model is not mirror symmetric")
    return blocks, (fixed, left, right)


def sector_matrices(blocks):
    odd = blocks["A"]-blocks["B"]
    even = np.block([[blocks["E"], np.sqrt(2)*blocks["C"]],
                     [np.sqrt(2)*blocks["G"], blocks["A"]+blocks["B"]]])
    return even, odd


def negative_couplings(blocks, names):
    """List negative couplings by the class used in the manuscript."""
    fixed, left, right = names
    routes = {"cross": [], "within": [], "shared": []}
    a, b = blocks["A"], blocks["B"]
    for i in range(len(left)):
        for k in range(len(left)):
            if b[i, k] < -TOL:
                routes["cross"].append([right[k], left[i], float(b[i, k])])
            if i != k and a[i, k] < -TOL:
                routes["within"].append([left[k], left[i], float(a[i, k])])
    for i in range(len(fixed)):
        for k in range(len(left)):
            if blocks["C"][i, k] < -TOL:
                routes["shared"].append([left[k], fixed[i], float(blocks["C"][i, k])])
            if blocks["G"][k, i] < -TOL:
                routes["shared"].append([fixed[i], left[k], float(blocks["G"][k, i])])
        for k in range(len(fixed)):
            if i != k and blocks["E"][i, k] < -TOL:
                routes["shared"].append([fixed[k], fixed[i], float(blocks["E"][i, k])])
    return routes


def analyze_state(model, x):
    blocks, index = pair_blocks(model, x)
    even, odd = sector_matrices(blocks)
    routes = negative_couplings(blocks, index)
    named = {key: [[model.species[s], model.species[t], v] for s, t, v in rows]
             for key, rows in routes.items()}
    record = {"state": np.asarray(x).tolist(),
              "alpha_even": spectral_abscissa(even),
              "alpha_odd": spectral_abscissa(odd),
              "alpha_pair_sum": spectral_abscissa(blocks["A"]+blocks["B"]),
              "spectrum": {name: sorted(np.linalg.eigvals(m).real.tolist())
                           for name, m in (("even", even), ("odd", odd),
                                           ("pair_sum", blocks["A"]+blocks["B"]))},
              "negative_couplings": named,
              "cooperative": not any(routes.values())}
    # The comparison theorem must hold whenever its hypotheses do.
    if record["cooperative"] and record["alpha_odd"] > record["alpha_even"]+1e-9:
        raise AssertionError("Cooperative racemate violates the sector comparison")
    return record


def producer_closure(model, rows=None):
    """Largest species/channel pair that survives the two deletion rules."""
    index = {s: i for i, s in enumerate(model.species)}
    rows = set(index[s] for s in (model.internal if rows is None else rows))
    cols = set(range(len(model.channels)))
    while True:
        ordered = sorted(rows)
        new_cols = {j for j in cols
                    if model.minus[ordered, j].sum() > 0
                    and model.plus[ordered, j].sum() > 0}
        new_rows = {i for i in rows if any(model.net[i, j] > 0 for j in new_cols)}
        if new_cols == cols and new_rows == rows:
            break
        rows, cols = new_rows, new_cols
    return ([model.species[i] for i in sorted(rows)],
            [model.channels[j].id for j in sorted(cols)])


def reference_racemate(p, modules):
    """Positive racemate that is stable in the even sector, when it exists."""
    model = parent(p, modules)
    chosen = []
    for x in racemic_parent_states(p, modules):
        if x[1] <= 1e-9:
            continue
        record = analyze_state(model, x)
        if record["alpha_even"] < -1e-8:
            chosen.append(record)
    if len(chosen) > 1:
        raise ValueError("More than one positive even-stable racemate")
    return chosen[0] if chosen else None


def support_records(mechanism):
    p = POINTS[mechanism]
    stored = json.loads((RESULTS/f"{mechanism}_modules.json").read_text())
    cores = {tuple(r["modules"]): r["productive_core_count"] for r in stored}
    passing = {tuple(r["modules"]): bool(r["clean_chiral_instability"]
                                         and r["stable_chiral_endpoints"])
               for r in stored}
    records = []
    for modules in parent_supports():
        species, channels = producer_closure(parent(p, modules))
        if not species and cores[modules]:
            raise AssertionError("Empty closure but a stored productive core")
        records.append({"modules": list(modules),
                        "bits": "".join("1" if m in modules else "0" for m in MODULES),
                        "productive_core_count": cores[modules],
                        "closure_species": species, "closure_channels": channels,
                        "passes_phenotype": passing[modules],
                        "racemate": reference_racemate(p, modules)})
    return records


def control_records():
    out = {}
    a = control_a()
    m = (1+np.sqrt(5))/2
    out["control_a"] = {"closure": producer_closure(a),
                        "racemate": analyze_state(a, np.array([m, m]))}
    for eps in (0.0, 1e-3):
        b = control_b(eps)
        out[f"control_b_eps_{eps:g}"] = {"closure": producer_closure(b),
                                         "racemate": analyze_state(b, toy_racemate(eps))}
    limit = replace(POINTS["destructive"], kmon=0.0)
    out["destructive_parent_zero_reverse"] = {
        "closure": producer_closure(parent(limit)),
        "racemate": reference_racemate(limit, MODULES)}
    return out


def route_label(record):
    if record is None:
        return "--"
    routes = record["negative_couplings"]
    letters = [letter for letter, key in (("c", "cross"), ("h", "within"),
                                          ("s", "shared")) if routes[key]]
    return ",".join(letters) if letters else "none"


def number(value):
    return "--" if value is None else f"{value:.4f}"


def write_table(records):
    """One-column table: shared core count, then alpha_- and classes per mechanism."""
    lines = ["% Generated by Experiment/Autocatalytic/Homochiral/couplings.py; do not edit.",
             r"\begin{ruledtabular}", r"\begin{tabular}{l c r l r l}",
             r" & & \multicolumn{2}{c}{Constructive} & \multicolumn{2}{c}{Destructive}\\",
             r"\cline{3-4}\cline{5-6}",
             r"Support & Cores & $\alpha_-$ & Neg. & $\alpha_-$ & Neg.\\", r"\hline"]
    for con, des in zip(records["constructive"], records["destructive"]):
        if con["productive_core_count"] != des["productive_core_count"]:
            raise AssertionError("Mechanisms with the same support differ in cores")
        cells = [con["bits"], str(con["productive_core_count"])]
        for r in (con, des):
            rac = r["racemate"]
            if rac is None:
                cells += ["--", "--"]
                continue
            if rac["alpha_even"] >= 0:
                raise AssertionError("Listed racemate is not even-stable")
            odd = number(rac["alpha_odd"])
            cells += [r"$\mathbf{"+odd+"}$" if r["passes_phenotype"] else "$"+odd+"$",
                      route_label(rac)]
        lines.append(" & ".join(cells)+r" \\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}"]
    (PAPER/"support_table.tex").write_text("\n".join(lines)+"\n")


def write_macros(records, controls, empty_closure, coreless, routes):
    rows = [r for group in records.values() for r in group]
    with_state = [r for r in rows if r["racemate"]]
    negative = [r for r in with_state if not r["racemate"]["cooperative"]]
    minimal = next(r for r in records["constructive"] if r["bits"] == "1010")["racemate"]
    full = {m: next(r for r in records[m] if r["bits"] == "1111")["racemate"]
            for m in POINTS}
    values = {
        "SupportCount": len(rows), "EmptyClosureCount": empty_closure,
        "CorelessCount": coreless, "RacemateCount": len(with_state),
        "CooperativeCount": len(with_state)-len(negative),
        "NegativeCount": len(negative),
        "NegativeStableCount": sum(not r["passes_phenotype"] for r in negative),
        "PassingCount": sum(r["passes_phenotype"] for r in rows),
        "ConPairSum": f"{minimal['alpha_pair_sum']:.6f}",
        "ConPairOdd": f"{minimal['alpha_odd']:.6f}",
        "ConPairEven": f"{minimal['alpha_even']:.4f}",
        "ConSharedEntry": f"{minimal['negative_couplings']['shared'][0][2]:.4f}",
        "ConFullPairSum": f"{full['constructive']['alpha_pair_sum']:.4f}",
        "ConFullOdd": f"{full['constructive']['alpha_odd']:.4f}",
        "DesFullPairSum": f"{full['destructive']['alpha_pair_sum']:.4f}",
        "DesFullOdd": f"{full['destructive']['alpha_odd']:.4f}",
        "ToyCrossEntry": f"{controls['control_b_eps_0']['racemate']['negative_couplings']['cross'][0][2]:.0f}",
        "RouteDraws": routes["draws"], "RouteCases": routes["cases"],
        "RouteDisagreements": routes["disagreements"],
        "RouteConHetInterference": routes["counts"].get(
            "constructive with heteroligation interference", 0),
        "RouteConHetResource": routes["counts"].get(
            "constructive with heteroligation resource", 0),
    }
    lines = ["% Generated by Experiment/Autocatalytic/Homochiral/couplings.py; do not edit."]
    lines += [rf"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items()]
    (PAPER/"couplings.tex").write_text("\n".join(lines)+"\n")


def route_margin(p, modules, x):
    """Closed-form sign of -alpha(A+B) for the model at a racemate.

    Nonnegative margin means interference, negative means resource competition.
    """
    modules = frozenset(modules)
    k1 = p.k1 if "matched" in modules else 0
    k2 = p.k2 if "mismatched" in modules else 0
    a1 = p.a1 if "homoligation" in modules else 0
    a2 = p.a2 if "heteroligation" in modules else 0
    a, b = p.ab
    precursor, m, dimer = x[0], x[1], x[3]
    k, force = p.u+(k1+k2)*dimer, a*precursor-b*m
    gamma, gamma2 = p.phi+a1*p.kdim, p.phi+a2*p.kdim
    damping = gamma*(p.phi+b*k)+4*a1*m*p.phi
    return damping+4*a2*m*p.phi*gamma/gamma2-2*a1*m*(k1+k2)*force


def check_routes(records):
    """The closed-form route agrees with alpha(A+B) on every racemate."""
    for mechanism, rows in records.items():
        for r in rows:
            rac = r["racemate"]
            if rac is None:
                continue
            margin = route_margin(POINTS[mechanism], r["modules"], rac["state"])
            if abs(rac["alpha_pair_sum"]) > 1e-9 and (margin >= 0) != (rac["alpha_pair_sum"] <= 0):
                raise AssertionError(("route", mechanism, r["bits"]))


def sampled_routes():
    """Route of every chirally unstable racemate on random rate sets.

    The draws use the two log-uniform domains of ``sampling.py``. For each
    draw and support the first positive racemate with alpha_+ < 0 < alpha_-
    is classified by alpha(A+B) and compared with ``route_margin``.
    """
    from sampling import BASE, EXTENDED, SUPPORTS, draw
    rng = np.random.default_rng(ROUTE_SEED)
    counts, disagreements, cases = {}, 0, 0
    for domain in (BASE, EXTENDED):
        for mechanism in POINTS:
            for _ in range(ROUTE_DRAWS):
                p = draw(rng, mechanism, domain)
                for modules in SUPPORTS:
                    try:
                        states = racemic_parent_states(p, modules)
                    except RuntimeError:
                        continue
                    model = parent(p, modules)
                    for x in states:
                        if x[1] <= 1e-9:
                            continue
                        info = analyze_state(model, x)
                        if not info["alpha_even"] < -1e-8 < 1e-8 < info["alpha_odd"]:
                            continue
                        pair_sum = info["alpha_pair_sum"]
                        margin = route_margin(p, modules, x)
                        if abs(pair_sum) > 1e-9 and (margin >= 0) != (pair_sum <= 0):
                            disagreements += 1
                        route = "interference" if pair_sum <= 0 else "resource"
                        het = "with" if "heteroligation" in modules else "without"
                        key = f"{mechanism} {het} heteroligation {route}"
                        counts[key] = counts.get(key, 0)+1
                        cases += 1
                        break
    return {"seed": ROUTE_SEED, "draws": 4*ROUTE_DRAWS, "cases": cases,
            "disagreements": disagreements, "counts": counts}


def closure_counterexample():
    """Support 0001 with reverse ligation: nonempty closure but no core."""
    model = parent(replace(POINTS["constructive"], kdim=0.1), ("heteroligation",))
    species, _ = producer_closure(model)
    return {"model_counterexample": {"support": "0001", "kdim": 0.1,
                                     "closure_species": species,
                                     "cores": len(productive_cores(model)["cores"])}}


def main():
    records = {m: support_records(m) for m in POINTS}
    controls = control_records()
    cooperative_unstable = [
        (m, r["bits"]) for m, rows in records.items() for r in rows
        if r["racemate"] and r["racemate"]["cooperative"]
        and r["racemate"]["alpha_odd"] > 1e-8 > r["racemate"]["alpha_even"]]
    if cooperative_unstable:
        raise AssertionError(cooperative_unstable)
    check_routes(records)
    empty_closure = sum(not r["closure_species"] for rows in records.values() for r in rows)
    coreless = sum(r["productive_core_count"] == 0 for rows in records.values() for r in rows)
    routes = sampled_routes()
    if routes["disagreements"]:
        raise AssertionError(("sampled routes", routes["disagreements"]))
    summary = {"supports": records, "controls": controls,
               "supports_with_empty_closure": empty_closure,
               "supports_without_core": coreless,
               "closure_counterexample": closure_counterexample()["model_counterexample"],
               "sampled_routes": routes,
               "route_legend": {"c": "negative cross-enantiomer entry of B",
                                "h": "negative off-diagonal entry of A (same handedness)",
                                "s": "negative coupling through a mirror-fixed species"}}
    (RESULTS/"couplings.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_table(records)
    write_macros(records, controls, empty_closure, coreless, routes)
    for mechanism, rows in records.items():
        for r in rows:
            rac = r["racemate"]
            print(mechanism, r["bits"], r["productive_core_count"],
                  "closure" if r["closure_species"] else "empty",
                  number(rac and rac["alpha_even"]), number(rac and rac["alpha_odd"]),
                  route_label(rac), "PASS" if r["passes_phenotype"] else "")
    for name, r in controls.items():
        print(name, r["closure"][0], number(r["racemate"]["alpha_even"]),
              number(r["racemate"]["alpha_odd"]), route_label(r["racemate"]))
    print("empty closures", empty_closure, "coreless supports", coreless)
    print("sampled routes", json.dumps(routes))


if __name__ == "__main__":
    main()
