"""Sign structure of racemic Jacobians and a producer-closure core screen.

This module adds two checks to the pilot without changing ``run.py``.

1. In mirror-pair coordinates a racemic Jacobian has the block form
   ``[[E, C, C], [G, A, B], [G, B, A]]``. The odd block is ``A-B`` and the even
   block is ``[[E, sqrt(2) C], [sqrt(2) G, A+B]]``. When every off-diagonal
   entry of the even block is nonnegative and ``B`` is entrywise nonnegative,
   the odd spectral abscissa cannot exceed the even one. Each support is
   therefore labeled by the negative couplings it contains.
2. Repeatedly deleting species without a producing channel and channels
   without an internal reactant and product leaves a closure that contains
   every productive subsystem. An empty closure excludes productive cores.

Run from the repository root::

    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalyic/sign_structure.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis import racemic_parent_states, spectral_abscissa, toy_racemate
from models import (MODULES, ParentParameters, control_a, control_b, parent,
                    parent_supports)

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
PAPER = HERE.parents[1]/"paper"/"autocatalytic"
TOL = 1e-12
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
    letters = [letter for letter, key in (("c", "cross"), ("w", "within"),
                                          ("s", "shared")) if routes[key]]
    return ",".join(letters) if letters else "none"


def number(value):
    return "--" if value is None else f"{value:.4f}"


def write_table(records):
    lines = ["% Generated by Experiment/Autocatalyic/sign_structure.py; do not edit.",
             r"\begin{tabular}{l rrrl c rrrl}", r"\toprule",
             r" & \multicolumn{4}{c}{Constructive} && \multicolumn{4}{c}{Destructive}\\",
             r"\cmidrule{2-5}\cmidrule{7-10}",
             r"Support & Cores & $\alpha_+$ & $\alpha_-$ & Neg. && "
             r"Cores & $\alpha_+$ & $\alpha_-$ & Neg.\\", r"\midrule"]
    for con, des in zip(records["constructive"], records["destructive"]):
        cells = [con["bits"]]
        for i, r in enumerate((con, des)):
            rac = r["racemate"]
            even, odd = (number(rac and rac[key]) for key in ("alpha_even", "alpha_odd"))
            if rac:
                even = "$"+even+"$"
                odd = r"$\mathbf{"+odd+"}$" if r["passes_phenotype"] else "$"+odd+"$"
            cells += [str(r["productive_core_count"]), even, odd, route_label(rac)]
            if i == 0:
                cells.append("")
        lines.append(" & ".join(cells)+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER/"support_table.tex").write_text("\n".join(lines)+"\n")


def write_macros(records, controls, empty_closure, coreless):
    rows = [r for group in records.values() for r in group]
    with_state = [r for r in rows if r["racemate"]]
    negative = [r for r in with_state if not r["racemate"]["cooperative"]]
    minimal = next(r for r in records["constructive"] if r["bits"] == "1010")["racemate"]
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
        "ToyCrossEntry": f"{controls['control_b_eps_0']['racemate']['negative_couplings']['cross'][0][2]:.0f}",
    }
    lines = ["% Generated by Experiment/Autocatalyic/sign_structure.py; do not edit."]
    lines += [rf"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items()]
    (PAPER/"sign_results.tex").write_text("\n".join(lines)+"\n")


def main():
    records = {m: support_records(m) for m in POINTS}
    controls = control_records()
    cooperative_unstable = [
        (m, r["bits"]) for m, rows in records.items() for r in rows
        if r["racemate"] and r["racemate"]["cooperative"]
        and r["racemate"]["alpha_odd"] > 1e-8 > r["racemate"]["alpha_even"]]
    if cooperative_unstable:
        raise AssertionError(cooperative_unstable)
    empty_closure = sum(not r["closure_species"] for rows in records.values() for r in rows)
    coreless = sum(r["productive_core_count"] == 0 for rows in records.values() for r in rows)
    summary = {"supports": records, "controls": controls,
               "supports_with_empty_closure": empty_closure,
               "supports_without_core": coreless,
               "route_legend": {"c": "negative cross-enantiomer entry of B",
                                "w": "negative off-diagonal entry of A",
                                "s": "negative coupling through a mirror-fixed species"}}
    (RESULTS/"sign_structure.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_table(records)
    write_macros(records, controls, empty_closure, coreless)
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


if __name__ == "__main__":
    main()
