"""Negative-coupling classes of chiral models from the literature.

Each model is written as an explicit mass-action network in the form in
which it is usually analyzed, with chemostatted species absorbed into the
rate coefficients. The rate values are illustrative. They are chosen so that
a positive racemate with a chiral instability exists, and they are not taken
from the cited papers. Run from the repository root::

    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalyic/literature_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import root

from analysis import racemic_parent_states, spectral_abscissa, toy_racemate
from models import Channel, Network, control_b, parent
from sign_structure import POINTS, negative_couplings, pair_blocks, sector_matrices

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
PAPER = HERE.parents[1]/"paper"/"autocatalytic"


def mirrored(make):
    """Build the channels for both hands from a function of (own, other)."""
    return make("L", "D")+make("D", "L")


def frank_flow(k=1.0, mu=1.0, phi=0.1, a0=1.0):
    """Frank's scheme in a flow reactor: A+L->2L and L+D->waste."""
    channels = mirrored(lambda s, o: [
        Channel("growth_"+s, {"A": 1, s: 1}, {s: 2}, k, "growth", "k")])
    channels += [Channel("antagonism", {"L": 1, "D": 1}, {}, mu, "antagonism", "mu")]
    return Network("frank", ("A", "L", "D"), (0, 2, 1), channels,
                   (phi*a0, 0, 0), phi, ("L", "D"), {}), (0.25, 0.15, 0.15)


def kondepudi_nelson(source=0.01, decay=0.1, growth=1.0, saturation=0.1, k3=1.0):
    """S+T<->X, S+T+X<->2X, X_L+X_D->P with S, T chemostatted and P removed."""
    channels = mirrored(lambda s, o: [
        Channel("auto_"+s, {s: 1}, {s: 2}, growth, "autocatalysis", "k2 S T"),
        Channel("auto_back_"+s, {s: 2}, {s: 1}, saturation, "autocatalysis", "k-2")])
    channels += [Channel("pair", {"L": 1, "D": 1}, {}, k3, "antagonism", "k3")]
    return Network("kondepudi_nelson", ("L", "D"), (1, 0), channels,
                   source, decay, ("L", "D"), {}), (0.8, 0.8)


def limited_enantioselectivity(source=0.01, decay=0.1, k2=1.0, k2r=0.1, k3=0.1, k3r=2.0):
    """A<->L, A+L<->2L, A+L<->L+D with A chemostatted and unconstrained rates."""
    channels = mirrored(lambda s, o: [
        Channel("auto_"+s, {s: 1}, {s: 2}, k2, "enantioselective", "k2 A"),
        Channel("auto_back_"+s, {s: 2}, {s: 1}, k2r, "enantioselective", "k-2"),
        Channel("cross_"+s, {s: 1}, {s: 1, o: 1}, k3, "non-enantioselective", "k3 A"),
        Channel("cross_back_"+s, {s: 1, o: 1}, {s: 1}, k3r, "non-enantioselective", "k-3")])
    return Network("limited_enantioselectivity", ("L", "D"), (1, 0), channels,
                   source, decay, ("L", "D"), {}), (0.5, 0.5)


def recycling(k=1.0, back=0.1, total=1.0):
    """Closed system with A+2L->3L and L->A, after Saito and Hyuga."""
    channels = mirrored(lambda s, o: [
        Channel("auto_"+s, {"A": 1, s: 2}, {s: 3}, k, "autocatalysis", "k"),
        Channel("recycle_"+s, {s: 1}, {"A": 1}, back, "recycling", "lambda")])
    m = (1+np.sqrt(1-8*back/(k*total**2)))*total/4
    return Network("recycling", ("A", "L", "D"), (0, 2, 1), channels,
                   0, 0, ("L", "D"), {}), (total-2*m, m, m)


def solve_racemate(model, guess):
    fixed = [i for i, j in enumerate(model.mirror) if i == j]
    left = [i for i, j in enumerate(model.mirror) if i < j]

    def expand(v):
        x = np.zeros(len(model.species))
        x[fixed] = v[:len(fixed)]
        x[left] = v[len(fixed):]
        x[model.mirror[left]] = v[len(fixed):]
        return x

    start = np.asarray(guess, float)[fixed+left]
    if not np.any(model.loss) and not np.any(model.feed):
        return np.asarray(guess, float)  # closed system: analytic state supplied
    solution = root(lambda v: model.rhs(0, expand(v))[fixed+left], start, tol=1e-13)
    x = expand(solution.x)
    if not solution.success or np.max(np.abs(model.rhs(0, x))) > 1e-10 or np.min(x) <= 0:
        raise RuntimeError(f"No positive racemate for {model.name}")
    return x


def classify(model, x, closed=False):
    blocks, index = pair_blocks(model, x)
    even, odd = sector_matrices(blocks)
    routes = negative_couplings(blocks, index)
    even_values = np.linalg.eigvals(even)
    if closed:  # drop the zero eigenvalue of the conserved total
        even_values = np.delete(even_values, np.argmin(np.abs(even_values)))
    a, b = blocks["A"], blocks["B"]
    metzler_a = bool(np.all(a-np.diag(np.diag(a)) >= -1e-12))
    return {"racemate": np.asarray(x).tolist(),
            "mirror_fixed_species": [model.species[i] for i in index[0]],
            "alpha_even": float(np.max(even_values.real)),
            "alpha_odd": spectral_abscissa(odd),
            "alpha_pair_sum": spectral_abscissa(a+b),
            "classes": [k for k in ("cross", "within", "shared") if routes[k]],
            "inhibition_only": bool(metzler_a and np.all(b <= 1e-12) and np.any(b < -1e-12))}


def records():
    out = {}
    for label, (model, guess), closed in (
            ("Frank, flow reactor", frank_flow(), False),
            ("Kondepudi--Nelson", kondepudi_nelson(), False),
            ("Limited enantioselectivity", limited_enantioselectivity(), False),
            ("Autocatalysis with recycling", recycling(), True)):
        out[label] = classify(model, solve_racemate(model, guess), closed)
    out["Control B"] = classify(control_b(), toy_racemate())
    for label, mechanism, modules in (
            ("Constructive 1010", "constructive", ("matched", "homoligation")),
            ("Destructive 0110", "destructive", ("mismatched", "homoligation"))):
        p = POINTS[mechanism]
        model = parent(p, modules)
        for x in racemic_parent_states(p, modules):
            if x[1] > 1e-9:
                record = classify(model, x)
                if record["alpha_even"] < -1e-8:
                    out[label] = record
    return out


def main():
    data = records()
    for label, r in data.items():
        if not (r["alpha_even"] < 0 < r["alpha_odd"] and r["classes"]):
            raise AssertionError(f"{label} is not a chirally unstable example: {r}")
        if r["inhibition_only"] and r["alpha_pair_sum"] > r["alpha_odd"]+1e-10:
            raise AssertionError(f"{label} violates the comparison under inhibition")
    (RESULTS/"literature_models.json").write_text(json.dumps(data, indent=2)+"\n")
    letters = {"cross": "c", "within": "w", "shared": "s"}
    lines = ["% Generated by Experiment/Autocatalyic/literature_models.py; do not edit.",
             r"\begin{tabular}{l l r r l c}", r"\toprule",
             r"Model & Mirror-fixed & $\alpha_+$ & $\alpha_-$ & Classes & Inhibition only\\",
             r"\midrule"]
    for label, r in data.items():
        lines.append(" & ".join([
            label, ", ".join(f"${s}$" for s in r["mirror_fixed_species"]) or "none",
            f"${r['alpha_even']:.3f}$", f"${r['alpha_odd']:.3f}$",
            ",".join(letters[c] for c in r["classes"]),
            "yes" if r["inhibition_only"] else "no"])+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER/"literature_table.tex").write_text("\n".join(lines)+"\n")
    for label, r in data.items():
        print(label, r["mirror_fixed_species"], round(r["alpha_even"], 4),
              round(r["alpha_odd"], 4), r["classes"], r["inhibition_only"])


if __name__ == "__main__":
    main()
