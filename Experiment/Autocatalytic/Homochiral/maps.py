"""Stability of the racemate over the preference ratio and reverse conversion.

For the full model of each mechanism, the reference rates are kept and two
coefficients are varied: the catalytic preference ratio (k1/k2 with
k2 = 0.75, or k2/k1 with k1 = 10) and the reverse conversion factor beta,
including beta = 0. Every grid point is classified by its positive racemic
states as chirally unstable (alpha_+ < 0 < alpha_-), stable, or without a
positive even-stable racemate. Cores are counted at beta = 0 and at one
positive beta, since the core set does not change for beta > 0.

Run from the repository root::

    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalytic/Homochiral/maps.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis import productive_cores, racemic_parent_states, stability
from models import ParentParameters, parent

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
SETUPS = {
    "constructive": (ParentParameters(), "k1", 0.75, np.linspace(0.05, 10, 200)),
    "destructive": (ParentParameters(mechanism="destructive", k1=10., k2=80.), "k2", 10.,
                    np.linspace(0.05, 12, 240)),
}
BETAS = np.concatenate([[0.0], np.logspace(-6, 0, 97)])
CHIRAL, STABLE, NONE = 2, 1, 0


def classify(p):
    """Label of the grid point and the largest alpha_- of an even-stable racemate."""
    model = parent(p)
    label, odd = NONE, float("nan")
    for x in racemic_parent_states(p):
        if x[1] <= 1e-9:
            continue
        info = stability(model, x)
        if info["alpha_even"] >= -1e-10:
            continue
        odd = info["alpha_odd"] if np.isnan(odd) else max(odd, info["alpha_odd"])
        label = CHIRAL if odd > 1e-10 else STABLE
    return label, odd


def scan(mechanism):
    base, field, scale, ratios = SETUPS[mechanism]
    grid = np.zeros((len(BETAS), len(ratios)), dtype=int)
    odd = np.full(grid.shape, np.nan)
    for i, beta in enumerate(BETAS):
        for j, ratio in enumerate(ratios):
            grid[i, j], odd[i, j] = classify(
                replace(base, kmon=float(beta), **{field: scale*ratio}))
    cores = {label: len(productive_cores(parent(replace(base, kmon=beta)))["cores"])
             for label, beta in (("zero", 0.0), ("positive", 0.01))}
    return {"ratio": ratios.tolist(), "beta": BETAS.tolist(), "label": grid.tolist(),
            "alpha_odd": [[None if np.isnan(v) else v for v in row] for row in odd],
            "cores": cores, "codes": {"chiral": CHIRAL, "stable": STABLE, "none": NONE}}


def scan_lines():
    """Sector abscissae along the archived scans (beta = 0.01), for the figure.

    alpha(A+B) is the even abscissa with the precursor held fixed and decides
    the route of a chiral instability (interference when it is negative).
    """
    from couplings import pair_blocks, sector_matrices
    out = {}
    for mechanism, (base, field, scale, _) in SETUPS.items():
        rows = []
        for ratio in np.arange(0.05, (10 if mechanism == "constructive" else 12)+1e-9, 0.05):
            p = replace(base, **{field: scale*ratio})
            model = parent(p)
            for x in racemic_parent_states(p):
                if x[1] <= 1e-9:
                    continue
                blocks, _ = pair_blocks(model, x)
                even, odd = sector_matrices(blocks)
                values = [max(np.linalg.eigvals(m).real)
                          for m in (even, odd, blocks["A"]+blocks["B"])]
                if values[0] < -1e-10:
                    rows.append([float(ratio)]+[float(v) for v in values])
        out[mechanism] = {"columns": ["ratio", "alpha_even", "alpha_odd", "alpha_pair_sum"],
                          "rows": rows}
    (RESULTS/"scan_lines.json").write_text(json.dumps(out))
    return out


def main():
    out = {mechanism: scan(mechanism) for mechanism in SETUPS}
    (RESULTS/"maps.json").write_text(json.dumps(out))
    scan_lines()
    for mechanism, record in out.items():
        grid = np.array(record["label"])
        print(mechanism, record["cores"], {k: int((grid == v).sum())
                                          for k, v in record["codes"].items()})


if __name__ == "__main__":
    main()
