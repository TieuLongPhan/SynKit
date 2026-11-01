"""Controls, thresholds, and all 16 module supports of both mechanisms.

Computes the two control networks, the threshold scans, every module
support at the two reference points with cores, racemic states and seeded
endpoints, racemization, and the destructive model at zero reverse
conversion.

The study is regenerated from the repository root, with
PYTHONDONTWRITEBYTECODE=1, in this order::

    python Experiment/Autocatalytic/Homochiral/supports.py
    python Experiment/Autocatalytic/Homochiral/couplings.py
    python Experiment/Autocatalytic/Homochiral/sampling.py
    python Experiment/Autocatalytic/Homochiral/bifurcation.py
    python Experiment/Autocatalytic/Homochiral/thermodynamics.py
    python Experiment/Autocatalytic/Homochiral/literature.py
    python Experiment/Autocatalytic/Homochiral/reproduction.py
    python Experiment/Autocatalytic/Homochiral/maps.py
    python Experiment/Autocatalytic/Homochiral/figures.py
    pytest -q -o addopts='' Experiment/Autocatalytic/Homochiral

tools.py is optional and needs local clones of Autogatito and BiRNe. The
LaTeX fragments in results/latex hold the numbers that main.tex copies.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from analysis import (chiral_phenotype, paired_trajectories, parent_criterion, parent_odd_matrix,
                      positive_racemic_state, productive_cores, racemic_parent_states,
                      stability, toy_racemate)
from models import (ParentParameters, control_a, control_b, parent,
                    parent_supports)

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
# LaTeX fragments with generated numbers; main.tex carries a copy of them.
PAPER = HERE/"results"/"latex"
SEED = 1729


def save_json(name, obj):
    (RESULTS/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def save_csv(name, rows):
    if not rows:
        return
    with (RESULTS/name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_trajectory(name, model, times, plus, minus):
    rows = []
    for j, t in enumerate(times):
        row = {"time": t}
        for suffix, trajectory in (("plus", plus), ("minus", minus)):
            row.update({s+"_"+suffix: trajectory[i, j] for i, s in enumerate(model.species)})
        rows.append(row)
    save_csv(name, rows)


def controls():
    a = control_a()
    rac_a = np.repeat((1+np.sqrt(5))/2, 2)
    sa = stability(a, rac_a)
    cores_a = productive_cores(a)
    b = control_b()
    rac_b = toy_racemate()
    sb = stability(b, rac_b)
    cores_b = productive_cores(b)
    paired, times, yp, ym = paired_trajectories(b, rac_b)
    save_trajectory("control_b_trajectory.csv", b, times, yp, ym)
    reversal = []
    for eps in (0., 1e-6, 1e-3, 1e-2):
        model = control_b(eps)
        rac = toy_racemate(eps)
        pair, _, _, _ = paired_trajectories(model, rac)
        structure = cores_b if eps == 0 else productive_cores(model)
        reversal.append({"epsilon": eps, "racemate": stability(model, rac),
                         "structural": structure, "paired": pair})
    race = []
    for kr in (0., .25, .45, .55):
        model = control_b(kr=kr)
        info = stability(model, rac_b)
        horizon = max(4000., 30/max(abs(info["alpha_odd"]), 1e-4))
        pair, _, _, _ = paired_trajectories(model, rac_b, horizon=horizon)
        race.append({"kr": kr, "racemate": info, "paired": pair})
    assert abs(sa["alpha_odd"]+np.sqrt(5)) < 1e-12
    assert len(cores_a["cores"]) == 2 and not cores_b["cores"]
    assert abs(sb["alpha_odd"]-(-6+np.sqrt(37))) < 1e-12
    assert sb["alpha_even"] < 0 < sb["alpha_odd"]
    assert paired["plus"]["alpha_full"] < 0
    assert race[-1]["racemate"]["alpha_odd"] < 0
    result = {"control_a": {"model": a.to_record(), "racemate": sa, "structural": cores_a},
              "control_b": {"model": b.to_record(), "racemate": sb,
                            "structural": cores_b, "paired": paired},
              "reverse_channel_continuation": reversal,
              "racemization": race, "toy_kr_critical": .5}
    save_json("controls.json", result)
    return result, (times, yp, ym)


def threshold_curve(mechanism):
    if mechanism == "constructive":
        base = ParentParameters()
        grid = np.linspace(0, 10, 201)
        parameters = lambda ratio: replace(base, k1=.75*ratio)
    else:
        base = ParentParameters(mechanism=mechanism, k1=10., k2=80.)
        grid = np.linspace(0, 12, 241)
        parameters = lambda ratio: replace(base, k2=10.*ratio)
    rows = []
    for ratio in grid:
        p = parameters(ratio)
        model = parent(p)
        for branch, x in enumerate(racemic_parent_states(p)):
            state = stability(model, x)
            odd = parent_odd_matrix(p, x)
            numeric_odd = model.q_odd.T @ model.jacobian(0, x) @ model.q_odd
            # odd ordering is monomers, homodimers, heterodimers.
            if np.max(np.abs(numeric_odd[:2, :2]-odd)) > 1e-10:
                raise RuntimeError("Analytic odd block disagrees with compiled kinetics")
            criterion = parent_criterion(p, x)
            rows.append({"mechanism": mechanism, "ratio": ratio, "branch": branch,
                         "monomer": x[1], "precursor": x[0],
                         "alpha_even": state["alpha_even"], "alpha_odd": state["alpha_odd"],
                         "residual_inf": state["residual_inf"],
                         "criterion_margin": criterion["margin"],
                         "positive_monomers": x[1] > 1e-9})
    def margin(ratio):
        p = parameters(ratio)
        return parent_criterion(p, positive_racemic_state(p))["margin"]
    # The constructive system also has extinction and an activation saddle.
    # Bracket the chiral crossing on the actual positive even-stable branch.
    active = [r for r in rows if r["positive_monomers"] and r["alpha_even"] < -1e-8]
    brackets = [(a["ratio"], b["ratio"]) for a, b in zip(active, active[1:])
                if a["criterion_margin"] <= 0 < b["criterion_margin"]]
    if len(brackets) != 1:
        raise RuntimeError(f"Expected one chiral crossing in declared grid, found {brackets}")
    bracket = brackets[0]
    critical = brentq(margin, *bracket, xtol=1e-12)
    p = parameters(critical)
    x = positive_racemic_state(p)
    return {"mechanism": mechanism, "critical_ratio": critical,
            "critical_parameters": parent(p).metadata["parameters"],
            "racemate_at_threshold": stability(parent(p), x),
            "ratio_grid": [float(grid[0]), float(grid[-1]), len(grid)],
            "racemic_branches_recorded": len(rows)}, rows


def module_comparison(p, structural):
    records = []
    for modules in parent_supports():
        model = parent(p, modules)
        states = []
        trajectories = []
        for x in racemic_parent_states(p, modules):
            info = stability(model, x)
            info["criterion"] = parent_criterion(p, x, modules)
            info["clean_chiral_instability"] = bool(x[1] > 1e-9
                and info["alpha_even"] < -1e-8 and info["alpha_odd"] > 1e-8)
            if info["clean_chiral_instability"]:
                horizon = max(4000., 30/info["alpha_odd"])
                pair, times, yp, ym = paired_trajectories(model, x, horizon=horizon)
                trajectories.append(pair)
            states.append(info)
        cores = [c for c in structural["cores"] if set(c["modules"]) <= set(modules)]
        clean = any(s["clean_chiral_instability"] for s in states)
        realized = any(chiral_phenotype(t) for t in trajectories)
        records.append({"modules": list(modules), "channel_ids": [r.id for r in model.channels],
                        "parameters": model.metadata["parameters"],
                        "racemic_states": states, "paired_trajectories": trajectories,
                        "productive_core_count": len(cores), "productive_cores": cores,
                        "clean_chiral_instability": clean, "stable_chiral_endpoints": realized})
    for record in records:
        support = set(record["modules"])
        proper_passing = [r["modules"] for r in records if set(r["modules"]) < support
                          and r["clean_chiral_instability"] and r["stable_chiral_endpoints"]]
        record["proper_passing_subsets"] = proper_passing
        record["minimal_at_inherited_rates"] = (record["clean_chiral_instability"]
                                                and record["stable_chiral_endpoints"]
                                                and not proper_passing)
    save_json(p.mechanism+"_modules.json", records)
    return records


def robustness(p):
    x = positive_racemic_state(p)
    critical = parent_criterion(p, x)["kr_critical"]
    rows, detailed = [], []
    for fraction in (0., .25, .5, .75, .9, 1.1):
        actual = replace(p, kr=fraction*critical)
        model = parent(actual)
        info = stability(model, x)
        horizon = max(4000., 30/max(abs(info["alpha_odd"]), 1e-5))
        pair, _, _, _ = paired_trajectories(model, x, horizon=horizon)
        endpoint = pair["plus"]
        rows.append({"mechanism": p.mechanism, "kr_fraction": fraction,
                     "kr": actual.kr, "alpha_odd": info["alpha_odd"],
                     "alpha_endpoint": endpoint["alpha_full"],
                     "residual_inf": endpoint["residual_inf"], **endpoint["metrics"]})
        detailed.append({"parameters": model.metadata["parameters"],
                         "racemate": info, "paired": pair})
    hydrolysis = []
    for kdim in (0., 1e-6, 1e-3, 1e-2, 1e-1):
        actual = replace(p, kdim=kdim)
        model = parent(actual)
        for state in racemic_parent_states(actual):
            if state[1] > 1e-9:
                info = stability(model, state)
                info["criterion"] = parent_criterion(actual, state)
                hydrolysis.append({"kdim": kdim, "racemate": info})
    save_csv(p.mechanism+"_racemization.csv", rows)
    report = {"kr_critical": critical, "racemization": detailed,
              "hydrolysis_convention": "chemically explicit LD/DL -> L+D",
              "hydrolysis_racemic_scan": hydrolysis}
    save_json(p.mechanism+"_robustness.json", report)
    return report, rows


def destructive_reverse_limit(p, positive_structure):
    """Continue the actual parent from zero reverse conversion to its baseline.

    All positive kmon values have the same structural support, so exhaustive
    enumeration is reused; rates and nonlinear dynamics are recalculated.
    """
    records, rows = [], []
    for beta in (0., 1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
        actual = replace(p, kmon=beta)
        model = parent(actual)
        structure = productive_cores(model) if beta == 0 else positive_structure
        racemate = positive_racemic_state(actual)
        info = stability(model, racemate)
        pair, _, _, _ = paired_trajectories(model, racemate,
                                          horizon=max(4000., 30/info["alpha_odd"]))
        records.append({"parameters": model.metadata["parameters"],
                        "structural": structure, "racemate": info, "paired": pair})
        rows.append({"kmon": beta, "core_count": len(structure["cores"]),
                     "alpha_even": info["alpha_even"], "alpha_odd": info["alpha_odd"],
                     "alpha_endpoint": pair["plus"]["alpha_full"],
                     "paired_mirror_error": pair["paired_trajectory_mirror_error"],
                     **pair["plus"]["metrics"]})
    if records[0]["structural"]["cores"] or not all(r["racemate"]["alpha_odd"] > 0 for r in records):
        raise RuntimeError("Parent reverse-limit diagnostic failed")
    save_json("destructive_reverse_limit.json", records)
    save_csv("destructive_reverse_limit.csv", rows)
    return records, rows


def paper_assets(summary, comparisons):
    c = summary["controls"]
    numbers = {
        "ToyOdd": c["control_b"]["racemate"]["alpha_odd"],
        "ToyEven": c["control_b"]["racemate"]["alpha_even"],
        "ToyEE": c["control_b"]["paired"]["plus"]["metrics"]["ee_monomer"],
        "ToyStable": c["control_b"]["paired"]["plus"]["alpha_full"],
        "ConstructThreshold": summary["thresholds"]["constructive"]["critical_ratio"],
        "DestructThreshold": summary["thresholds"]["destructive"]["critical_ratio"],
        "ConstructRace": summary["robustness"]["constructive"]["kr_critical"],
        "DestructRace": summary["robustness"]["destructive"]["kr_critical"],
        "ParentIrrevOdd": summary["destructive_reverse_limit"][0]["racemate"]["alpha_odd"],
        "ParentIrrevEE": summary["destructive_reverse_limit"][0]["paired"]["plus"]["metrics"]["ee_monomer"],
        "ParentIrrevStable": summary["destructive_reverse_limit"][0]["paired"]["plus"]["alpha_full"],
    }
    (PAPER/"supports.tex").write_text("% Generated by Experiment/Autocatalytic/Homochiral/supports.py; do not edit.\n"+
        "\n".join("\\newcommand{\\"+name+"}{"+format(value, ".8g")+"}" for name, value in numbers.items())+"\n")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    print("Controls and core enumeration...", flush=True)
    control, _ = controls()
    bases = {"constructive": ParentParameters(),
             "destructive": ParentParameters(mechanism="destructive", k1=10., k2=80.)}
    structural, comparisons, thresholds, robust = {}, {}, {}, {}
    for mechanism, parameters in bases.items():
        model = parent(parameters)
        save_json(mechanism+"_model.json", model.to_record())
        print(f"Cores, threshold, and all 16 module supports: {mechanism}...", flush=True)
        structural[mechanism] = productive_cores(model)
        save_json(mechanism+"_cores.json", structural[mechanism])
        thresholds[mechanism], _ = threshold_curve(mechanism)
        comparisons[mechanism] = module_comparison(parameters, structural[mechanism])
        robust[mechanism], _ = robustness(parameters)
    print("Destructive model at zero reverse conversion...", flush=True)
    reverse_records, _ = destructive_reverse_limit(bases["destructive"], structural["destructive"])
    save_json("thresholds.json", thresholds)
    paper_assets({"controls": control, "thresholds": thresholds, "robustness": robust,
                  "destructive_reverse_limit": reverse_records}, comparisons)


if __name__ == "__main__":
    main()
