"""Run the complete bounded pilot and regenerate its paper assets.

Usage from repository root:
    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalyic/run.py
"""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/synkit-autocatalytic-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from analysis import (integrate_endpoint, metrics, paired_trajectories, parent_criterion, parent_odd_matrix,
                      positive_racemic_state, productive_cores, racemic_parent_states,
                      stability, toy_racemate)
from models import (MODULES, ParentParameters, control_a, control_b, parent,
                    parent_supports)
from reference_check import check_reference, source_manifest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE/"results"
PAPER = ROOT/"paper"/"autocatalytic"
FIGURES = PAPER/"figures"
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


def syncrn_roundtrip(model):
    # Use the repository's existing compiler; preserve both incidence sides.
    sys.path.insert(0, str(ROOT))
    from synkit.CRN.Structure.syncrn import SynCRN

    def side(mapping):
        return "+".join((str(n) if n != 1 else "")+s for s, n in mapping.items())

    strings = [side(r.reactants)+">>"+side(r.products) for r in model.channels]
    syn = SynCRN.from_reaction_strings(strings)
    for rid, channel in zip(syn.reaction_ids, model.to_record()["channels"]):
        # SynCRN's graph serializer preserves source_attrs; use that public
        # carrier rather than assuming canonical metadata is serialized.
        syn.reactions[rid].source_attrs["pilot"] = channel
    recovered = SynCRN.from_digraph(syn.to_digraph())
    matrix = recovered.to_stoichiometric_matrices()
    labels = [recovered.species[sid].label for sid in matrix["species_order"]]
    order = [labels.index(s) for s in model.species if s in labels]
    expected = [model.species.index(s) for s in model.species if s in labels]
    passed = (np.array_equal(matrix["S_minus"][order], model.minus[expected])
              and np.array_equal(matrix["S_plus"][order], model.plus[expected]))
    # Metadata is carried by source attributes on a graph roundtrip.
    metadata_checks = []
    for rid, channel in zip(recovered.reaction_ids, model.channels):
        reaction = recovered.reactions[rid]
        source = reaction.source_attrs
        meta = source["pilot"]
        metadata_checks.append(meta.get("rate") == channel.rate)
    if not passed or not all(metadata_checks):
        raise RuntimeError("SynCRN roundtrip lost incidence or kinetic metadata")
    return {"incidence_preserved": passed, "rate_metadata_preserved": all(metadata_checks),
            "channels": len(strings), "species_labels": labels,
            "adapter_scope": "existing SynCRN string compiler and directed-graph roundtrip"}


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
    save_csv(mechanism+"_racemic_scan.csv", rows)
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
                tag = "-".join(m for m in MODULES if m in modules)
                save_trajectory(p.mechanism+"_"+tag+"_trajectory.csv", model, times, yp, ym)
            states.append(info)
        cores = [c for c in structural["cores"] if set(c["modules"]) <= set(modules)]
        clean = any(s["clean_chiral_instability"] for s in states)
        realized = any(t["plus"]["alpha_full"] < -1e-8
                       and abs(t["plus"]["metrics"]["ee_monomer"] or 0) > 1e-3
                       and t["plus"]["metrics"]["monomer_concentration"] > 1e-6
                       and t["paired_trajectory_mirror_error"] < 1e-6 for t in trajectories)
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


def tolerance_checks(control, comparisons, bases):
    cases = [("control_b", control_b(), control["control_b"]["paired"])]
    for mechanism, parameters in bases.items():
        record = next(r for r in comparisons[mechanism] if len(r["modules"]) == 4)
        cases.append((mechanism, parent(parameters), record["paired_trajectories"][0]))
    reports = []
    for name, model, pair in cases:
        original = pair["plus"]
        tighter, _, _ = integrate_endpoint(model, np.asarray(pair["initial_plus"]),
                                          horizon=original["horizon"], rtol=1e-10, atol=1e-13)
        difference = float(np.max(np.abs(np.asarray(tighter["state"])-original["state"])))
        ee_difference = abs(tighter["metrics"]["ee_monomer"]-original["metrics"]["ee_monomer"])
        if difference > 1e-7 or ee_difference > 1e-7:
            raise RuntimeError(f"Endpoint tolerance convergence failed for {name}")
        reports.append({"model": name, "standard_rtol": 2e-9, "standard_atol": 2e-12,
                        "tighter_rtol": 1e-10, "tighter_atol": 1e-13,
                        "endpoint_max_abs_difference": difference,
                        "ee_monomer_difference": ee_difference, "tighter_endpoint": tighter})
    save_json("tolerance_validation.json", reports)
    return reports


def make_figures(control, toy, scans, comparisons, race_rows, parent_reverse_rows):
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1), constrained_layout=True)
    times, yp, ym = toy
    axes[0].plot(times, (yp[0]-yp[1])/(yp[0]+yp[1]), label="+ perturbation")
    axes[0].plot(times, (ym[0]-ym[1])/(ym[0]+ym[1]), label="mirror perturbation")
    axes[0].set(xlabel="Time", ylabel="Monomer ee", xlim=(0, 500), title="Selective-removal control")
    axes[0].legend(frameon=False, fontsize=8)
    reverse = control["reverse_channel_continuation"]
    eps = [r["epsilon"] for r in reverse]
    axes[1].plot(eps, [r["racemate"]["alpha_odd"] for r in reverse], "o-")
    axes[1].set_xscale("symlog", linthresh=1e-6)
    axes[1].set(xlabel="Reverse coefficient epsilon", ylabel="Odd growth rate", title="Continuous dynamics")
    axes[2].plot(eps, [len(r["structural"]["cores"]) for r in reverse], "o-")
    axes[2].set_xscale("symlog", linthresh=1e-6)
    axes[2].set(xlabel="Reverse coefficient epsilon", ylabel="Productive core count", title="Discrete support change")
    fig.savefig(FIGURES/"controls.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1), constrained_layout=True)
    for ax, (threshold, rows) in zip(axes, scans):
        positive = [r for r in rows if r["positive_monomers"] and r["alpha_even"] < -1e-8]
        ax.plot([r["ratio"] for r in positive], [r["alpha_odd"] for r in positive], label="odd sector")
        ax.plot([r["ratio"] for r in positive], [r["alpha_even"] for r in positive], label="even sector")
        ax.axhline(0, color=".5", lw=.7)
        ax.axvline(threshold["critical_ratio"], color=".5", ls="--", lw=.8)
        ratio_label = "k1/k2" if threshold["mechanism"] == "constructive" else "k2/k1"
        ax.set(xlabel=ratio_label, ylabel="Spectral abscissa", title=threshold["mechanism"].capitalize())
        ax.legend(frameon=False)
    fig.savefig(FIGURES/"parent_thresholds.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), constrained_layout=True)
    for ax, (mechanism, records) in zip(axes, comparisons.items()):
        data = np.array([[r["productive_core_count"] > 0, r["clean_chiral_instability"],
                          r["stable_chiral_endpoints"], r["minimal_at_inherited_rates"]] for r in records])
        ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        labels = ["".join(str(int(m in r["modules"])) for m in MODULES) for r in records]
        ax.set_yticks(range(len(records)), labels)
        ax.set_xticks(range(4), ["Core", "Odd instability", "Stable chiral pair", "Minimal"], rotation=35, ha="right")
        ax.set(ylabel="Modules: matched, mismatched, homo, hetero", title=mechanism.capitalize())
    fig.savefig(FIGURES/"module_comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1), constrained_layout=True)
    for mechanism, rows in race_rows.items():
        axes[0].plot([r["kr_fraction"] for r in rows], [abs(r["ee_monomer"]) for r in rows], "o-", label=mechanism)
        axes[1].plot([r["kr_fraction"] for r in rows], [r["peptide_residue_outflow"] for r in rows], "o-", label=mechanism)
    axes[0].set(xlabel="kr / critical kr", ylabel="Absolute monomer ee")
    axes[1].set(xlabel="kr / critical kr", ylabel="Peptide residue outflow")
    for ax in axes:
        ax.axvline(1, color=".5", ls="--", lw=.8)
        ax.legend(frameon=False)
    fig.savefig(FIGURES/"racemization.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1), constrained_layout=True)
    beta = [r["kmon"] for r in parent_reverse_rows]
    axes[0].plot(beta, [r["alpha_odd"] for r in parent_reverse_rows], "o-", label="Odd growth rate")
    axes[0].plot(beta, [r["alpha_endpoint"] for r in parent_reverse_rows], "s-", label="Endpoint stability")
    axes[0].axhline(0, color=".5", lw=.7)
    axes[0].set(ylabel="Spectral abscissa")
    axes[0].legend(frameon=False)
    axes[1].plot(beta, [r["ee_monomer"] for r in parent_reverse_rows], "o-", color="tab:orange")
    axes[1].set(ylabel="Monomer ee")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1e-8)
        ax.set_xlabel("Reverse conversion factor kmon")
        ax.axvline(0, color=".5", lw=.7)
    fig.savefig(FIGURES/"parent_reverse_limit.pdf")
    plt.close(fig)


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
        "ReferenceRHS": max(r.get("rhs_max_abs_error", 0) for r in summary["reference"]["checks"]),
        "ReferenceTrajectory": max(r.get("finite_time_max_abs_error", 0) for r in summary["reference"]["checks"]),
        "ParentIrrevOdd": summary["destructive_reverse_limit"][0]["racemate"]["alpha_odd"],
        "ParentIrrevEE": summary["destructive_reverse_limit"][0]["paired"]["plus"]["metrics"]["ee_monomer"],
        "ParentIrrevStable": summary["destructive_reverse_limit"][0]["paired"]["plus"]["alpha_full"],
    }
    (PAPER/"results.tex").write_text("% Generated by Experiment/Autocatalyic/run.py; do not edit.\n"+
        "\n".join("\\newcommand{\\"+name+"}{"+format(value, ".8g")+"}" for name, value in numbers.items())+"\n")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Mechanism & Modules & Cores & $\alpha_-$ & $|ee_m|$ & Minimal \\", r"\midrule"]
    for mechanism, records in comparisons.items():
        for record in records:
            if not record["clean_chiral_instability"]:
                continue
            state = next(s for s in record["racemic_states"] if s["clean_chiral_instability"])
            ee = abs(record["paired_trajectories"][0]["plus"]["metrics"]["ee_monomer"])
            bits = "".join(str(int(m in record["modules"])) for m in MODULES)
            lines.append(f"{mechanism.capitalize()} & {bits} & {record['productive_core_count']} & "
                         f"{state['alpha_odd']:.6f} & {ee:.6f} & "
                         +( "yes" if record["minimal_at_inherited_rates"] else "no")+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER/"module_table.tex").write_text("\n".join(lines)+"\n")


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    print("Checking independently compiled reference C equations...", flush=True)
    reference = check_reference(SEED)
    save_json("source_manifest.json", source_manifest())
    save_json("reference_validation.json", reference)
    print("Validating controls and complete small-support enumeration...", flush=True)
    control, toy = controls()
    bases = {"constructive": ParentParameters(),
             "destructive": ParentParameters(mechanism="destructive", k1=10., k2=80.)}
    structural, comparisons, thresholds, scans, robust, race_rows, roundtrips = {}, {}, {}, [], {}, {}, {}
    for mechanism, parameters in bases.items():
        model = parent(parameters)
        save_json(mechanism+"_model.json", model.to_record())
        roundtrips[mechanism] = syncrn_roundtrip(model)
        print(f"Enumerating internal productive supports: {mechanism}...", flush=True)
        structural[mechanism] = productive_cores(model)
        save_json(mechanism+"_cores.json", structural[mechanism])
        print(f"Computing reference threshold and all 16 module supports: {mechanism}...", flush=True)
        threshold, rows = threshold_curve(mechanism)
        thresholds[mechanism] = threshold
        scans.append((threshold, rows))
        comparisons[mechanism] = module_comparison(parameters, structural[mechanism])
        print(f"Testing racemization and positive hydrolysis: {mechanism}...", flush=True)
        robust[mechanism], race_rows[mechanism] = robustness(parameters)
    print("Continuing the destructive parent through its zero-reverse limit...", flush=True)
    reverse_records, reverse_rows = destructive_reverse_limit(bases["destructive"], structural["destructive"])
    convergence = tolerance_checks(control, comparisons, bases)
    summary = {"schema_version": 1, "seed": SEED, "reference": reference,
               "controls": control, "thresholds": thresholds, "robustness": robust,
               "syncrn_roundtrips": roundtrips,
               "destructive_reverse_limit": reverse_records,
               "tolerance_validation": convergence,
               "module_summary": {m: {"supports_evaluated": len(rs),
                    "supports_with_productive_cores": sum(r["productive_core_count"] > 0 for r in rs),
                    "supports_with_clean_chiral_instability": sum(r["clean_chiral_instability"] for r in rs),
                    "supports_with_stable_chiral_pairs": sum(r["stable_chiral_endpoints"] for r in rs),
                    "minimal_supports_at_inherited_rates": [r["modules"] for r in rs if r["minimal_at_inherited_rates"]]}
                    for m, rs in comparisons.items()},
               "scope": {"rate_quantifier": "two declared inherited parameter points; no existential kinetic completeness claim",
                         "internal_productivity": "L,D,LL,DD,LD,DL; P and reactor input/loss excluded",
                         "external_backends": {"Autogatito": "not executed", "Autogato": "not executed", "BiRNe": "not executed"},
                         "thermodynamics": "shared conversion equilibrium ratios enforced; physical compatibility of synthesis and ligation conditions not established"}}
    print("Writing results, figures, and manuscript tables...", flush=True)
    make_figures(control, toy, scans, comparisons, race_rows, reverse_rows)
    paper_assets(summary, comparisons)
    save_json("summary.json", summary)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib", "networkx", "pytest")}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = sorted(HERE.glob("*.py"))+sorted((HERE/"reference").glob("*.c"))
    save_json("provenance.json", {"python": platform.python_version(), "platform": platform.platform(),
              "packages": versions, "git_head": commit, "worktree_was_dirty": True,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "syncrn_adapter_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((ROOT/"synkit/CRN/Structure").glob("*.py"))},
              "plan_sha256": hashlib.sha256((PAPER/"chiral_symmetry_crn_research_plan.md").read_bytes()).hexdigest(),
              "seed": SEED, "command": "PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalyic/run.py"})
    (HERE/"environment-lock.txt").write_text("# Environment used for this pilot; does not replace the repository environment.\n"+
                                          f"# Python {platform.python_version()}\n"+
                                          "\n".join(f"{k}=={v}" for k,v in versions.items())+"\n")
    print(json.dumps(summary["module_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
