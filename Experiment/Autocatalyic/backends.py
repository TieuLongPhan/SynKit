"""Run the published structural tools on the pilot networks.

The tools are not vendored. Clone them next to each other and point
``AUTOCAT_BACKENDS`` at that directory::

    git clone https://github.com/hollyritch/Bi.R.Ne
    git clone https://github.com/hollyritch/MRChordlessCircuits
    AUTOCAT_BACKENDS=/path/to/clones python Experiment/Autocatalyic/backends.py

BiRNe needs SymPy, SciPy, NetworkX and tqdm. Autogatito additionally needs
python-libsbml and Cython, and ``python setup.py build_ext --inplace`` inside
``MRChordlessCircuits/autogatito``. ``AUTOCAT_PYTHON`` selects the interpreter
used for both tools (default: the current one).

Each network is exported from the same ``Network`` object that the kinetic
analysis uses. Feed and first-order loss become explicit inflow and outflow
reactions for BiRNe, which analyzes the open system. For Autogatito only the
internal reactions are exported and non-internal species are left out, which
is the boundary convention of the manuscript.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis import productive_cores
from models import MODULES, control_a, control_b, parent
from sign_structure import POINTS

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
WORK = Path(os.environ.get("AUTOCAT_WORK", "/tmp/synkit-autocatalytic-backends"))
PYTHON = os.environ.get("AUTOCAT_PYTHON", sys.executable)


def networks():
    con, des = POINTS["constructive"], POINTS["destructive"]
    return {
        "control_a": control_a(),
        "control_b": control_b(0.0),
        "control_b_reverse": control_b(1e-3),
        "constructive_1010": parent(con, ("matched", "homoligation")),
        "destructive_0110": parent(des, ("mismatched", "homoligation")),
        "destructive_zero_reverse": parent(replace(des, kmon=0.0)),
        "constructive_full": parent(con, MODULES),
        "destructive_full": parent(des, MODULES),
    }


def side(counts):
    return " + ".join((f"{n}{s}" if n > 1 else s) for s, n in counts.items())


def active_species(model):
    """Species that take part in a reaction or are fed."""
    used = (model.minus.sum(axis=1)+model.plus.sum(axis=1) > 0) | (model.feed > 0)
    return [s for s, keep in zip(model.species, used) if keep]


def birne_input(model):
    species = active_species(model)
    index = {s: i for i, s in enumerate(model.species)}
    lines = ["Metabolites:"]+[f"{s}: {s}" for s in species]+["", "Reactions:"]
    for s in species:
        if model.feed[index[s]] > 0:
            lines.append(f"feed_{s}: -> {s}")
    for s in species:
        if model.loss[index[s]] > 0:
            lines.append(f"loss_{s}: {s} ->")
    for r in model.channels:
        lines.append(f"{r.id}: {side(r.reactants)} -> {side(r.products)}")
    return "\n".join(lines)+"\n"


def run_birne(name, model, root, timeout):
    out = WORK/"birne"/name
    out.mkdir(parents=True, exist_ok=True)
    source = out/"network.txt"
    source.write_text(birne_input(model))
    try:
        subprocess.run([PYTHON, "-W", "ignore", "coefficientAnalyzer.py", "-i", str(source),
                        "-o", str(out)], cwd=root/"Bi.R.Ne", check=True, timeout=timeout,
                       capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_seconds": timeout}
    text = (out/"output.txt").read_text()
    blocks = re.split(r">>>> Current coefficient is:", text)[1:]
    cores = []
    for block in blocks:
        reactions = re.search(r"Reactions are (\[.*?\])\n", block)
        species = re.search(r"Metabolites are (\[.*?\])\n", block)
        cores.append({
            "autocatalytic": "It is autocatalytic" in block,
            "reactions": re.findall(r"\(\d+, '([^']+)'\)", reactions.group(1)),
            "species": re.findall(r"\(\d+, \('([^']+)'", species.group(1)),
            "oscillatory": "oscillat" in block.lower() and "No osz" not in block})
    return {"status": "completed",
            "consistent": "The network is consistent" in text,
            "nondegenerate": "non-degnerate" in text,
            "admits_stability": "The network admits stability" in text,
            "admits_multistationarity": "Network admits multistationarity" in text,
            "unstable_cores": cores,
            "autocatalytic_count": sum(c["autocatalytic"] for c in cores),
            "nonautocatalytic_count": sum(not c["autocatalytic"] for c in cores)}


def sbml(model):
    """SBML level 3 with internal species only and catalysts on both sides."""
    import libsbml
    document = libsbml.SBMLDocument(3, 1)
    sb = document.createModel()
    sb.setId(model.name)
    compartment = sb.createCompartment()
    compartment.setId("c")
    compartment.setConstant(True)
    compartment.setSize(1)
    internal = set(model.internal)
    for s in model.internal:
        species = sb.createSpecies()
        species.setId(s)
        species.setName(s)
        species.setCompartment("c")
        for setter in (species.setConstant, species.setBoundaryCondition,
                       species.setHasOnlySubstanceUnits):
            setter(False)
        species.setInitialConcentration(1.0)
    kept = []
    for r in model.channels:
        reactants = {s: n for s, n in r.reactants.items() if s in internal}
        products = {s: n for s, n in r.products.items() if s in internal}
        if not reactants and not products:
            continue
        reaction = sb.createReaction()
        reaction.setId(r.id)
        reaction.setReversible(False)
        reaction.setFast(False) if hasattr(reaction, "setFast") else None
        for counts, create in ((reactants, reaction.createReactant),
                               (products, reaction.createProduct)):
            for s, n in counts.items():
                ref = create()
                ref.setSpecies(s)
                ref.setStoichiometry(float(n))
                ref.setConstant(True)
        kept.append(r.id)
    return libsbml.writeSBMLToString(document), kept


def run_autogatito(name, model, root, timeout):
    tool = root/"MRChordlessCircuits"/"autogatito"
    out = WORK/"autogatito"/name
    out.mkdir(parents=True, exist_ok=True)
    text, _ = sbml(model)
    source = out/"network.xml"
    source.write_text(text)
    trees = tool/"PickleFiles"/name
    shutil.rmtree(trees, ignore_errors=True)
    for stale in out.glob("partitionTreeData*.pkl"):
        stale.unlink()
    run = dict(cwd=tool, check=True, timeout=timeout, capture_output=True, text=True)
    # A cutoff of zero keeps single explicitly autocatalytic reactions.
    subprocess.run([PYTHON, "partitionNetwork.py", "-i", str(source), "-c", "0",
                    "-t", "2", "-o", name], **run)
    cores = set()
    for tree in sorted(trees.glob("partitionTree*.pkl")) if trees.exists() else []:
        subprocess.run([PYTHON, "partitionAnalysis.py", "-x", str(source), "-i",
                        str(tree.relative_to(tool)), "-s", name, "-b", "1000", "-e", "1000",
                        "-c", "-o", str(out)], **run)
        number = tree.stem.replace("partitionTree", "")
        with (out/f"partitionTreeData{number}.pkl").open("rb") as handle:
            parameters = pickle.load(handle)[0]
        names = {v: k for k, v in parameters["vertexIDs"].items()}
        for core in parameters["AutocatalyticCores"]:
            cores.add((tuple(sorted({names[m] for m, _ in core})),
                       tuple(sorted({names[r] for _, r in core}))))
    return {"status": "completed",
            "cores": sorted([list(species), list(reactions)] for species, reactions in cores)}


def ours(model):
    cores = productive_cores(model)["cores"]
    return sorted([sorted(c["species"]), sorted(c["channels"])] for c in cores)


def main():
    root = os.environ.get("AUTOCAT_BACKENDS")
    if not root:
        raise SystemExit("Set AUTOCAT_BACKENDS to the directory with the cloned tools")
    root = Path(root)
    revisions = {}
    for repo in ("Bi.R.Ne", "MRChordlessCircuits"):
        if (root/repo).exists():
            revisions[repo] = subprocess.run(["git", "-C", str(root/repo), "rev-parse", "HEAD"],
                                             capture_output=True, text=True).stdout.strip()
    timeout = int(os.environ.get("AUTOCAT_TIMEOUT", "1800"))
    records = {}
    for name, model in networks().items():
        internal = set(model.internal)
        record = {"species": active_species(model),
                  "reactions": [r.id for r in model.channels], "lp_cores": ours(model)}
        if "MRChordlessCircuits" in revisions:
            record["autogatito"] = run_autogatito(name, model, root, timeout)
            record["autogatito_equals_lp"] = record["autogatito"]["cores"] == record["lp_cores"]
        if "Bi.R.Ne" in revisions:
            record["birne"] = b = run_birne(name, model, root, timeout)
            if b["status"] == "completed":
                # BiRNe treats every species as internal. Restrict to the
                # internal set of the manuscript before comparing.
                inside = sorted([sorted(c["species"]), sorted(c["reactions"])]
                                for c in b["unstable_cores"]
                                if c["autocatalytic"] and set(c["species"]) <= internal)
                record["birne_internal_autocatalytic"] = inside
                record["birne_equals_lp"] = inside == record["lp_cores"]
        records[name] = record
        b = record.get("birne", {})
        print(name, "| LP", len(record["lp_cores"]),
              "| Autogatito", len(record.get("autogatito", {}).get("cores", [])),
              record.get("autogatito_equals_lp"),
              "| BiRNe autocatalytic", b.get("autocatalytic_count"),
              "internal", len(record.get("birne_internal_autocatalytic", [])),
              record.get("birne_equals_lp"),
              "non-autocatalytic", b.get("nonautocatalytic_count"), flush=True)
    (RESULTS/"backends.json").write_text(json.dumps(
        {"revisions": revisions, "networks": records}, indent=2)+"\n")
    write_table(records)


def write_table(records):
    labels = {"control_a": "Control A", "control_b": "Control B",
              "control_b_reverse": r"Control B, $\epsilon=10^{-3}$",
              "constructive_1010": "Constructive 1010",
              "destructive_0110": "Destructive 0110",
              "destructive_zero_reverse": r"Destructive 1111, $\beta=0$",
              "constructive_full": "Constructive 1111",
              "destructive_full": "Destructive 1111"}
    lines = ["% Generated by Experiment/Autocatalyic/backends.py; do not edit.",
             r"\begin{tabular}{l ccc cc}", r"\toprule",
             r" & \multicolumn{3}{c}{Cores on $X_{\rm int}$} & "
             r"\multicolumn{2}{c}{BiRNe unstable cores}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"Network & LP & Autogatito & BiRNe & autocatalytic & non-autocatalytic\\",
             r"\midrule"]
    for name, label in labels.items():
        r = records[name]
        b = r.get("birne", {})
        lines.append(" & ".join([
            label, str(len(r["lp_cores"])),
            str(len(r["autogatito"]["cores"])) if "autogatito" in r else "--",
            str(len(r["birne_internal_autocatalytic"])) if "birne_internal_autocatalytic" in r else "--",
            str(b.get("autocatalytic_count", "--")),
            str(b.get("nonautocatalytic_count", "--"))])+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    paper = HERE.parents[1]/"paper"/"autocatalytic"
    (paper/"backend_table.tex").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
