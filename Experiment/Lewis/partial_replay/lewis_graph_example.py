"""Run full/partial forward/backward replay directly on frozen Lewis graphs.

python -m Experiment.Lewis.partial_replay.lewis_graph_example --backend python
python -m Experiment.Lewis.partial_replay.lewis_graph_example --backend cpp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem

from synkit.Graph.Morphism import LewisLabelledGraph, llg_from_its
from synkit.IO import rsmi_to_its
from synkit.Rule import SynRule
from synkit.Synthesis.Reactor import WildcardEngine, build_native_reactor


REACTION = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"
ROOT = Path(__file__).resolve().parents[3]


def identity(reaction):
    endpoints = []
    for side in reaction.split(">>"):
        molecule = Chem.MolFromSmiles(side)
        if molecule is None:
            raise ValueError(f"Invalid example endpoint: {side}")
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        endpoints.append(Chem.MolToSmiles(molecule))
    return tuple(endpoints)


def run(backend="python"):
    graph = rsmi_to_its(REACTION, format="tuple")
    left = llg_from_its(graph, "left", electron_complete=True)
    right = llg_from_its(graph, "right", electron_complete=True)
    partial_left = LewisLabelledGraph.from_electron_graph(
        left.to_networkx().subgraph((1, 2)).copy()
    )
    partial_right = LewisLabelledGraph.from_electron_graph(
        right.to_networkx().subgraph((1, 3)).copy()
    )
    absolute = SynRule(graph, implicit_h=False)
    options = {"backend": backend, "electron_diagnostics": True}
    if backend == "cpp":
        options["library_path"] = str(build_native_reactor(ROOT / "build/partial/native"))
    expected = identity(REACTION)
    records = []
    for name, host, invert, partial in (
        ("full_forward", left, False, False),
        ("full_backward", right, True, False),
        ("partial_forward", partial_left, False, True),
        ("partial_backward", partial_right, True, True),
    ):
        engine = WildcardEngine.from_lewis_graph(
            host, absolute, invert=invert, partial=partial, **options
        )
        outputs = engine.smarts_list
        # Serialization retains the original reaction orientation even when
        # execution is backward; the recovered reactants are the left endpoint.
        recovered = expected in {identity(output) for output in outputs}
        mismatches = sum(report["mismatch_count"] for report in engine.diagnostics)
        if not recovered or mismatches:
            raise AssertionError(f"Lewis graph example failed: {name}")
        records.append({"case": name, "outputs": outputs,
                        "recovered_endpoint": "reactants" if invert else "products",
                        "exact_recovery": recovered, "electron_mismatches": mismatches})
    # Erasing common H deliberately represents the generic O-R boundary family.
    symbolic = WildcardEngine.from_lewis_graph(
        partial_left, SynRule(graph, implicit_h=True), **options
    )
    if not symbolic.its_list or not all(
        graph.graph["partial_completion"]["attachment_ports"] for graph in symbolic.its_list
    ):
        raise AssertionError("Expected a persistent symbolic boundary.")
    return {"backend": backend, "mapped_reaction": REACTION,
            "recovery_identity": "Exact chemical endpoints with atom-map annotations removed; "
                                 "unobserved atoms receive fresh map numbers.",
            "absolute_hydrogen_rule": records,
            "normalized_hydrogen_rule_symbolic_family": symbolic.smarts_list}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    args = parser.parse_args()
    print(json.dumps(run(args.backend), indent=2))


if __name__ == "__main__":
    main()
