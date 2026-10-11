#!/usr/bin/env python
"""Bounded published-checkpoint search for the qualitative oxygen-origin case.

The model does not receive or predict isotope abundances. A supplied initial
oxygen anchor is tracked through its mapped outputs and scored qualitatively.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter

import networkx as nx
import numpy as np
from rdkit import Chem
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.flower_checkpoint_audit import COMMIT, _digest  # noqa: E402
from Experiment.MTG.flower_history_audit import _audit_graph  # noqa: E402
from Experiment.MTG.isotope_case import candidates, OBSERVATION  # noqa: E402


def run(args):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.flower, text=True
    ).strip()
    assert commit == COMMIT
    initial = candidates()[0]["rsmi"].split(">>")[0]
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    molecule = Chem.AddHs(Chem.MolFromSmiles(initial, parser))
    original_to_model = {
        a.GetAtomMapNum(): a.GetIdx() + 1
        for a in molecule.GetAtoms()
        if a.GetAtomMapNum()
    }
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    root = Chem.MolToSmiles(molecule, isomericSmiles=False, allHsExplicit=True)
    protocol = {
        "schema": "synkit.mtg-flower-isotope-beam-protocol/1",
        "initial": initial,
        "model_input": root,
        "initial_labelled_oxygen_anchor": original_to_model[2],
        "initial_carbonyl_carbon_anchor": original_to_model[1],
        "isotope_handling": "Initial label transported through supplied maps, external to isotope-blind model input.",
        "seed": 0,
        "samples_per_expansion": 16,
        "beam_size": 3,
        "nbest": 3,
        "max_depth": 6,
        "model_commit": commit,
        "checkpoint_sha256": _digest(args.checkpoint),
        "observation_source": OBSERVATION["source"],
        "limits": (
            "CPU calibration with reduced beam and depth; no population or "
            "negative-chemistry claim from truncation."
        ),
    }
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    if args.protocol.exists():
        assert json.loads(args.protocol.read_text()) == protocol
    else:
        args.protocol.write_text(json.dumps(protocol, indent=2) + "\n")
    sys.path.insert(0, str(args.flower))
    import beam_predict
    from train import init_model

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_num_threads(4)
    loaded = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    config = loaded["args"]
    config.local_rank, config.device = -1, torch.device("cpu")
    config.load_from, config.model_path = None, "/tmp/mtg-flower-model-work"
    config.num_workers, config.test_batch_size = 0, 4096
    for key, value in (
        ("sample_size", 16),
        ("beam_size", 3),
        ("nbest", 3),
        ("max_depth", 6),
    ):
        setattr(config, key, value)
    model, flow, _ = init_model(config)
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in loaded["state_dict"].items()}
    )
    model.eval()
    graph = nx.DiGraph()
    graph.add_node(root, depth=1)
    started = perf_counter()
    with torch.no_grad():
        beam_predict.beam_search(
            config, model, flow, {0: [root]}, [(graph, root, (initial, []))]
        )
    oxygen, carbon = original_to_model[2], original_to_model[1]
    roles = []
    target = Chem.MolToSmiles(Chem.MolFromSmiles("O=C(Oc1ccccc1)c1ccccc1"))
    for index, smiles in enumerate(graph):
        mol = Chem.MolFromSmiles(smiles, parser)
        atoms = {a.GetAtomMapNum(): a for a in mol.GetAtoms()}
        label = atoms[oxygen]
        bonds = {
            a.GetAtomMapNum(): mol.GetBondBetweenAtoms(
                label.GetIdx(), a.GetIdx()
            ).GetBondTypeAsDouble()
            for a in label.GetNeighbors()
        }
        role = "other"
        if bonds.get(carbon) == 2:
            role = "carbonyl"
        elif (
            len(bonds) == 2
            and all(value == 1 for value in bonds.values())
            and all(atoms[n].GetSymbol() == "C" for n in bonds)
        ):
            role = "ether"
        projected = Chem.RemoveHs(Chem.Mol(mol))
        for atom in projected.GetAtoms():
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
        product_present = target in Chem.MolToSmiles(
            projected, isomericSmiles=False
        ).split(".")
        roles.append(
            {
                "node": index,
                "role": role,
                "terminal_self_loop": graph.has_edge(smiles, smiles),
                "phenyl_benzoate_present": product_present,
                "qualitative_observation_compatible": (
                    (role == "carbonyl") if product_present else None
                ),
                "mapped_smiles": smiles,
            }
        )
    report = {
        "schema": "synkit.mtg-flower-isotope-beam/1",
        "protocol": protocol,
        "protocol_sha256": _digest(args.protocol),
        "script_sha256": _digest(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "threads": 4,
            "packages": {
                key: metadata.version(key)
                for key in (
                    "torch",
                    "rdkit",
                    "numpy",
                    "networkx",
                    "torchdiffeq",
                    "iteround",
                )
            },
        },
        "elapsed_seconds": perf_counter() - started,
        "audit": _audit_graph(graph, root, (initial, []), set()),
        "anchored_oxygen_roles": roles,
        "product_nodes": sum(row["phenyl_benzoate_present"] for row in roles),
        "edges": [
            {
                "source": list(graph).index(a),
                "target": list(graph).index(b),
                "rank": int(data["rank"]),
                "count": int(data["count"]),
            }
            for a, b, data in graph.edges(data=True)
        ],
        "claim_limits": (
            "Initial oxygen anchor is supplied experimental intervention information. Model "
            "scores are not isotope kinetics. A bounded search failing to reach product "
            "cannot refute the independent observation."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["audit"]["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flower", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
