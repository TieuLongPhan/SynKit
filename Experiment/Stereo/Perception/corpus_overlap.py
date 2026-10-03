"""Count development-corpus overlap without equating rows with independent molecules."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from Experiment.Stereo.Chirality.published import DATASET, load_dataset
from Experiment.Stereo.datasets import ROTA, load_cip, load_rota
from Experiment.Stereo.provenance import capture_source, finish_source, sha256


def structure_keys(smiles):
    """Keep isotopes and charges when erasing configuration and atom-map identifiers."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("SMILES parse failure")
    molecule = Chem.RemoveHs(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    configured = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    Chem.RemoveStereochemistry(molecule)
    constitution = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    scaffold_key = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)
    return configured, constitution, scaffold_key or None


def audit(cip_path):
    """Return task-specific row, representational, constitution, and scaffold counts."""
    source = capture_source()
    inputs = {
        "acs": (DATASET, [(row["ID"], row["Input SMILES"]) for row in load_dataset()]),
        "cip": (cip_path, [(row["ID"], row["SMILES"]) for row in load_cip(cip_path)]),
        "rota": (ROTA, [(f"RotA-{index:04d}", row["SMILES"]) for index, row in enumerate(load_rota(), 1)]),
    }
    corpora, indices = {}, {}
    for name, (path, rows) in inputs.items():
        configured, constitutions, scaffolds = defaultdict(list), defaultdict(list), defaultdict(list)
        failures = []
        for identifier, smiles in rows:
            try:
                configured_key, constitution, scaffold = structure_keys(smiles)
            except ValueError as error:
                failures.append({"id": identifier, "reason": str(error)})
                continue
            configured[configured_key].append(identifier)
            constitutions[constitution].append(identifier)
            if scaffold:
                scaffolds[scaffold].append(identifier)
        indices[name] = constitutions
        corpora[name] = {
            "input_sha256": sha256(path), "rows": len(rows),
            "parsed_rows": len(rows) - len(failures), "parse_failures": failures,
            "unique_rdkit_configured_representations": len(configured),
            "unique_stereo_neutral_constitutions": len(constitutions),
            "unique_nonempty_murcko_scaffolds": len(scaffolds),
            "acyclic_rows_without_murcko_scaffold": sum(
                len(ids) for key, ids in constitutions.items()
                if MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(key)).GetNumAtoms() == 0
            ),
            "duplicate_constitution_groups": [
                {"constitution_sha256_index": hashlib.sha256(key.encode()).hexdigest(), "ids": ids}
                for key, ids in sorted(constitutions.items()) if len(ids) > 1
            ],
        }
    pairwise = {}
    names = sorted(corpora)
    for position, first in enumerate(names):
        for second in names[position + 1:]:
            overlap = sorted(indices[first].keys() & indices[second].keys())
            pairwise[f"{first}:{second}"] = {
                "shared_constitutions": len(overlap),
                "rows_in_first": sum(len(indices[first][key]) for key in overlap),
                "rows_in_second": sum(len(indices[second][key]) for key in overlap),
                "groups": [
                    {"constitution_sha256_index": hashlib.sha256(key.encode()).hexdigest(),
                     first: indices[first][key], second: indices[second][key]} for key in overlap
                ],
            }
    all_keys = set().union(*(index.keys() for index in indices.values()))
    return {
        "schema": "synkit.stereo-corpus-overlap/1", "provenance": finish_source(source),
        "identity_contract": {
            "representation": (
                "RDKit canonical SMILES after standard parsing, RemoveHs and atom-map erasure"
            ),
            "constitution": "stereo erased; isotopes, charges, hydrogen counts and aromaticity retained",
            "resonance_normalization": False, "charge_neutralization": False,
            "equality": "full canonical strings compared in memory; output digests are lookup indices only",
            "structure_redistribution": False,
            "configured_count_boundary": (
                "parser-visible stereo only; not a complete configured-stereograph identity count"
            ),
            "scaffold": "RDKit Murcko scaffold; empty acyclic scaffolds excluded from unique scaffold counts",
        },
        "corpora": corpora, "pairwise_overlap": pairwise,
        "total_rows": sum(item["rows"] for item in corpora.values()),
        "unique_union_constitutions": len(all_keys),
        "assistance_contracts": {
            "retained_perception": "focal configuration neutral; neighboring configurations supplied",
            "rota_recovery": "family conditioned by source category; not a family-blind evaluation",
            "cip_smiles_only": (
                "family categories used for scoring strata; target labels used only for scoring"
            ),
            "cip_coordinate": (
                "source family categories select coordinate projections; target labels used only for scoring"
            ),
        },
        "generalization_boundary": (
            "All three corpora are development evidence; post hoc row splits do not create an untouched holdout."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cip-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    report = audit(args.cip_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": report["total_rows"],
                      "unique_constitutions": report["unique_union_constitutions"],
                      "overlap": {key: value["shared_constitutions"]
                                  for key, value in report["pairwise_overlap"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
