"""Probe release-specific non-tetrahedral canonical SMILES representation.

This is a bounded collection of formal representation checks. It does not
establish experimental configurations or a theorem about RDKit's canonizer.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time

from rdkit import Chem, rdBase


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(smiles):
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, parameters)
    if molecule is None:
        raise ValueError("RDKit refused a declared probe fixture: " + smiles)
    return molecule


def serialize(molecule):
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def run_probe(fixtures_path):
    started = time.perf_counter()
    source_path = Path(__file__)
    original_flag = Chem.GetAllowNontetrahedralChirality()
    inputs_before = checksum(fixtures_path)
    source_before = checksum(source_path)
    fixtures = json.loads(Path(fixtures_path).read_text())
    if fixtures["schema"] != "synkit.rdkit-representation-fixtures/1":
        raise ValueError("Unknown representation fixture schema.")
    groups = []
    repository = next((parent for parent in source_path.resolve().parents if (parent / ".git").exists()), None)
    git_evidence = {}
    if repository is not None:
        git_evidence = {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
            ).strip(),
            "working_tree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repository,
            )),
        }
    Chem.SetAllowNontetrahedralChirality(True)
    try:
        for group in fixtures["groups"]:
            count = group["state_count"]
            prefix = group["tag_prefix"]
            examples = group["equivalent_documented_encodings"]
            if count <= 0 or len(examples) != count or len(set(examples)) != count:
                raise ValueError("Documented state coverage omits or duplicates a fixture.")
            equivalent_outputs = [serialize(parse(value)) for value in examples]
            records = []
            for state in range(1, count + 1):
                smiles = group["ordered_ligand_template"].format(state)
                molecule = parse(smiles)
                canonical = serialize(molecule)
                atom_count = molecule.GetNumAtoms()
                outputs = set()
                discrepancies = []
                permutations = 0
                for order in itertools.permutations(range(atom_count)):
                    observed = serialize(Chem.RenumberAtoms(molecule, order))
                    outputs.add(observed)
                    permutations += 1
                    if observed != canonical and len(discrepancies) < 3:
                        discrepancies.append({"atom_order": list(order), "canonical_smiles": observed})
                stripped = Chem.Mol(molecule)
                Chem.RemoveStereochemistry(stripped)
                records.append({
                    "state": state, "input_smiles": smiles,
                    "canonical_smiles": canonical, "atom_count": atom_count,
                    "atom_permutations_checked": permutations,
                    "distinct_outputs_after_renumbering": sorted(outputs),
                    "numbering_invariant": outputs == {canonical},
                    "round_trip_preserved": serialize(parse(canonical)) == canonical,
                    "configured_tag_written": "@" + prefix in canonical,
                    "stereo_removed_smiles": serialize(stripped),
                    "discrepancy_examples": discrepancies,
                })
            state_outputs = {record["canonical_smiles"] for record in records}
            stripped_outputs = {record["stereo_removed_smiles"] for record in records}
            groups.append({
                "family": group["family"], "states": count, "records": records,
                "documented_encoding_outputs": equivalent_outputs,
                "documented_equivalent_encodings_agree": len(set(equivalent_outputs)) == 1,
                "distinct_configurations_retained": len(state_outputs) == count,
                "removing_stereo_collapses_configurations": len(stripped_outputs) == 1,
                "passed": len(set(equivalent_outputs)) == 1 and len(state_outputs) == count
                and len(stripped_outputs) == 1
                and all(record["numbering_invariant"] and record["round_trip_preserved"]
                        and record["configured_tag_written"] for record in records),
            })
    finally:
        Chem.SetAllowNontetrahedralChirality(original_flag)
    if checksum(fixtures_path) != inputs_before or checksum(source_path) != source_before:
        raise RuntimeError("Probe source or fixtures changed during execution.")
    return {
        "schema": "synkit.rdkit-representation-probe/1",
        "provenance": {**git_evidence, "python": platform.python_version(), "rdkit": rdBase.rdkitVersion,
                       "probe_source_sha256": source_before, "input_sha256": inputs_before,
                       "source_unchanged_during_run": True, "inputs_unchanged_during_run": True},
        "primary_reference": fixtures["source"],
        "protocol": {"canonical": True, "isomericSmiles": True, "removeHs": False,
                     "sanitize": True, "allow_nontetrahedral_chirality": True,
                     "atom_orders": "all permutations for every numbered distinct-ligand state",
                     "naming_assessed": False, "coordinate_interpretation_assessed": False,
                     "chemical_truth_assessed": False},
        "groups": groups,
        "summary": {"families": len(groups), "states": sum(group["states"] for group in groups),
                    "atom_permutations_checked": sum(record["atom_permutations_checked"]
                                                     for group in groups for record in group["records"]),
                    "passed_families": sum(group["passed"] for group in groups)},
        "passed": bool(groups) and all(group["passed"] for group in groups),
        "scope": fixtures["scope"], "wall_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_probe(args.fixtures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], **report["summary"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
