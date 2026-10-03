"""Diagnose ACS comparator differences using explicitly matched frozen inventories.

Full upstream descriptors are transferred losslessly into the generic SynKit
configured-graph canonizer. The separate planar-inventory ablation does not
claim to preserve the original represented molecular information.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import networkx as nx
from rdkit import Chem, RDLogger

from Experiment.Stereo.Canonicalization.global_local import _case_time_limit
from Experiment.Stereo.Chirality.published import EXPECTED_SHA256, load_dataset
from Experiment.Stereo.provenance import capture_source, finish_source
from synkit.Chem.Molecule.chirality import classify_molecular_chirality
from synkit.Graph.Stereo import canonicalize_configured_stereograph, mirror_stereo_registry
from synkit.Graph.Stereo.interchange import stereomolgraph_registry_to_synkit

IDS = ("VS246", "VS247", "VS248", "VS299")


def _upstream_hashes(checkout):
    return {str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((checkout / "src/stereomolgraph").rglob("*.py"))}


def _mirror_equal(graph, registry):
    options = {"atom_color": "color", "bond_color": (), "enumerate_automorphism_group": False}
    original = canonicalize_configured_stereograph(graph, registry.values(), **options)
    reflected = canonicalize_configured_stereograph(graph, mirror_stereo_registry(registry).values(), **options)
    if not all(result.complete and result.exact for result in (original, reflected)):
        raise ValueError("Incomplete matched-inventory canonicalization.")
    return original.canonical_code == reflected.canonical_code


def diagnose(row, converter, matcher, timeout_seconds):
    source_molecule = Chem.MolFromSmiles(row["Input SMILES"])
    molecule = Chem.AddHs(source_molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    upstream = converter(stereo_complete=True, use_atom_map_number=True)(molecule)
    graph = nx.Graph()
    graph.add_nodes_from((atom.GetAtomMapNum(), {"color": atom.GetSymbol()}) for atom in molecule.GetAtoms())
    graph.add_edges_from((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())
                         for bond in molecule.GetBonds())
    with _case_time_limit(timeout_seconds):
        bridge = stereomolgraph_registry_to_synkit(upstream, source_molecule=molecule, projection="all")
        if not bridge.lossless or bridge.exclusions:
            raise ValueError("Full inventory transfer must be lossless without exclusions.")
        original = upstream == upstream.enantiomer()
        unrefined = next(matcher(upstream, upstream.enantiomer(), stereo=True), None) is not None
        synkit = _mirror_equal(graph, bridge.descriptors)
        ablated = upstream.copy()
        for bond in tuple(ablated.bond_stereo):
            ablated.delete_bond_stereo(bond)
        retained = {key: value for key, value in bridge.descriptors.items() if value.descriptor_class != "planar_bond"}
        ablated_upstream = ablated == ablated.enantiomer()
        ablated_synkit = _mirror_equal(graph, retained)
    specialized = classify_molecular_chirality(source_molecule)
    agrees = original == unrefined == synkit and ablated_upstream == ablated_synkit
    return {
        "id": row["ID"], "manual_label": row["manual"], "status": "complete",
        "specialized_synkit_label": specialized.classification.value,
        "specialized_synkit_descriptor_count": specialized.descriptor_count,
        "full_upstream_atom_descriptors": len(upstream.atom_stereo),
        "full_upstream_bond_descriptors": len(upstream.bond_stereo),
        "full_inventory_transfer_lossless": bridge.lossless,
        "upstream_full_inventory_mirror_equal": original,
        "upstream_unrefined_mapping_mirror_equal": unrefined,
        "synkit_full_inventory_mirror_equal": synkit,
        "upstream_without_planar_inventory_mirror_equal": ablated_upstream,
        "synkit_without_planar_inventory_mirror_equal": ablated_synkit,
        "matched_inventory_algorithm_agreement": agrees,
        "inventory_ablation_changes_verdict": original != ablated_upstream,
        "generic_descriptor_wire_fields": [
            {"family": value.descriptor_class, "atoms": value.atoms, "parity": value.parity}
            for value in bridge.descriptors.values()
        ],
    }


def benchmark(repository, timeout_seconds=30):
    source = capture_source()
    checkout = repository.resolve()
    sys.path.insert(0, str(checkout / "src"))
    from stereomolgraph.rdmol2graph import RDMol2StereoMolGraph
    from stereomolgraph.algorithms.isomorphism import vf2pp_all_isomorphisms

    upstream_hashes = _upstream_hashes(checkout)
    records = []
    for row in load_dataset():
        if row["ID"] not in IDS:
            continue
        try:
            records.append(diagnose(row, RDMol2StereoMolGraph, vf2pp_all_isomorphisms, timeout_seconds))
        except (TimeoutError, ValueError) as error:
            records.append({"id": row["ID"], "status": "incomplete", "reason": str(error)})
    complete = len(records) == len(IDS) and all(record["status"] == "complete" for record in records)
    passed = complete and all(record["matched_inventory_algorithm_agreement"] for record in records)
    if upstream_hashes != _upstream_hashes(checkout):
        raise RuntimeError("Comparator source changed during the diagnostic.")
    return {
        "schema": "synkit.comparator-inventory-diagnostic/1", "dataset_sha256": EXPECTED_SHA256,
        "selection": "four retrospectively identified ACS mismatches; development diagnosis, not holdout",
        "input_contract": "explicit H; element/connectivity; full upstream configured inventory; typed LP resolution",
        "timeout_seconds_per_record": timeout_seconds, "provenance": finish_source(source),
        "upstream": {"checkout": str(checkout), "source_sha256": upstream_hashes,
                     "source_unchanged_during_run": True,
                     "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()},
        "records": records, "complete": complete, "passed": passed,
        "claim_boundary": ("Aligned inventories isolate the comparison algorithms. Planar-inventory ablation "
                           "changes represented information; it does not prove which molecular convention is correct."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereomolgraph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    result = benchmark(args.stereomolgraph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "complete": result["complete"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
