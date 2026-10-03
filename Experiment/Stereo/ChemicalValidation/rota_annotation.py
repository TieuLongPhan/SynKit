"""Independent structural adjudication of the prospectively frozen RotA sample.

This is an automated constitutional review of supplied/projected loci. It
does not certify axial geometry, a rotational barrier, nitrogen inversion
time or experimental persistence. All neighboring orientations are neutral.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import time

import networkx as nx
from rdkit import Chem

from Experiment.Stereo.Canonicalization.independent_model import State, formal_states
from Experiment.Stereo.Canonicalization.independent_orbits import reference_partition
from Experiment.Stereo.Canonicalization.independent_transport import _Queries
from Experiment.Stereo.ChemicalValidation.common import ROOT, provenance, replay_typed_map, sha256, wall_limit
from Experiment.Stereo.ChemicalValidation.coordination import ChemicalInputUnresolved, source_molecule
from Experiment.Stereo.ChemicalValidation.negative import negative_graph

MANIFEST = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/rota_adjudication_selection.json"
PROTOCOL = {
    'axis_topology_boundary': (
        'Preserve every supplied axis support for mathematical configuration tests. Record bond '
        'type and hybridization separately; aromatic ring edges are not evidence of ordinary '
        'single-bond atropisomerism.'
    ),
    'scope': (
        'All sixty frozen loci, including eight non-axis category projections; no status-based '
        'reselection.'
    ),
    'input_assistance': (
        'Frozen axis support/port inventory for 52 loci; source categories identify four atom '
        'pairs and four spiro centers.'
    ),
    'orientation_boundary': (
        'Constitution alone; focal and neighboring configurations, atom maps and CIP labels do '
        'not supply truth.'
    ),
    'axis_truth': (
        'Original typed maps fix both endpoints and exchange a declared terminal reference pair; '
        'witnesses or exhausted bounded queries decide terminal ligand equivalence.'
    ),
    'configured_truth': (
        'Independent complete local choices and exact global partitions determine configured '
        'pose distinction; separate from terminal ligand equivalence.'
    ),
    'atom_pair_truth': (
        'Independently verify adjacency and four-port C/N tetrahedral inventories; classify each '
        'focal center and the joint four-assignment space.'
    ),
    'spiro_truth': (
        'Two actual cycles intersect only at the four-port carbon center; independently classify '
        'its tetrahedral configurations.'
    ),
    'nitrogen_lp_boundary': (
        'Only neutral nonaromatic N with three total valence units, zero radicals and one hidden '
        'H in the selected atom-pair cases receives one Lewis LP port.'
    ),
    'physical_truth': (
        'No barrier, coordinate orientation or persistence annotation is inferred; unknown '
        'without separately sourced evidence.'
    ),
    'review_boundary': (
        'Automated independent mathematical reference, not independent human chemical review.'
    ),
    'reference_budget': {'seconds': 10, 'max_queries_or_maps': 50000, 'max_rss_mib': 1024},
}


def _tetrahedron(molecule, center, graph):
    atom = molecule.GetAtomWithIdx(center)
    refs = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
    hydrogen = atom.GetTotalNumHs()
    if atom.GetHybridization() != Chem.HybridizationType.SP3 or atom.GetNumRadicalElectrons():
        raise ChemicalInputUnresolved("projected_atom_not_supported_tetrahedral_topology")
    if atom.GetSymbol() == "C" and atom.GetFormalCharge() == 0:
        if len(refs) + hydrogen != 4 or hydrogen not in (0, 1):
            raise ChemicalInputUnresolved("carbon_tetrahedral_ports_not_certified")
        if hydrogen:
            refs.append(f"@H:{center}")
    elif (atom.GetSymbol() == "N" and atom.GetFormalCharge() == 0 and not atom.GetIsAromatic() and
          atom.GetTotalValence() == 3 and hydrogen == 1 and len(refs) == 2):
        refs.extend((f"@H:{center}", f"@LP:{center}"))
        graph.nodes[center]["lone_pairs"] = 1
    else:
        raise ChemicalInputUnresolved("projected_tetrahedral_electronic_ports_outside_declared_boundary")
    graph.nodes[center]["hcount"] = hydrogen
    return State("tetrahedral", (center, *refs), 1)


def _axis_inventory(record, molecule, graph, loci):
    if len(record["input_supports"]) != 1 or record["input_supports"][0]["family"] != "atrop_axis":
        raise ChemicalInputUnresolved("frozen_rota_support_not_one_axis")
    support = record["input_supports"][0]["support"]
    path = tuple(support["path"])
    left, right = map(tuple, support["terminal_frames"])
    if len(path) != 2 or tuple(sorted(path)) != tuple(sorted(loci)) or not graph.has_edge(*path):
        raise ChemicalInputUnresolved("frozen_rota_axis_path_mismatch")
    if len(left) != 2 or len(right) != 2 or len(set((*left, *path, *right))) != 6:
        raise ChemicalInputUnresolved("frozen_rota_axis_ports_not_injective")
    for center, other, frame in ((path[0], path[1], left), (path[1], path[0], right)):
        if set(graph[center]) - {other} != set(frame):
            raise ChemicalInputUnresolved("frozen_rota_terminal_inventory_not_equal_neighbors")
    seed = State("atrop_bond", (*left, *path, *right), 1)
    facts = {"axis_path": path, "terminal_frames": (left, right),
             "terminal_hybridizations": [str(molecule.GetAtomWithIdx(index).GetHybridization()) for index in path],
             "terminal_hidden_hydrogens": [molecule.GetAtomWithIdx(index).GetTotalNumHs() for index in path],
             "source_bond_type": graph.edges[path]["color"][0],
             "ordinary_single_bond_topology": graph.edges[path]["color"][0] == "SINGLE",
             "geometric_axis_certified": False}
    return graph, (seed,), "axis", facts


def source_inventory(record):
    """Validate frozen support against constitution without a perception call."""
    molecule = source_molecule(record["smiles"])
    if molecule.GetNumAtoms() != record["atom_count"]:
        raise ChemicalInputUnresolved("frozen_rota_atom_count_mismatch")
    graph = negative_graph(molecule)
    loci = tuple(map(int, record["locus"].split("-")))
    if any(index < 0 or index >= len(graph) for index in loci):
        raise ChemicalInputUnresolved("frozen_rota_locus_index_out_of_range")
    if record["input_supports"]:
        return _axis_inventory(record, molecule, graph, loci)
    if record["source_category"] == "Chiral atom pair" and len(loci) == 2:
        if not graph.has_edge(*loci):
            raise ChemicalInputUnresolved("projected_atom_pair_not_adjacent")
        seeds = tuple(_tetrahedron(molecule, center, graph) for center in loci)
        return graph, seeds, "tetrahedral_pair", {"centers": loci, "pair_adjacency_verified": True}
    if record["source_category"] == "Spiral atom and chain" and len(loci) == 1:
        center = loci[0]
        seed = _tetrahedron(molecule, center, graph)
        cycles = [cycle for cycle in nx.cycle_basis(graph) if center in cycle]
        witnesses = [pair for pair in combinations(cycles, 2) if set(pair[0]) & set(pair[1]) == {center}]
        if not witnesses:
            raise ChemicalInputUnresolved("spiro_cycle_intersection_not_certified")
        return graph, (seed,), "tetrahedral_spiro", {"center": center, "spiro_cycle_witness": witnesses[0]}
    raise ChemicalInputUnresolved("frozen_projected_family_not_independently_supported")


def terminal_exchange_truth(graph, seed):
    """Exact original constitutional maps, including all coupled branch moves."""
    left, right = seed.atoms[2:4]
    frames = (seed.atoms[:2], seed.atoms[4:])
    queries = _Queries(graph, "color", "color", time.monotonic() + 10, 1024, 50000)
    witnesses = []
    with wall_limit(10):
        for first, second in frames:
            mapping = queries.query({left: left, right: right, first: second})
            if mapping is not None:
                replay_typed_map(graph, graph, mapping)
            witnesses.append(None if mapping is None else sorted(mapping.items()))
    return {"carrier_status": "symmetry_related" if any(witness is not None for witness in witnesses) else "confirmed",
            "terminal_exchange_witnesses": witnesses,
            "negative_queries_exhausted": [witness is None for witness in witnesses],
            "scope": "Axis-fixed terminal ligand equivalence, separate from global configured pose distinction."}


def _reference(graph, seeds):
    choices = tuple(formal_states(seed) for seed in seeds)
    options = {"method": "direct_single_carrier_transport", "max_queries": 50000} if len(seeds) == 1 else {
        "max_automorphisms": 50000,
    }
    result = reference_partition(graph, choices, atom_color="color", bond_color="color",
                                 timeout_seconds=10, max_rss_mb=1024, **options)
    if result["status"] != "complete":
        raise ChemicalInputUnresolved("independent_rota_reference_incomplete:" + result["reason"])
    return result


def annotate(record):
    base = {"source_id": record["source_id"], "source_row_zero_based": record["source_row_zero_based"],
            "locus": record["locus"], "source_category": record["source_category"], "stratum": record["stratum"],
            "algorithm_confirmed_rejected_status_read": False,
            "experimental_persistence": "unknown_no_separate_barrier_or_time_evidence"}
    try:
        graph, seeds, family, facts = source_inventory(record)
        reference = _reference(graph, seeds)
        if family == "axis":
            carrier = terminal_exchange_truth(graph, seeds[0])
        else:
            local = [_reference(graph, (seed,)) for seed in seeds]
            carrier = {"center_statuses": ["confirmed" if len(result["partition"]) == 2 else "symmetry_related"
                                           for result in local], "focal_center_references": local,
                       "scope": "Focal constitutional tetrahedral classes, with neighboring orientation neutral."}
        return {**base, "status": "complete", "independent_family": family, "input_facts": facts,
                "configured_seeds": [asdict(seed) for seed in seeds], "reference": reference,
                "configured_class_count": len(reference["partition"]), "constitutional_carrier": carrier}
    except (ChemicalInputUnresolved, TimeoutError) as error:
        return {**base, "status": "unresolved", "reason": str(error)}


def build_report():
    start = provenance()
    source = json.loads(MANIFEST.read_text())
    records = [annotate(record) for record in source["records"]]
    unchanged = start["source_sha256"] == provenance()["source_sha256"]
    return {"schema": "synkit.frozen-rota-structural-adjudication/1",
            "frozen_utc": datetime.now(timezone.utc).isoformat(), "provenance": start,
            "selection_manifest": str(MANIFEST.relative_to(ROOT)), "selection_manifest_sha256": sha256(MANIFEST),
            "protocol": PROTOCOL, "production_evaluated": False, "records": records,
            "outcomes": dict(Counter(record["status"] for record in records)),
            "family_counts": dict(Counter(record.get("independent_family", "unresolved") for record in records)),
            "sources_unchanged_during_annotation": unchanged,
            "annotation_valid": unchanged and len(records) == 60,
            "review_limit": ("Automated structural adjudication; source corpus is development evidence, "
                             "not a new holdout.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"annotation_valid": report["annotation_valid"], "outcomes": report["outcomes"],
                      "families": report["family_counts"]}))
    return 0 if report["annotation_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
