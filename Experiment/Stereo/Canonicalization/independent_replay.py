"""Replay archived local-orbit witnesses against original typed constitutions.

Input adapters rebuild the runner's frozen supports, outside reference truth.
This audit verifies positive maps and stored partition/mirror consistency; it
does not replace the independently tested exhaustion of negative queries.
"""

import argparse
import json
from pathlib import Path

from rdkit import Chem, RDLogger

from Experiment.Stereo.Canonicalization import configuration_free_local as local
from Experiment.Stereo.Canonicalization.independent_model import from_descriptor, inventory_key, state_key, transport
from Experiment.Stereo.Canonicalization.independent_orbits import (
    ATOM_KEYS, BOND_KEYS, _mirror_classes, _mirror_table, _selector,
    _states_and_lookup, apply_table, color_signature,
)
from Experiment.Stereo.provenance import ROOT, capture_source, finish_source, sha256


def _inputs(task, record_id, rows):
    if task == "internal":
        fixture = next(value for value in local._fixture_catalog() if value.family == record_id)
        return fixture.graph, (fixture.seed,), "color", "color"
    molecule = Chem.MolFromSmiles(rows[record_id])
    if molecule is None:
        raise ValueError(f"Cannot reconstruct input:{record_id}")
    _, supplied = local._rdkit_graph_and_registry(molecule)
    neutral = local._stereo_neutral(molecule)
    graph, _ = local._rdkit_graph_and_registry(neutral)
    identifiers = local._rdkit_identifier_map(neutral)
    unique = {local.descriptor_id(seed): seed for descriptor in supplied.values()
              for seed in (local._fixed_descriptor_seed(descriptor),)}
    for element in local.detect_potential_stereo_elements(neutral):
        try:
            seed = local._seed_from_potential(element, neutral, identifiers)
        except (TypeError, ValueError):
            continue
        unique.setdefault(local.descriptor_id(seed), seed)
    return graph, tuple(unique[key] for key in sorted(unique)), ATOM_KEYS, BOND_KEYS


def _validate_map(graph, mapping, node_tokens, edge_tokens):
    if set(mapping) != set(graph) or set(mapping.values()) != set(graph):
        raise ValueError("Witness is not a full vertex bijection.")
    if any(node_tokens[node] != node_tokens[image] for node, image in mapping.items()):
        raise ValueError("Witness changes an original typed node color.")
    if any(edge_tokens.get(frozenset((mapping[a], mapping[b]))) != edge_tokens[frozenset((a, b))]
           for a, b in graph.edges):
        raise ValueError("Witness changes an original edge or typed edge color.")


def replay_carrier(graph, seed, carrier, atom_color, bond_color):
    """Check every positive map using original attributes and transported states."""
    grouped = {}
    for descriptor in local.all_local_arrangements(seed):
        value = from_descriptor(descriptor)
        grouped.setdefault(repr(state_key(value)), value)
    choices = (tuple(grouped[key] for key in sorted(grouped)),)
    states, lookup, inventory = _states_and_lookup(choices)
    reference = carrier["independent_reference"]
    if reference["status"] != "complete":
        raise ValueError(f"Incomplete reference:{carrier['carrier_id']}")
    assignments = tuple((index,) for index in range(len(states[0])))
    if tuple(tuple(value) for value in reference["assignments"]) != assignments:
        raise ValueError("Archived formal assignment inventory differs from original inputs.")
    partition = tuple(tuple(tuple(member) for member in group) for group in reference["partition"])
    members = [member for group in partition for member in group]
    if len(members) != len(assignments) or set(members) != set(assignments):
        raise ValueError("Archived partition is not a disjoint complete assignment partition.")
    class_of = {member: index for index, group in enumerate(partition) for member in group}
    node_color, edge_color = _selector(atom_color), _selector(bond_color)
    node_tokens = {node: color_signature(node_color(data)) for node, data in graph.nodes(data=True)}
    edge_tokens = {frozenset((a, b)): color_signature(edge_color(data)) for a, b, data in graph.edges(data=True)}
    targets = []
    for witness in reference["merge_witnesses"]:
        source, target = tuple(witness["source_assignment"]), tuple(witness["target_assignment"])
        mapping = dict(witness["atom_mapping"])
        _validate_map(graph, mapping, node_tokens, edge_tokens)
        image = transport(states[0][source[0]], mapping)
        if (state_key(image) != state_key(states[0][target[0]]) or (repr(inventory_key(image)),) != inventory):
            raise ValueError("Witness does not replay its configuration and frozen inventory.")
        if class_of[source] != class_of[target]:
            raise ValueError("Witness connects assignments in different archived classes.")
        targets.append(target)
    if len(targets) != len(assignments) or set(targets) != set(assignments):
        raise ValueError("Every assignment requires exactly one archived merge witness.")
    mirror_table = _mirror_table(states, lookup)
    mirrors = _mirror_classes(partition, mirror_table)
    if json.loads(json.dumps(mirrors)) != reference["mirror_classes"]:
        raise ValueError("Archived mirror-class relation differs from independent reconstruction.")
    expected_images = [list(apply_table(value, mirror_table)) for value in assignments]
    if expected_images != reference["mirror_assignment_images"]:
        raise ValueError("Archived mirror assignment images differ from independent reconstruction.")
    return len(targets)


def replay(path, *, cip_path=None, acs_path=local.ACS_DATASET, rota_path=local.ROTA):
    source = capture_source()
    report = json.loads(path.read_text())
    if report["schema"] != "synkit.configuration-free-local-canonicalization/2":
        raise ValueError("A current local reference report is required.")
    if not report["summary"]["independent_oracles_complete"] or report["summary"]["carriers_failed"]:
        raise ValueError("An incomplete or failed local report cannot pass replay.")
    hashes = report["provenance"]["source_sha256"]
    listed_hashes_match = all(sha256(ROOT / name) == digest for name, digest in hashes.items())
    if not listed_hashes_match:
        raise ValueError("Executed source files differ from the archived report.")
    task = report["task"]
    rows = {} if task == "internal" else dict(local._select_rows(
        task, acs_path=acs_path, cip_path=cip_path, rota_path=rota_path,
    )[0])
    counts = {"records": 0, "carriers": 0, "witnesses": 0}
    for record in report["records"]:
        graph, seeds, atom_color, bond_color = _inputs(task, record["id"], rows)
        if len(seeds) != len(record["carriers"]):
            raise ValueError(f"Frozen carrier inventory differs:{record['id']}")
        for seed, carrier in zip(seeds, record["carriers"], strict=True):
            if seed.descriptor_class != carrier["family"]:
                raise ValueError("Frozen carrier family order differs.")
            counts["witnesses"] += replay_carrier(graph, seed, carrier, atom_color, bond_color)
            counts["carriers"] += 1
        counts["records"] += 1
    return {"schema": "synkit.local-reference-witness-replay/1", "input_report": str(path),
            "input_sha256": sha256(path), "report_source_commit": report["provenance"]["source_commit"],
            "listed_report_source_hashes_match": listed_hashes_match, "counts": counts, "passed": True,
            "scope": ("positive witness replay and stored partition/mirror consistency; "
                      "negative exhaustion tested separately"),
            "provenance": finish_source(source)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--cip-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    result = replay(args.report, cip_path=args.cip_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"counts": result["counts"], "passed": result["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
