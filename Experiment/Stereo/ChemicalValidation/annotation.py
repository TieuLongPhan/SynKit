"""Freeze independent chemical truth before any production evaluation.

Run with ``python -m Experiment.Stereo.ChemicalValidation.annotation``.
The report includes complete bounded assignment partitions for each accepted
coordination constitution and independently replayed structural negatives.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import networkx as nx

from Experiment.Stereo.Canonicalization.independent_model import (
    State, formal_states, inventory_key, state_key, transport,
)
from Experiment.Stereo.Canonicalization.independent_orbits import reference_partition
from Experiment.Stereo.ChemicalValidation.common import ROOT, provenance, replay_typed_map, sha256, wall_limit
from Experiment.Stereo.ChemicalValidation.coordination import ChemicalInputUnresolved, read_coordination
from Experiment.Stereo.ChemicalValidation.negative import replay_negative

COORDINATION_MANIFEST = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/chemical_coordination_selection.json"
NEGATIVE_MANIFEST = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence/chemical_negative_selection.json"


def state_from_wire(value):
    data = dict(value)
    for field in ("atoms", "path", "support"):
        data[field] = tuple(data[field])
    data["frames"] = tuple((center, tuple(refs), relation) for center, refs, relation in data["frames"])
    return State(**data)


def align_inventory(source, target, source_states, target_states):
    """Find one original typed map preserving the complete supplied inventory.

    This alignment is only a change of atom namespace. The subsequent full
    constitutional automorphism enumeration decides configuration equality.
    """
    expected = sorted(repr(inventory_key(state)) for state in target_states)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        source, target, node_match=lambda left, right: left["color"] == right["color"],
        edge_match=lambda left, right: left["color"] == right["color"],
    )
    with wall_limit(10):
        for count, mapping in enumerate(matcher.isomorphisms_iter(), start=1):
            if count > 50000:
                raise ChemicalInputUnresolved("source_alignment_map_limit")
            replay_typed_map(source, target, mapping)
            mapped = tuple(transport(state, mapping) for state in source_states)
            if sorted(repr(inventory_key(state)) for state in mapped) == expected:
                return mapping, count
    raise ChemicalInputUnresolved("source_configuration_inventory_not_alignable")


def _assignment(states, choices):
    lookup = {repr(state_key(state)): (carrier, index)
              for carrier, local in enumerate(choices) for index, state in enumerate(local)}
    result = [None] * len(choices)
    for state in states:
        carrier, index = lookup[repr(state_key(state))]
        if result[carrier] is not None:
            raise ChemicalInputUnresolved("ambiguous_aligned_carrier")
        result[carrier] = index
    if any(index is None for index in result):
        raise ChemicalInputUnresolved("aligned_source_missing_supplied_carrier")
    return tuple(result)


def annotate_coordination(records):
    accepted, unresolved, groups = [], [], {}
    for record in records:
        try:
            source = read_coordination(record, ROOT)
            key = record["constitution_key"]
            if key not in groups:
                choices = tuple(formal_states(state) for state in source.states)
                reference = reference_partition(source.graph, choices, atom_color="color", bond_color="color",
                                                timeout_seconds=10, max_automorphisms=50000, max_rss_mb=1024)
                if reference["status"] != "complete":
                    raise ChemicalInputUnresolved("constitutional_reference_incomplete:" + reference["reason"])
                class_of = {tuple(assignment): index for index, members in enumerate(reference["partition"])
                            for assignment in members}
                groups[key] = {"anchor": source, "choices": choices, "reference": reference,
                               "class_of": class_of, "group_id": len(groups), "anchor_id": record["id"]}
            group = groups[key]
            mapping, count = align_inventory(source.graph, group["anchor"].graph,
                                             source.states, group["anchor"].states)
            states = tuple(transport(state, mapping) for state in source.states)
            coordinate_states = tuple(transport(state, mapping) for state in source.coordinate_states)
            assignment = _assignment(states, group["choices"])
            reflected_assignment = _assignment(coordinate_states, group["choices"])
            expected_mirror = tuple(group["reference"]["mirror_assignment_images"][
                group["reference"]["assignments"].index(assignment)])
            if expected_mirror != reflected_assignment:
                raise ChemicalInputUnresolved("coordinate_reflection_disagrees_with_independent_frame_action")
            expected, mirror_expected = group["class_of"][assignment], group["class_of"][reflected_assignment]
            accepted.append({"id": record["id"], "status": "certified", "group_id": group["group_id"],
                             "source_sha256": record["sha256"], "source_alignment": sorted(mapping.items()),
                             "alignment_candidates_examined": count, "source_audit": source.audit,
                             "aligned_states": [asdict(state) for state in states],
                             "aligned_coordinate_mirror_states": [asdict(state) for state in coordinate_states],
                             "assignment": assignment, "reflected_assignment": reflected_assignment,
                             "reference_class": expected, "reference_mirror_class": mirror_expected,
                             "expected_mirror_relation": "achiral" if expected == mirror_expected else "chiral"})
        except (ChemicalInputUnresolved, TimeoutError) as error:
            unresolved.append({"id": record["id"], "status": "unresolved", "reason": str(error),
                               "source_sha256": record["sha256"],
                               "source_local_annotations": len(record["source_local_orientation_annotations"]),
                               "production_evaluated": False})
    references = [{"group_id": group["group_id"], "constitution_key": key,
                   "anchor_id": group["anchor_id"], "reference": group["reference"]}
                  for key, group in groups.items()]
    local_frames = sum(item["source_audit"]["supplied_local_frames_retained"] for item in accepted)
    return {"selected_count": len(records), "certified_count": len(accepted), "unresolved_count": len(unresolved),
            "accepted": accepted, "unresolved": unresolved, "constitutional_groups": references,
            "retained_source_local_frames": local_frames,
            "mirror_relations": dict(Counter(item["expected_mirror_relation"] for item in accepted))}


def build_annotation():
    start = provenance()
    coordination = json.loads(COORDINATION_MANIFEST.read_text())
    negatives = json.loads(NEGATIVE_MANIFEST.read_text())
    negative_truth = []
    for record in negatives["records"]:
        _molecule, truth = replay_negative(record)
        negative_truth.append({"id": record["id"], "truth": truth})
    report = {"schema": "synkit.frozen-chemical-annotation/1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "production_evaluated": False, "provenance": start,
              "coordination_manifest": str(COORDINATION_MANIFEST.relative_to(ROOT)),
              "coordination_manifest_sha256": sha256(COORDINATION_MANIFEST),
              "negative_manifest": str(NEGATIVE_MANIFEST.relative_to(ROOT)),
              "negative_manifest_sha256": sha256(NEGATIVE_MANIFEST),
              "truth_boundary": ("Source-generated configurations and constitutional symmetry; "
                                 "experimental persistence not assessed."),
              "coordination": annotate_coordination(coordination["records"]), "negative_truth": negative_truth}
    report["sources_unchanged_during_annotation"] = start["source_sha256"] == provenance()["source_sha256"]
    covered = len(report["coordination"]["accepted"]) + len(report["coordination"]["unresolved"])
    report["annotation_valid"] = (report["sources_unchanged_during_annotation"] and len(negative_truth) == 48 and
                                  report["coordination"]["selected_count"] == 40 and
                                  covered == 40)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_annotation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"annotation_valid": report["annotation_valid"],
                      "certified_coordination": report["coordination"]["certified_count"],
                      "unresolved_coordination": report["coordination"]["unresolved_count"],
                      "negative_truth": len(report["negative_truth"])}))
    return 0 if report["annotation_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
