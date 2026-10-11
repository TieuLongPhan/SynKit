"""Evaluate production against an independently frozen annotation report."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import networkx as nx

from Experiment.Stereo.Canonicalization.exact_code_store import ExactCodeStore
from Experiment.Stereo.Canonicalization.independent_model import assignment_key, formal_states, to_descriptor, transport
from Experiment.Stereo.Canonicalization.independent_orbits import reference_partition
from Experiment.Stereo.Canonicalization.independent_runner import _code
from Experiment.Stereo.Canonicalization.runtime import peak_rss_mib
from Experiment.Stereo.ChemicalValidation.annotation import (
    COORDINATION_MANIFEST, NEGATIVE_MANIFEST, _assignment, state_from_wire,
)
from Experiment.Stereo.ChemicalValidation.common import ROOT, provenance, replay_typed_map, sha256
from Experiment.Stereo.ChemicalValidation.coordination import read_coordination
from Experiment.Stereo.ChemicalValidation.negative import evaluate_negative


def _check_frozen_sources(report):
    if (report["schema"] != "synkit.frozen-chemical-annotation/1" or not report["annotation_valid"] or
            report["production_evaluated"] or not report["sources_unchanged_during_annotation"]):
        raise ValueError("chemical_annotation_not_independently_frozen")
    for key, path in (("coordination_manifest_sha256", COORDINATION_MANIFEST),
                      ("negative_manifest_sha256", NEGATIVE_MANIFEST)):
        if report[key] != sha256(path):
            raise ValueError("chemical_annotation_selection_hash_mismatch")
    manifest = json.loads(NEGATIVE_MANIFEST.read_text())
    for key, digest in (("source_report", "source_sha256"), ("query_manifest", "query_manifest_sha256")):
        if sha256(ROOT / manifest[key]) != manifest[digest]:
            raise ValueError("negative_source_or_query_manifest_hash_mismatch")
    selected = json.loads(COORDINATION_MANIFEST.read_text())["records"]
    coordination = report["coordination"]
    covered = coordination["accepted"] + coordination["unresolved"]
    ids = [record["id"] for record in covered]
    if len(ids) != len(set(ids)) or set(ids) != {record["id"] for record in selected}:
        raise ValueError("frozen_annotation_does_not_cover_every_selected_input_once")
    if (coordination["certified_count"] != len(coordination["accepted"]) or
            coordination["unresolved_count"] != len(coordination["unresolved"])):
        raise ValueError("frozen_annotation_false_coverage_count")
    if [record["id"] for record in report["negative_truth"]] != [record["id"] for record in manifest["records"]]:
        raise ValueError("frozen_annotation_negative_scope_mismatch")


def _replay_group_references(annotation, parsed):
    """Recompute every full reference partition before reading production codes."""
    checks = []
    groups = annotation["coordination"]["constitutional_groups"]
    group_ids = [group["group_id"] for group in groups]
    accepted = annotation["coordination"]["accepted"]
    if len(group_ids) != len(set(group_ids)) or set(group_ids) != {record["group_id"] for record in accepted}:
        raise ValueError("frozen_reference_group_scope_mismatch")
    for group in groups:
        source = parsed[group["anchor_id"]]
        choices = tuple(formal_states(state) for state in source.states)
        fresh = reference_partition(source.graph, choices, atom_color="color", bond_color="color",
                                    timeout_seconds=10, max_automorphisms=50000, max_rss_mb=1024)
        old = group["reference"]
        fields = ("assignments", "partition", "mirror_assignment_images", "mirror_classes", "formal_local_class_counts")
        equal = (fresh["status"] == old["status"] == "complete" and all(
            json.loads(json.dumps(fresh[field])) == old[field] for field in fields
        ))
        if not equal:
            raise ValueError("frozen_reference_partition_failed_independent_replay")
        _check_record_classes([record for record in accepted if record["group_id"] == group["group_id"]],
                              choices, fresh)
        checks.append({"group_id": group["group_id"], "complete_partition_and_mirror_replayed": True,
                       "assignment_count": len(fresh["assignments"]),
                       "constitutional_maps": fresh["constitutional_automorphisms"]})
    return checks


def _check_record_classes(records, choices, reference):
    class_of = {tuple(assignment): index for index, members in enumerate(reference["partition"])
                for assignment in members}
    for record in records:
        values = tuple(state_from_wire(value) for value in record["aligned_states"])
        mirrored = tuple(state_from_wire(value) for value in record["aligned_coordinate_mirror_states"])
        source, reflection = _assignment(values, choices), _assignment(mirrored, choices)
        relation = "achiral" if class_of[source] == class_of[reflection] else "chiral"
        if (list(source) != record["assignment"] or list(reflection) != record["reflected_assignment"] or
                class_of[source] != record["reference_class"] or
                class_of[reflection] != record["reference_mirror_class"] or
                relation != record["expected_mirror_relation"]):
            raise ValueError("frozen_source_class_not_justified_by_independent_partition")


def _evaluate_coordination(record, source, graph, store):
    from synkit.Graph.Stereo import mirror_configured_descriptor

    mapping = dict(record["source_alignment"])
    replay_typed_map(source.graph, graph, mapping)
    states = tuple(state_from_wire(value) for value in record["aligned_states"])
    mirrors = tuple(state_from_wire(value) for value in record["aligned_coordinate_mirror_states"])
    replayed = tuple(transport(state, mapping) for state in source.states)
    replayed_mirrors = tuple(transport(state, mapping) for state in source.coordinate_states)
    original_matches = assignment_key(replayed) == assignment_key(states)
    mirror_matches = assignment_key(replayed_mirrors) == assignment_key(mirrors)
    if not original_matches or not mirror_matches:
        raise ValueError("frozen_configured_annotation_disagrees_with_source_replay")
    descriptors = tuple(to_descriptor(state) for state in states)
    renamed = {node: 10000 + index for index, node in enumerate(reversed(tuple(graph)))}
    tasks = (("source", graph, descriptors),
             ("coordinate_mirror", graph, tuple(to_descriptor(state) for state in mirrors)),
             ("production_mirror", graph, tuple(mirror_configured_descriptor(value) for value in descriptors)),
             ("renumbered", nx.relabel_nodes(graph, renamed),
              tuple(to_descriptor(transport(state, renamed)) for state in states)))
    observed = {}
    timings = {}
    for name, working, values in tasks:
        started = time.monotonic()
        observed[name] = store.intern(_code(working, values, {"atom_color": "color", "bond_color": "color"}, 5))
        timings[name] = time.monotonic() - started
        if peak_rss_mib() > 1024:
            raise MemoryError("chemical_evaluation_process_peak_rss_limit:1024MiB")
    observed_relation = "achiral" if observed["source"] == observed["coordinate_mirror"] else "chiral"
    digests = {name: store.index_digest(identifier) for name, identifier in observed.items()}
    mirrors_agree = observed["coordinate_mirror"] == observed["production_mirror"]
    return {"id": record["id"], "status": "complete", "group_id": record["group_id"],
            "code_interning_ids": observed, "code_index_digests": digests,
            "certificate_wall_seconds": timings, "reference_class": record["reference_class"],
            "reference_mirror_class": record["reference_mirror_class"],
            "expected_mirror_relation": record["expected_mirror_relation"],
            "observed_mirror_relation": observed_relation,
            "renumbering_agreement": observed["source"] == observed["renumbered"],
            "coordinate_and_production_mirror_agreement": mirrors_agree,
            "passed": (observed_relation == record["expected_mirror_relation"] and
                       observed["source"] == observed["renumbered"] and
                       observed["coordinate_mirror"] == observed["production_mirror"])}


def _partition_checks(records):
    """Check both equality directions for every observed source/mirror class."""
    reference_to_observed, observed_to_reference, issues = {}, {}, []
    for record in records:
        if record["status"] != "complete":
            continue
        for mode, field in (("source", "reference_class"), ("coordinate_mirror", "reference_mirror_class")):
            expected = (record["group_id"], record[field])
            observed = record["code_interning_ids"][mode]
            if expected in reference_to_observed and reference_to_observed[expected] != observed:
                issues.append({"id": record["id"], "mode": mode, "reason": "one_reference_class_split"})
            if observed in observed_to_reference and observed_to_reference[observed] != expected:
                issues.append({"id": record["id"], "mode": mode, "reason": "distinct_reference_classes_merged"})
            reference_to_observed[expected] = observed
            observed_to_reference[observed] = expected
    return {"observed_reference_classes": len(reference_to_observed), "issues": issues, "passed": not issues,
            "scope": ("All certified source configurations and their coordinate mirrors; "
                      "not all unsourced formal poses.")}


def evaluate(annotation_path):
    started = time.monotonic()
    annotation = json.loads(annotation_path.read_text())
    _check_frozen_sources(annotation)
    start = provenance()
    manifest = json.loads(COORDINATION_MANIFEST.read_text())
    selected = {record["id"]: record for record in manifest["records"]}
    parsed = {record["id"]: read_coordination(selected[record["id"]], ROOT)
              for record in annotation["coordination"]["accepted"]}
    reference_checks = _replay_group_references(annotation, parsed)
    anchors = {group["group_id"]: parsed[group["anchor_id"]].graph
               for group in annotation["coordination"]["constitutional_groups"]}
    results = []
    with ExactCodeStore() as store:
        for record in annotation["coordination"]["accepted"]:
            try:
                result = _evaluate_coordination(record, parsed[record["id"]], anchors[record["group_id"]], store)
            except (TimeoutError, MemoryError, RuntimeError) as error:
                result = {"id": record["id"], "status": "production_refusal", "passed": False,
                          "reason": f"{type(error).__name__}: {error}"}
            results.append(result)
        storage_bytes = store.bytes_on_disk
    negatives = []
    for record in json.loads(NEGATIVE_MANIFEST.read_text())["records"]:
        try:
            negatives.append({"status": "complete", **evaluate_negative(record)})
        except (TimeoutError, MemoryError, RuntimeError) as error:
            negatives.append({"id": record["id"], "status": "production_refusal", "passed": False,
                              "reason": f"{type(error).__name__}: {error}"})
    partitions = _partition_checks(results)
    unchanged = start["source_sha256"] == provenance()["source_sha256"]
    return {"schema": "synkit.sourced-chemical-validation/1", "provenance": start,
            "annotation_path": str(annotation_path), "annotation_sha256": sha256(annotation_path),
            "coordination": {"selected_count": len(selected), "certified_count": len(parsed),
                             "unresolved": annotation["coordination"]["unresolved"], "records": results,
                             "outcomes": dict(Counter(record["status"] for record in results)),
                             "observed_partition": partitions, "exact_certificate_store_bytes": storage_bytes},
            "independent_reference_replay": reference_checks,
            "negative_controls": {"selected_count": 48, "records": negatives,
                                  "outcomes": dict(Counter(record["status"] for record in negatives))},
            "sources_unchanged_during_run": unchanged, "seconds": time.monotonic() - started,
            "claim_boundary": ("Configured source-generated structures and structural negatives; "
                               "no experimental persistence claim."),
            "passed": (unchanged and len(results) == 38 and len(negatives) == 48 and partitions["passed"] and
                       all(record["passed"] for record in results + negatives))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.annotation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "coordination": report["coordination"]["outcomes"],
                      "negatives": report["negative_controls"]["outcomes"], "seconds": report["seconds"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
