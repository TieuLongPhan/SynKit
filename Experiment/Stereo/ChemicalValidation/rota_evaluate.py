"""Compare production with the frozen sixty-case independent RotA annotation."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from rdkit import Chem

from Experiment.Stereo.Canonicalization.independent_fixtures import Fixture
from Experiment.Stereo.Canonicalization.independent_model import formal_states
from Experiment.Stereo.Canonicalization.independent_orbits import partition_agreement, partition_from_labels
from Experiment.Stereo.Canonicalization.independent_runner import _production_assignments
from Experiment.Stereo.ChemicalValidation.common import provenance, replay_typed_map, sha256, wall_limit
from Experiment.Stereo.ChemicalValidation.coordination import source_molecule
from Experiment.Stereo.ChemicalValidation.rota_annotation import MANIFEST, annotate, source_inventory


def _without_elapsed(value):
    if isinstance(value, dict):
        return {key: _without_elapsed(item) for key, item in value.items() if key != "seconds"}
    if isinstance(value, (tuple, list)):
        return [_without_elapsed(item) for item in value]
    return value


def _carrier_conclusions(record, annotation):
    """Replay axial witnesses without requiring one particular valid automorphism."""
    carrier = _without_elapsed(annotation["constitutional_carrier"])
    if annotation["independent_family"] != "axis":
        return carrier
    graph, seeds, _family, _facts = source_inventory(record)
    seed = seeds[0]
    left, right = seed.atoms[2:4]
    frames = (seed.atoms[:2], seed.atoms[4:])
    witnesses = carrier["terminal_exchange_witnesses"]
    if len(witnesses) != len(frames):
        raise ValueError("rota_terminal_witness_count_mismatch")
    for witness, (first, second) in zip(witnesses, frames, strict=True):
        if witness is None:
            continue
        mapping = dict(witness)
        if len(mapping) != len(witness):
            raise ValueError("rota_terminal_witness_duplicate_domain")
        replay_typed_map(graph, graph, mapping)
        if (mapping[left] != left or mapping[right] != right or
                mapping[first] != second or mapping[second] != first):
            raise ValueError("rota_terminal_witness_does_not_fix_axis_and_exchange_frame")
    present = [witness is not None for witness in witnesses]
    expected_status = "symmetry_related" if any(present) else "confirmed"
    if (carrier["negative_queries_exhausted"] != [not value for value in present] or
            carrier["carrier_status"] != expected_status):
        raise ValueError("rota_terminal_witness_conclusion_mismatch")
    return {**carrier, "terminal_exchange_witnesses": present}


def replay_frozen_annotation(annotation, selected):
    """Recreate all structural conclusions before any production call."""
    if (annotation["schema"] != "synkit.frozen-rota-structural-adjudication/1" or
            not annotation["annotation_valid"] or annotation["production_evaluated"] or
            not annotation["sources_unchanged_during_annotation"] or
            annotation["selection_manifest_sha256"] != sha256(MANIFEST)):
        raise ValueError("rota_annotation_not_independently_frozen")
    expected_keys = [(record["source_id"], record["locus"]) for record in selected]
    observed_keys = [(record["source_id"], record["locus"]) for record in annotation["records"]]
    if expected_keys != observed_keys or len(set(observed_keys)) != 60:
        raise ValueError("rota_annotation_does_not_cover_all_sixty_frozen_loci_once")
    fields = ("status", "independent_family", "input_facts", "configured_seeds", "configured_class_count",
              "experimental_persistence")
    reference_fields = ("assignments", "partition", "mirror_assignment_images", "mirror_classes",
                        "formal_local_class_counts")
    for old, record in zip(annotation["records"], selected, strict=True):
        fresh = annotate(record)
        if fresh["status"] != old["status"] or fresh["status"] != "complete":
            raise ValueError("rota_structural_annotation_cannot_be_completely_replayed")
        if any(_without_elapsed(fresh[field]) != _without_elapsed(old[field]) for field in fields):
            raise ValueError("rota_frozen_structural_truth_disagrees_with_independent_replay")
        if _carrier_conclusions(record, fresh) != _carrier_conclusions(record, old):
            raise ValueError("rota_frozen_structural_truth_disagrees_with_independent_replay")
        if any(_without_elapsed(fresh["reference"][field]) != old["reference"][field] for field in reference_fields):
            raise ValueError("rota_frozen_partition_disagrees_with_independent_replay")
    return {"all_sixty_references_replayed": True, "case_keys": expected_keys}


def _expected_carriers(annotation, seeds):
    if annotation["independent_family"] == "axis":
        return [("axis", frozenset(seeds[0].atoms[2:4]), annotation["constitutional_carrier"]["carrier_status"])]
    return [("tetrahedral", seed.atoms[0], status) for seed, status in
            zip(seeds, annotation["constitutional_carrier"]["center_statuses"], strict=True)]


def compare_perception(record, annotation, seeds):
    """Match declared carrier semantics, separate from global configuration truth."""
    from Experiment.Stereo.Perception.family_executors import neutralize_rdkit_configuration
    from Experiment.Stereo.Perception.full_detection import _pair_text, _rota_predicted_loci
    from synkit.Chem.Molecule.stereo_perception import detect_potential_stereo_elements

    molecule = neutralize_rdkit_configuration(source_molecule(record["smiles"]))
    expected = _expected_carriers(annotation, seeds)
    order = tuple(reversed(range(molecule.GetNumAtoms())))
    renumbering = {old: new for new, old in enumerate(order)}
    observations, checks = [], []
    for working, mapping in ((molecule, {node: node for node in range(molecule.GetNumAtoms())}),
                             (Chem.RenumberAtoms(molecule, order), renumbering)):
        with wall_limit(10):
            elements = detect_potential_stereo_elements(working, include_extended_ring_axes=True)
        current = []
        for family, support, status in expected:
            if family == "axis":
                target = {mapping[node] for node in support}
                matching = [element for element in elements if element.element_type.value == "atrop_axis" and
                            set(element.support.path) == target]
                label = sorted(target)
            else:
                target = mapping[support]
                matching = [element for element in elements if element.element_type.value == "tetrahedral" and
                            element.support.center == target]
                label = target
            observed = [element.carrier_status.value for element in matching]
            current.append({"family": family, "support": label, "expected": status, "observed": observed})
            negative_tetrahedral_omitted = family == "tetrahedral" and status == "symmetry_related" and not observed
            current[-1]["negative_tetrahedral_omitted"] = negative_tetrahedral_omitted
            checks.append(observed == [status] or negative_tetrahedral_omitted)
        confirmed = [element for element in elements if element.carrier_status.value == "confirmed"]
        projected = _rota_predicted_loci(working, confirmed, record["source_category"], require_confirmed_support=True)
        target_locus = _pair_text(mapping[int(value)] for value in record["locus"].split("-"))
        expected_confirmation = all(status == "confirmed" for _family, _support, status in expected)
        projected_confirmation = target_locus in projected
        current.append({"family": "category_projection", "locus": target_locus,
                        "expected_confirmed": expected_confirmation, "observed_confirmed": projected_confirmation,
                        "geometry_only_support_cannot_confirm": True})
        checks.append(projected_confirmation == expected_confirmation)
        observations.append(current)
    return {"original": observations[0], "renumbered": observations[1], "passed": all(checks),
            "neighbor_orientation": "neutral", "extended_ring_axis_candidates": True}


def evaluate_case(record, annotation):
    started = time.monotonic()
    graph, seeds, family, _facts = source_inventory(record)
    choices = tuple(formal_states(seed) for seed in seeds)
    reference = dict(annotation["reference"])
    reference["assignments"] = tuple(tuple(value) for value in reference["assignments"])
    fixture = Fixture(record["source_id"] + ":" + record["locus"], graph, seeds, "Frozen RotA structural adjudication")
    observed, mirrors, renamed, issues, indices, disk_size = _production_assignments(
        fixture, choices, reference, {"atom_color": "color", "bond_color": "color"}, 5,
    )
    complete = len(observed) == len(reference["assignments"]) and not issues
    agreement = complete and partition_agreement(reference, reference["assignments"], observed)
    mirror_agreement = False
    if complete:
        codes = dict(zip(reference["assignments"], observed, strict=True))
        expected_mirrors = [codes[tuple(assignment)] for assignment in reference["mirror_assignment_images"]]
        mirror_agreement = mirrors == expected_mirrors
    try:
        perception = compare_perception(record, annotation, seeds)
    except TimeoutError as error:
        perception = {"passed": False, "reason": str(error)}
    return {"source_id": record["source_id"], "locus": record["locus"], "independent_family": family,
            "status": "complete" if complete else "production_refusal", "issues": issues,
            "production_assignments_completed": len(observed), "full_partition_agreement": agreement,
            "production_mirror_agreement": mirror_agreement, "renumbering_checks": len(renamed),
            "renumbering_failures": renamed.count(False),
            "production_partition": partition_from_labels(reference["assignments"], observed) if complete else None,
            "certificate_sha256_indices": indices, "temporary_full_certificate_store_bytes": disk_size,
            "perception": perception, "experimental_persistence": "unknown",
            "passed": agreement and mirror_agreement and all(renamed) and perception["passed"],
            "seconds": time.monotonic() - started}


def evaluate(annotation_path):
    started = time.monotonic()
    start = provenance()
    annotation = json.loads(annotation_path.read_text())
    selected = json.loads(MANIFEST.read_text())["records"]
    replay = replay_frozen_annotation(annotation, selected)
    results = [evaluate_case(record, truth) for record, truth in zip(selected, annotation["records"], strict=True)]
    unchanged = start["source_sha256"] == provenance()["source_sha256"]
    return {"schema": "synkit.rota-structural-production-comparison/1", "provenance": start,
            "annotation_path": str(annotation_path), "annotation_sha256": sha256(annotation_path),
            "independent_replay": replay, "records": results,
            "outcomes": dict(Counter(record["status"] for record in results)),
            "full_partition_failures": sum(not record["full_partition_agreement"] for record in results),
            "perception_failures": sum(not record["perception"]["passed"] for record in results),
            "complete_assignment_triplets": sum(record["production_assignments_completed"] for record in results),
            "sources_unchanged_during_run": unchanged, "seconds": time.monotonic() - started,
            "claim_boundary": ("Assisted structural development sample, not independent human chemical review. "
                               "Seven aromatic supports remain conditional configured-graph cases; "
                               "physical persistence unknown."),
            "passed": unchanged and len(results) == 60 and all(record["passed"] for record in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.annotation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "outcomes": report["outcomes"],
                      "partition_failures": report["full_partition_failures"],
                      "perception_failures": report["perception_failures"], "seconds": report["seconds"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
