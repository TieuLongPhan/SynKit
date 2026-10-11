"""Reconcile frozen chemical truth, production partitions and manuscript counts.

These checks validate archived evidence. They neither infer chemical truth nor
replace the independent annotations or their complete production replays.
"""

from collections import Counter
import itertools


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(records, fields):
    """Retain every case once, allowing several loci from one source molecule."""
    result = {tuple(record[field] for field in fields): record for record in records}
    require(bool(records) and len(result) == len(records), "Chemical coverage is empty or duplicates a case.")
    return result


def partition(value):
    groups = [frozenset(tuple(assignment) for assignment in group) for group in value]
    flattened = [tuple(assignment) for group in value for assignment in group]
    require(all(groups) and len(flattened) == len(set(flattened)), "Chemical partition repeats or omits members.")
    return frozenset(groups)


def reference_contract(reference):
    """Require a complete Cartesian population, partition and mirror involution."""
    require(reference["status"] == "complete", "Chemical independent reference is incomplete.")
    counts = reference["formal_local_class_counts"]
    require(bool(counts) and all(type(count) is int and count > 0 for count in counts),
            "Chemical reference has invalid local class counts.")
    assignments = [tuple(value) for value in reference["assignments"]]
    expected = set(itertools.product(*(range(count) for count in counts)))
    require(len(assignments) == len(expected) and set(assignments) == expected,
            "Chemical reference omits or duplicates an assignment.")
    groups = partition(reference["partition"])
    require(set().union(*groups) == expected, "Chemical partition does not cover the assignment population.")
    images = [tuple(value) for value in reference["mirror_assignment_images"]]
    require(len(images) == len(assignments) and set(images) == expected,
            "Chemical mirror does not cover the assignment population.")
    mirror = dict(zip(assignments, images))
    require(all(mirror[mirror[value]] == value for value in assignments), "Chemical mirror is not involutive.")
    class_of = {tuple(value): i for i, group in enumerate(reference["partition"]) for value in group}
    class_images = []
    for group in reference["partition"]:
        image = {class_of[mirror[tuple(value)]] for value in group}
        require(len(image) == 1, "Chemical mirror is not constant on a configuration class.")
        class_images.append(next(iter(image)))
    require(reference["mirror_classes"]["class_images"] == class_images,
            "Chemical class mirror does not reproduce from assignments.")
    require(reference["composition"]["failures"] == reference["mirror_commutation"]["failures"] == 0,
            "Chemical reference transport checks failed.")
    return class_of


def coordination_annotation(report):
    require(report["annotation_valid"] and not report["production_evaluated"]
            and report["sources_unchanged_during_annotation"], "Chemical truth was not frozen independently.")
    data = report["coordination"]
    accepted = indexed(data["accepted"], ("id",))
    unresolved = indexed(data["unresolved"], ("id",)) if data["unresolved"] else {}
    require(not (accepted.keys() & unresolved.keys())
            and data["selected_count"] == len(accepted) + len(unresolved)
            and data["certified_count"] == len(accepted) and data["unresolved_count"] == len(unresolved),
            "Chemical selected/certified/refused coverage is inconsistent.")
    require(all(not record["production_evaluated"] for record in unresolved.values()),
            "A frozen geometry refusal was production evaluated.")
    groups = indexed(data["constitutional_groups"], ("group_id",))
    require(set(groups) == {(r["group_id"],) for r in accepted.values()}, "Chemical group coverage is inconsistent.")
    classes = {key: reference_contract(group["reference"]) for key, group in groups.items()}
    for record in accepted.values():
        class_of = classes[(record["group_id"],)]
        require(class_of[tuple(record["assignment"])] == record["reference_class"]
                and class_of[tuple(record["reflected_assignment"])] == record["reference_mirror_class"],
                "Chemical source class disagrees with the frozen complete partition.")
        expected = "achiral" if record["reference_class"] == record["reference_mirror_class"] else "chiral"
        require(record["expected_mirror_relation"] == expected, "Chemical source mirror relation is inconsistent.")
        for key in ("geometry", "reflected_geometry"):
            geometry = record["source_audit"][key]
            require(geometry["normalized_rmsd"] <= geometry["rmsd_limit"]
                    and geometry["next_inequivalent_pose_gap"] >= geometry["pose_gap_minimum"],
                    "A certified source geometry fails its frozen refusal criteria.")
    require(data["retained_source_local_frames"] == sum(
        r["source_audit"]["supplied_local_frames_retained"] for r in accepted.values()),
        "Chemical retained local-frame count is inconsistent.")
    require(data["mirror_relations"] == dict(Counter(r["expected_mirror_relation"] for r in accepted.values())),
            "Chemical source mirror counts are inconsistent.")
    indexed(report["negative_truth"], ("id",))


def coordination_production(report):
    require(report["passed"] and report["sources_unchanged_during_run"], "Chemical production did not pass.")
    data = report["coordination"]
    records = indexed(data["records"], ("id",))
    require(data["certified_count"] == len(records)
            and data["selected_count"] == len(records) + len(data["unresolved"]),
            "Chemical production coverage is inconsistent.")
    require(data["outcomes"] == dict(Counter(r["status"] for r in records.values())),
            "Chemical production outcomes are inconsistent.")
    reference_to_output, output_to_reference = {}, {}
    for record in records.values():
        codes = record["code_interning_ids"]
        require(record["status"] == "complete" and record["passed"]
                and record["renumbering_agreement"] and record["coordinate_and_production_mirror_agreement"]
                and codes["source"] == codes["renumbered"]
                and codes["coordinate_mirror"] == codes["production_mirror"],
                "Chemical production lacks a completed mirror/renumbering comparison.")
        observed = "achiral" if codes["source"] == codes["coordinate_mirror"] else "chiral"
        require(record["observed_mirror_relation"] == record["expected_mirror_relation"] == observed,
                "Chemical production mirror flags disagree with full-code equality.")
        for mode, field in (("source", "reference_class"), ("coordinate_mirror", "reference_mirror_class")):
            expected, output = (record["group_id"], record[field]), codes[mode]
            require(reference_to_output.get(expected, output) == output
                    and output_to_reference.get(output, expected) == expected,
                    "Chemical full-code equality splits or merges a reference class.")
            reference_to_output[expected], output_to_reference[output] = output, expected
    observed = data["observed_partition"]
    require(observed["passed"] and not observed["issues"]
            and observed["observed_reference_classes"] == len(reference_to_output),
            "Chemical observed-class coverage is inconsistent.")
    require(all(r["complete_partition_and_mirror_replayed"] for r in report["independent_reference_replay"]),
            "Chemical independent reference replay did not complete.")
    _negative_production(report["negative_controls"])


def _negative_production(data):
    records = indexed(data["records"], ("id",))
    require(data["selected_count"] == len(records)
            and data["outcomes"] == dict(Counter(r["status"] for r in records.values())),
            "Negative-control coverage is inconsistent.")
    for record in records.values():
        truth = record["truth"]
        witness_fields = ("full_typed_bijection_replayed", "axis_endpoints_fixed",
                          "outside_ring_fixed", "terminal_branches_exchanged")
        require(record["status"] == "complete" and record["passed"]
                and all(truth[key] for key in witness_fields),
                "Negative control lacks a complete constitutional symmetry witness.")
        mapping = dict(record["renumbering"])
        require(len(mapping) == len(record["renumbering"]) and set(mapping) == set(mapping.values()),
                "Negative-control renumbering is not a bijection.")
        endpoints = set(record["axis_endpoints"])
        modes = (("production_original", endpoints),
                 ("production_renumbered", {mapping[v] for v in endpoints}))
        for field, expected in modes:
            items = record[field]
            require(len(items) == 1 and items[0]["carrier_status"] == "symmetry_related"
                    and set(items[0]["path"]) == expected,
                    "Negative control reports a false confirmed or mismapped axis.")


def rota_annotation(report):
    require(report["annotation_valid"] and not report["production_evaluated"]
            and report["sources_unchanged_during_annotation"], "RotA truth was not frozen independently.")
    records = indexed(report["records"], ("source_id", "locus"))
    require(report["outcomes"] == dict(Counter(r["status"] for r in records.values()))
            and report["family_counts"] == dict(Counter(r["independent_family"] for r in records.values())),
            "RotA family/outcome counts are inconsistent.")
    for record in records.values():
        require(record["status"] == "complete" and not record["algorithm_confirmed_rejected_status_read"],
                "RotA annotation is incomplete or used the algorithm's confirmation status.")
        reference_contract(record["reference"])
        require(record["configured_class_count"] == len(record["reference"]["partition"]),
                "RotA configured-class count is inconsistent.")
        carrier = record["constitutional_carrier"]
        if record["independent_family"] == "axis":
            facts = record["input_facts"]
            require(facts["ordinary_single_bond_topology"] == (facts["source_bond_type"] == "SINGLE")
                    and not facts["geometric_axis_certified"], "RotA topology was promoted to geometric truth.")
            witnessed = any(value is not None for value in carrier["terminal_exchange_witnesses"])
            expected = "symmetry_related" if witnessed else "confirmed"
            require(carrier["carrier_status"] == expected
                    and (witnessed or all(carrier["negative_queries_exhausted"])),
                    "RotA axis truth lacks a witness or exhausted negative queries.")
        else:
            focal = carrier["focal_center_references"]
            require(len(focal) == len(carrier["center_statuses"]), "RotA focal carrier coverage is inconsistent.")
            for reference, status in zip(focal, carrier["center_statuses"]):
                reference_contract(reference)
                require(status == ("confirmed" if len(reference["partition"]) == 2 else "symmetry_related"),
                        "RotA focal carrier status disagrees with its independent partition.")


def rota_production(report):
    require(report["passed"] and report["sources_unchanged_during_run"], "RotA production did not pass.")
    records = indexed(report["records"], ("source_id", "locus"))
    replay = report["independent_replay"]
    require(replay["all_sixty_references_replayed"]
            and len(replay["case_keys"]) == len(records)
            and {tuple(key) for key in replay["case_keys"]} == set(records), "RotA replay coverage is inconsistent.")
    require(report["outcomes"] == dict(Counter(r["status"] for r in records.values()))
            and not report["full_partition_failures"] and not report["perception_failures"],
            "RotA production outcomes or failures are inconsistent.")
    for record in records.values():
        require(record["status"] == "complete" and record["passed"] and not record["issues"]
                and record["full_partition_agreement"] and record["production_mirror_agreement"]
                and record["renumbering_checks"] == record["production_assignments_completed"]
                and record["renumbering_failures"] == 0 and record["perception"]["passed"],
                "RotA production lacks complete partition, mirror, numbering or perception evidence.")
        groups = partition(record["production_partition"])
        require(sum(map(len, groups)) == record["production_assignments_completed"],
                "RotA production assignment coverage disagrees with its partition.")
        _perception_contract(record["perception"])
    require(report["complete_assignment_triplets"] == sum(r["production_assignments_completed"]
                                                          for r in records.values()),
            "RotA complete assignment count is inconsistent.")


def _perception_contract(perception):
    for mode in ("original", "renumbered"):
        require(bool(perception[mode]), "RotA perception lacks a supplied evidence mode.")
        for check in perception[mode]:
            if check["family"] == "category_projection":
                require(check["expected_confirmed"] == check["observed_confirmed"]
                        and check["geometry_only_support_cannot_confirm"],
                        "RotA category projection contradicts its confirmation contract.")
            else:
                omitted = (check["family"] == "tetrahedral" and check["expected"] == "symmetry_related"
                           and not check["observed"] and check["negative_tetrahedral_omitted"])
                require(check["observed"] == [check["expected"]] or omitted,
                        "RotA observed carrier contradicts its expected status.")


VALIDATORS = {
    "synkit.frozen-chemical-annotation/1": coordination_annotation,
    "synkit.sourced-chemical-validation/1": coordination_production,
    "synkit.frozen-rota-structural-adjudication/1": rota_annotation,
    "synkit.rota-structural-production-comparison/1": rota_production,
}


def summarize(annotation, production, rota_truth, rota_results):
    """Join exact case identities before deriving numerical manuscript values."""
    for report in (annotation, production, rota_truth, rota_results):
        VALIDATORS[report["schema"]](report)
    _join_coordination(annotation, production)
    _join_rota(rota_truth, rota_results)
    coordination, observed = annotation["coordination"], production["coordination"]
    records = coordination["accepted"]
    axes = [record for record in rota_truth["records"] if record["independent_family"] == "axis"]
    return {
        "coordination": {
            "selected": coordination["selected_count"], "certified": len(records),
            "unresolved": len(coordination["unresolved"]), "constitutions": len(coordination["constitutional_groups"]),
            "retained_tetrahedral_frames": coordination["retained_source_local_frames"],
            "reference_assignments": sum(len(g["reference"]["assignments"])
                                         for g in coordination["constitutional_groups"]),
            "reference_classes": sum(len(g["reference"]["partition"])
                                     for g in coordination["constitutional_groups"]),
            "source_achiral": coordination["mirror_relations"]["achiral"],
            "source_chiral": coordination["mirror_relations"]["chiral"],
            "code_computations": sum(len(r["code_interning_ids"]) for r in observed["records"]),
            "observed_source_mirror_classes": observed["observed_partition"]["observed_reference_classes"],
            "maximum_certified_rmsd": max(r["source_audit"]["geometry"]["normalized_rmsd"] for r in records),
            "minimum_certified_pose_gap": min(r["source_audit"]["geometry"]["next_inequivalent_pose_gap"]
                                              for r in records),
        },
        "negatives": {"selected": len(annotation["negative_truth"]),
                      "passed": len(production["negative_controls"]["records"]),
                      "perception_computations": 2 * len(production["negative_controls"]["records"])},
        "rota": {
            "selected": len(rota_truth["records"]), "complete": len(rota_results["records"]),
            "source_records": len({r["source_id"] for r in rota_truth["records"]}),
            "strata": len({r["stratum"] for r in rota_truth["records"]}),
            "axis_supports": len(axes), "tetrahedral_pairs": rota_truth["family_counts"]["tetrahedral_pair"],
            "spiro_centers": rota_truth["family_counts"]["tetrahedral_spiro"],
            "ordinary_single_bond_axes": sum(r["input_facts"]["ordinary_single_bond_topology"] for r in axes),
            "aromatic_axes": sum(r["input_facts"]["source_bond_type"] == "AROMATIC" for r in axes),
            "terminal_related_axes": sum(r["constitutional_carrier"]["carrier_status"] == "symmetry_related"
                                         for r in axes),
            "terminal_confirmed_axes": sum(r["constitutional_carrier"]["carrier_status"] == "confirmed"
                                           for r in axes),
            "formal_assignments": sum(len(r["reference"]["assignments"]) for r in rota_truth["records"]),
            "sum_per_case_classes": sum(r["configured_class_count"] for r in rota_truth["records"]),
            "code_computations": 3 * rota_results["complete_assignment_triplets"],
            "perception_computations": 2 * len(rota_results["records"]),
        },
    }


def _join_coordination(annotation, production):
    truth = indexed(annotation["coordination"]["accepted"], ("id",))
    observed = indexed(production["coordination"]["records"], ("id",))
    require(set(truth) == set(observed), "Chemical production substituted or omitted a frozen source case.")
    for key, record in truth.items():
        require(all(record[field] == observed[key][field] for field in
                    ("group_id", "reference_class", "reference_mirror_class", "expected_mirror_relation")),
                "Chemical production changed its frozen independent truth.")
    require(annotation["coordination"]["unresolved"] == production["coordination"]["unresolved"],
            "Chemical production omitted or reclassified frozen refusals.")
    negative_truth = indexed(annotation["negative_truth"], ("id",))
    negatives = indexed(production["negative_controls"]["records"], ("id",))
    require(set(negative_truth) == set(negatives)
            and all(record["truth"] == negatives[key]["truth"] for key, record in negative_truth.items()),
            "Production negative controls differ from frozen truth.")
    groups = indexed(annotation["coordination"]["constitutional_groups"], ("group_id",))
    replay = indexed(production["independent_reference_replay"], ("group_id",))
    require(set(groups) == set(replay) and all(replay[key]["assignment_count"]
            == len(group["reference"]["assignments"]) for key, group in groups.items()),
            "Chemical full-reference replay coverage differs from frozen groups.")


def _join_rota(annotation, production):
    truth = indexed(annotation["records"], ("source_id", "locus"))
    observed = indexed(production["records"], ("source_id", "locus"))
    require(set(truth) == set(observed), "RotA production substituted or omitted a frozen locus.")
    for key, record in truth.items():
        result = observed[key]
        require(record["independent_family"] == result["independent_family"]
                and partition(record["reference"]["partition"]) == partition(result["production_partition"])
                and len(record["reference"]["assignments"]) == result["production_assignments_completed"],
                "RotA production partition differs from its full independent reference.")
        carrier = record["constitutional_carrier"]
        expected = ([carrier["carrier_status"]] if record["independent_family"] == "axis"
                    else carrier["center_statuses"])
        for mode in ("original", "renumbered"):
            checks = result["perception"][mode]
            statuses = [check["expected"] for check in checks if check["family"] != "category_projection"]
            projections = [check for check in checks if check["family"] == "category_projection"]
            require(statuses == expected and len(projections) == 1
                    and projections[0]["expected_confirmed"] == all(status == "confirmed" for status in expected),
                    "RotA perception changed its independently frozen carrier truth.")
