"""Reconcile task-specific protocol summaries from archived records."""

from collections import Counter


def check(condition, message):
    if not condition:
        raise ValueError(message)


def unique(records, key="id"):
    check(bool(records) and len({r[key] for r in records}) == len(records),
          "Protocol coverage is empty or duplicates a record.")


def relations(report):
    records, totals = report["records"], report["totals"]
    unique(records)
    check(all(r["correct"] == (r["expected"] == r["observed"]) for r in records),
          "Relation correctness flags disagree with expected/observed relations.")
    expected = {"cases": len(records), "correct": sum(r["correct"] for r in records),
                "expected_relations": dict(Counter(r["expected"] for r in records)),
                "observed_relations": dict(Counter(r["observed"] for r in records))}
    check(totals == expected, "Relation totals do not reproduce from records.")


def conformance(report):
    records, summary = report["records"], report["summary"]
    unique(records)
    scored = [r for r in records if r["status"] in ("passed", "failed")]
    expected = {"records": len(records), "scored_records": len(scored),
                "passed": sum(r["status"] == "passed" for r in scored),
                "failed": sum(r["status"] == "failed" for r in scored),
                "errors": sum(r["status"] == "error" for r in records),
                "not_applicable": sum(r["status"] == "not_applicable" for r in records)}
    check(report["dataset"]["records"] == len(records)
          and all(summary[key] == value for key, value in expected.items()),
          "Conformance coverage/outcomes do not reproduce from records.")
    for field, prefix in (("configuration_check", "configuration"), ("reason_check", "reason")):
        checks = [r for r in scored if r[field] in ("passed", "failed")]
        check(summary[prefix + "_checks"] == len(checks)
              and summary[prefix + "_checks_passed"] == sum(r[field] == "passed" for r in checks),
              "Conformance evidence checks do not reproduce.")


def rota_loci(report):
    records = report["records"]
    unique(records)
    valid = [r for r in records if not r["parse_failure"]]
    check(report["dataset"]["records"] == len(records), "RotA endpoint report omits source rows.")
    for record in valid:
        reference = {tuple(value) for value in record["reference_pairs"]}
        predicted = {tuple(axis["pair"]) for axis in record["predicted_axes"]}
        check({tuple(value) for value in record["true_positive_pairs"]} == reference & predicted
              and {tuple(value) for value in record["false_negative_pairs"]} == reference - predicted
              and {tuple(value) for value in record["false_positive_pairs"]} == predicted - reference,
              "RotA endpoint scores disagree with retained pair sets.")
        check(record["exact_pair_set"] == (reference == predicted)
              and record["all_reference_pairs_recovered"] == (reference <= predicted),
              "RotA endpoint recovery flags disagree with retained pair sets.")
    fields = {"reference_loci": "reference_pairs", "predicted_loci": "predicted_axes",
              "true_positive_loci": "true_positive_pairs", "expanded_reference_paths": "reference_expanded_paths",
              "expanded_true_positive_paths": "true_positive_expanded_paths"}
    expected = {key: sum(len(r[field]) for r in valid) for key, field in fields.items()}
    expected.update({"parsed_records": len(valid), "parse_failures": len(records) - len(valid),
                     "records_with_all_reference_loci": sum(r["all_reference_pairs_recovered"] for r in valid),
                     "records_with_exact_locus_set": sum(r["exact_pair_set"] for r in valid)})
    check(all(report["totals"][key] == value for key, value in expected.items()),
          "RotA endpoint totals do not reproduce from records.")


def elements(report):
    records = report["records"]
    unique(records)
    ids = [r["id"] for r in records] + report["parse_failures"]
    check(len(set(ids)) == len(ids) == report["dataset"]["records"], "Element audit omits or duplicates source rows.")
    for field, value in report["totals"].items():
        if field == "renumbering_checked_records":
            expected = sum(r["renumbering_invariant"] is not None for r in records)
        elif field == "renumbering_invariant_records":
            expected = sum(r["renumbering_invariant"] is True for r in records)
        else:
            expected = sum(len(r[field]) for r in records)
        check(value == expected, "Element audit totals do not reproduce from retained records.")


def negatives(report):
    records, summary = report["records"], report["summary"]
    unique(records, "fixture_id")
    counts = Counter(r["proof_kind"] for r in records)
    check(summary["records"] == len(records)
          and summary["axis_fixed_automorphism_negatives"] == counts["axis_fixed_automorphism"]
          and summary["no_supported_axis_negatives"] == counts["no_supported_axis"],
          "Synthetic negative coverage does not reproduce from fixtures.")
    observed = sum(axis["carrier_status"] == "confirmed" for r in records for axis in r["axes"])
    check(summary["confirmed_axis_false_positives"] == observed, "Synthetic negative confirmation count disagrees.")


def cip_contract(report):
    groups = report["conflicting_input_groups"]
    ids = [identifier for group in groups for identifier in group["record_ids"]]
    check(len(ids) == len(set(ids)) == report["conflicting_records"], "CIP input-conflict coverage is inconsistent.")
    losses = 0
    for group in groups:
        sets = group["expected_label_sets"]
        covered = [identifier for labels in sets for identifier in labels["record_ids"]]
        check(len(sets) > 1 and set(covered) == set(group["record_ids"]) and len(covered) == len(set(covered)),
              "CIP conflicting label-set coverage is inconsistent.")
        losses += len(covered) - max(len(labels["record_ids"]) for labels in sets)
    check(report["maximum_exact_records_from_supplied_smiles"] == report["dataset"]["records"] - losses,
          "CIP deterministic input ceiling does not reproduce from conflicts.")
    check(report["complete_300_record_assignment_identifiable"] == (not groups),
          "CIP SMILES-only identifiability contradicts its conflicts.")


VALIDATORS = {
    "synkit.stereoisomer-relation-conformance/1": relations,
    "synkit.stereo-perception-conformance-report/1": conformance,
    "synkit.rota-axis-locus-benchmark/1": rota_loci,
    "synkit.stereo-element-audit/1": elements,
    "synkit.rota-synthetic-negative-controls/1": negatives,
    "synkit.cip-input-contract-audit/2": cip_contract,
}
