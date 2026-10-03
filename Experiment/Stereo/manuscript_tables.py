"""Generate numerical claims only from checksummed reports with explicit scopes."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from Experiment.Stereo.chemical_result_contracts import VALIDATORS as CHEMICAL_VALIDATORS
from Experiment.Stereo.protocol_result_contracts import VALIDATORS as PROTOCOL_VALIDATORS

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = "synkit.manuscript-result-manifest/1"


def pointer(value, path):
    """Read an RFC 6901 JSON pointer, refusing a missing field."""
    if not path:
        return value
    if not path.startswith("/"):
        raise ValueError(f"Invalid JSON pointer: {path}")
    for token in path[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _validate_cip(report):
    records = report["records"]
    coverage = report["coverage"]
    if len(records) != coverage["parsed_records"]:
        raise ValueError("CIP parsed denominator is inconsistent.")
    ids = [record["id"] for record in records] + report["parse_failures"]
    if len(set(ids)) != len(ids) or len(ids) != coverage["input_records"]:
        raise ValueError("CIP coverage omits or duplicates input records.")
    correct, wrong, missing, emitted = 0, 0, 0, 0
    for record in records:
        expected, predicted = set(record["expected"]), set(record["predicted"])
        if record["exact"] != (expected == predicted):
            raise ValueError("CIP exact-record flag disagrees with label sets.")
        correct += len(expected & predicted)
        wrong += len(predicted - expected)
        missing += len(expected - predicted)
        emitted += len(predicted)
    measured = {"records": len(records), "exact_records": sum(item["exact"] for item in records),
                "true_positive_labels": correct, "wrong_labels": wrong,
                "missing_labels": missing, "predicted_labels": emitted}
    if any(report["overall"][key] != value for key, value in measured.items()):
        raise ValueError("CIP summary does not reproduce from retained label sets.")
    if report["protocol"] == "smiles_only" and report["dataset"]["coordinate_sha256"] is not None:
        raise ValueError("SMILES-only report has borrowed coordinate provenance.")


def _validate_rdkit_representation(report):
    """Reconcile finite geometry state coverage and exhaustive numbering totals."""
    groups = report["groups"]
    families = [group["family"] for group in groups]
    if not groups or len(set(families)) != len(groups):
        raise ValueError("RDKit representation probe omits or duplicates a family.")
    permutations = 0
    for group in groups:
        records, states = group["records"], group["states"]
        if len(records) != states or {r["state"] for r in records} != set(range(1, states + 1)):
            raise ValueError("RDKit representation probe omits or duplicates a numbered state.")
        if len(group["documented_encoding_outputs"]) != states:
            raise ValueError("RDKit documented encoding coverage is inconsistent.")
        agree = len(set(group["documented_encoding_outputs"])) == 1
        distinct = len({r["canonical_smiles"] for r in records}) == states
        collapsed = len({r["stereo_removed_smiles"] for r in records}) == 1
        declared = (group["documented_equivalent_encodings_agree"],
                    group["distinct_configurations_retained"],
                    group["removing_stereo_collapses_configurations"])
        if (agree, distinct, collapsed) != declared:
            raise ValueError("RDKit representation flags disagree with retained full strings.")
        for record in records:
            if record["atom_permutations_checked"] != math.factorial(record["atom_count"]):
                raise ValueError("RDKit exhaustive numbering coverage is incomplete.")
            invariant = record["distinct_outputs_after_renumbering"] == [record["canonical_smiles"]]
            if record["numbering_invariant"] != invariant:
                raise ValueError("RDKit numbering flag disagrees with retained full strings.")
            permutations += record["atom_permutations_checked"]
    expected = {"families": len(groups), "states": sum(g["states"] for g in groups),
                "atom_permutations_checked": permutations, "passed_families": sum(g["passed"] for g in groups)}
    if report["summary"] != expected:
        raise ValueError("RDKit representation summary does not reproduce from records.")


def _validate_detection(report):
    """Check coverage and task-specific scores against the retained locus sets."""
    contracts = {
        "acs_stereomolgraph_molecular_chirality": ("reference_supplied_loci", "predicted_loci", "recovered_loci"),
        "chiralfinder_rota": ("reference_pairs", "predicted_pairs", "true_positive_loci"),
        "cip_validation_suite": ("reference_positions", "predicted_positions", "true_positive_loci"),
    }
    rows = 0
    for name, (reference_field, prediction_suffix, correct_field) in contracts.items():
        dataset = report["datasets"][name]
        records = dataset["records"]
        if len(records) != dataset["dataset_records"] or len({r["id"] for r in records}) != len(records):
            raise ValueError("Detection coverage omits or duplicates a dataset row.")
        parsed = [r for r in records if not r["parse_failure"]]
        evaluated = [r for r in parsed if r["detection_error"] is None]
        if dataset["parsed_records"] != len(parsed) or dataset["evaluated_records"] != len(evaluated):
            raise ValueError("Detection parse/evaluation coverage does not reproduce.")
        rows += len(records)
        retained_prefix = ("neighbor_configuration_assisted" if name == "cip_validation_suite"
                           else "local_neighbor_orientation_retained")
        for setting, prediction in (("configuration_erased", "broad_" + prediction_suffix),
                                    ("local_neighbor_orientation_retained", retained_prefix + "_" + prediction_suffix)):
            _validate_detection_setting(dataset["two_setting_comparison"][setting], evaluated,
                                        reference_field, prediction)
        for prefix in ("", retained_prefix + "_"):
            _validate_detection_renumbering(dataset[prefix + "renumbering"], evaluated, prefix)
        if "configuration_erased_miss_structure" in dataset:
            _validate_detection_misses(dataset, evaluated, reference_field, "broad_" + prediction_suffix)
        if name == "cip_validation_suite":
            _validate_detection_families(dataset, evaluated)
        for mode in ("broad", "confirmed"):
            pairs = [(set(r[reference_field]), set(r[mode + "_" + prediction_suffix])) for r in evaluated]
            expected = {"reference_loci": sum(len(reference) for reference, _ in pairs),
                        correct_field: sum(len(reference & predicted) for reference, predicted in pairs)}
            if name != "acs_stereomolgraph_molecular_chirality":
                expected["predicted_loci"] = sum(len(predicted) for _, predicted in pairs)
            if any(dataset[mode][key] != value for key, value in expected.items()):
                raise ValueError("Detection summary disagrees with retained reference/prediction sets.")
    if report["coverage"]["dataset_rows"] != rows or report["coverage"]["pooled_accuracy_reported"]:
        raise ValueError("Detection total coverage or task pooling is inconsistent.")


def _validate_detection_setting(summary, records, reference_field, prediction_field):
    pairs = [(set(r[reference_field]), set(r[prediction_field])) for r in records]
    expected = {"records": len(records), "reference_loci": sum(len(a) for a, _ in pairs),
                "predicted_loci": sum(len(b) for _, b in pairs),
                "true_positive_loci": sum(len(a & b) for a, b in pairs),
                "missed_loci": sum(len(a - b) for a, b in pairs),
                "additional_detected_loci": sum(len(b - a) for a, b in pairs),
                "exact_records": sum(a == b for a, b in pairs),
                "records_with_additional_detected_loci": sum(bool(b - a) for a, b in pairs)}
    if any(summary[key] != value for key, value in expected.items()):
        raise ValueError("Detection setting counts disagree with retained locus sets.")


def _validate_detection_renumbering(summary, records, prefix):
    checked = [r for r in records if r[prefix + "renumbering_invariant"] is not None]
    expected = {"checked_records": len(checked),
                "invariant_records": sum(r[prefix + "renumbering_invariant"] is True for r in checked),
                "failures": [{"id": r["id"], "issues": r[prefix + "renumbering_issues"]}
                             for r in checked if r[prefix + "renumbering_invariant"] is False]}
    if summary != expected:
        raise ValueError("Detection renumbering counts disagree with record outcomes.")


def _validate_detection_misses(dataset, records, reference_field, prediction_field):
    details = []
    for record in records:
        retained = record["configuration_erased_tetrahedral_miss_details"]
        expected = {value for value in set(record[reference_field]) - set(record[prediction_field])
                    if value.startswith("tetrahedral:")}
        if len(retained) != len(expected) or {detail["locus"] for detail in retained} != expected:
            raise ValueError("Detection miss explanations omit or duplicate a tetrahedral locus.")
        details.extend(retained)
    fields = {"by_configuration_erased_status": "configuration_erased_status",
              "by_atom_element": "atom_element", "by_ring_membership": "ring_membership"}
    expected = {key: dict(Counter(detail[field] for detail in details)) for key, field in fields.items()}
    expected["tetrahedral_misses_audited"] = len(details)
    expected["by_available_local_neighbor_frame_families"] = dict(Counter(
        "+".join(detail["available_local_neighbor_frame_families"]) or "none" for detail in details))
    if dataset["configuration_erased_miss_structure"] != expected:
        raise ValueError("Detection structural miss counts disagree with retained explanations.")


def _validate_detection_families(dataset, records):
    predictions = {"broad": "broad_predicted_positions", "confirmed": "confirmed_predicted_positions",
                   "neighbor_configuration_assisted": "neighbor_configuration_assisted_predicted_positions"}
    for family, modes in dataset["by_reference_class"].items():
        for mode, summary in modes.items():
            pairs = [({v for v in r["reference_positions"] if v.startswith(family + ":")},
                      set(r[predictions[mode]])) for r in records]
            expected = {"reference_loci": sum(len(a) for a, _ in pairs),
                        "recovered_loci": sum(len(a & b) for a, b in pairs)}
            if any(summary[key] != value for key, value in expected.items()):
                raise ValueError("Detection family counts disagree with retained reference positions.")


def _validate_controlled_costs(report):
    """Reconcile planned trials, censoring and full-code comparisons from records."""
    records, repetitions = report["records"], report["protocol"]["repetitions"]
    grouped, completed = defaultdict(list), defaultdict(set)
    for record in records:
        grouped[record["id"]].append(record)
        if record["status"] == "complete":
            completed[record["id"]].add(record["mode"])
        elif "full_code_intern_id" in record:
            raise ValueError("A censored or refused trial contains a completed code.")
    planned = {(mode, repetition) for mode in ("generators", "complete_group")
               for repetition in range(repetitions)}
    for trials in grouped.values():
        if len(trials) != len(planned) or {(r["mode"], r["repetition"]) for r in trials} != planned:
            raise ValueError("Controlled-cost coverage omits or duplicates planned trials.")
        if len({r["full_code_intern_id"] for r in trials if r["status"] == "complete"}) > 1:
            raise ValueError("Completed output modes or repetitions disagree on full codes.")
    if not grouped or report["cases_with_complete_code"] != len(completed):
        raise ValueError("Completed-case denominator does not reproduce.")
    if report["cases_with_both_output_modes_complete"] != sum(len(modes) == 2 for modes in completed.values()):
        raise ValueError("Matched-mode denominator does not reproduce.")
    _validate_cost_statistics(records, report["summary"])


def _validate_cost_statistics(records, summaries):
    """Exclude censored records when independently recomputing every statistic."""
    for mode, summary in summaries.items():
        trials = [r for r in records if r["mode"] == mode]
        if summary["planned"] != len(trials) or summary["outcomes"] != dict(Counter(r["status"] for r in trials)):
            raise ValueError("Controlled-cost outcomes do not reproduce from trials.")
        for field, measured in summary["completed_only"].items():
            values = []
            for trial in trials:
                if trial["status"] != "complete":
                    continue
                value = trial.get(field)
                if value is None and trial.get("primary_members") and field in trial["primary_members"][0]:
                    value = sum(member[field] for member in trial["primary_members"])
                if value is not None:
                    values.append(value)
            values.sort()
            expected = {"count": len(values), "median": statistics.median(values) if values else None,
                        "p95": values[(95 * len(values) + 99) // 100 - 1] if values else None,
                        "maximum": values[-1] if values else None}
            if measured != expected:
                raise ValueError("Controlled-cost completed-only statistics include inconsistent values.")


def _validate_registration(report):
    """Check full partition equality and the declared registration size sweep."""
    protocol, records = report["protocol"], report["records"]
    planned = {(size, repetition, method) for size in protocol["sizes"]
               for repetition in range(protocol["repetitions"]) for method in ("pairwise", "certificates")}
    if len(records) != len(planned) or {(r["size"], r["repetition"], r["method"]) for r in records} != planned:
        raise ValueError("Registration coverage omits or duplicates planned pipelines.")
    for record in records:
        if record["status"] != "complete":
            continue
        observed = {tuple(sorted(group)) for group in record["observed_partition"]}
        expected = {tuple(sorted(group)) for group in record["expected_partition"]}
        flat = [member for group in record["observed_partition"] for member in group]
        if observed != expected or len(flat) != len(set(flat)) or set(flat) != set(range(record["size"])):
            raise ValueError("Registration groups do not reproduce the complete independent partition.")
        if not record["partition_passed"] or record["group_count"] != len(observed):
            raise ValueError("Registration correctness flag or group denominator is inconsistent.")
    for row in report["summary"]:
        medians = {}
        for method in ("pairwise", "certificates"):
            trials = [r for r in records if r["size"] == row["size"] and r["method"] == method]
            if row[method]["outcomes"] != dict(Counter(r["status"] for r in trials)):
                raise ValueError("Registration outcomes do not reproduce.")
            values = sorted(r["total_seconds"] for r in trials if r["status"] == "complete")
            measured = {"count": len(values), "median": statistics.median(values) if values else None,
                        "p95": values[(95 * len(values) + 99) // 100 - 1] if values else None,
                        "maximum": values[-1] if values else None}
            if row[method]["total_seconds_completed_only"] != measured:
                raise ValueError("Registration runtime statistics do not reproduce.")
            medians[method] = measured["median"]
        ratio = medians["pairwise"] / medians["certificates"] if all(medians.values()) else None
        if row["pairwise_over_certificates_median_ratio"] != ratio:
            raise ValueError("Registration runtime ratio does not reproduce.")


def _required_correctness(report):
    """Apply the correctness contract declared by each supported report schema."""
    schema = report["schema"]
    if schema == "synkit.independent-stereograph-audit/1":
        valid = bool(report["records"]) and report["passed"]
        valid &= all(record["passed"] for record in report["records"])
        valid &= report["sources_unchanged_during_run"] and report["mutation_controls"]["passed"]
        valid &= report["summary"]["cases"] == len(report["records"])
    elif schema == "synkit.configuration-free-local-canonicalization/2":
        summary = report["summary"]
        valid = summary["complete"] and summary["independent_oracles_complete"]
        valid &= summary["carriers_failed"] == 0 and summary["parse_failures"] == 0
        valid &= summary["carriers"] > 0
    elif schema == "synkit.configuration-free-global-abc/2":
        valid = bool(report["records"]) and all(record["passed"] for record in report["records"])
        valid &= report["summary"]["cases_selected"] == len(report["records"])
        valid &= report["summary"]["complete"]
    elif schema == "synkit.controlled-stereo-costs/1":
        valid = report["study_valid"] and report["planned_runs_executed"]
        valid &= report["provenance"]["source_unchanged_during_run"]
        valid &= not report["full_code_disagreements"] and not report["unexpected_outcomes"]
    elif schema == "synkit.curated-stereo-registration/1":
        valid = report["passed"] and report["provenance"]["source_unchanged_during_run"]
        complete = all(r["status"] == "complete" and r["partition_passed"] for r in report["records"])
        valid &= bool(report["records"]) and complete
        boundary = report["boundary_examples"]
        valid &= boundary["strict_partial_input_refuses"]
        valid &= boundary["strict_partial_input_status"] == "incomplete"
        valid &= boundary["passed"] and all(r["passed"] and r["expected_equal"] == r["observed_equal"]
                                            for r in boundary["profile_pair"])
    elif schema == "synkit.rdkit-representation-probe/1":
        valid = report["passed"] and bool(report["groups"])
        valid &= report["provenance"]["source_unchanged_during_run"]
        valid &= report["provenance"]["inputs_unchanged_during_run"]
        valid &= all(group["passed"] and group["documented_equivalent_encodings_agree"]
                     and group["distinct_configurations_retained"]
                     and group["removing_stereo_collapses_configurations"]
                     and all(r["numbering_invariant"] and r["round_trip_preserved"]
                             and r["configured_tag_written"] for r in group["records"])
                     for group in report["groups"])
    elif schema in CHEMICAL_VALIDATORS:
        # The schema-specific validator has checked each complete record first.
        valid = True
    elif schema == "synkit.chemical-manuscript-summary/1":
        valid = report["passed"]
    elif schema == "synkit.stereoisomer-relation-conformance/1":
        valid = report["totals"]["correct"] == report["totals"]["cases"]
    elif schema == "synkit.stereo-perception-conformance-report/1":
        valid = report["summary"]["passed"] == report["summary"]["records"]
    elif schema == "synkit.rota-synthetic-negative-controls/1":
        valid = report["summary"]["confirmed_axis_false_positives"] == 0
    else:
        raise ValueError(f"No required-correctness contract is implemented for {schema}.")
    return valid


def validate_report(report, specification):
    """Enforce schema, declared scope, and required correctness before claiming numbers."""
    if report.get("schema") != specification["schema"]:
        raise ValueError("Report schema differs from the frozen manifest.")
    if not specification.get("scope") or not specification.get("expected"):
        raise ValueError("Reports require a named scope and explicit scope expectations.")
    for path, expected in specification["expected"].items():
        if pointer(report, path) != expected:
            raise ValueError(f"Report contract differs at {path}.")
    validator = {
        "synkit.cip-native-validation/4": _validate_cip,
        "synkit.controlled-stereo-costs/1": _validate_controlled_costs,
        "synkit.curated-stereo-registration/1": _validate_registration,
        "synkit.rdkit-representation-probe/1": _validate_rdkit_representation,
        "synkit.full-stereo-carrier-detection/4": _validate_detection,
        **CHEMICAL_VALIDATORS, **PROTOCOL_VALIDATORS,
    }.get(report["schema"])
    if validator is not None:
        validator(report)
    if specification.get("required_correctness", False):
        valid = _required_correctness(report)
        if not valid:
            raise ValueError("Required correctness report contains failed, capped, or incomplete work.")


def _validate_linked_inputs(report, root):
    """Check report-specific immutable inputs before deriving a numerical claim."""
    if report["schema"] == "synkit.chemical-manuscript-summary/1":
        from Experiment.Stereo.chemical_results import validate_summary
        validate_summary(report, root)
    if report["schema"] == "synkit.controlled-stereo-costs/1":
        inputs = root / report["inputs"]["path"]
        if hashlib.sha256(inputs.read_bytes()).hexdigest() != report["inputs"]["sha256"]:
            raise ValueError("Controlled-cost input manifest checksum differs.")
        frozen = json.loads(inputs.read_text())
        if {r["id"] for r in report["records"]} != {case["id"] for case in frozen["cases"]}:
            raise ValueError("Controlled-cost trials omit or replace a frozen input.")


def build_ledger(manifest, root=ROOT):
    """Resolve every claim to an exact artifact, field, source, and task denominator."""
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Unknown numerical manifest schema.")
    ledger, seen = [], {}
    for specification in manifest["reports"]:
        path = root / specification["path"]
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != specification["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {path}")
        report = json.loads(payload)
        validate_report(report, specification)
        _validate_linked_inputs(report, root)
        for claim in specification["claims"]:
            identifier = claim["id"]
            if not re.fullmatch(r"[A-Za-z]+", identifier):
                raise ValueError(f"Claim identifier is not a safe LaTeX macro: {identifier}")
            value = pointer(report, claim["pointer"])
            if type(value) not in (int, float):
                raise ValueError(f"Claim is not numerical: {identifier}")
            denominator = pointer(report, claim["denominator"]) if claim.get("denominator") else ""
            identity = (value, denominator, specification["scope"])
            if identifier in seen and seen[identifier] != identity:
                raise ValueError(f"Contradictory duplicate claim: {identifier}")
            if identifier in seen:
                continue
            seen[identifier] = identity
            ledger.append({
                "claim_id": identifier, "value": value, "denominator": denominator,
                "unit": claim.get("unit", "count"), "scope": specification["scope"],
                "report": specification["path"], "sha256": digest,
                "schema": report["schema"], "json_pointer": claim["pointer"],
                "source_commit": report.get("provenance", {}).get("source_commit", "historical-unrecorded"),
            })
    if not ledger:
        raise ValueError("The manifest contains no supported numerical claims.")
    return ledger


def write_ledger(ledger, destination):
    """Write machine-readable values and importable TeX macros from one ledger."""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "numerical_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    with (destination / "numerical_ledger.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ledger)
    lines = ["% Generated by Experiment/Stereo/manuscript_tables.py; do not edit values."]
    for claim in ledger:
        lines.append("\\newcommand{\\" + claim["claim_id"] + "}{" + str(claim["value"]) + "}")
    (destination / "numerical_values.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ledger = build_ledger(json.loads(args.manifest.read_text()))
    write_ledger(ledger, args.output_dir)
    print(json.dumps({"claims": len(ledger), "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
