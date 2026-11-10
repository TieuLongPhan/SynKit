"""Independently recount a streamed extraction/replay audit and report failures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

from .benchmark import digest
from .fragment_metrics import METRICS


def analyze(directory):
    summary = json.loads((directory / "summary.json").read_text())
    rows = 0
    admitted = 0
    with_ports = 0
    refusals = Counter()
    metrics = defaultdict(Counter)
    comparison = Counter()
    failures = []
    misses = []
    refusal_examples = defaultdict(list)
    with gzip.open(directory / "records.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            rows += 1
            if record["status"] != "extracted":
                refusals[record["code"]] += 1
                if len(refusal_examples[record["code"]]) < 3:
                    refusal_examples[record["code"]].append({
                        "id": record["r_id"], "error": record["error"],
                    })
                continue
            admitted += 1
            with_ports += bool(record["extraction"]["ports"])
            comparison.update(record.get("backend_comparison", {}))
            for key, run in record["runs"].items():
                tally = metrics[key]
                tally["attempted"] += 1
                tally[run["status"]] += 1
                tally["withheld_cases"] += run["withheld_components"] > 0
                tally["symbolic_cases"] += run.get("symbolic", False)
                tally["electron_mismatch_cases"] += run.get("electron_mismatches", 0) > 0
                if "first_application_exact" in run:
                    tally["first_application_exact"] += run["first_application_exact"]
                    tally["full_search_retries"] += run.get("full_search_retry", False)
                    tally["nonexhaustive_returned_searches"] += run.get("search_complete") is False
                for metric in METRICS:
                    tally[metric] += run[metric]
                if run["withheld_components"] > 0:
                    tally["withheld_exact"] += run["exact_set"]
                    tally["withheld_largest_gt"] += run["largest_gt_recovered"]
                    tally["withheld_max_fragment"] += run["max_fragment_match"]
                    tally["withheld_symbolic"] += run.get("symbolic", False)
                if run["status"] != "complete":
                    failures.append({"id": record["r_id"], "run": key,
                                     "type": run.get("error_type"), "error": run.get("error")})
                elif not run["exact_set"] and "/full/" in key:
                    misses.append({"id": record["r_id"], "run": key,
                                   "mappings": run["mapping_count"],
                                   "outcome_count": run["outcome_count"],
                                   "largest_gt_recovered": run["largest_gt_recovered"],
                                   "outcome_sample": run["outcomes"]})
    assert rows == summary["source_rows"]
    assert admitted == summary["extracted"] and with_ports == summary["with_ports"]
    assert dict(refusals) == summary["extraction_refusals"]
    assert comparison == Counter(summary["backend_comparison"])
    for key, expected in summary["metrics"].items():
        for name, value in expected.items():
            if name != "source_rows":
                assert metrics[key][name] == value, (key, name, value, metrics[key][name])
    result = {
        "source_rows": rows, "admitted": admitted, "with_ports": with_ports,
        "extraction_refusals": dict(refusals),
        "metrics": {key: dict(value) for key, value in sorted(metrics.items())},
        "comparison": dict(comparison), "execution_failures": failures,
        "full_input_nonexact": misses, "refusal_examples": dict(refusal_examples),
        "summary_sha256": digest(directory / "summary.json"),
        "records_sha256": digest(directory / "records.jsonl.gz"),
        "settings": summary["settings"],
        "analysis_script_sha256": digest(__file__),
    }
    (directory / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    fields = ("run", "attempted", "complete", *METRICS, "withheld_cases",
              "withheld_exact", "withheld_largest_gt", "withheld_symbolic",
              "symbolic_cases", "electron_mismatch_cases", "timeout", "error", "native_refusal")
    fields += ("first_application_exact", "full_search_retries", "nonexhaustive_returned_searches")
    with (directory / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, value in sorted(metrics.items()):
            writer.writerow({"run": key, **{name: value[name] for name in fields[1:]}})
    lines = ["Wildcard extraction and bidirectional replay", "===========================================", "",
             f"Selected source records: {rows:,}.", f"Admitted extractions: {admitted:,}; with wildcard ports: {with_ports:,}.", "",
             "Extraction refusals:", ""]
    if summary["settings"].get("recovery_only"):
        lines[6:6] = ["Native recovery audit: return the first valid full-source application.",
            "If its endpoint misses GT, retry full native enumeration within the same budget.",
            "GT is used only by the scorer; the C++ matcher receives no GT target.",
            "A returned witness proves recovery, not exhaustive enumeration or prediction.",
            "All recorded inputs are supplied; this does not test withholding input components.", ""]
    lines += [f"* {key}: {value:,}" for key, value in refusals.most_common()]
    lines += ["", "Complete recorded input (exact whole endpoint / admitted):", ""]
    for key in sorted(metrics):
        if "/full/" in key:
            tally = metrics[key]
            lines.append(f"* {key}: {tally['exact_set']:,}/{admitted:,}; completed {tally['complete']:,}; largest GT {tally['largest_gt_recovered']:,}.")
    lines += ["", "Largest-component input, restricted to cases with actual withholding:", ""]
    for key in sorted(metrics):
        if "/partial_largest/" in key:
            tally = metrics[key]
            lines.append(f"* {key}: {tally['withheld_exact']:,}/{tally['withheld_cases']:,} exact; largest GT {tally['withheld_largest_gt']:,}; symbolic {tally['withheld_symbolic']:,}.")
    lines += ["", f"Backend comparison: {dict(comparison)}.",
              f"Execution failures: {len(failures):,}; completed full-input nonexact runs: {len(misses):,}.", "",
              "Scores are unranked self-replay with atom maps removed from hosts and identities.",
              "Refusals remain in the source denominator. Unresolved wildcard predicates are",
              "not expanded into concrete molecules for recovery scores. See analysis.json for",
              "all execution-failure and completed full-input miss identifiers.", ""]
    if summary["settings"].get("recovery_only"):
        lines += ["Witness search counts:", ""]
        for key, tally in sorted(metrics.items()):
            lines.append(f"* {key}: {tally['first_application_exact']:,} exact first applications; "
                         f"{tally['full_search_retries']:,} full native retries; "
                         f"{tally['nonexhaustive_returned_searches']:,} nonexhaustive searches returned.")
        lines += ["", "Completion counts refer to returned executions. Search exhaustiveness is stored",
                  "separately per record. No failure or search limit proves absence of a GT witness.", ""]
    (directory / "report.rst").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"execution_failures", "full_input_nonexact", "refusal_examples"}}, indent=2))


if __name__ == "__main__":
    main()
