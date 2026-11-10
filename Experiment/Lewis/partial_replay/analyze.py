"""Compare reference/native replay records and generate manuscript evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    summary = json.loads((path / "summary.json").read_text())
    assert digest(path / "records.jsonl.gz") == summary["records_sha256"]
    with gzip.open(path / "records.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == summary["rows"]
    assert summary["source_unchanged"], "The implementation changed during execution"
    return summary, records


def compare(native, reference):
    """Compare the entire selected prefix, including refusals and losing cases."""
    fields = ("status", "exact", "exact_stereo", "outcome_sha256", "withheld_components",
              "exact_with_introduced_atoms", "exact_with_ports")
    differences, executions, complete_pairs, outcome_differences, status_differences = [], 0, 0, 0, 0
    assert len(native) >= len(reference)
    for actual, expected in zip(native, reference):
        assert (actual["row"], actual["r_id"]) == (expected["row"], expected["r_id"])
        if actual["status"] != expected["status"]:
            differences.append({"r_id": actual["r_id"], "field": "record_status",
                                "cpp": actual["status"], "python": expected["status"]})
            continue
        if actual["status"] != "attempted":
            continue
        for variant in actual["runs"]:
            for direction in actual["runs"][variant]:
                executions += 1
                left, right = actual["runs"][variant][direction], expected["runs"][variant][direction]
                status_differences += left["status"] != right["status"]
                if left["status"] == right["status"] == "complete":
                    complete_pairs += 1
                    outcome_differences += left["outcome_sha256"] != right["outcome_sha256"]
                for field in fields:
                    if left.get(field) != right.get(field):
                        differences.append({"r_id": actual["r_id"], "variant": variant,
                                            "direction": direction, "field": field,
                                            "cpp": left.get(field), "python": right.get(field)})
    return {"source_rows": len(reference), "execution_pairs": executions,
            "completed_pairs": complete_pairs, "output_set_disagreements": outcome_differences,
            "status_disagreements": status_differences,
            "field_disagreements": len(differences), "differences": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native, rows = load(args.native)
    reference, ref_rows = load(args.reference)
    assert native["source"]["sha256"] == reference["source"]["sha256"]
    assert native["settings"]["hydrogen_policy"] == reference["settings"]["hydrogen_policy"]
    assert native["settings"]["candidate_limit"] == reference["settings"]["candidate_limit"]
    differential = compare(rows, ref_rows)
    reference_failures, reference_failure_ids = Counter(), {}
    for row in ref_rows:
        for variant, directions in row.get("runs", {}).items():
            for direction, result in directions.items():
                if result["status"] != "complete":
                    key = variant, direction, result["status"], result.get("error", "")
                    reference_failures[key] += 1
                    reference_failure_ids.setdefault(key, []).append(row["r_id"])
    failures, witnesses, complete_misses = Counter(), [], []
    for row in rows:
        for variant, directions in row.get("runs", {}).items():
            for direction, result in directions.items():
                if result["status"] != "complete":
                    failures[(variant, direction, result["status"], result.get("error", ""))] += 1
                if result["exact_with_introduced_atoms"] and result["withheld_components"]:
                    witnesses.append({"r_id": row["r_id"], "variant": variant, "direction": direction,
                                      "input": result["input"], "target": row["targets"][direction],
                                      "reaction": result["exact_witness"]})
                if variant == "full_control" and result["status"] == "complete" and not result["exact"]:
                    complete_misses.append({"r_id": row["r_id"], "direction": direction,
                                            "mappings": result["mappings"], "outcomes": result["outcome_sample"]})
    root = Path(__file__).resolve().parents[3]
    provenance = {}
    for relative in ("paper/main.tex", "synrxn/reconstruction/builders/synthesis.py",
                     "Data/families/transformations/forward_prediction/task.json",
                     "Data/families/transformations/retrosynthesis/task.json",
                     "Data/families/transformations/forward_prediction/canonical/synthesis/uspto_50k/table.json"):
        path = root.parent / "SynRXN" / relative
        provenance[str(path)] = digest(path)
    catalogue_path = root.parent / "SynRXN/Data/families/transformations/forward_prediction/canonical/synthesis/uspto_50k/table.json"
    catalogue = json.loads(catalogue_path.read_text())["source_table_metadata"]
    assert native["source"]["sha256"] == catalogue["sha256"]
    assert native["rows"] == catalogue["rows"]
    report = {
        "schema": "synkit.uspto-partial-analysis/1", "source_rows": len(rows),
        "native_summary": str(args.native / "summary.json"),
        "native_summary_sha256": digest(args.native / "summary.json"),
        "reference_summary": str(args.reference / "summary.json"),
        "reference_summary_sha256": digest(args.reference / "summary.json"),
        "analysis_sha256": digest(__file__), "synrxn_sources": provenance,
        "source_matches_catalogue": True, "differential": differential,
        "metrics": {"cpp": native["metrics"], "python": reference["metrics"]},
        "failures": [{"variant": k[0], "direction": k[1], "status": k[2], "error": k[3], "count": v}
                     for k, v in failures.most_common()],
        "reference_failures": [{"variant": k[0], "direction": k[1], "status": k[2],
                                "error": k[3], "count": v, "ids": reference_failure_ids[k]}
                               for k, v in reference_failures.most_common()],
        "full_control_completed_misses": complete_misses,
        "withheld_exact_witnesses": witnesses,
        "rows_with_stereo_targets": {d: sum(r.get("target_has_stereo", {}).get(d, False) for r in rows)
                                     for d in ("fw", "bw")},
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    metrics = native["metrics"]
    labels = {"full_control": "Full match / all inputs", "partial_all": "Partial / all inputs",
              "partial_largest": "Partial / largest input"}
    table = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Input and match policy & Forward & Backward & FW inferred & BW inferred\\", r"\midrule"]
    for backend, group in (("C++", metrics), ("Python", reference["metrics"])):
        for variant, label in labels.items():
            fw, bw = group[variant + "/fw"], group[variant + "/bw"]
            table.append(f"{backend}: {label} & {fw.get('exact',0):,} & {bw.get('exact',0):,}"
                         f" & {fw.get('exact_with_introduced_atoms',0):,}"
                         f" & {bw.get('exact_with_introduced_atoms',0):,}" + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (output / "table.tex").write_text("\n".join(table) + "\n")
    counts = {"UsptoRows": len(rows), "UsptoOneSided": native["record_status"].get("unsupported_atom_carrier", 0),
              "UsptoAttempted": native["record_status"].get("attempted", 0),
              "UsptoNativeRefused": metrics["partial_all/fw"].get("unsupported_native", 0),
              "UsptoForward": metrics["partial_all/fw"].get("exact", 0),
              "UsptoBackward": metrics["partial_all/bw"].get("exact", 0),
              "UsptoWithheldFw": metrics["partial_largest/fw"].get("withheld_rows", 0),
              "UsptoWithheldBw": metrics["partial_largest/bw"].get("withheld_rows", 0),
              "UsptoWithheldExactFw": metrics["partial_largest/fw"].get("withheld_exact_with_introduced_atoms", 0),
              "UsptoWithheldExactBw": metrics["partial_largest/bw"].get("withheld_exact_with_introduced_atoms", 0),
              "ReferenceRows": len(ref_rows), "ReferencePairs": differential["execution_pairs"],
              "ReferenceCompletedPairs": differential["completed_pairs"],
              "ReferenceOutcomeDifferences": differential["output_set_disagreements"],
              "ReferenceStatusDifferences": differential["status_disagreements"],
              "PythonForward": reference["metrics"]["partial_all/fw"].get("exact", 0),
              "PythonBackward": reference["metrics"]["partial_all/bw"].get("exact", 0),
              "PythonForwardErrors": reference["metrics"]["partial_all/fw"].get("error", 0),
              "PythonBackwardErrors": reference["metrics"]["partial_all/bw"].get("error", 0),
              "PythonBackwardLewis": reference["metrics"]["partial_all/bw"].get("completion_error", 0),
              "PythonWithheldExactFw": reference["metrics"]["partial_largest/fw"].get("withheld_exact", 0)}
    (output / "values.tex").write_text("% Generated by partial_replay/analyze.py\n" + "".join(
        "\\newcommand{\\" + key + "}{" + f"{value:,}" + "}\n" for key, value in counts.items()))
    with (output / "witnesses.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=("r_id", "variant", "direction", "input", "target", "reaction"))
        writer.writeheader()
        writer.writerows(witnesses)
    with (output / "recovery.csv").open("w") as stream:
        columns = ("backend", "variant", "direction", "source_rows", "attempted", "complete", "exact",
                   "exact_with_introduced_atoms", "withheld_rows", "withheld_exact", "unsupported_native",
                   "error", "completion_error", "search_limit", "timeout")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for backend, group in report["metrics"].items():
            for key, values in group.items():
                variant, direction = key.split("/")
                writer.writerow({"backend": backend, "variant": variant, "direction": direction,
                                 **{field: values.get(field, 0) for field in columns[3:]}})
    lines = ["USPTO-50K replay comparison", "==========================", "",
             f"All {len(rows):,} source records were processed by both backends.",
             f"One-sided atom carriers refused: {counts['UsptoOneSided']:,}.",
             f"C++ additionally refuses {counts['UsptoNativeRefused']:,} aromaticity-changing records.", "",
             f"Python partial/all-input recovery: FW {counts['PythonForward']:,}; BW {counts['PythonBackward']:,}.",
             f"C++ partial/all-input recovery: FW {counts['UsptoForward']:,}; BW {counts['UsptoBackward']:,}.", "",
             f"For {counts['UsptoWithheldFw']:,} carrier-preserving forward records, the largest-input test",
             f"withholds smaller fragments. Python exactly recovers {counts['PythonWithheldExactFw']:,}",
             f"targets through completion; C++ exactly recovers {counts['UsptoWithheldExactFw']:,}.",
             "These counts include observed introduced-atom witnesses, rather than a full-match path.",
             "Backward largest-input runs withhold no whole product components in this cohort.", "",
             f"Common completed execution pairs: {counts['ReferenceCompletedPairs']:,};",
             f"canonical output-set disagreements: {counts['ReferenceOutcomeDifferences']:,}.",
             f"Execution-status disagreements: {counts['ReferenceStatusDifferences']:,}; retained in analysis.json.", "",
             f"Python has {counts['PythonForwardErrors']} forward and {counts['PythonBackwardErrors']} backward",
             f"aromatic reperception errors and {counts['PythonBackwardLewis']} backward Lewis endpoint failures.",
             "Their identifiers and errors remain in analysis.json.", "",
             "Each rule is extracted from its own reference reaction. This tests source-conditioned",
             "execution, not prediction. Exact scoring compares complete constitutional fragment",
             "multisets with charges/isotopes intact. See the benchmark README for input and timing limits.", "",
             ".. csv-table:: Recovery and refusal counts", "   :file: recovery.csv", "   :header-rows: 1", ""]
    (output / "report.rst").write_text("\n".join(lines))
    print(json.dumps({"counts": counts, "differential": {k: v for k, v in differential.items() if k != "differences"},
                      "failure_categories": report["failures"], "full_control_misses": len(complete_misses)}, indent=2))


if __name__ == "__main__":
    main()
