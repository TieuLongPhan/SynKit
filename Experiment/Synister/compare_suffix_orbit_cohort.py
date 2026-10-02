"""Compare late and incremental product-orbit filtering in suffix spectra."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def _mapping_set(run, benchmark_id):
    return {
        tuple(mapping)
        for mapping in load(run / f"{benchmark_id}.synister_cp.maps.json")
    }


def compare(control, treatment, mode):
    before, after = load(control / "protocol.json"), load(treatment / "protocol.json")
    for protocol in (before, after):
        if (
            protocol.get("target_mode") != mode
            or protocol.get("cohort_size") != 100
            or protocol.get("seconds_per_method") != 10.0
            or protocol.get("cp_suffix_spectrum") is not True
        ):
            raise ValueError(f"unexpected {mode} benchmark protocol")
    if before.get("cp_suffix_spectrum_orbit_pruning") is not False:
        raise ValueError("control must use late suffix orbit filtering")
    if after.get("cp_suffix_spectrum_orbit_pruning") is not True:
        raise ValueError("treatment must use incremental suffix orbit filtering")
    for key in (
        "inputs_sha256",
        "prepared_seeds_sha256",
        "specific_cd_target_manifest_sha256",
        "source_sha256",
        "mapping_cap",
        "workers",
        "cpu_affinity",
        "thread_environment",
    ):
        if before.get(key) != after.get(key):
            raise ValueError(f"paired protocol mismatch: {key}")

    base_records = {
        item["benchmark_id"]: item
        for item in (load(p) for p in control.glob("reaction_*.result.json"))
    }
    new_records = {
        item["benchmark_id"]: item
        for item in (load(p) for p in treatment.glob("reaction_*.result.json"))
    }
    if len(base_records) != 100 or base_records.keys() != new_records.keys():
        raise ValueError("paired runs must contain the same 100 reactions")

    rows, complete_pairs = [], []
    equal_complete = 0
    for benchmark_id in sorted(base_records):
        base = base_records[benchmark_id]["methods"]["synister_cp"]
        new = new_records[benchmark_id]["methods"]["synister_cp"]
        if base["target_mode"] != mode or new["target_mode"] != mode:
            raise ValueError(f"wrong target mode for {benchmark_id}")
        if mode == "specific_cd" and base["target_doubled_cd"] != new["target_doubled_cd"]:
            raise ValueError(f"specific CD mismatch for {benchmark_id}")
        if mode == "minimal" and base["complete"] and new["complete"]:
            if base["minimum_doubled_cd"] != new["minimum_doubled_cd"]:
                raise ValueError(f"minimum cost mismatch for {benchmark_id}")
        both_complete = bool(base["complete"] and new["complete"])
        equal = None
        if both_complete:
            equal = _mapping_set(control, benchmark_id) == _mapping_set(
                treatment, benchmark_id
            )
            if not equal:
                raise ValueError(f"complete mapping sets differ for {benchmark_id}")
            equal_complete += 1
            complete_pairs.append((base["seconds"], new["seconds"]))
        base_stats = base.get("statistics", {}).get("search", {})
        new_stats = new.get("statistics", {}).get("search", {})
        ratio = base["seconds"] / new["seconds"] if new["seconds"] else None
        rows.append(
            {
                "mode": mode,
                "reaction": benchmark_id,
                "control_complete": base["complete"],
                "treatment_complete": new["complete"],
                "control_termination": base["termination"],
                "treatment_termination": new["termination"],
                "control_seconds": base["seconds"],
                "treatment_seconds": new["seconds"],
                "control_over_treatment_ratio": ratio,
                "control_mapping_count": base["mapping_count"],
                "treatment_mapping_count": new["mapping_count"],
                "complete_mapping_sets_equal": equal,
                "control_suffix_calls": base_stats.get("suffix_spectrum_calls", 0),
                "treatment_suffix_calls": new_stats.get("suffix_spectrum_calls", 0),
                "control_late_orbits_pruned": base_stats.get(
                    "suffix_spectrum_orbits_pruned", 0
                ),
                "treatment_branches_pruned": new_stats.get(
                    "suffix_spectrum_orbit_branches_pruned", 0
                ),
            }
        )
    ratios = [base / new for base, new in complete_pairs if new > 0]
    summary = {
        "mode": mode,
        "cases": 100,
        "control_complete": sum(row["control_complete"] for row in rows),
        "treatment_complete": sum(row["treatment_complete"] for row in rows),
        "jointly_complete": len(complete_pairs),
        "jointly_complete_equal_mapping_sets": equal_complete,
        "treatment_faster_on_jointly_complete": sum(value > 1 for value in ratios),
        "median_control_over_treatment_ratio": statistics.median(ratios) if ratios else None,
        "control_seconds_sum_jointly_complete": sum(a for a, _ in complete_pairs),
        "treatment_seconds_sum_jointly_complete": sum(b for _, b in complete_pairs),
        "control_suffix_calls": sum(row["control_suffix_calls"] for row in rows),
        "treatment_suffix_calls": sum(row["treatment_suffix_calls"] for row in rows),
        "treatment_pruned_branches": sum(row["treatment_branches_pruned"] for row in rows),
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimal-control", type=Path, required=True)
    parser.add_argument("--minimal-treatment", type=Path, required=True)
    parser.add_argument("--specific-control", type=Path, required=True)
    parser.add_argument("--specific-treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [
        compare(args.minimal_control, args.minimal_treatment, "minimal"),
        compare(args.specific_control, args.specific_treatment, "specific_cd"),
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [row for _, group in results for row in group]
    with (args.output / "per_reaction.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = {summary["mode"]: summary for summary, _ in results}
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Incremental suffix orbit pruning benchmark",
        "",
        "Paired PABS runs on the frozen 100-reaction cohort with saved seeds, a",
        "10-second per-method limit, and a 100,000-map cap. Only suffix orbit",
        "filtering differs. Ratios above 1 favor incremental filtering.",
        "",
    ]
    for summary, group in results:
        lines.extend(
            [
                f"## {summary['mode']}",
                "",
                "| Reaction | Control done | Treatment done | Control s | Treatment s | Base/new | Maps equal | Base maps | New maps | Base orbits | New branches |",
                "|---|:---:|:---:|---:|---:|---:|:---:|---:|---:|---:|---:|",
            ]
        )
        for row in group:
            ratio = row["control_over_treatment_ratio"]
            lines.append(
                f"| {row['reaction']} | {row['control_complete']} | {row['treatment_complete']} | "
                f"{row['control_seconds']:.3f} | {row['treatment_seconds']:.3f} | "
                f"{'—' if ratio is None else f'{ratio:.3f}'} | "
                f"{'—' if row['complete_mapping_sets_equal'] is None else row['complete_mapping_sets_equal']} | "
                f"{row['control_mapping_count']} | {row['treatment_mapping_count']} | "
                f"{row['control_late_orbits_pruned']} | {row['treatment_branches_pruned']} |"
            )
        lines.extend(
            [
                "",
                f"Summary: control/treatment complete {summary['control_complete']}/"
                f"{summary['treatment_complete']}; jointly complete {summary['jointly_complete']}; "
                f"equal complete map sets {summary['jointly_complete_equal_mapping_sets']}; "
                f"median control/treatment ratio {summary['median_control_over_treatment_ratio']}; "
                f"suffix calls {summary['control_suffix_calls']}/"
                f"{summary['treatment_suffix_calls']}; branches pruned "
                f"{summary['treatment_pruned_branches']}.",
                "",
            ]
        )
    (args.output / "per_reaction.md").write_text("\n".join(lines))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
