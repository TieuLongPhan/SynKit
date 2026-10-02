"""Compare frozen CP-off and separator-spectrum 100-reaction runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def mapping_set(run, key):
    return {
        tuple(mapping)
        for mapping in json.loads((run / f"{key}.synister_cp.maps.json").read_text())
    }


def compare_mode(control, spectrum, mode):
    control_protocol = read_json(control / "protocol.json")
    spectrum_protocol = read_json(spectrum / "protocol.json")
    expected = {
        "target_mode": mode,
        "cohort_size": 100,
        "seconds_per_method": 10.0,
        "inputs_sha256": control_protocol["inputs_sha256"],
        "prepared_seeds_sha256": control_protocol["prepared_seeds_sha256"],
        "specific_cd_target_manifest_sha256": control_protocol.get(
            "specific_cd_target_manifest_sha256"
        ),
    }
    for protocol in (control_protocol, spectrum_protocol):
        for key, value in expected.items():
            if protocol.get(key) != value:
                raise ValueError(f"{key} mismatch in {mode} run: {protocol.get(key)!r}")
    if control_protocol.get("cp_separator_spectrum") is not False:
        raise ValueError("control run unexpectedly enables separator spectrum")
    if spectrum_protocol.get("cp_separator_spectrum") is not True:
        raise ValueError("spectrum run does not enable separator spectrum")
    if control_protocol["source_sha256"] != spectrum_protocol["source_sha256"]:
        raise ValueError("control and treatment source snapshots differ")
    for key in (
        "seconds_per_method",
        "workers",
        "cp_branch_order",
        "cp_pairwise_edge_bounds",
        "cp_reactant_symmetry",
        "cp_factor_spectrum_bounds",
        "cp_suffix_spectrum",
        "mapping_cap",
        "cpu_affinity",
        "thread_environment",
        "single_thread",
    ):
        if control_protocol.get(key) != spectrum_protocol.get(key):
            raise ValueError(f"protocol mismatch outside separator option: {key}")
    for run in (control, spectrum):
        audit = read_json(run / "audit.json")
        if not audit["all_recorded_outputs_consistent"]:
            raise ValueError(f"independent output audit failed: {run}")
        if (
            audit["paired_solver_records"] != 100
            or audit["preparation_or_worker_errors"]
        ):
            raise ValueError(f"incomplete benchmark audit: {run}")

    control_records = {
        record["benchmark_id"]: record
        for record in (
            read_json(path) for path in control.glob("reaction_*.result.json")
        )
    }
    spectrum_records = {
        record["benchmark_id"]: record
        for record in (
            read_json(path) for path in spectrum.glob("reaction_*.result.json")
        )
    }
    if set(control_records) != set(spectrum_records) or len(control_records) != 100:
        raise ValueError("control and treatment do not contain the same 100 reactions")

    rows = []
    complete_pairs = []
    equal_complete = 0
    for key in sorted(control_records):
        base = control_records[key]["methods"]["synister_cp"]
        trial = spectrum_records[key]["methods"]["synister_cp"]
        if base["target_mode"] != mode or trial["target_mode"] != mode:
            raise ValueError(f"wrong target mode in record {key}")
        if (
            mode == "specific_cd"
            and base["target_doubled_cd"] != trial["target_doubled_cd"]
        ):
            raise ValueError(f"specific-CD target mismatch in {key}")
        if (
            mode == "minimal"
            and base["complete"]
            and trial["complete"]
            and base["minimum_doubled_cd"] != trial["minimum_doubled_cd"]
        ):
            raise ValueError(f"complete minimum costs differ in {key}")
        both_complete = bool(base["complete"] and trial["complete"])
        same_maps = None
        if both_complete:
            base_maps = mapping_set(control, key)
            trial_maps = mapping_set(spectrum, key)
            same_maps = base_maps == trial_maps
            if not same_maps:
                raise ValueError(f"complete mapping sets differ for {mode} {key}")
            equal_complete += 1
            complete_pairs.append((float(base["seconds"]), float(trial["seconds"])))
        base_seconds = float(base["seconds"])
        trial_seconds = float(trial["seconds"])
        search = trial.get("statistics", {}).get("search", {})
        rows.append(
            {
                "mode": mode,
                "reaction": key,
                "target_doubled_cd": base.get("target_doubled_cd"),
                "control_complete": base["complete"],
                "separator_complete": trial["complete"],
                "control_termination": base["termination"],
                "separator_termination": trial["termination"],
                "control_minimum_doubled_cd": base["minimum_doubled_cd"],
                "separator_minimum_doubled_cd": trial["minimum_doubled_cd"],
                "control_seconds": base_seconds,
                "separator_seconds": trial_seconds,
                "control_over_separator_speed_ratio": (
                    base_seconds / trial_seconds if trial_seconds else None
                ),
                "control_mapping_count": base["mapping_count"],
                "separator_mapping_count": trial["mapping_count"],
                "complete_mapping_sets_equal": same_maps,
                "separator_calls": search.get("separator_spectrum_calls", 0),
                "separator_skipped": search.get("separator_spectrum_skipped", 0),
                "separator_states": search.get("separator_spectrum_states", 0),
                "separator_search_seconds": search.get(
                    "separator_spectrum_seconds", 0.0
                ),
            }
        )

    ratios = [base / trial for base, trial in complete_pairs if trial > 0]
    summary = {
        "mode": mode,
        "cases": 100,
        "control_complete": sum(row["control_complete"] for row in rows),
        "separator_complete": sum(row["separator_complete"] for row in rows),
        "control_minimum_proved": (
            sum(row["control_minimum_doubled_cd"] is not None for row in rows)
            if mode == "minimal"
            else None
        ),
        "separator_minimum_proved": (
            sum(row["separator_minimum_doubled_cd"] is not None for row in rows)
            if mode == "minimal"
            else None
        ),
        "jointly_complete": len(complete_pairs),
        "jointly_complete_mapping_sets_equal": equal_complete,
        "separator_faster_on_jointly_complete": sum(ratio > 1 for ratio in ratios),
        "control_over_separator_median_seconds_ratio": (
            statistics.median(ratios) if ratios else None
        ),
        "control_seconds_sum_jointly_complete": sum(a for a, _ in complete_pairs),
        "separator_seconds_sum_jointly_complete": sum(b for _, b in complete_pairs),
        "separator_total_calls": sum(row["separator_calls"] for row in rows),
        "separator_total_states": sum(row["separator_states"] for row in rows),
        "separator_total_search_seconds": sum(
            row["separator_search_seconds"] for row in rows
        ),
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimal-control", type=Path, required=True)
    parser.add_argument("--minimal-spectrum", type=Path, required=True)
    parser.add_argument("--specific-control", type=Path, required=True)
    parser.add_argument("--specific-spectrum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = [
        compare_mode(args.minimal_control, args.minimal_spectrum, "minimal"),
        compare_mode(args.specific_control, args.specific_spectrum, "specific_cd"),
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = [row for _, rows in results for row in rows]
    columns = list(all_rows[0])
    with (args.output / "per_reaction.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    summaries = {item["mode"]: item for item, _ in results}
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    lines = [
        "# Separator-spectrum paired comparison",
        "",
        "The control and treatment use the same frozen code snapshot, 100 inputs,",
        "saved seeds and (in specific-CD mode) frozen targets, a 10-second per-method",
        "cap. Mapping outputs were independently audited. Ratios are control time",
        "divided by separator time; values above 1 favor separator-spectrum.",
        "",
    ]
    for summary, rows in results:
        lines.extend(
            [
                f"## {summary['mode']}",
                "",
                "| Reaction | Control | Separator | Base s | Sep s | Base/Sep | Base maps | Sep maps | Sets equal | Calls | States | Sep s* |",
                "|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            ratio = row["control_over_separator_speed_ratio"]
            equal = row["complete_mapping_sets_equal"]
            lines.append(
                "| {reaction} | {control_complete} | {separator_complete} | "
                "{control_seconds:.3f} | {separator_seconds:.3f} | {ratio} | "
                "{control_mapping_count} | {separator_mapping_count} | {equal} | "
                "{separator_calls} | {separator_states} | {separator_search_seconds:.3f} |".format(
                    **row,
                    ratio="" if ratio is None else f"{ratio:.3f}",
                    equal="" if equal is None else str(equal),
                )
            )
        lines.extend(
            [
                "",
                f"Summary: separator completed {summary['separator_complete']}/100; "
                f"control completed {summary['control_complete']}/100; "
                f"jointly complete {summary['jointly_complete']}/100; "
                f"median control/separator ratio "
                f"{summary['control_over_separator_median_seconds_ratio']:.3f}; "
                f"equal complete map sets "
                f"{summary['jointly_complete_mapping_sets_equal']}/"
                f"{summary['jointly_complete']}; separator calls/states/search seconds "
                f"{summary['separator_total_calls']}/"
                f"{summary['separator_total_states']}/"
                f"{summary['separator_total_search_seconds']:.3f}.",
                "",
            ]
        )
    (args.output / "per_reaction.md").write_text("\n".join(lines))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
