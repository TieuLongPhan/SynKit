"""Frozen, repeated PABS/cost-table/reward-frontier development benchmark."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

from Experiment.Synister.benchmark_artifacts import (
    environment,
    freeze_source,
    save,
    verify_source,
)

ARMS = ("pabs", "cost_table", "reward_frontier")


def worker(task):
    os.sched_setaffinity(0, {task["cpu"]})
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    from Experiment.Synister.global_milp import doubled_distance
    from Experiment.Synister.mapping_check import check_mappings
    from synkit.Chem.Mapper.identifiability import parse_reaction
    from synkit.Chem.Mapper.exact.propagation import (
        PropagationConfig,
        enumerate_synister_cp_mappings,
    )

    output = Path(task["output"])
    checked = 0
    for item in task["rows"]:
        row, index = item["row"], item["index"]
        benchmark_id = row["benchmark_id"]
        r, p = parse_reaction(row["reaction"])
        graphs = [r.graph(), p.graph()]
        seed = task["seeds"][benchmark_id]["mapping"]
        target = task["targets"][benchmark_id]["target_doubled_cd"]
        if doubled_distance(r, p, seed) != target:
            raise ValueError("Frozen target failed literal seed rescoring")
        attempts, comparisons = [], []
        for mode_index, mode in enumerate(("minimal", "specific_cd")):
            for repetition in range(task["repeats"]):
                order = list(ARMS[repetition % 3 :] + ARMS[: repetition % 3])
                if (index + mode_index) % 2:
                    order.reverse()
                outputs = {}
                trial = {}
                for arm in order:
                    config = PropagationConfig(
                        suffix_spectrum=arm != "pabs",
                        suffix_spectrum_representation=(
                            "reward_frontier"
                            if arm == "reward_frontier"
                            else "cost_table"
                        ),
                        suffix_spectrum_orbit_pruning=False,
                    )
                    maps = []
                    started = time.perf_counter()
                    result = enumerate_synister_cp_mappings(
                        graphs,
                        CD="minimal" if mode == "minimal" else target / 2,
                        binary=False,
                        initial_mapping=seed,
                        max_bijections=None,
                        max_mappings=100000,
                        tolerance=0,
                        time_limit_seconds=task["seconds"],
                        compute_minimum_cost=mode == "minimal",
                        symmetry_pruning=True,
                        expand_symmetry=True,
                        symmetry_node_properties=("charges", "hcounts"),
                        collect_mappings=False,
                        mapping_callback=lambda mapping, cost: maps.append(
                            tuple(mapping)
                        ),
                        config=config,
                    )
                    elapsed = time.perf_counter() - started
                    expected = (
                        None
                        if result.minimum_cost is None
                        else int(2 * result.minimum_cost)
                    )
                    if mode == "specific_cd":
                        expected = target
                    if len(set(maps)) != len(maps):
                        raise ValueError("Duplicate expanded mapping")
                    count = check_mappings(r, p, maps, expected)
                    checked += count
                    outputs[arm] = set(maps)
                    payload = json.dumps(sorted(maps), separators=(",", ":")).encode()
                    key = f"{benchmark_id}.{mode}.{repetition}.{arm}"
                    with gzip.open(output / (key + ".maps.json.gz"), "wb") as stream:
                        stream.write(payload)
                    record = {
                        "reaction": benchmark_id,
                        "mode": mode,
                        "repeat": repetition,
                        "arm": arm,
                        "order": order,
                        "cpu": task["cpu"],
                        "complete": result.complete,
                        "termination": result.truncation_reason or result.status,
                        "minimum_doubled_cd": (
                            None
                            if result.minimum_cost is None
                            else int(2 * result.minimum_cost)
                        ),
                        "seconds": elapsed,
                        "mapping_count": len(maps),
                        "mapping_sha256": sha256(payload).hexdigest(),
                        "checked_maps": count,
                        "statistics": result.backend_statistics,
                    }
                    save(output / (key + ".result.json"), record)
                    attempts.append(record)
                    trial[arm] = record
                for left in ARMS[:-1]:
                    a, b = trial[left], trial["reward_frontier"]
                    equal = None
                    if a["complete"] and b["complete"]:
                        if a["minimum_doubled_cd"] != b["minimum_doubled_cd"]:
                            raise ValueError("Complete minimum costs disagree")
                        equal = outputs[left] == outputs["reward_frontier"]
                        if not equal:
                            raise ValueError("Complete expanded mapping sets disagree")
                    elif mode == "specific_cd" or (
                        a["minimum_doubled_cd"] is not None
                        and a["minimum_doubled_cd"] == b["minimum_doubled_cd"]
                    ):
                        if (
                            a["complete"]
                            and not outputs["reward_frontier"] <= outputs[left]
                        ):
                            raise ValueError(
                                "Partial reward output is outside complete reference"
                            )
                        if (
                            b["complete"]
                            and not outputs[left] <= outputs["reward_frontier"]
                        ):
                            raise ValueError(
                                "Partial reference output is outside complete reward output"
                            )
                    comparisons.append(
                        {
                            "mode": mode,
                            "repeat": repetition,
                            "baseline": left,
                            "both_complete": a["complete"] and b["complete"],
                            "equal": equal,
                        }
                    )
        save(
            output / (benchmark_id + ".case.json"),
            {
                "reaction": benchmark_id,
                "attempts": attempts,
                "comparisons": comparisons,
            },
        )
    return {"cpu": task["cpu"], "cases": len(task["rows"]), "checked_maps": checked}


def report(output, rows, repeats, seconds):
    cases = [
        json.loads((output / (row["benchmark_id"] + ".case.json")).read_text())
        for row in rows
    ]
    summaries, table = {}, []
    for mode in ("minimal", "specific_cd"):
        mode_attempts = [
            a for case in cases for a in case["attempts"] if a["mode"] == mode
        ]
        comparisons = [
            c for case in cases for c in case["comparisons"] if c["mode"] == mode
        ]
        summary = {
            "cases": len(rows),
            "repeats": repeats,
            "complete_attempts": {
                arm: sum(a["complete"] for a in mode_attempts if a["arm"] == arm)
                for arm in ARMS
            },
            "minimum_proved_attempts": {
                arm: sum(
                    a["minimum_doubled_cd"] is not None
                    for a in mode_attempts
                    if a["arm"] == arm
                )
                for arm in ARMS
            },
            "comparisons": {},
        }
        medians, all_complete = {}, {}
        for case in cases:
            key = case["reaction"]
            attempts = [a for a in case["attempts"] if a["mode"] == mode]
            medians[key] = {
                arm: statistics.median(
                    a["seconds"] for a in attempts if a["arm"] == arm
                )
                for arm in ARMS
            }
            all_complete[key] = {
                arm: all(a["complete"] for a in attempts if a["arm"] == arm)
                for arm in ARMS
            }
            row = {"mode": mode, "reaction": key}
            for arm in ARMS:
                selected = [a for a in attempts if a["arm"] == arm]
                row[arm + "_complete"] = sum(a["complete"] for a in selected)
                row[arm + "_median_seconds"] = medians[key][arm]
                row[arm + "_median_states"] = statistics.median(
                    a["statistics"]["search"].get("suffix_spectrum_states", 0)
                    for a in selected
                )
                row[arm + "_median_prepare_seconds"] = statistics.median(
                    a["statistics"]["search"].get("suffix_spectrum_prepare_seconds", 0)
                    for a in selected
                )
                row[arm + "_median_reconstruction_seconds"] = statistics.median(
                    a["statistics"]["search"].get(
                        "suffix_spectrum_reconstruction_seconds", 0
                    )
                    for a in selected
                )
            table.append(row)
        for baseline in ARMS[:-1]:
            joint = [
                key
                for key in medians
                if all_complete[key][baseline] and all_complete[key]["reward_frontier"]
            ]
            ratios = [
                medians[key][baseline] / medians[key]["reward_frontier"]
                for key in joint
            ]
            compare = [c for c in comparisons if c["baseline"] == baseline]
            summary["comparisons"][baseline] = {
                "jointly_complete_all_repeats_cases": len(joint),
                "jointly_complete_trials": sum(c["both_complete"] for c in compare),
                "equal_complete_mapping_sets": sum(c["equal"] is True for c in compare),
                "median_baseline_over_reward_ratio": (
                    statistics.median(ratios) if ratios else None
                ),
                "reward_faster_cases": sum(ratio > 1 for ratio in ratios),
                "baseline_sum_median_seconds_joint": sum(
                    medians[key][baseline] for key in joint
                ),
                "reward_sum_median_seconds_joint": sum(
                    medians[key]["reward_frontier"] for key in joint
                ),
            }
        summaries[mode] = summary
    with (output / "per_reaction.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = [
        "# Reward frontier repeated development benchmark",
        "",
        f"{repeats} counterbalanced repeats; {seconds:g} seconds per query and a 100,000-map cap.",
        "Each lane uses one distinct CPU and single-thread numerical libraries.",
        "Literal output checking occurs outside the timed solver calls.",
        "",
        "| Mode | Reaction | PABS complete | Cost complete | Reward complete | PABS median s | Cost median s | Reward median s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['mode']} | {row['reaction']} | {row['pabs_complete']}/{repeats} | {row['cost_table_complete']}/{repeats} | {row['reward_frontier_complete']}/{repeats} | {row['pabs_median_seconds']:.4f} | {row['cost_table_median_seconds']:.4f} | {row['reward_frontier_median_seconds']:.4f} |"
        )
    (output / "per_reaction.md").write_text("\n".join(lines) + "\n")
    save(output / "summary.json", summaries)
    return summaries


def run(output, repeats, seconds):
    root = Path(__file__).resolve().parents[2]
    all_rows = json.loads(
        (root / "paper/synister/evidence/enumeration_main_v1/inputs.json").read_text()
    )
    selected = []
    groups = sorted({(r["source"], r["size_bin"]) for r in all_rows})
    for source, size in groups:
        bucket = [r for r in all_rows if (r["source"], r["size_bin"]) == (source, size)]
        selected.extend(
            sorted(bucket, key=lambda r: sha256(r["reaction"].encode()).hexdigest())[:2]
        )
    diagnostic_ids = {"reaction_038", "reaction_082", "reaction_093", "reaction_099"}
    selected_ids = {r["benchmark_id"] for r in selected}
    selected.extend(
        r for r in all_rows if r["benchmark_id"] in diagnostic_ids - selected_ids
    )
    selected.sort(key=lambda r: r["benchmark_id"])
    seeds_path = (
        root
        / "Experiment/Synister/runs/separator_spectrum_100_minimal_v1/prepared_seeds.json"
    )
    targets_path = (
        root
        / "Experiment/Synister/runs/separator_spectrum_100_specific_v1/specific_cd_target_manifest.json"
    )
    seeds = json.loads(seeds_path.read_text())
    targets = json.loads(targets_path.read_text())["targets"]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "frozen_source"
    hashes = freeze_source(root, source)
    cpus = sorted(os.sched_getaffinity(0))[:4]
    save(output / "environment.json", environment())
    save(output / "inputs.json", selected)
    save(
        output / "protocol.json",
        {
            "selection": "Two smallest reaction hashes per source/size bin, plus 038/082/093/099 diagnostics; no outcome selection",
            "cases": len(selected),
            "diagnostic_ids": sorted(diagnostic_ids),
            "seconds_per_query": seconds,
            "repeats": repeats,
            "cpus": cpus,
            "arms": ARMS,
            "mapping_cap": 100000,
            "seeds_sha256": sha256(seeds_path.read_bytes()).hexdigest(),
            "targets_sha256": sha256(targets_path.read_bytes()).hexdigest(),
            "source_sha256": hashes,
            "suffix_max_calls": 2,
            "suffix_max_states": 20000,
            "suffix_max_seconds_per_call": 0.002,
            "suffix_max_reward": 4096,
            "order": "rotated across repeats, reversed by case/mode parity",
        },
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(source),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )

    def lane(lane_index):
        task = {
            "output": str(output),
            "cpu": cpus[lane_index],
            "seconds": seconds,
            "repeats": repeats,
            "rows": [
                {"row": row, "index": index}
                for index, row in enumerate(selected)
                if index % len(cpus) == lane_index
            ],
            "seeds": seeds,
            "targets": targets,
        }
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "Experiment.Synister.reward_frontier_benchmark",
                "--worker",
            ],
            input=json.dumps(task),
            capture_output=True,
            text=True,
            cwd=source,
            env=env,
        )
        (output / f"lane_{lane_index}.log").write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(
                f"Benchmark lane {lane_index} failed: {process.stderr[-2000:]}"
            )
        result = json.loads(process.stdout)
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        results = list(pool.map(lane, range(len(cpus))))
    verify_source(source, hashes)
    summary = report(output, selected, repeats, seconds)
    save(
        output / "audit.json",
        {
            "all_checked_outputs_consistent": True,
            "independently_rescored_maps": sum(r["checked_maps"] for r in results),
            "case_records": len(selected),
            "attempts": len(selected) * 2 * repeats * len(ARMS),
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))), flush=True)
    else:
        if args.output is None or args.repeats < 1 or args.seconds <= 0:
            parser.error("output is required and repeat/time budgets must be positive")
        run(args.output, args.repeats, args.seconds)
