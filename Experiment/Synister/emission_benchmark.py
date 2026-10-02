"""Paired output benchmark isolating the product-only emission implementation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
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

ARMS = ("control", "fast")


def worker(task):
    os.sched_setaffinity(0, {task["cpu"]})
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    from Experiment.Synister.global_milp import doubled_distance
    from Experiment.Synister.mapping_check import check_mappings
    from synkit.Chem.Mapper.identifiability import parse_reaction
    from synkit.Chem.Mapper.exact.propagation import enumerate_synister_cp_mappings
    import synkit.Chem.Mapper.exact.propagation as propagation
    from synkit.Chem.Mapper.exact.propagation_search import PropagatedSearch

    output = Path(task["output"])
    spec = importlib.util.spec_from_file_location(
        "synkit.Chem.Mapper.exact._emission_control",
        output / "control_propagation_search.py",
    )
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    methods = {
        "control": control.PropagatedSearch.accept,
        "fast": PropagatedSearch.accept,
    }
    original_initialize = PropagatedSearch.__init__
    captured_group = []

    def initialize(search, *args, **kwargs):
        original_initialize(search, *args, **kwargs)
        captured_group[:] = [search.group]

    PropagatedSearch.__init__ = initialize
    original_automorphisms = propagation.bounded_automorphism_permutations
    original_subgroup = propagation.bounded_generated_subgroup
    checked = 0
    for item in task["rows"]:
        row, index = item["row"], item["index"]
        key = row["benchmark_id"]
        r, p = parse_reaction(row["reaction"])
        graphs = [r.graph(), p.graph()]
        if task["freeze_symmetry"]:
            discovery_started = time.perf_counter()
            permutations, discovered = original_automorphisms(
                graphs[1],
                False,
                limit=256,
                timeout_seconds=0.2,
                max_search_nodes=10000,
                node_properties=("charges", "hcounts"),
            )
            group = original_subgroup(
                permutations, max_order=256, deadline=discovery_started + 0.25
            )
            propagation.bounded_automorphism_permutations = lambda *args, **kwargs: (
                permutations,
                discovered,
            )
            propagation.bounded_generated_subgroup = lambda *args, **kwargs: group
            save(
                output / (key + ".symmetry.json"),
                {
                    "group": group,
                    "complete_discovery": discovered,
                    "discovery_seconds_outside_timed_calls": time.perf_counter()
                    - discovery_started,
                },
            )
        seed = task["seeds"][key]["mapping"]
        target = task["targets"][key]["target_doubled_cd"]
        if doubled_distance(r, p, seed) != target:
            raise ValueError("Frozen target differs from the literal seed distance")
        attempts, comparisons = [], []
        for mode_index, mode in enumerate(task["modes"]):
            for repeat in range(task["repeats"]):
                order = ARMS if (repeat + index + mode_index) % 2 == 0 else ARMS[::-1]
                trial, maps_by_arm = {}, {}
                for arm in order:
                    PropagatedSearch.accept = methods[arm]
                    captured_group.clear()
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
                        mapping_callback=lambda mapping, cost: maps.append(mapping),
                    )
                    elapsed = time.perf_counter() - started
                    minimum = (
                        None
                        if result.minimum_cost is None
                        else int(2 * result.minimum_cost)
                    )
                    expected = minimum if mode == "minimal" else target
                    if len(set(maps)) != len(maps):
                        raise ValueError("Duplicate mapping in expanded output")
                    checked += check_mappings(r, p, maps, expected)
                    payload = json.dumps(maps, separators=(",", ":")).encode()
                    name = f"{key}.{mode}.{repeat}.{arm}"
                    with gzip.open(output / (name + ".maps.json.gz"), "wb") as stream:
                        stream.write(payload)
                    record = {
                        "reaction": key,
                        "mode": mode,
                        "repeat": repeat,
                        "arm": arm,
                        "order": order,
                        "cpu": task["cpu"],
                        "seconds": elapsed,
                        "complete": result.complete,
                        "termination": result.truncation_reason or result.status,
                        "mapping_count": len(maps),
                        "minimum_doubled_cd": minimum,
                        "symmetry_group_order": result.symmetry_group_order,
                        "symmetry_group_sha256": (
                            sha256(json.dumps(captured_group[0]).encode()).hexdigest()
                            if captured_group
                            else None
                        ),
                        "ordered_mapping_sha256": sha256(payload).hexdigest(),
                        "statistics": result.backend_statistics,
                    }
                    save(output / (name + ".result.json"), record)
                    trial[arm], maps_by_arm[arm] = record, maps
                    attempts.append(record)
                a, b = trial["control"], trial["fast"]
                if (
                    a["minimum_doubled_cd"] is not None
                    and b["minimum_doubled_cd"] is not None
                    and a["minimum_doubled_cd"] != b["minimum_doubled_cd"]
                ):
                    raise ValueError("Proved minimum costs disagree")
                same_shell = mode == "specific_cd" or (
                    a["minimum_doubled_cd"] is not None
                    and a["minimum_doubled_cd"] == b["minimum_doubled_cd"]
                )
                if a["complete"] and b["complete"]:
                    if not same_shell or set(maps_by_arm["control"]) != set(
                        maps_by_arm["fast"]
                    ):
                        raise ValueError("Complete shell outputs disagree")
                same_fixed_output = same_shell and all(
                    trial[arm]["termination"] == "mapping_limit" for arm in ARMS
                )
                equal_order = maps_by_arm["control"] == maps_by_arm["fast"]
                if (
                    same_fixed_output
                    and a["symmetry_group_sha256"] is not None
                    and a["symmetry_group_sha256"] == b["symmetry_group_sha256"]
                    and not equal_order
                ):
                    raise ValueError("Output order changed at the same mapping cap")
                if same_shell:
                    for complete_arm, partial_arm in (
                        ("control", "fast"),
                        ("fast", "control"),
                    ):
                        if trial[complete_arm]["complete"] and not set(
                            maps_by_arm[partial_arm]
                        ) <= set(maps_by_arm[complete_arm]):
                            raise ValueError(
                                "Partial output lies outside the complete shell"
                            )
                comparisons.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "both_complete": a["complete"] and b["complete"],
                        "same_fixed_output": same_fixed_output,
                        "ordered_outputs_equal": equal_order,
                    }
                )
        save(
            output / (key + ".case.json"),
            {"reaction": key, "attempts": attempts, "comparisons": comparisons},
        )
    return {"cpu": task["cpu"], "cases": len(task["rows"]), "checked_maps": checked}


def report(output, rows, modes, repeats, seconds):
    cases = [
        json.loads((output / (r["benchmark_id"] + ".case.json")).read_text())
        for r in rows
    ]
    table, summary = [], {}
    for mode in modes:
        attempts = [a for c in cases for a in c["attempts"] if a["mode"] == mode]
        comparisons = [a for c in cases for a in c["comparisons"] if a["mode"] == mode]
        joint_complete, fixed_output = [], []
        for case in cases:
            selected = [a for a in case["attempts"] if a["mode"] == mode]
            row = {"mode": mode, "reaction": case["reaction"]}
            for arm in ARMS:
                arm_rows = [r for r in selected if r["arm"] == arm]
                row[arm + "_complete"] = sum(r["complete"] for r in arm_rows)
                row[arm + "_median_seconds"] = statistics.median(
                    r["seconds"] for r in arm_rows
                )
                row[arm + "_mapping_counts"] = sorted(
                    {r["mapping_count"] for r in arm_rows}
                )
                row[arm + "_termination"] = sorted({r["termination"] for r in arm_rows})
            table.append(row)
            pair = (row["control_median_seconds"], row["fast_median_seconds"])
            if all(r["complete"] for r in selected):
                joint_complete.append(pair)
            checks = [c for c in case["comparisons"] if c["mode"] == mode]
            if all(
                c["same_fixed_output"] and c["ordered_outputs_equal"] for c in checks
            ):
                fixed_output.append(pair)

        def timings(pairs):
            baseline, fast = (sum(p[i] for p in pairs) for i in (0, 1))
            return {
                "cases": len(pairs),
                "control_sum_median_seconds": baseline,
                "fast_sum_median_seconds": fast,
                "time_reduction_percent": (
                    None if not baseline else 100 * (1 - fast / baseline)
                ),
            }

        summary[mode] = {
            "cases": len(rows),
            "repeats": repeats,
            "complete_attempts": {
                arm: sum(r["complete"] for r in attempts if r["arm"] == arm)
                for arm in ARMS
            },
            "minimum_proved_attempts": {
                arm: sum(
                    r["minimum_doubled_cd"] is not None
                    for r in attempts
                    if r["arm"] == arm
                )
                for arm in ARMS
            },
            "equal_ordered_trials": sum(
                c["ordered_outputs_equal"] for c in comparisons
            ),
            "joint_complete": timings(joint_complete),
            "same_capped_output": timings(fixed_output),
        }
    save(output / "summary.json", summary)
    with (output / "per_reaction.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = [
        "# Product-only emission benchmark",
        "",
        f"{repeats} repeats; {seconds:g}-second solver limit; 100,000-map cap.",
        "Times include solver setup, proof when requested, enumeration and callbacks; see protocol for symmetry-discovery timing.",
        "Output checks occur outside timed calls. The control differs only in the accept method.",
        "",
        "| Mode | Reaction | Control complete | Fast complete | Control median s | Fast median s | Control maps | Fast maps |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in table:
        lines.append(
            f"| {row['mode']} | {row['reaction']} | {row['control_complete']}/{repeats} | {row['fast_complete']}/{repeats} | {row['control_median_seconds']:.4f} | {row['fast_median_seconds']:.4f} | {row['control_mapping_counts']} | {row['fast_mapping_counts']} |"
        )
    (output / "per_reaction.md").write_text("\n".join(lines) + "\n")
    return summary


def run(args):
    root = Path(__file__).resolve().parents[2]
    rows = json.loads(
        (root / "paper/synister/evidence/enumeration_main_v1/inputs.json").read_text()
    )
    if args.cases != "all":
        ids = {"reaction_" + i.strip().zfill(3) for i in args.cases.split(",")}
        rows = [r for r in rows if r["benchmark_id"] in ids]
        if {r["benchmark_id"] for r in rows} != ids:
            raise ValueError("Unknown benchmark case")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "frozen_source"
    hashes = freeze_source(root, frozen)
    control = (
        root
        / "Experiment/Synister/runs/reward_frontier_development_v2/frozen_source/synkit/Chem/Mapper/exact/propagation_search.py"
    )
    shutil.copyfile(control, output / "control_propagation_search.py")
    seed_path = (
        root
        / "Experiment/Synister/runs/separator_spectrum_100_minimal_v1/prepared_seeds.json"
    )
    target_path = (
        root
        / "Experiment/Synister/runs/separator_spectrum_100_specific_v1/specific_cd_target_manifest.json"
    )
    seeds = json.loads(seed_path.read_text())
    targets = json.loads(target_path.read_text())["targets"]
    cpus = sorted(os.sched_getaffinity(0))[: min(4, len(rows))]
    save(output / "environment.json", environment())
    save(output / "inputs.json", rows)
    save(
        output / "protocol.json",
        {
            "cases": len(rows),
            "selection": args.cases,
            "modes": args.modes,
            "repeats": args.repeats,
            "seconds_per_query": args.seconds,
            "mapping_cap": 100000,
            "cpus": cpus,
            "source_sha256": hashes,
            "control_source_sha256": sha256(control.read_bytes()).hexdigest(),
            "seeds_sha256": sha256(seed_path.read_bytes()).hexdigest(),
            "targets_sha256": sha256(target_path.read_bytes()).hexdigest(),
            "isolation": "Only PropagatedSearch.accept is exchanged within each dedicated worker; ordinary PABS configuration in both arms",
            "symmetry_discovery": (
                "One verified subgroup per case, prepared outside timed calls and replayed identically in both arms"
                if args.freeze_symmetry
                else "Normal budgeted discovery in every timed query; subgroup hashes recorded"
            ),
            "order": "Alternating by repeat/case/mode parity; same logical CPU for each pair",
            "python": sys.version,
        },
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(frozen),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )

    def lane(index):
        task = {
            "output": str(output),
            "cpu": cpus[index],
            "seconds": args.seconds,
            "repeats": args.repeats,
            "modes": args.modes,
            "freeze_symmetry": args.freeze_symmetry,
            "rows": [
                {"row": row, "index": i}
                for i, row in enumerate(rows)
                if i % len(cpus) == index
            ],
            "seeds": seeds,
            "targets": targets,
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "Experiment.Synister.emission_benchmark",
                "--worker",
            ],
            input=json.dumps(task),
            capture_output=True,
            text=True,
            cwd=frozen,
            env=env,
        )
        (output / f"lane_{index}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr[-3000:])
        payload = json.loads(result.stdout)
        print(json.dumps(payload), flush=True)
        return payload

    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        results = list(pool.map(lane, range(len(cpus))))
    verify_source(frozen, hashes)
    if (
        sha256((output / "control_propagation_search.py").read_bytes()).hexdigest()
        != sha256(control.read_bytes()).hexdigest()
    ):
        raise ValueError("Control source changed")
    summary = report(output, rows, args.modes, args.repeats, args.seconds)
    save(
        output / "audit.json",
        {
            "all_outputs_consistent": True,
            "independently_rescored_maps": sum(r["checked_maps"] for r in results),
            "cases": len(rows),
            "attempts": len(rows) * len(args.modes) * args.repeats * 2,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", default="038,040,041")
    parser.add_argument("--freeze-symmetry", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("minimal", "specific_cd"),
        default=["minimal", "specific_cd"],
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))), flush=True)
    else:
        if args.output is None or args.repeats < 1 or args.seconds <= 0:
            parser.error("output and positive repeat/time budgets are required")
        run(args)
