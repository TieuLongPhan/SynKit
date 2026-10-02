"""Frozen, paired legacy/Synister-CP comparison on the existing 100 reactions."""

import argparse
from importlib.metadata import version
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys


def save(path, value):
    """Publish an immutable, durable result artifact."""
    from Experiment.Synister.propagation_phases import publish

    publish(path, value)


def worker(task):
    """Run an isolated solver, validation, or seed-preparation stage."""
    from Experiment.Synister.propagation_phases import perform

    return perform(task)


def summarize(directory, total):
    """Report completion and paired time comparisons without excluding timeouts."""
    import statistics

    records = [
        json.loads(p.read_text())
        for p in sorted(directory.glob("reaction_*.result.json"))
    ]
    paired = [
        r for r in records if set(r.get("methods", {})) == {"legacy", "synister_cp"}
    ]
    complete = [
        r
        for r in paired
        if all(
            m["complete"] and m.get("validation_complete", True)
            for m in r["methods"].values()
        )
    ]
    ratios = [
        r["methods"]["legacy"]["seconds"] / r["methods"]["synister_cp"]["seconds"]
        for r in complete
    ]
    result = {
        "expected_pairs": total,
        "target_mode": json.loads((directory / "protocol.json").read_text()).get(
            "target_mode", "minimal"
        ),
        "recorded_pairs": len(records),
        "errors": [r for r in records if "error" in r],
        "both_complete": len(complete),
        "complete": {
            name: sum(
                r["methods"][name]["complete"]
                and r["methods"][name].get("validation_complete", True)
                for r in paired
            )
            for name in ("legacy", "synister_cp")
        },
        "minimum_proved": {
            name: sum(
                r["methods"][name]["minimum_doubled_cd"] is not None for r in paired
            )
            for name in ("legacy", "synister_cp")
        },
        "new_faster_both_complete": sum(ratio > 1 for ratio in ratios),
        "median_legacy_over_new_seconds_both_complete": (
            statistics.median(ratios) if ratios else None
        ),
    }
    if any(r.get("schema_version") == 2 for r in records):
        result["solver_complete"] = {
            name: sum(r["methods"][name]["complete"] for r in paired)
            for name in ("legacy", "synister_cp")
        }
        result["validation_incomplete"] = {
            name: sum(
                not r["methods"][name].get("validation_complete", False) for r in paired
            )
            for name in ("legacy", "synister_cp")
        }
        result["unknown_solver_outcomes"] = {
            name: sum(
                not r["methods"][name].get("solver_outcome_known", False)
                for r in paired
            )
            for name in ("legacy", "synister_cp")
        }
        result["total_solver_seconds_both_complete"] = {
            name: sum(r["methods"][name]["seconds"] for r in complete)
            for name in ("legacy", "synister_cp")
        }
    return result


def attempt(
    index_row,
    directory,
    source,
    seconds,
    seeds,
    env,
    cp_branch_order,
    cp_pairwise_edge_bounds,
    cp_reactant_symmetry,
    cp_separator_spectrum,
    cp_factor_spectrum_bounds,
    cp_suffix_spectrum,
    cp_suffix_spectrum_orbit_pruning,
    target_by_id=None,
):
    """Compare both engines, retaining every per-method interrupted phase."""
    from Experiment.Synister.propagation_phases import invoke, run_method

    index, row = index_row
    task = dict(
        row,
        seconds=seconds,
        directory=str(directory),
        reverse=bool(index % 2),
        branch_order=cp_branch_order,
        pairwise_edge_bounds=cp_pairwise_edge_bounds,
        reactant_symmetry_pruning=cp_reactant_symmetry,
        separator_spectrum=cp_separator_spectrum,
        factor_spectrum_bounds=cp_factor_spectrum_bounds,
        suffix_spectrum=cp_suffix_spectrum,
        suffix_spectrum_orbit_pruning=cp_suffix_spectrum_orbit_pruning,
        target_mode="specific_cd" if target_by_id is not None else "minimal",
    )
    if target_by_id is not None:
        task["target_doubled_cd"] = target_by_id[row["benchmark_id"]]
    result = {"schema_version": 2, "benchmark_id": row["benchmark_id"], "methods": {}}
    if seeds is None:
        prepared = invoke(dict(task, stage="seed"), source, env, 60)
        path = directory / (row["benchmark_id"] + ".seed.json")
        if not path.exists():
            save(
                directory / (row["benchmark_id"] + ".result.json"),
                dict(result, error=prepared),
            )
            return row["benchmark_id"]
        prediction = json.loads(path.read_text())
        task["prepared_seed"] = prediction["mapping"]
        result["seed_seconds"] = prediction["seconds"]
    else:
        task["prepared_seed"] = seeds[row["benchmark_id"]]["mapping"]
        result["seed_seconds"] = None
    result["seed"] = task["prepared_seed"]
    order = ["synister_cp", "legacy"] if task["reverse"] else ["legacy", "synister_cp"]
    for method in order:
        result["methods"][method] = run_method(dict(task, method=method), source, env)
    save(directory / (row["benchmark_id"] + ".result.json"), result)
    return row["benchmark_id"]


def run(
    directory,
    seconds,
    workers,
    prepared_seeds=None,
    cases=None,
    solver_source=None,
    cp_branch_order="default",
    input_manifest=None,
    target_manifest=None,
    cp_pairwise_edge_bounds=False,
    cp_reactant_symmetry=False,
    cp_separator_spectrum=False,
    cp_factor_spectrum_bounds=False,
    cp_suffix_spectrum=False,
    cp_suffix_spectrum_orbit_pruning=True,
):
    root = Path(__file__).resolve().parents[2]
    directory = directory.resolve()
    inputs = root / "paper/synister/evidence/enumeration_main_v1/inputs.json"
    external_manifest = None
    if input_manifest is None:
        rows = json.loads(inputs.read_text())
    else:
        from Experiment.Synister.propagation_external_cohort import verify_manifest

        input_manifest = input_manifest.resolve()
        external_manifest = json.loads(input_manifest.read_text())
        rows = verify_manifest(external_manifest, root)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "frozen_source"
    hashes = {}
    for base in ("synkit", "Experiment/Synister"):
        origin = (
            solver_source if base == "synkit" and solver_source is not None else root
        )
        for path in sorted((origin / base).rglob("*.py")):
            relative = path.relative_to(origin)
            if (
                any(part.startswith(".venv") for part in relative.parts)
                or "runs" in relative.parts
            ):
                continue
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[str(relative)] = sha256(path.read_bytes()).hexdigest()
    if external_manifest is not None:
        save(directory / "external_input_manifest.json", external_manifest)
    if cases is not None:
        unknown = set(cases) - {r["benchmark_id"] for r in rows}
        if unknown:
            raise ValueError(f"Unknown input IDs: {sorted(unknown)}")
        rows = [r for r in rows if r["benchmark_id"] in cases]
    save(directory / "inputs.json", rows)
    seeds = json.loads(prepared_seeds.read_text()) if prepared_seeds else None
    if seeds is not None:
        seeds = {r["benchmark_id"]: seeds[r["benchmark_id"]] for r in rows}
        if set(seeds) != {r["benchmark_id"] for r in rows}:
            raise ValueError("Prepared seeds must cover exactly the selected cohort")
        for row in rows:
            if seeds[row["benchmark_id"]]["reaction"] != row["reaction"]:
                raise ValueError("Prepared seed reaction differs from selected input")
        save(directory / "prepared_seeds.json", seeds)
    target_by_id = None
    frozen_targets = None
    if target_manifest is not None:
        if seeds is None:
            raise ValueError("A frozen specific-CD run requires shared prepared seeds")
        from Experiment.Synister.global_milp import doubled_distance
        from synkit.Chem.Mapper.identifiability import parse_reaction

        frozen_targets = json.loads(target_manifest.read_text())
        target_rows = frozen_targets.get("targets", {})
        if not {r["benchmark_id"] for r in rows} <= set(target_rows):
            raise ValueError("Specific-CD target manifest misses a selected input")
        if frozen_targets.get("rule") != "literal doubled CD of the frozen shared seed":
            raise ValueError("Unsupported specific-CD target rule")
        if (
            frozen_targets.get("inputs_sha256")
            != sha256(inputs.read_bytes()).hexdigest()
        ):
            raise ValueError(
                "Specific-CD manifest is not bound to the frozen input file"
            )
        if (
            frozen_targets.get("prepared_seeds_sha256")
            != sha256(prepared_seeds.read_bytes()).hexdigest()
        ):
            raise ValueError(
                "Specific-CD manifest is not bound to the shared seed file"
            )
        target_by_id = {}
        for row in rows:
            benchmark_id = row["benchmark_id"]
            target = target_rows[benchmark_id]
            reaction_hash = sha256(row["reaction"].encode()).hexdigest()
            if target.get("reaction_sha256") != reaction_hash:
                raise ValueError("Specific-CD target belongs to a different reaction")
            if target.get("seed_mapping") != seeds[benchmark_id]["mapping"]:
                raise ValueError("Specific-CD target uses a different frozen seed")
            r, p = parse_reaction(row["reaction"])
            expected = doubled_distance(r, p, target["seed_mapping"])
            if target.get("target_doubled_cd") != expected:
                raise ValueError(
                    "Specific-CD target failed independent literal rescoring"
                )
            target_by_id[benchmark_id] = expected
        save(directory / "specific_cd_target_manifest.json", frozen_targets)
    save(
        directory / "protocol.json",
        {
            "seconds_per_method": seconds,
            "target_mode": "specific_cd" if frozen_targets is not None else "minimal",
            "specific_cd_target_rule": (
                frozen_targets.get("rule") if frozen_targets is not None else None
            ),
            "specific_cd_target_manifest_sha256": (
                sha256(
                    (directory / "specific_cd_target_manifest.json").read_bytes()
                ).hexdigest()
                if frozen_targets is not None
                else None
            ),
            "specific_cd_source_inputs_sha256": (
                frozen_targets.get("inputs_sha256")
                if frozen_targets is not None
                else None
            ),
            "specific_cd_source_seeds_sha256": (
                frozen_targets.get("prepared_seeds_sha256")
                if frozen_targets is not None
                else None
            ),
            "cp_branch_order": cp_branch_order,
            "cp_pairwise_edge_bounds": cp_pairwise_edge_bounds,
            "cp_reactant_symmetry": cp_reactant_symmetry,
            "cp_separator_spectrum": cp_separator_spectrum,
            "cp_factor_spectrum_bounds": cp_factor_spectrum_bounds,
            "cp_suffix_spectrum": cp_suffix_spectrum,
            "cp_suffix_spectrum_orbit_pruning": cp_suffix_spectrum_orbit_pruning,
            "cp_suffix_spectrum_max_calls": 2 if cp_suffix_spectrum else None,
            "cp_suffix_spectrum_max_seconds_per_call": (
                0.002 if cp_suffix_spectrum else None
            ),
            "cp_suffix_spectrum_max_states": 20_000 if cp_suffix_spectrum else None,
            "workers": workers,
            "mapping_cap": 100000,
            "memory_gib_per_pair": 6,
            "single_thread": True,
            "seed": "saved shared SLAP" if seeds else "fresh shared SLAP",
            "prepared_seeds_sha256": (
                sha256(prepared_seeds.read_bytes()).hexdigest() if seeds else None
            ),
            "order": "counterbalanced by input index",
            "schema_version": 2,
            "cohort_size": len(rows),
            "validation_seconds": 30,
            "isolation": "independent search and validation subprocesses",
            "solver_source_origin": str(solver_source or root),
            "inputs_sha256": sha256(
                (directory / "inputs.json").read_bytes()
            ).hexdigest(),
            "original_cohort_sha256": (
                sha256(inputs.read_bytes()).hexdigest()
                if external_manifest is None
                else None
            ),
            "external_input_manifest_sha256": (
                sha256(
                    (directory / "external_input_manifest.json").read_bytes()
                ).hexdigest()
                if external_manifest is not None
                else None
            ),
            "external_input_manifest_origin": (
                str(input_manifest) if external_manifest is not None else None
            ),
            "source_sha256": hashes,
            "python": sys.version,
            "dependencies": {
                name: version(name) for name in ("numpy", "scipy", "rdkit", "networkx")
            },
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "thread_environment": {
                name: "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "saved_seed_artifact_sha256": (
                sha256((directory / "prepared_seeds.json").read_bytes()).hexdigest()
                if seeds
                else None
            ),
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

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, case in enumerate(
            pool.map(
                lambda item: attempt(
                    item,
                    directory,
                    source,
                    seconds,
                    seeds,
                    env,
                    cp_branch_order,
                    cp_pairwise_edge_bounds,
                    cp_reactant_symmetry,
                    cp_separator_spectrum,
                    cp_factor_spectrum_bounds,
                    cp_suffix_spectrum,
                    cp_suffix_spectrum_orbit_pruning,
                    target_by_id,
                ),
                enumerate(rows),
            ),
            1,
        ):
            if index % 10 == 0:
                print(json.dumps(summarize(directory, len(rows))), flush=True)
    result = summarize(directory, len(rows))
    save(directory / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepared-seeds", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--solver-source", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--specific-cd-target-manifest", type=Path)
    parser.add_argument(
        "--cp-branch-order",
        choices=("default", "pagerank", "impact", "contention"),
        default="default",
    )
    parser.add_argument("--cp-pairwise-edge-bounds", action="store_true")
    parser.add_argument("--cp-reactant-symmetry", action="store_true")
    parser.add_argument("--cp-separator-spectrum", action="store_true")
    parser.add_argument("--cp-factor-spectrum-bounds", action="store_true")
    parser.add_argument("--cp-suffix-spectrum", action="store_true")
    parser.add_argument(
        "--cp-no-suffix-spectrum-orbit-pruning",
        action="store_false",
        dest="cp_suffix_spectrum_orbit_pruning",
        default=True,
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
    else:
        run(
            args.directory,
            args.seconds,
            args.workers,
            args.prepared_seeds,
            args.cases,
            args.solver_source,
            args.cp_branch_order,
            args.input_manifest,
            args.specific_cd_target_manifest,
            args.cp_pairwise_edge_bounds,
            args.cp_reactant_symmetry,
            args.cp_separator_spectrum,
            args.cp_factor_spectrum_bounds,
            args.cp_suffix_spectrum,
            args.cp_suffix_spectrum_orbit_pruning,
        )
