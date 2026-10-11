"""Run bounded validation or full research reproduction without implicit raw stress."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

from Experiment.Stereo.provenance import ROOT, capture_source, finish_source, sha256
from Experiment.Stereo.cip_sources import COORDINATE_PATH, SMILES_PATH, provenance_inputs

PILOT_CASES = (
    "tetrahedral:distinct", "planar_bond:distinct", "tetrahedral:two:meso",
    "framework:coupled:opposite", "mixed:tetrahedral+octahedral:distinct",
)


def command(module, *args):
    """Build an argument array without shell interpolation."""
    return [sys.executable, "-m", "Experiment.Stereo." + module, *map(str, args)]


def stages(args):
    """Select the declared workload and retain separate outputs for every protocol."""
    out = args.output_dir.resolve()
    prior_art = ROOT / "Experiment/Stereo/Data/Revision/PaperEvidence"
    rdkit_probe = [sys.executable, str(ROOT / "Experiment/Stereo/rdkit_representation_probe.py"),
                   "--fixtures", str(prior_art / "rdkit_representation_fixtures.json"),
                   "--output", str(out / "rdkit_representation.json")]
    independent = command("Canonicalization.independent_orbits", "--output", out / "independent.json",
                          "--jobs", args.jobs)
    if args.tier == "bounded":
        for case in PILOT_CASES:
            independent += ["--case", case]
        return [
            ("tests", [sys.executable, "-m", "pytest", "-q",
                       "Test/Benchmark/Stereo/Canonicalization/test_independent_orbits.py",
                       "Test/Benchmark/Stereo/Canonicalization/test_independent_graph.py",
                       "Test/Benchmark/Stereo/Canonicalization/test_independent_transport.py",
                       "Test/Benchmark/Stereo/Canonicalization/test_independent_replay.py",
                       "Test/Benchmark/Stereo/Canonicalization/test_performance.py",
                       "Test/Benchmark/Stereo/Canonicalization/test_registration.py",
                       "Test/Benchmark/Stereo/Canonicalization/test_chemical_coordination.py",
                       "Test/Benchmark/Stereo/Canonicalization/test_chemical_truth.py",
                       "Test/Benchmark/Stereo/Canonicalization/test_rota_structural_annotation.py",
                       "Test/Benchmark/Stereo/Canonicalization/test_rota_structural_evaluation.py",
                       "Test/Benchmark/Stereo/test_result_contracts.py",
                       "Test/Benchmark/Stereo/test_report_provenance.py",
                       "Test/Benchmark/Stereo/test_rdkit_representation_contract.py",
                       "Test/Benchmark/Stereo/test_detection_result_contract.py",
                       "Test/Benchmark/Stereo/test_chemical_result_contracts.py",
                       "Test/Benchmark/Stereo/test_protocol_result_contracts.py",
                       "Test/Benchmark/Stereo/test_cip_sources.py",
                       "Test/Benchmark/Stereo/Perception/test_detection_constitution_identity.py",
                       "Test/Benchmark/Stereo/Perception/test_coordinate_transport_boundary.py",
                       "--junitxml=" + str(out / "tests.xml")]),
            ("independent", independent),
            ("rdkit_representation", rdkit_probe),
        ]
    if args.cip_path is None or args.coordinate_path is None:
        raise ValueError("Full reproduction requires the pinned SMILES and coordinate paths.")
    independent += ["--include-public-abc", "--cip-path", str(args.cip_path.resolve())]
    if args.tier == "raw-stress":
        if not args.allow_expensive_raw:
            raise ValueError("Raw stress requires --allow-expensive-raw and may take days.")
        return [("raw_" + budget, command(
            "Canonicalization.configuration_free_global", budget, "--enumeration-mode", "raw",
            "--allow-expensive-raw", "--cip-path", args.cip_path.resolve(), "--jobs", args.jobs,
            "--output", out / ("raw_" + budget + ".json"), "--timeout", 10,
            "--table", out / ("raw_" + budget + ".csv"),
            "--max-assignments", 20000000,
            *(["--case-id", "VS146"] if budget == "C" else []),
        )) for budget in ("B", "C")]
    selected = [("independent", independent)]
    for task in ("internal", "acs", "cip", "rota"):
        selected.append(("local_" + task, command(
            "Canonicalization.configuration_free_local", task, "--jobs", args.jobs,
            "--cip-path", args.cip_path.resolve(), "--timeout", 10,
            "--output", out / ("local_" + task + ".json"),
            "--table", out / ("local_" + task + ".csv"),
        )))
        selected.append(("replay_local_" + task, command(
            "Canonicalization.independent_replay", out / ("local_" + task + ".json"),
            "--cip-path", args.cip_path.resolve(),
            "--output", out / ("replay_local_" + task + ".json"),
        )))
    for budget in ("A", "B", "C"):
        selected.append(("formal_" + budget, command(
            "Canonicalization.configuration_free_global", budget, "--enumeration-mode", "formal",
            "--jobs", args.jobs, "--cip-path", args.cip_path.resolve(),
            "--output", out / ("formal_" + budget + ".json"),
            "--table", out / ("formal_" + budget + ".csv"),
        )))
    selected += [
        ("source_identity", command("Chirality.run", "acs", "--case-timeout", 5,
                                    "--identity-profile", "chemical", "--output", out / "source_identity.json")),
        ("exact_plus_global", command("Chirality.global_certificate_benchmark", "--case-timeout", 30,
                                      "--output", out / "exact_plus_global.json")),
        ("topology_completion", command("Chirality.published", "--output", out / "topology_completion.json",
                                        *(["--stereomolgraph", args.stereomolgraph] if args.stereomolgraph else []))),
        ("cip_smiles", command("Perception.cip_labels", "--cip-path", args.cip_path.resolve(),
                               "--output", out / "cip_smiles.json")),
        ("cip_coordinates", command("Perception.cip_labels", "--cip-path", args.cip_path.resolve(),
                                    "--cip-3d-path", args.coordinate_path.resolve(),
                                    "--output", out / "cip_coordinates.json")),
        ("overlap", command("Perception.corpus_overlap", "--cip-path", args.cip_path.resolve(),
                            "--output", out / "overlap.json")),
        ("relations", command("Chirality.run", "relations", "--output", out / "relations.json")),
    ]
    if args.stereomolgraph is not None:
        selected.append(("comparator_inventory", command(
            "Chirality.comparator_inventory", "--stereomolgraph", args.stereomolgraph,
            "--output", out / "comparator_inventory.json",
        )))
    for task, module in (("rota_loci", "axis_loci"), ("perception_conformance", "conformance"),
                         ("retained_detection", "full_detection"), ("elements", "stereo_elements"),
                         ("axial_negatives", "rota_synthetic_negatives"), ("cip_contract", "cip_input_contract")):
        extra = []
        if module in ("full_detection", "stereo_elements", "cip_input_contract"):
            extra += ["--cip-path", args.cip_path.resolve()]
        if module == "cip_input_contract":
            extra += ["--cip-3d-path", args.coordinate_path.resolve()]
        selected.append((task, command("Perception." + module, "--output", out / (task + ".json"), *extra)))
    performance_cap = max(1, min(14400, getattr(args, "stage_timeout", 14400) - 60))
    selected += [
        ("rdkit_representation", rdkit_probe),
        ("chemical_annotation", command("ChemicalValidation.annotation",
                                        "--output", out / "chemical_annotation.json")),
        ("chemical_validation", command("ChemicalValidation.evaluate", "--annotation", out / "chemical_annotation.json",
                                        "--output", out / "chemical_validation.json")),
        ("rota_annotation", command("ChemicalValidation.rota_annotation",
                                    "--output", out / "rota_annotation.json")),
        ("rota_validation", command("ChemicalValidation.rota_evaluate", "--annotation", out / "rota_annotation.json",
                                    "--output", out / "rota_validation.json")),
        ("chemical_results", command("chemical_results", "--coordination-annotation", out / "chemical_annotation.json",
                                     "--coordination-production", out / "chemical_validation.json",
                                     "--rota-annotation", out / "rota_annotation.json",
                                     "--rota-production", out / "rota_validation.json",
                                     "--output", out / "chemical_results.json")),
        ("controlled_costs", command("Canonicalization.performance", "--output", out / "controlled_costs.json",
                                     "--total-seconds", performance_cap)),
        ("registration", command("Canonicalization.registration", "--output", out / "registration.json")),
        ("cost_figures", command("Canonicalization.performance_figures", "--performance", out / "controlled_costs.json",
                                 "--registration", out / "registration.json", "--output-dir", out / "figures")),
    ]
    if args.tier == "protocols":
        protocol_names = {"relations", "rota_loci", "perception_conformance", "retained_detection",
                          "elements", "axial_negatives", "cip_contract"}
        return [(name, argv) for name, argv in selected if name in protocol_names]
    return selected


def run(args):
    """Write a stage manifest and stop on a failed or time-limited required stage."""
    selected = stages(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = capture_source(provenance_inputs(args.cip_path, args.coordinate_path))
    environment = {**os.environ, **{key: "1" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )}}
    records = []
    for name, argv in selected:
        print(json.dumps({"stage": name, "state": "started", "command": argv}), flush=True)
        started = time.monotonic()
        with (args.output_dir / (name + ".log")).open("w") as log:
            try:
                with subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True) as process:
                    try:
                        returncode = process.wait(timeout=args.stage_timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
                status = "complete" if returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status, returncode = "stage_timeout", None
        record = {"stage": name, "command": argv, "status": status, "returncode": returncode,
                  "wall_seconds": time.monotonic() - started}
        output = args.output_dir / (name + ".json")
        if output.exists():
            record["report_sha256"] = sha256(output)
        records.append(record)
        print(json.dumps(record), flush=True)
        if status != "complete":
            break
    report = {
        "schema": "synkit.stereo-reproduction-stages/1", "provenance": finish_source(source),
        "tier": args.tier, "jobs": args.jobs, "stage_timeout_seconds": args.stage_timeout,
        "selected_stage_names": [name for name, _ in selected], "records": records,
        "complete": len(records) == len(selected) and all(row["status"] == "complete" for row in records),
        "claim_boundary": "A completed stage sequence is not a chemical truth claim; report-specific gates apply.",
    }
    (args.output_dir / "stage_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("bounded", "protocols", "full", "raw-stress"), default="bounded")
    parser.add_argument("--cip-path", type=Path, default=SMILES_PATH)
    parser.add_argument("--coordinate-path", type=Path, default=COORDINATE_PATH)
    parser.add_argument("--stereomolgraph", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--stage-timeout", type=float, default=14400)
    parser.add_argument("--allow-expensive-raw", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.stage_timeout <= 0:
        parser.error("Jobs and stage timeout must be positive.")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
