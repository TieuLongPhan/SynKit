"""Durable solver reports and separately supervised output validation."""

from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


def publish(path, value):
    """Publish complete JSON durably, without replacing existing evidence."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def artifact(task, suffix):
    """Locate an immutable per-method stage artifact."""
    return Path(task["directory"]) / (
        task["benchmark_id"] + "." + task["method"] + suffix
    )


def solve(task):
    """Save the solver outcome before serializing or checking its mappings."""
    from synkit.Chem.Mapper.identifiability import parse_reaction
    from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
    from synkit.Chem.Mapper.exact.propagation import (
        PropagationConfig,
        enumerate_synister_cp_mappings,
    )
    from Experiment.Synister.global_milp import doubled_distance

    started = time.perf_counter()
    r, p = parse_reaction(task["reaction"])
    parsed = time.perf_counter()
    seed = task["prepared_seed"]
    seed_cost = doubled_distance(r, p, seed)
    target_mode = task.get("target_mode", "minimal")
    target_doubled_cd = task.get("target_doubled_cd")
    if target_mode not in {"minimal", "specific_cd"}:
        raise ValueError("target_mode must be 'minimal' or 'specific_cd'")
    if target_mode == "specific_cd":
        if (
            isinstance(target_doubled_cd, bool)
            or not isinstance(target_doubled_cd, int)
            or target_doubled_cd < 0
        ):
            raise ValueError("specific_cd requires a nonnegative integer doubled CD")
        if target_doubled_cd != seed_cost:
            raise ValueError("specific target must equal the frozen seed's literal CD")
    functions = {
        "legacy": enumerate_distance_mappings,
        "synister_cp": enumerate_synister_cp_mappings,
    }
    maps = []
    before = time.perf_counter()
    options = {}
    if task["method"] == "synister_cp":
        options["config"] = PropagationConfig(
            branch_order=task.get("branch_order", "default"),
            pairwise_edge_bounds=task.get("pairwise_edge_bounds", False),
            separator_spectrum=task.get("separator_spectrum", False),
            factor_spectrum_bounds=task.get("factor_spectrum_bounds", False),
            suffix_spectrum=task.get("suffix_spectrum", False),
            suffix_spectrum_orbit_pruning=task.get(
                "suffix_spectrum_orbit_pruning", True
            ),
        )
        options["reactant_symmetry_pruning"] = task.get(
            "reactant_symmetry_pruning", False
        )
    result = functions[task["method"]](
        [r.graph(), p.graph()],
        CD=("minimal" if target_mode == "minimal" else target_doubled_cd / 2),
        binary=False,
        initial_mapping=seed,
        max_bijections=None,
        max_mappings=100000,
        tolerance=0,
        compute_minimum_cost=target_mode == "minimal",
        time_limit_seconds=task["seconds"],
        symmetry_pruning=True,
        expand_symmetry=True,
        symmetry_node_properties=("charges", "hcounts"),
        collect_mappings=False,
        mapping_callback=lambda mapping, cost: maps.append(tuple(mapping)),
        **options,
    )
    elapsed = time.perf_counter() - before
    search_stats = (result.backend_statistics or {}).get("search", {})
    proof_seconds = search_stats.get("minimum_proof_seconds")
    enumeration_seconds = search_stats.get("enumeration_seconds")
    if task["method"] == "legacy":
        if proof_seconds == 0 and result.minimum_cost is None:
            # Legacy's nested minimum pass reports its attempted traversal under
            # traversal_seconds when it times out before returning a proof.
            proof_seconds = search_stats.get("traversal_seconds", 0.0)
        if enumeration_seconds is None:
            traversal = search_stats.get("traversal_seconds")
            enumeration_seconds = (
                traversal
                if result.minimum_cost is not None and traversal is not None
                else 0.0
            )
    record = {
        "schema_version": 2,
        "benchmark_id": task["benchmark_id"],
        "seed": seed,
        "seed_doubled_cd": seed_cost,
        "parse_seconds": parsed - started,
        "seed_check_seconds": before - parsed,
        "methods": {
            task["method"]: {
                "complete": result.complete,
                "target_mode": target_mode,
                "target_doubled_cd": (
                    None if target_mode == "minimal" else target_doubled_cd
                ),
                "minimum_doubled_cd": (
                    None
                    if result.minimum_cost is None
                    else int(2 * result.minimum_cost)
                ),
                "termination": result.truncation_reason or result.status,
                "seconds": elapsed,
                "mapping_count": len(maps),
                "mapping_sha256": None,
                "validation_complete": False,
                "visited_nodes": result.visited_nodes,
                "statistics": result.backend_statistics,
                "phase_seconds": {
                    "solver_total": elapsed,
                    "preprocessing": search_stats.get("preprocessing_seconds"),
                    "minimum_proof": proof_seconds,
                    "enumeration": enumeration_seconds,
                },
                "backend": result.backend,
            }
        },
    }
    publish(artifact(task, ".solver.json"), record)
    writing = time.perf_counter()
    encoded = json.dumps(maps, separators=(",", ":")).encode()
    path = artifact(task, ".maps.json")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    output = {
        "mapping_sha256": sha256(encoded).hexdigest(),
        "output_bytes": len(encoded),
        "writing_seconds": time.perf_counter() - writing,
    }
    publish(artifact(task, ".output.json"), output)
    return record


def validate(task):
    """Check a saved output independently under a separate phase budget."""
    from synkit.Chem.Mapper.identifiability import parse_reaction
    from Experiment.Synister.mapping_check import check_mappings

    before = time.perf_counter()
    record = json.loads(artifact(task, ".solver.json").read_text())
    output = json.loads(artifact(task, ".output.json").read_text())
    path = artifact(task, ".maps.json")
    if sha256(path.read_bytes()).hexdigest() != output["mapping_sha256"]:
        raise ValueError("Saved mappings changed")
    maps = json.loads(path.read_text())
    method = record["methods"][task["method"]]
    if len(maps) != method["mapping_count"] or len(set(map(tuple, maps))) != len(maps):
        raise ValueError("Missing or duplicate indexed output")
    r, p = parse_reaction(task["reaction"])
    expected_doubled_cd = (
        method["minimum_doubled_cd"]
        if method["target_mode"] == "minimal"
        else method["target_doubled_cd"]
    )
    count = check_mappings(r, p, maps, expected_doubled_cd)
    result = {
        "complete": True,
        "mapping_count": count,
        "target_mode": method["target_mode"],
        "target_doubled_cd": expected_doubled_cd,
        "mapping_sha256": output["mapping_sha256"],
        "seconds": time.perf_counter() - before,
    }
    publish(artifact(task, ".validation.json"), result)
    return result


def prepare_seed(task):
    """Prepare one shared feasible seed separately from either solver."""
    from synkit.Chem.Mapper.prediction_adapter import predict_slap

    before = time.perf_counter()
    result = {
        "mapping": predict_slap(task["reaction"])["mapping"],
        "seconds": time.perf_counter() - before,
    }
    publish(Path(task["directory"]) / (task["benchmark_id"] + ".seed.json"), result)
    return result


def perform(task):
    """Execute exactly one resource-limited phase in an isolated process."""
    limit = 6 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    return {"solve": solve, "validate": validate, "seed": prepare_seed}[task["stage"]](
        task
    )


def invoke(task, source, env, timeout):
    """Supervise one phase without conflating its deadline with another."""
    before = time.perf_counter()
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "Experiment.Synister.propagation_comparison",
                "--worker",
            ],
            input=json.dumps(task),
            text=True,
            capture_output=True,
            cwd=source,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"parent_timeout": True, "wall_seconds": time.perf_counter() - before}
    return {
        "parent_timeout": False,
        "wall_seconds": time.perf_counter() - before,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr[-4000:],
    }


def run_method(task, source, env):
    """Retain search metadata even if output publication or checking fails."""
    search = invoke(dict(task, stage="solve"), source, env, task["seconds"] + 15)
    path = artifact(task, ".solver.json")
    if not path.exists():
        return {
            "complete": False,
            "solver_outcome_known": False,
            "minimum_doubled_cd": None,
            "seconds": None,
            "mapping_count": None,
            "mapping_sha256": None,
            "validation_complete": False,
            "termination": (
                "parent_time_limit_search"
                if search["parent_timeout"]
                else "solver_error"
            ),
            "interrupted_phase": "search",
            "search_process": search,
        }
    record = json.loads(path.read_text())
    result = dict(
        record["methods"][task["method"]],
        solver_outcome_known=True,
        search_process=search,
    )
    output_path = artifact(task, ".output.json")
    if not output_path.exists():
        result.update(interrupted_phase="output", output_error=search)
        return result
    result.update(json.loads(output_path.read_text()))
    checking = invoke(
        dict(task, stage="validate"), source, env, task.get("validation_seconds", 30)
    )
    validation_path = artifact(task, ".validation.json")
    result["validation_process"] = checking
    if validation_path.exists():
        checked = json.loads(validation_path.read_text())
        result.update(
            validation_complete=checked["complete"],
            validation_seconds=checked["seconds"],
        )
    else:
        result.update(interrupted_phase="validation", validation_error=checking)
    return result
