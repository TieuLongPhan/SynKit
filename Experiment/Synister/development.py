"""Supervised development-only campaign with predictions frozen before search.

No confirmation protocol is implied. Each new run requires an unused directory.
All selected inputs and failed attempts remain in its accounting.
"""

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

from synkit.Chem.Mapper.prediction_adapter import unmapped_input

ROOT = Path(__file__).resolve().parents[2]


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_reaction(row):
    """Remove only this CSV's redundant terminal reaction-ID suffix."""
    value = row["mapped_reaction"]
    if "|" in value:
        value, suffix = value.rsplit("|", 1)
        if suffix != row["reaction_id"] or "|" in value:
            raise ValueError("Unexpected source reaction metadata suffix")
    return value


def save(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value))


def snapshot(output):
    sources = sorted((ROOT / "synkit/Chem/Mapper").rglob("*.py"))
    sources += sorted((ROOT / "Experiment/Synister").glob("*.py"))
    sources += sorted((ROOT / "Experiment/Synister/tests").glob("*.py"))
    archive = {str(path.relative_to(ROOT)): path.read_text() for path in sources}
    save(output / "source_snapshot.json", archive)
    return digest(encoded(archive))


def environment():
    import rxnmapper
    base = Path(rxnmapper.__file__).parent
    model_files = {
        str(p.relative_to(base)): digest(p.read_bytes())
        for p in sorted(base.rglob("*"))
        if p.is_file() and p.suffix in (".bin", ".safetensors", ".json", ".txt")
    }
    return {"python": sys.version, "executable": sys.executable,
            # Resolve by name: distributions() may expose setuptools' vendored
            # dist-info last, overwriting the active top-level package version.
            "packages": {d.metadata["Name"]: importlib.metadata.version(d.metadata["Name"])
                         for d in importlib.metadata.distributions()},
            "rxnmapper_resource_sha256": model_files}


def execute(task, timeout, output):
    started = time.monotonic()
    env = dict(os.environ)
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    env.update(PYTHONHASHSEED="0", TOKENIZERS_PARALLELISM="false")
    command = [sys.executable, "-m", "Experiment.Synister.worker"]
    try:
        result = subprocess.run(command, input=encoded(task), capture_output=True,
                                timeout=timeout, env=env, cwd=ROOT)
        stderr = result.stderr.decode(errors="replace")
        if result.returncode:
            record = {"status": "worker_failed", "returncode": result.returncode,
                      "stdout": result.stdout.decode(errors="replace")}
        else:
            try:
                record = json.loads(result.stdout)
            except (ValueError, UnicodeDecodeError):
                record = {"status": "invalid_worker_output",
                          "stdout": result.stdout.decode(errors="replace")}
    except subprocess.TimeoutExpired as exc:
        record = {"status": "hard_timeout", "deadline_seconds": timeout}
        stderr = (exc.stderr or b"").decode(errors="replace")
    record.update(stage=task["stage"], case_id=task["case_id"], stderr=stderr,
                  task_sha256=digest(encoded(task)), parent_seconds=time.monotonic() - started)
    save(output / f'{task["case_id"]}.{task["stage"]}.json', record)
    print(json.dumps({k: record[k] for k in ("case_id", "stage", "status", "parent_seconds")}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--search-seconds", type=float, default=10)
    parser.add_argument("--score-seconds", type=float, default=10)
    parser.add_argument("--prediction-deadline", type=float, default=30)
    parser.add_argument("--seed-policy", choices=("none", "best-frozen"), default="none")
    parser.add_argument("--score-backend", choices=("full-group", "support-stabilizer"), default="full-group")
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--confirmation-lock", type=Path)
    args = parser.parse_args()
    confirmation = None
    if args.confirmation_lock:
        from Experiment.Synister.confirmation_contract import validate_launch, validate_sources
        confirmation = validate_launch(args)
    if args.limit < 1 or not 1 <= args.workers <= 4:
        raise ValueError("Require a positive limit and 1–4 workers")
    if any(x <= 0 for x in (args.search_seconds, args.score_seconds, args.prediction_deadline)):
        raise ValueError("Budgets must be positive")
    if args.selection_manifest:
        selection = json.loads(args.selection_manifest.read_text())
        if selection["dataset_sha256"] != digest(args.dataset.read_bytes()):
            raise ValueError("Selection manifest does not match input dataset")
    args.output.mkdir(parents=True, exist_ok=False)
    records = args.output / "cases"
    records.mkdir()
    with gzip.open(args.dataset, "rt") as stream:
        from itertools import islice
        selected = list(islice(csv.DictReader(stream), args.limit))
    if confirmation and len(selected) != confirmation.get("settings", {"limit":1000})["limit"]:
        raise ValueError("Campaign requires exactly the locked input count")
    locked_scope = ("prospective_C2_replication" if confirmation and
                    confirmation["protocol"] == "identifiability-c2-rhea-v1"
                    else "prospective_C1_confirmation")
    inputs, eligible = [], []
    for index, row in enumerate(selected):
        case = {"case_id": f"case_{index:04d}", "reaction_id": row["reaction_id"],
                "source_line": row["source_line"], "source_row_sha256": digest(encoded(row))}
        try:
            case["reaction"] = unmapped_input(source_reaction(row))
            case["status"] = "eligible"
            eligible.append(case)
        except Exception as exc:
            case.update(status="input_rejected", error=str(exc))
        inputs.append(case)
    save(args.output / "inputs.json", inputs)
    manifest = {
        "protocol": "identifiability-development-v1", "scope": "development_only_first_file_rows",
        "dataset_sha256": digest(args.dataset.read_bytes()),
        "inputs_sha256": digest(encoded(inputs)), "selected": len(inputs), "eligible": len(eligible),
        "source_snapshot_sha256": snapshot(args.output), "environment": environment(),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "resources": "6 GiB address space per worker; one numerical thread; <=4 workers",
        "deadline_policy": "prediction: total process wall; search/score: cooperative budget plus 5s process overhead",
        "prediction_a": "slap-heavy-split-v1", "prediction_b": "rxnmapper-0.4.3",
    }
    if args.selection_manifest:
        manifest["protocol"] = "identifiability-selected-development-v1"
        manifest["scope"] = "development_only_preselected_inputs"
        manifest["selection_manifest_sha256"] = digest(args.selection_manifest.read_bytes())
        manifest["selection_scope"] = selection["scope"]
    if confirmation:
        manifest.update(protocol=confirmation["protocol"], scope=locked_scope,
                        execution_lock_sha256=digest(args.confirmation_lock.read_bytes()),
                        all_sources_sha256=confirmation["all_sources_sha256"])
        save(args.output / "execution_lock.json", confirmation)
        from Experiment.Synister.confirmation_contract import source_contents
        save(args.output / "all_sources.json", source_contents())
    save(args.output / "manifest.json", manifest)
    predictions = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [dict(case, stage=stage) for case in eligible for stage in ("slap", "rxnmapper")]
        for result in pool.map(lambda task: execute(task, args.prediction_deadline, records), tasks):
            predictions[(result["case_id"], result["stage"])] = result
        # Barrier: every prediction attempt exists before any candidate search.
        save(args.output / "prediction_freeze.json", {
            f"{case}.{stage}": digest(encoded(value)) for (case, stage), value in predictions.items()
        })
        if confirmation:
            validate_sources(confirmation)
        tasks = [dict(case, stage="exact", search_seconds=args.search_seconds) for case in eligible]
        if confirmation:
            for task in tasks:
                task["export_joint_labels"] = True
        if args.seed_policy == "best-frozen":
            from Experiment.Synister.coverage_followup import seed_from_predictions
            for task in tasks:
                mapping, seed = seed_from_predictions(task["reaction"], {
                    method: predictions[(task["case_id"], method)] for method in ("slap", "rxnmapper")})
                task.update(initial_mapping=mapping, seed_metadata=seed)
        save(args.output / "search_tasks.json", tasks)
        exact = list(pool.map(lambda task: execute(task, args.search_seconds + 5, records), tasks))
        if confirmation:
            validate_sources(confirmation)
        by_id = {case["case_id"]: case for case in eligible}
        score_tasks = []
        for result in exact:
            key = result["case_id"]
            a, b = predictions[(key, "slap")], predictions[(key, "rxnmapper")]
            if result["status"] == "complete" and a["status"] == b["status"] == "valid":
                score_tasks.append(dict(by_id[key], stage="score", score_seconds=args.score_seconds,
                                        score_backend=args.score_backend,
                                        labels=result["labels"], prediction_a=a["prediction"]["mapping"],
                                        prediction_b=b["prediction"]["mapping"]))
        save(args.output / "score_tasks.json", score_tasks)
        scores = list(pool.map(lambda task: execute(task, args.score_seconds + 5, records), score_tasks))
        if confirmation:
            validate_sources(confirmation)
    common_valid = sum(predictions[(c["case_id"], "slap")]["status"] == "valid"
                       and predictions[(c["case_id"], "rxnmapper")]["status"] == "valid" for c in eligible)
    resolved = [x for x in scores if x["status"] == "complete"]
    lower = sum((Fraction(x["lower"]["difference"]) for x in resolved), Fraction(0))
    upper = sum((Fraction(x["upper"]["difference"]) for x in resolved), Fraction(0))
    summary = {"scope": "development_only", "selected": len(inputs), "eligible": len(eligible),
               "common_valid_predictions": common_valid, "resolved_comparisons": len(resolved),
               "exact_searches_closed": sum(x["status"] == "complete" for x in exact),
               "positive_paired_width_cases": sum(Fraction(x["width"]) > 0 for x in resolved)}
    if confirmation:
        summary["scope"] = locked_scope
    if resolved:
        summary["resolved_conditional_envelope"] = [str(lower / len(resolved)), str(upper / len(resolved))]
    if common_valid:
        unknown = common_valid - len(resolved)
        summary["common_valid_outer_envelope"] = [str((lower - unknown) / common_valid),
                                                  str((upper + unknown) / common_valid)]
        summary["unresolved_weight"] = str(Fraction(unknown, common_valid))
    save(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
