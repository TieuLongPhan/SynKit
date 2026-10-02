"""New development search attempt reusing independently frozen predictions.

All original eligible cases are attempted, not only interesting/failed cases.
The fixed seed is the lower-CD valid frozen prediction, with lexicographic
mapping ties. It supplies an incumbent/order, never a fixed correspondence.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import shutil

from Experiment.Synister.development import digest, encoded, execute, save, snapshot
from synkit.Chem.Mapper.identifiability import extract_label, parse_reaction


def seed_from_predictions(reaction, predictions):
    r, p = parse_reaction(reaction)
    candidates = []
    for method, record in predictions.items():
        if record["status"] == "valid":
            mapping = tuple(record["prediction"]["mapping"])
            cost = extract_label(r, p, mapping).weighted_distance
            candidates.append((cost, mapping, method))
    if not candidates:
        return None, None
    cost, mapping, method = min(candidates)
    return list(mapping), {"method": method, "distance": str(cost),
                           "policy": "minimum-CD_then_lexicographic-mapping_then_method"}


def run(args):
    if not 1 <= args.workers <= 4 or args.search_seconds <= 0 or args.score_seconds <= 0:
        raise ValueError("Invalid worker count or budget")
    parent = args.parent.resolve()
    inputs = json.loads((parent / "inputs.json").read_text())
    old_manifest = json.loads((parent / "manifest.json").read_text())
    old_freeze = json.loads((parent / "prediction_freeze.json").read_text())
    assert digest((parent / "inputs.json").read_bytes()) == old_manifest["inputs_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    cases = args.output / "cases"
    cases.mkdir()
    shutil.copyfile(parent / "inputs.json", args.output / "inputs.json")
    predictions = {}
    for case in inputs:
        if case["status"] != "eligible":
            continue
        for method in ("slap", "rxnmapper"):
            key = f'{case["case_id"]}.{method}'
            source = parent / "cases" / f"{key}.json"
            assert digest(source.read_bytes()) == old_freeze[key]
            predictions[(case["case_id"], method)] = json.loads(source.read_text())
            shutil.copyfile(source, cases / source.name)
    shutil.copyfile(parent / "prediction_freeze.json", args.output / "prediction_freeze.json")
    manifest = dict(old_manifest)
    manifest.update(
        protocol="identifiability-seeded-development-v1",
        scope="development_only_same_frozen_predictions_new_search_attempt",
        parent_manifest_sha256=digest((parent / "manifest.json").read_bytes()),
        source_snapshot_sha256=snapshot(args.output),
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        seed_policy="minimum-CD of valid frozen predictions; lexicographic mapping then method ties",
        deadline_policy="search/score cooperative budget plus 5s external overhead allowance",
        environment_scope="inherited prediction environment; search executed by the recorded command interpreter",
        search_reused=args.reuse_search,
        score_backend=args.score_backend,
    )
    import sys
    manifest["search_python"] = sys.version
    manifest["search_executable"] = sys.executable
    save(args.output / "manifest.json", manifest)
    tasks = []
    for case in inputs:
        if case["status"] != "eligible":
            continue
        mapping, seed = seed_from_predictions(case["reaction"], {
            method: predictions[(case["case_id"], method)] for method in ("slap", "rxnmapper")})
        tasks.append(dict(case, stage="exact", search_seconds=args.search_seconds,
                          initial_mapping=mapping, seed_metadata=seed))
    save(args.output / "search_tasks.json", tasks)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if args.reuse_search:
            results = []
            for task in tasks:
                path = parent / "cases" / f'{task["case_id"]}.exact.json'
                results.append(json.loads(path.read_text()))
                shutil.copyfile(path, cases / path.name)
        else:
            results = list(pool.map(lambda task: execute(task, args.search_seconds + 5, cases), tasks))
        case_by_id = {x["case_id"]: x for x in inputs}
        score_tasks = []
        common_valid = 0
        for result in results:
            key = result["case_id"]
            a, b = predictions[(key, "slap")], predictions[(key, "rxnmapper")]
            valid = a["status"] == b["status"] == "valid"
            common_valid += valid
            if valid and result["status"] == "complete":
                score_tasks.append(dict(case_by_id[key], stage="score", score_seconds=args.score_seconds,
                                        score_backend=args.score_backend,
                                        labels=result["labels"], prediction_a=a["prediction"]["mapping"],
                                        prediction_b=b["prediction"]["mapping"]))
        scored = list(pool.map(lambda task: execute(task, args.score_seconds + 5, cases), score_tasks))
    resolved = [x for x in scored if x["status"] == "complete"]
    lower = sum((Fraction(x["lower"]["difference"]) for x in resolved), Fraction())
    upper = sum((Fraction(x["upper"]["difference"]) for x in resolved), Fraction())
    summary = {
        "scope": "development_only", "selected": len(inputs), "eligible": len(tasks),
        "common_valid_predictions": common_valid, "resolved_comparisons": len(resolved),
        "exact_searches_closed": sum(x["status"] == "complete" for x in results),
        "positive_paired_width_cases": sum(Fraction(x["width"]) > 0 for x in resolved),
    }
    if resolved:
        summary["resolved_conditional_envelope"] = [str(lower/len(resolved)), str(upper/len(resolved))]
    if common_valid:
        unknown = common_valid-len(resolved)
        summary["common_valid_outer_envelope"] = [str((lower-unknown)/common_valid), str((upper+unknown)/common_valid)]
        summary["unresolved_weight"] = str(Fraction(unknown, common_valid))
    save(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-seconds", type=float, default=10)
    parser.add_argument("--score-seconds", type=float, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-backend", choices=("full-group", "support-stabilizer"), default="full-group")
    parser.add_argument("--reuse-search", action="store_true", help="Copy parent exact-search records without rerunning them")
    run(parser.parse_args())
