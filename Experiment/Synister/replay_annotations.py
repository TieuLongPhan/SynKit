"""Bounded computational replay of frozen annotation results.

Uses the production orbit/canonical engines: reproducibility, not an
independent mathematical proof. Independent witness checks are a separate audit.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

from Experiment.Synister.development import digest, encoded, execute, save, snapshot
from Experiment.Synister.confirmation_contract import source_contents


def compare(old, new):
    """Require every previously completed result, allowing new closures only."""
    assert old["reference"] == new["reference"], "Reference cost/membership changed"
    for name, metric in old["metrics"].items():
        if metric["status"] != "complete":
            continue
        replay = new["metrics"][name]
        assert replay["status"] == "complete", f"Replay unresolved: {name}"
        for field in ("width", "fixed_labels", "label_orbits"):
            assert metric[field] == replay[field], (name, field)
        for bound in ("lower", "upper"):
            assert metric[bound]["difference"] == replay[bound]["difference"], (name, bound)
        if metric.get("reference_status") == "complete":
            assert replay.get("reference_status") == "complete", name
            for field in ("reference_orbit_in_minimum", "reference_scores"):
                assert metric[field] == replay[field], (name, field)
    if old["structure"]["status"] == "complete":
        # Serialization normalization accounts for tuple/list interchange.
        assert json.dumps(old["structure"], sort_keys=True) == json.dumps(new["structure"], sort_keys=True)
    if old["policies"]:
        assert json.dumps(old["policies"], sort_keys=True) == json.dumps(new["policies"], sort_keys=True)


def replay_case(task):
    from synkit.Chem.Mapper.annotation_evaluation import analyze_annotations
    from synkit.Chem.Mapper.identifiability import parse_reaction
    from synkit.Chem.Mapper.prediction_adapter import align_mapped_prediction

    r, p = parse_reaction(task["reaction"])
    reference = align_mapped_prediction(task["reaction"], task["reference"])
    assert json.loads(json.dumps(asdict(reference))) == task["old"]["reference_input"]["alignment"]
    output = analyze_annotations(
        r, p, [x["mapping"] for x in task["joint_labels"]],
        task["prediction_a"], task["prediction_b"],
        joint_labels_complete=task["joint_labels_complete"], minimum=task["minimum"],
        reference_mapping=reference.mapping, metric_seconds=20, canonical_seconds=1)
    compare(task["old"], output)
    return {"status": "verified", "reference_mapping_in_minimum": output["reference"]["mapping_in_minimum"],
            "metrics_replayed": [name for name, value in task["old"]["metrics"].items()
                                 if value["status"] == "complete"],
            "policies_replayed": list(task["old"]["policies"]),
            "replayed": output}


def run(directory, references, output, primary=None):
    read = lambda path: json.loads(path.read_text())
    manifest = read(directory / "manifest.json")
    assert digest(references.read_bytes()) == manifest["references_sha256"]
    refs = {x["reaction_id"]: x["mapped_reaction"] for x in read(references)}
    tasks = read(directory / "annotation_tasks.json")
    secondary = manifest.get("scope") == "C1_secondary_descriptive_annotations"
    if secondary:
        assert primary is not None, "C1 reused searches require the primary archive"
        assert digest((primary / "manifest.json").read_bytes()) == manifest["parent_manifest_sha256"]
        assert digest((directory / "annotation_tasks.json").read_bytes()) == manifest["annotation_tasks_sha256"]
        assert digest((directory / "all_sources.json").read_bytes()) == manifest["all_sources_sha256"]
        assert digest((directory / "parent_records.json").read_bytes()) == manifest["parent_records_sha256"]
        for name, expected in read(directory / "parent_records.json").items():
            assert digest((directory / "cases" / name).read_bytes()) == expected
            assert digest((primary / "cases" / name).read_bytes()) == expected
        witness_audit = read(directory / "audit.json")
        assert witness_audit["manifest_sha256"] == digest((directory / "manifest.json").read_bytes())
        assert witness_audit["summary_sha256"] == digest((directory / "summary.json").read_bytes())
        searches = read(primary / "search_tasks.json")
    else:
        searches = read(directory / "search_tasks.json")
    assert all("reference" not in task for task in searches)
    output.mkdir(parents=True, exist_ok=False)
    cases = output / "cases"
    cases.mkdir()
    bindings, jobs, skipped = {}, [], []
    for task in tasks:
        key = task["case_id"]
        assert task["reference"] == refs[task["reaction_id"]]
        exact_path = directory / "cases" / f"{key}.exact.json"
        exact = read(exact_path)
        assert task["joint_labels"] == exact["joint_labels"]
        assert task["minimum"] == exact["minimum"]
        assert task["joint_labels_complete"] == exact["joint_labels_complete"]
        for name, method in (("a", "slap"), ("b", "rxnmapper")):
            path = directory / "cases" / f"{key}.{method}.json"
            assert task[f"prediction_{name}"] == read(path)["prediction"]["mapping"]
            bindings[str(path)] = digest(path.read_bytes())
        path = directory / "cases" / f"{key}.annotations.json"
        old = read(path)
        assert old["task_sha256"] == digest(encoded(task))
        task["old"] = old
        task["stage"] = "annotation_replay"
        bindings[str(path)] = digest(path.read_bytes())
        bindings[str(exact_path)] = digest(exact_path.read_bytes())
        if old["status"] != "evaluated" or old.get("reference_input", {}).get("status") != "valid":
            skipped.append({"case_id": key, "status": old["status"],
                            "reason": "no evaluated result with valid reference to replay"})
        else:
            jobs.append(task)
    save(output / "all_sources.json", source_contents())
    sources_hash = digest((output / "all_sources.json").read_bytes())
    save(output / "skipped.json", skipped)
    save(output / "replay_tasks.json", jobs)
    save(output / "manifest.json", {
        "scope": "computational replay using same engines; not independent proof",
        "parent_manifest_sha256": digest((directory / "manifest.json").read_bytes()),
        "references_sha256": digest(references.read_bytes()), "artifact_sha256": bindings,
        "source_snapshot_sha256": snapshot(output),
        "all_sources_sha256": sources_hash,
        "tasks_sha256": digest((output / "replay_tasks.json").read_bytes()),
        "skipped_sha256": digest((output / "skipped.json").read_bytes()),
        "resources": "4 processes, 6 GiB each, 180s external deadline per case; 20s per metric; 1s per canonical query"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda task: execute(task, 180, cases), jobs))
    assert digest(encoded(source_contents())) == sources_hash
    assert all(digest(Path(path).read_bytes()) == value for path, value in bindings.items())
    summary = {"selected_annotation_tasks": len(tasks), "not_replayable": len(skipped),
               "attempts": len(results), "status": dict(Counter(x["status"] for x in results)),
               "reference_mapping_outside_minimum": sum(not x["reference_mapping_in_minimum"]
                                                          for x in results if x["status"] == "verified"),
               "all_attempted_verified": bool(results) and all(x["status"] == "verified" for x in results),
               "all_verified": not skipped and bool(results) and all(x["status"] == "verified" for x in results)}
    save(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary", type=Path)
    args = parser.parse_args()
    run(args.directory, args.references, args.output, args.primary)
