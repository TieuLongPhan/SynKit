"""Execute the prespecified 150-case R1 binary-objective sensitivity cohort."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import shutil

from Experiment.Synister.audit_development import sha
from Experiment.Synister.binary_sensitivity import seed
from Experiment.Synister.confirmation_contract import source_contents
from Experiment.Synister.development import digest, encoded, execute, save
from Experiment.Synister.freeze_environment import closure, ROOTS
from synkit.Chem.Mapper.identifiability import parse_reaction


def select(rows):
    assert len({r["original_id"] for r in rows}) == len(rows)
    return sorted(rows, key=lambda r: (digest(("synister-r1-objective-v1\0"+r["original_id"]).encode()),
                                      r["original_id"]))[:150]


def run(parent, selection, protocol, output):
    read = lambda p: json.loads(p.read_text())
    assert sha(protocol) == "7e3bd15154ce6eabba80277a32fb5875bf72821bcf0831d86691e0453c00f9be"
    assert sha(selection) == "d01c56e6eaab47232cdb2feefe45fac20c894683b739a66b2d8467e1423bf363"
    audit, manifest = read(parent / "audit.json"), read(parent / "manifest.json")
    assert sha(parent / "manifest.json") == audit["manifest_sha256"]
    assert sha(parent / "summary.json") == audit["summary_sha256"]
    assert sha(parent / "inputs.json") == manifest["inputs_sha256"]
    inputs = {r["reaction_id"]: r for r in read(parent / "inputs.json")}
    selected = select(read(selection))
    assert len(selected) == 150
    freeze = read(parent / "prediction_freeze.json")
    output.mkdir(parents=True, exist_ok=False)
    records = output / "cases"
    records.mkdir()
    tasks, bindings, predictions = [], {}, {}
    for row in selected:
        case = inputs[row["original_id"]]
        assert row["reaction"] == case["reaction"]
        key = case["case_id"]
        preds = {}
        for stage in ("slap", "rxnmapper", "exact", "score"):
            name = f"{key}.{stage}.json"
            path = parent / "cases" / name
            bindings[name] = sha(path)
            shutil.copyfile(path, records / name)
            if stage in ("slap", "rxnmapper"):
                assert sha(path) == freeze[f"{key}.{stage}"]
                preds[stage] = read(path)
        predictions[key] = preds
        mapping, info = seed(*parse_reaction(case["reaction"]), preds)
        tasks.append(dict(case, stage="binary_exact", search_seconds=60.0,
                          initial_mapping=mapping, seed_metadata=info))
    save(output / "selection.json", selected)
    save(output / "search_tasks.json", tasks)
    save(output / "parent_records.json", bindings)
    save(output / "all_sources.json", source_contents())
    source_hash = sha(output / "all_sources.json")
    save(output / "manifest.json", {"scope": "R1 secondary binary-objective sensitivity",
         "protocol_sha256": sha(protocol), "parent_manifest_sha256": sha(parent / "manifest.json"),
         "parent_audit_sha256": sha(parent / "audit.json"), "parent_selection_sha256": sha(selection),
         "selection_sha256": sha(output / "selection.json"), "search_tasks_sha256": sha(output / "search_tasks.json"),
         "parent_records_sha256": sha(output / "parent_records.json"), "all_sources_sha256": source_hash,
         "packages": closure(ROOTS), "settings": {"workers": 4, "search_internal": 60,
         "search_external": 65, "score_internal": 30, "score_external": 35, "memory_gib": 6,
         "numerical_threads": 1, "emitted_map_cap": 100000, "product_compression": False},
         "policy": "original weighted labels and frozen predictions; independent binary optimization; no replacements"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        exact = list(pool.map(lambda task: execute(task,65,records), tasks))
        assert digest(encoded(source_contents())) == source_hash
        jobs = []
        by_id = {r["case_id"]: r for r in inputs.values()}
        for result in exact:
            key = result["case_id"]
            preds = predictions[key]
            if result["status"] == "complete" and all(p["status"] == "valid" for p in preds.values()):
                jobs.append(dict(by_id[key], stage="score", score_seconds=30.0,
                                 score_backend="support-stabilizer", labels=result["labels"],
                                 prediction_a=preds["slap"]["prediction"]["mapping"],
                                 prediction_b=preds["rxnmapper"]["prediction"]["mapping"]))
        # Keep the copied primary score records untouched in cases/.
        score_records = output / "binary_scores"
        score_records.mkdir()
        save(output / "score_tasks.json", jobs)
        scores = list(pool.map(lambda task: execute(task,35,score_records), jobs))
    assert digest(encoded(source_contents())) == source_hash
    assert all(sha(records/name) == value for name,value in bindings.items())
    summary = {"selected":150,"binary_search_status":dict(Counter(x["status"] for x in exact)),
               "binary_score_attempts":len(jobs),"binary_score_status":dict(Counter(x["status"] for x in scores)),
               "scope":"R1 execution accounting only; scientific audit and cohort report pending"}
    save(output / "summary.json", summary)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for name in ("parent","selection","protocol","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    args=parser.parse_args()
    run(args.parent,args.selection,args.protocol,args.output)
