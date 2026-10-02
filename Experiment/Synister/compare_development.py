"""Verify cross-attempt parity without pooling distinct development contracts."""

import argparse
import json
from pathlib import Path

from Experiment.Synister.audit_development import sha


def compare(old, new):
    assert sha(old / "inputs.json") == sha(new / "inputs.json")
    assert sha(old / "prediction_freeze.json") == sha(new / "prediction_freeze.json")
    cases = json.loads((old / "inputs.json").read_text())
    search_shared, score_shared, exact_copies = [], [], []
    old_closed, new_closed = [], []
    for case in cases:
        if case["status"] != "eligible":
            continue
        name = case["case_id"]
        paths = [directory / "cases" / f"{name}.exact.json" for directory in (old, new)]
        a, b = (json.loads(path.read_text()) for path in paths)
        if sha(paths[0]) == sha(paths[1]):
            exact_copies.append(name)
        if a["status"] == "complete":
            old_closed.append(name)
        if b["status"] == "complete":
            new_closed.append(name)
        if a["status"] == b["status"] == "complete":
            assert a["minimum"] == b["minimum"], name
            labels = lambda result: {tuple(tuple(edit[:2]) for edit in x["label"]["typed_bond_edits"])
                                     for x in result["labels"]}
            assert labels(a) == labels(b), name
            search_shared.append(name)
        paths = [directory / "cases" / f"{name}.score.json" for directory in (old, new)]
        if not all(path.exists() for path in paths):
            continue
        a, b = (json.loads(path.read_text()) for path in paths)
        if a["status"] == b["status"] == "complete":
            for field in ("width", "fixed_bond_labels", "bond_label_orbits"):
                assert a[field] == b[field], (name, field)
            for side in ("lower", "upper"):
                assert a[side]["difference"] == b[side]["difference"], (name, side)
            score_shared.append(name)
    return {
        "scope": "paired development attempt parity; not a speed benchmark or confirmation",
        "old_manifest_sha256": sha(old / "manifest.json"), "new_manifest_sha256": sha(new / "manifest.json"),
        "comparison_source_sha256": sha(Path(__file__)),
        "old_search_closures": len(old_closed), "new_search_closures": len(new_closed),
        "shared_search_closures_agree": len(search_shared),
        "shared_score_closures_agree": len(score_shared),
        "byte_identical_search_records": len(exact_copies),
        "lost_search_closure_ids": sorted(set(old_closed)-set(new_closed)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.old, args.new)
    if args.output:
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2))
