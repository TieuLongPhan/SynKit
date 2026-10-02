"""Retrospective terminal availability, not hypothetical shorter-budget reruns."""

import argparse
import json
import math
from pathlib import Path

from Experiment.Synister.resource_report import report as resource_report, sha

THRESHOLDS = (1, 10, 30, 60, 65, 100)
FLAGS = ("minimum_proved", "enumeration_complete", "symmetry_search_complete",
         "joint_labels_complete")


def elapsed(record):
    value = record.get("parent_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Missing or invalid parent elapsed time")
    return value


def checkpoints(search, score):
    search_time = elapsed(search)
    score_time = elapsed(score) if score is not None else None
    if search.get("status") != "complete" or not all(search.get(k) is True for k in FLAGS):
        return None, None
    combined = (search_time + score_time
                if score is not None and score.get("status") == "complete" else None)
    return search_time, combined


def summarize(rows):
    result = {"selected": len(rows), "thresholds": []}
    for threshold in THRESHOLDS:
        counts = [sum(row[i] is not None and row[i] <= threshold for row in rows)
                  for i in (0, 1)]
        result["thresholds"].append({"seconds": threshold, "search_available": counts[0],
                                     "search_plus_score_available": counts[1],
                                     "search_unavailable": len(rows)-counts[0],
                                     "search_plus_score_unavailable": len(rows)-counts[1]})
    result["final_available"] = {
        "search": sum(row[0] is not None for row in rows),
        "search_plus_score": sum(row[1] is not None for row in rows)}
    return result


def report(directory, protocol):
    resources = resource_report(directory)
    inputs = json.loads((directory / "inputs.json").read_text())
    rows = []
    for item in inputs:
        base = directory / "cases" / item["case_id"]
        search = json.loads(Path(str(base) + ".exact.json").read_text())
        score_path = Path(str(base) + ".score.json")
        score = json.loads(score_path.read_text()) if score_path.exists() else None
        rows.append(checkpoints(search, score))
    return {"scope": __doc__, "profile": summarize(rows),
            "protocol_sha256": sha(protocol), "reporter_sha256": sha(Path(__file__)),
            "inputs_sha256": sha(directory / "inputs.json"),
            "sources_sha256": sha(directory / "all_sources.json"),
            "manifest_sha256": sha(directory / "manifest.json"),
            "summary_sha256": sha(directory / "summary.json"),
            "primary_audit_sha256": resources["primary_audit_sha256"],
            "record_hashes": resources["record_hashes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.directory, args.protocol)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result["profile"], indent=2))
