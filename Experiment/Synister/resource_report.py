"""Descriptive saved-attempt resources, not a matched runtime benchmark."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(records):
    result = {"attempts": len(records),
              "status_counts": dict(Counter(r["status"] for r in records))}
    for field in ("parent_seconds", "worker_seconds", "peak_rss_kib"):
        values = [r[field] for r in records if field in r]
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Invalid resource measurement: {field}")
        result[field] = {"observed": len(values), "missing": len(records)-len(values),
                         "median": median(values) if values else None,
                         "maximum": max(values) if values else None}
    return result


def report(directory):
    read = lambda path: json.loads(path.read_text())
    audit = read(directory / "audit.json")
    for name in ("manifest", "summary"):
        assert sha(directory / f"{name}.json") == audit[f"{name}_sha256"]
    inputs = read(directory / "inputs.json")
    groups, bindings, expected = {}, {}, set()
    for case in inputs:
        key = case["case_id"]
        for stage in ("slap", "rxnmapper", "exact", "score"):
            path = directory / "cases" / f"{key}.{stage}.json"
            if not path.exists():
                if stage != "score":
                    raise ValueError(f"Missing required attempt: {path}")
                continue
            record = read(path)
            assert record["case_id"] == key and record["stage"] == stage
            expected.add(path.name)
            bindings[path.name] = sha(path)
            groups.setdefault(stage, []).append(record)
    assert expected == {p.name for p in (directory / "cases").iterdir()}
    return {"scope": __doc__, "selected_inputs": len(inputs),
            "primary_audit_sha256": sha(directory / "audit.json"),
            "reporter_sha256": sha(Path(__file__)),
            "interpretation": "All saved attempts, including failures. Parent elapsed time includes subprocess overhead; worker time covers the worker invocation. RSS is per-process high-water memory, not aggregate concurrent memory or address-space usage. Missing measurements are not zero. Medians condition on observed measurements, not completion. No cross-method speed claim.",
            "stages": {stage: summarize(rows) for stage, rows in groups.items()},
            "record_hashes": bindings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result["stages"], indent=2))
