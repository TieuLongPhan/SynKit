"""Recompute literal distance sets and reconcile every saved B0 query."""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from Experiment.Synister.all_distance_oracle import (
    binary_endpoint, literal_sets, map_digest, targets_for, weighted_cases,
)
from Experiment.Synister.worked_oracle import REACTION
from synkit.Chem.Mapper.identifiability import Endpoint, parse_reaction


def endpoint(record):
    return Endpoint(tuple(record["atomic_numbers"]), tuple(record["charges"]),
                    tuple(record["hcounts"]), tuple(tuple(b) for b in record["bonds"]))


def audit(directory):
    digest = lambda path: sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((directory/"manifest.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    selection = json.loads((directory/"real_selection.json").read_text())
    if digest(directory/"manifest.json") != summary["manifest_sha256"] or digest(directory/"cases.jsonl") != summary["records_sha256"]:
        raise ValueError("Manifest or record digest mismatch")
    if digest(directory/"real_selection.json") != manifest["real_selection_sha256"]:
        raise ValueError("Selection digest mismatch")
    sources = json.loads((directory/"sources.json").read_text())
    if {p: sha256(text.encode()).hexdigest() for p, text in sources.items()} != manifest["source_sha256"]:
        raise ValueError("Source snapshot mismatch")
    expected = {}
    for index in range(manifest["binary_pairs_requested"]):
        a, b = divmod(index, 64)
        expected[f"binary_{a:02d}_{b:02d}"] = (binary_endpoint(a), binary_endpoint(b), True)
    for key, r, p in weighted_cases(manifest["weighted_pairs_requested"], manifest["seed"]):
        expected[key] = (r, p, False)
    if manifest["worked_reaction_requested"]:
        expected["worked_84_1"] = (*parse_reaction(REACTION), False)
    for index, row in enumerate(selection["selected"]):
        expected[f"real_{index:03d}"] = (*parse_reaction(row["reaction"]), False)
    seen, totals, groups = set(), Counter(), {}
    with (directory/"cases.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            key = record["case_id"]
            if key in seen or key not in expected:
                raise ValueError("Repeated or unexpected case")
            seen.add(key)
            r, p = endpoint(record["reactant"]), endpoint(record["product"])
            if (r, p, record["binary"]) != expected[key]:
                raise ValueError("Executed endpoint differs from input selection")
            oracle = literal_sets(r, p)
            minimum = min(oracle)
            histogram = {str(k): len(v) for k, v in sorted(oracle.items())}
            if histogram != record["histogram"] or record["minimum_doubled_cd"] != minimum:
                raise ValueError("Literal distance partition disagrees")
            if record["compatible_maps"] != sum(map(len, oracle.values())):
                raise ValueError("Compatible permutation accounting differs")
            required = {(target, expanded) for target in ["minimal", *targets_for(oracle, record["binary"])]
                        for expanded in (False, True)}
            identities = [(q["target_doubled_cd"], q["symmetry_expansion"]) for q in record["queries"]]
            if set(identities) != required or len(identities) != len(required):
                raise ValueError("Missing or duplicate distance/mode query")
            hashes = {cost: map_digest(mappings) for cost, mappings in oracle.items()}
            for q in record["queries"]:
                cost = minimum if q["target_doubled_cd"] == "minimal" else q["target_doubled_cd"]
                count, hashed = len(oracle.get(cost, ())), hashes.get(cost, map_digest(()))
                if (q["expected_count"] != count or q["observed_count"] != count or
                    q["expected_sha256"] != hashed or q["observed_sha256"] != hashed or
                    not q["complete"] or not q["passed"] or not q["minimum_ok"] or
                    any(q[f"{name}_count"] for name in ("missing", "extra", "duplicate"))):
                    raise ValueError(f"Mapping-set comparison failed for {key}")
                totals["queries"] += 1
                totals["empty_queries"] += count == 0
            group = key.split("_")[0]
            stats = groups.setdefault(group, Counter())
            stats["cases"] += 1
            stats["queries"] += len(record["queries"])
            stats["compatible_maps"] += record["compatible_maps"]
            totals["cases"] += 1
            totals["compatible_maps"] += record["compatible_maps"]
    if seen != set(expected) or any(summary[key] != value for key, value in totals.items()):
        raise ValueError("Case/summary accounting mismatch")
    if not summary["all_passed"] or summary["failed_cases"] or summary["failed_queries"]:
        raise ValueError("Summary records a failure")
    return {"schema": "synister.all-distance-audit.v1", "all_verified": True,
            **dict(totals), "groups": {k: dict(v) for k, v in groups.items()},
            "records_sha256": digest(directory/"cases.jsonl"),
            "summary_sha256": digest(directory/"summary.json"),
            "manifest_sha256": digest(directory/"manifest.json"),
            "auditor_sha256": digest(Path(__file__)),
            "scope": "Fresh literal oracle recomputation and saved complete-set result checks; not a fresh production-search rerun"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.directory)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
