"""Compare every ablation against independently scored literal mapping sets."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

from Experiment.Synister.ablation_worker import CONFIGURATIONS, enumerate_variant
from Experiment.Synister.all_distance_oracle import (
    binary_endpoint, literal_sets, map_digest, targets_for, weighted_cases,
)
from Experiment.Synister.enumeration_benchmark import encode
from Experiment.Synister.worked_oracle import REACTION
from synkit.Chem.Mapper.identifiability import parse_reaction


def check(case_id, r, p, *, binary=False):
    expected = literal_sets(r, p)
    records = []
    for target in ["minimal", *targets_for(expected, binary)]:
        wanted = expected.get(min(expected) if target == "minimal" else target, set())
        for variant in CONFIGURATIONS:
            observed = enumerate_variant(r, p, variant=variant, target=target)
            maps = observed.pop("mappings")
            actual = set(maps)
            passed = observed["complete"] and actual == wanted and len(actual) == len(maps)
            if target == "minimal":
                passed = passed and observed["minimum_doubled_cd"] == min(expected)
            records.append({"target_doubled_cd": target, "variant": variant,
                            "passed": passed, "expected_count": len(wanted),
                            "observed_count": len(actual), "missing_count": len(wanted-actual),
                            "extra_count": len(actual-wanted), "duplicate_count": len(maps)-len(actual),
                            "expected_sha256": map_digest(wanted), "observed_sha256": map_digest(actual),
                            "seed_doubled_cd": observed["seed_doubled_cd"],
                            "seed_swaps": observed["seed_swaps"],
                            "variant_seconds": observed["variant_seconds"]})
    return {"case_id": case_id, "queries": records, "passed": all(r["passed"] for r in records)}


def run(output, *, binary_pairs=64, weighted=32, worked=True):
    output.mkdir(parents=True, exist_ok=False)
    source_paths = sorted(Path("synkit/Chem/Mapper").rglob("*.py"))
    source_paths += [Path(__file__), Path("Experiment/Synister/ablation_worker.py"),
                     Path("Experiment/Synister/all_distance_oracle.py"),
                     Path("Experiment/Synister/enumeration_benchmark.py"),
                     Path("Experiment/Synister/global_milp.py"),
                     Path("Experiment/Synister/worked_oracle.py")]
    (output/"sources.json").write_text(encode({str(p): p.read_text() for p in source_paths}))
    # Hash selection over all 4096 graph pairs, independent of search outcomes.
    pairs = sorted(((a, b) for a in range(64) for b in range(64)),
                   key=lambda x: sha256(f"{x[0]},{x[1]}".encode()).hexdigest())[:binary_pairs]
    (output/"manifest.json").write_text(encode({
        "schema": "synister.ablation-literal-controls.v1", "variants": list(CONFIGURATIONS),
        "binary_pairs": pairs, "weighted_cases": weighted, "weighted_seed": 20260920,
        "worked": worked, "sources_sha256": sha256((output/"sources.json").read_bytes()).hexdigest(),
        "scope": "All distances on an input-selected binary subset, seeded weighted graphs and the worked reaction; not exhaustive over all graphs"}))
    started = time.perf_counter()
    def cases():
        for a, b in pairs:
            yield f"binary_{a}_{b}", binary_endpoint(a), binary_endpoint(b), True
        for name, r, p in weighted_cases(weighted):
            yield name, r, p, False
        if worked:
            r, p = parse_reaction(REACTION)
            yield "worked_84_1", r, p, False
    counts = Counter()
    with (output/"cases.jsonl").open("x") as stream:
        for name, r, p, binary in cases():
            record = check(name, r, p, binary=binary)
            stream.write(json.dumps(record, allow_nan=False)+"\n")
            stream.flush()
            counts["cases"] += 1
            counts["queries"] += len(record["queries"])
            counts["failed_cases"] += not record["passed"]
            counts["failed_queries"] += sum(not q["passed"] for q in record["queries"])
            print(json.dumps({"case": name, "passed": record["passed"]}), flush=True)
    summary = {**counts, "all_passed": counts["failed_queries"] == 0,
               "elapsed_seconds": time.perf_counter()-started,
               "records_sha256": sha256((output/"cases.jsonl").read_bytes()).hexdigest(),
               "manifest_sha256": sha256((output/"manifest.json").read_bytes()).hexdigest()}
    (output/"summary.json").write_text(encode(summary))
    print(encode(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-pairs", type=int, default=64)
    parser.add_argument("--weighted", type=int, default=32)
    parser.add_argument("--no-worked", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.binary_pairs <= 4096 or args.weighted < 0:
        parser.error("Invalid case counts")
    result = run(args.output, binary_pairs=args.binary_pairs, weighted=args.weighted, worked=not args.no_worked)
    raise SystemExit(0 if result["all_passed"] else 1)
