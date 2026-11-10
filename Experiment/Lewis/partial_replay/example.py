"""Inspect a real USPTO replay: python -m Experiment.Lewis.partial_replay.example."""

import argparse
import csv
import gzip
import json
from pathlib import Path

from .benchmark import ROOT, initialize, process
from synkit.Synthesis.Reactor import build_native_reactor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", default="uspto_50k_232")
    parser.add_argument("--backend", choices=("cpp", "python"), default="cpp")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT.parent / "SynRXN/Data/tables/canonical/synthesis/uspto_50k.csv.gz")
    args = parser.parse_args()
    with gzip.open(args.dataset, "rt") as stream:
        selected = next(((i, row) for i, row in enumerate(csv.DictReader(stream))
                         if row["r_id"] == args.id), None)
    if selected is None:
        parser.error(f"Record {args.id} is absent from the source")
    library = build_native_reactor(ROOT / "build/partial/native") if args.backend == "cpp" else None
    initialize({"backend": args.backend, "library": str(library) if library else None, "seconds": 10,
                "candidate_limit": 2048, "hydrogen_policy": "absolute"})
    result = process(selected)
    print(result["r_id"], result["status"])
    print("Reaction:", selected[1]["aam"])
    if "runs" not in result:
        print(result["error"])
        print("Left-only maps:", result.get("left_only_maps"))
        print("Right-only maps:", result.get("right_only_maps"))
        return
    print("Center:", result["core_atoms"], "atoms,", result["core_edges"], "edges")
    for variant, directions in result["runs"].items():
        for direction, value in directions.items():
            print(variant, direction, json.dumps({key: value[key] for key in
                  ("status", "exact", "withheld_components", "exact_with_introduced_atoms", "error")
                  if key in value}))
            if variant == "partial_largest" and value["exact"]:
                print("Witness:", value["exact_witness"])


if __name__ == "__main__":
    main()
