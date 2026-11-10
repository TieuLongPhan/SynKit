"""Control: replay observed forward endpoints without closed-carrier admission.

This deliberately uses SynReactor's existing paired-presence rewriting, not
the atom-preserving WildcardEngine/C++ profile. It asserts observed product
recovery, not the identity or conservation of unreported by-products.
"""

import argparse
from collections import Counter
import csv
import gzip
import itertools
import json
from pathlib import Path
from time import perf_counter

from .benchmark import ROOT, canonical_fragments, contains_fragments, deadline, digest, initialize, prepare
from synkit.Synthesis.Reactor import SynReactor, WildcardEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT.parent / "SynRXN/Data/tables/canonical/synthesis/uspto_50k.csv.gz")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path to preserve earlier evidence")
    initialize({})
    counts, records, started = Counter(), [], perf_counter()
    with gzip.open(args.dataset, "rt") as stream:
        for position, row in enumerate(itertools.islice(csv.DictReader(stream), args.limit)):
            record = {"r_id": row["r_id"], "row": position}
            try:
                with deadline(5):
                    rule, inputs, targets, _, metadata = prepare(row["aam"], "absolute")
                    record.update(left_only_maps=metadata["left_only_maps"],
                                  right_only_maps=metadata["right_only_maps"])
                    if metadata["right_only_maps"]:
                        raise ValueError("This control requires complete observed reactant atoms")
                    engine = SynReactor(
                        ".".join(inputs["fw"]), rule, partial=False, explicit_h=False,
                        automorphism=False, dedup_its=False, stereo_mode="ignore",
                        embed_threshold=2048,
                    )
                    outputs = {canonical_fragments(s.split(">>")[1]) for s in engine.smarts_list}
                    record.update(status="complete", exact=targets["fw"] in outputs,
                                  recorded_product_contained=any(contains_fragments(s, targets["fw"]) for s in outputs),
                                  outcomes=len(outputs))
                    counts["completed"] += 1
                    counts["exact"] += record["exact"]
                    counts["product_side_incomplete_exact"] += bool(metadata["left_only_maps"]) and record["exact"]
            except (BaseException,) as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                record.update(status="error", error_type=type(error).__name__, error=str(error))
                counts["errors"] += 1
            records.append(record)
    # Demonstrate why observed product recovery and whole-output equality differ.
    template = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"
    output = canonical_fragments(WildcardEngine("CBr.[OH-]", template, partial=False).smarts_list[0].split(">>")[1])
    target = canonical_fragments("CO")
    report = {
        "schema": "synkit.forward-admission-diagnostic/1",
        "source": {"path": str(args.dataset.resolve()), "sha256": digest(args.dataset),
                   "selected_rows": len(records), "selection": "prefix in stored order"},
        "engine": "Python SynReactor tuple legacy profile; full matching; absolute center H",
        "script_sha256": digest(__file__), "counts": dict(counts), "records": records,
        "wall_seconds": perf_counter() - started,
        "interpretation": "Observed forward endpoint control, distinct from the closed WildcardEngine/C++ profile",
        "complete_product_example": {"output": ".".join(output), "recorded_product": ".".join(target),
                                     "whole_multiset_equal": output == target,
                                     "recorded_product_contained": contains_fragments(output, target)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": dict(counts), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
