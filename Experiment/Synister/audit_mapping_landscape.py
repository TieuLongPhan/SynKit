"""Replay every saved small-example map/orbit/ITS distribution."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from Experiment.Synister.mapping_landscape import analyze, toy_endpoints
from Experiment.Synister.worked_oracle import REACTION
from synkit.Chem.Mapper.identifiability import parse_reaction


def audit(directory):
    summary = json.loads((directory/"summary.json").read_text())
    if sha256((directory/"sources.json").read_bytes()).hexdigest() != summary["sources_sha256"]:
        raise ValueError("Changed source snapshot")
    cases = {"synthetic_five_vertices": toy_endpoints(), "worked_84_1": parse_reaction(REACTION)}
    if {c["case_id"] for c in summary["cases"]} != set(cases) or len(summary["cases"]) != len(cases):
        raise ValueError("Missing or unexpected example")
    checked = []
    for case in summary["cases"]:
        name = case["case_id"]
        path = directory/f"{name}.json"
        if sha256(path.read_bytes()).hexdigest() != case["record_sha256"]:
            raise ValueError("Changed landscape record")
        stored = json.loads(path.read_text())
        recomputed = json.loads(json.dumps(analyze(name, *cases[name])))
        stored.pop("elapsed_seconds")
        recomputed.pop("elapsed_seconds")
        if stored != recomputed or not recomputed["all_classifications_complete"]:
            raise ValueError("Full mapping/orbit/ITS replay differs")
        minimum = next(r for r in recomputed["rows"] if r["doubled_cd"] == recomputed["minimum_doubled_cd"])
        for key in ("compatible_maps", "product_group_order", "independent_class_comparisons", "all_classifications_complete"):
            if case[key] != recomputed[key]:
                raise ValueError("Summary case accounting mismatch")
        if case["minimum"] != minimum:
            raise ValueError("Minimum summary mismatch")
        checked.append({"case_id": name, "compatible_maps": recomputed["compatible_maps"],
                        "distance_rows": len(recomputed["rows"]), "product_orbits": len(recomputed["orbits"]),
                        "nonempty_distances": sum(bool(r["indexed_maps"]) for r in recomputed["rows"])})
    if not summary["all_passed"]:
        raise ValueError("Summary does not report successful validation")
    return {"all_verified": True, "cases": checked,
            "scope": "Complete literal-map, full product-group and independently checked ITS-class replay",
            "summary_sha256": sha256((directory/"summary.json").read_bytes()).hexdigest(),
            "auditor_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
