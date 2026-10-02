"""Replay C1 frame/rank accounting and inspect every selected source input."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

from Experiment.Synister.development import digest, save
from Experiment.Synister.select_confirmation import normalized, rows
from Experiment.Synister.select_development import endpoint_key, group
from synkit.Chem.Mapper.identifiability import parse_reaction
from synkit.Chem.Mapper.numerical_scope import validate_study_domain


def audit(directory, source, d1, adapter, previous):
    read = lambda name: json.loads((directory / name).read_text())
    manifest = read("manifest.json")
    assert digest(source.read_bytes()) == manifest["source_sha256"]["composite"]
    assert digest(previous.read_bytes()) == manifest["source_sha256"]["synister"]
    for path in (d1, adapter):
        assert digest(path.read_bytes()) == manifest["exclusion_inputs_sha256"][str(path)]
    for name in ("selection", "frame", "accounting", "references", "source_snapshot"):
        assert digest((directory / f"{name}.json").read_bytes()) == manifest[f"{name}_sha256"]
    assert digest((directory / "inputs.csv.gz").read_bytes()) == manifest["dataset_sha256"]
    selected, frame, accounting = read("selection.json"), read("frame.json"), read("accounting.json")
    references = {x["reaction_id"]: x["mapped_reaction"] for x in read("references.json")}
    source_rows = rows(source)
    raw = {x["r_id"]: x for x in source_rows}
    assert len(raw) == len(source_rows) == len(accounting)
    assert {x["r_id"] for x in accounting} == set(raw)
    excluded = {group(x["reaction_id"]) for x in rows(previous)}
    excluded.update(x["source_sequence"] for x in json.loads(d1.read_text()))
    excluded.update(x["source_sequence"] for x in json.loads(adapter.read_text())["records"])
    assert len(excluded) == manifest["excluded_source_groups"]
    candidates = defaultdict(list)
    eligible_statuses = {"selected", "frame_not_selected", "nonrepresentative_row", "duplicate_representative_endpoint"}
    for entry in accounting:
        original = raw[entry["r_id"]]
        assert entry["original_id"] == original["original_id"]
        assert entry["source_sequence"] == group(original["original_id"])
        assert (entry["status"] == "previous_source_group") == (entry["source_sequence"] in excluded)
        if entry["status"] in eligible_statuses:
            candidates[entry["source_sequence"]].append(entry)
    # Reconstruct both ranking stages without calling the selector's choose().
    def h(prefix, value):
        return hashlib.sha256((prefix + "\0" + value).encode()).hexdigest()
    representatives = {g: min(values, key=lambda x: (h("synister-c1-row-v1", x["r_id"]), x["r_id"]))
                       for g, values in candidates.items()}
    unique = {}
    for g in sorted(representatives):
        value = representatives[g]
        unique.setdefault(value["endpoint_sha256"], value)
    expected = sorted(unique.values(), key=lambda x: (h("synister-c1-group-v1", x["source_sequence"]), x["source_sequence"]))
    assert [x["r_id"] for x in expected] == [x["r_id"] for x in frame]
    assert selected == frame[:1000] and len(selected) == 1000
    assert manifest["frame_groups"] == len(frame)
    assert len({x["endpoint_sha256"] for x in selected}) == len(selected)
    for entry in selected:
        original = raw[entry["r_id"]]
        reaction = normalized(original["original_id"], original["ground_truth"])
        assert reaction == entry["reaction"]
        assert endpoint_key(reaction) == entry["endpoint_sha256"]
        r, p = parse_reaction(reaction)
        validate_study_domain(r, p)
        assert len(r.atomic_numbers) == entry["heavy_atoms"]
        assert normalized(entry["original_id"], references[entry["original_id"]]) == reaction
        assert references[entry["original_id"]] == original["ground_truth"]
        assert entry["source_sequence"] not in excluded
    with gzip.open(directory / "inputs.csv.gz", "rt") as stream:
        csv_values = list(csv.DictReader(stream))
    assert csv_values == [dict(source_line=x["r_id"], reaction_id=x["original_id"], mapped_reaction=x["reaction"]) for x in selected]
    assert dict(Counter(x["status"] for x in accounting)) == manifest["row_accounting"]
    return {"scope": "independent rank/accounting replay and selected-source/input/reference checks; unselected input eligibility and historical endpoint exclusions not independently recomputed",
            "manifest_sha256": digest((directory / "manifest.json").read_bytes()),
            "selected": len(selected), "frame_groups": len(frame), "status": "verified"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("directory", "source", "d1", "adapter", "previous", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.source, args.d1, args.adapter, args.previous)
    save(args.output, result)
    print(json.dumps(result, indent=2))
