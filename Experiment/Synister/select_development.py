"""Input-only, source-disjoint and endpoint-deduplicated D1 selection."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

from rdkit import Chem, rdBase

from Experiment.Synister.development import digest, save, source_reaction
from synkit.Chem.Mapper.prediction_adapter import unmapped_input
from synkit.Chem.Mapper.identifiability import parse_reaction

SALT = "synister-identifiability-d1-v1"
EXPECTED = {
    "synister": "e40647847169a7fc98af1aab44fa81c7a64deb70b77085ae3deb3925e39642b0",
    "composite": "75f34235813ce9accff59134536f6e1f0f6af5c9a16a381e7a75e96b0f52d203",
}


def group(identifier):
    if ":" not in identifier:
        raise ValueError("Expected audited FlowER source-sequence identifier")
    return identifier.split(":", 1)[0]


def endpoint_key(reaction):
    """Unoriented full-endpoint key includes every component and multiplicity."""
    sides = [Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)
             for s in reaction.split(">>")]
    return digest("\0".join(sorted(sides)).encode())


def stratum(n):
    return next((name for limit, name in ((20, "le20"), (40, "21to40"), (60, "41to60"),
                                          (80, "61to80")) if n <= limit), "gt80")


def select(args):
    for kind in EXPECTED:
        assert digest(getattr(args, kind).read_bytes()) == EXPECTED[kind], f"Source hash mismatch: {kind}"
    if args.per_stratum < 1:
        raise ValueError("Positive per-stratum count required")
    args.output.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.synister, "rt") as f:
        previous = list(csv.DictReader(f))
    excluded = {group(row["reaction_id"]) for row in previous}
    adapter = json.loads(args.adapter_selection.read_text())
    excluded.update(row["source_sequence"] for row in adapter["records"])
    previous_endpoints = set()
    rejected_previous = Counter()
    with rdBase.BlockLogs():
        for row in previous:
            try:
                previous_endpoints.add(endpoint_key(unmapped_input(source_reaction(row))))
            except ValueError as exc:
                rejected_previous[str(exc)] += 1
    candidates, accounting, counts = {}, [], Counter()
    with gzip.open(args.composite, "rt") as f, rdBase.BlockLogs():
        for row in csv.DictReader(f):
            source_group = group(row["original_id"])
            record = {"r_id": row["r_id"], "original_id": row["original_id"], "source_sequence": source_group}
            if source_group in excluded:
                record["status"] = "previous_source_group"
            else:
                try:
                    raw = source_reaction({"reaction_id": row["original_id"], "mapped_reaction": row["ground_truth"]})
                    reaction = unmapped_input(raw)
                    key = endpoint_key(reaction)
                    n = len(parse_reaction(reaction)[0].atomic_numbers)
                    record.update(endpoint_sha256=key, heavy_atoms=n, stratum=stratum(n))
                    if key in previous_endpoints:
                        record["status"] = "previous_endpoint_pair"
                    else:
                        record["status"] = "eligible_before_group_selection"
                        rank = hashlib.sha256(f'{SALT}\0{row["original_id"]}'.encode()).hexdigest()
                        record["rank"] = rank
                        candidate = dict(record, reaction=reaction, reference=raw)
                        if source_group not in candidates or rank < candidates[source_group]["rank"]:
                            candidates[source_group] = candidate
                except ValueError as exc:
                    record.update(status="unsupported_input", reason=str(exc))
            counts[record["status"]] += 1
            accounting.append(record)
    # One representative per group, then full-endpoint/reversal deduplication.
    distinct, seen = [], set()
    for candidate in sorted(candidates.values(), key=lambda x: x["rank"]):
        if candidate["endpoint_sha256"] not in seen:
            distinct.append(candidate)
            seen.add(candidate["endpoint_sha256"])
    chosen, selected_counts = [], Counter()
    for candidate in distinct:
        if selected_counts[candidate["stratum"]] < args.per_stratum:
            chosen.append(candidate)
            selected_counts[candidate["stratum"]] += 1
    assert len({x["source_sequence"] for x in chosen}) == len(chosen)
    assert not ({x["source_sequence"] for x in chosen} & excluded)
    save(args.output / "references.json", [{"reaction_id": x["original_id"], "mapped_reaction": x["reference"]} for x in chosen])
    public = [{k: v for k, v in x.items() if k != "reference"} for x in chosen]
    save(args.output / "selection.json", public)
    save(args.output / "accounting.json", accounting)
    with (args.output / "inputs.csv.gz").open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=("source_line", "reaction_id", "mapped_reaction"))
                writer.writeheader()
                for x in chosen:
                    writer.writerow({"source_line": x["r_id"], "reaction_id": x["original_id"], "mapped_reaction": x["reaction"]})
    manifest = {
        "scope": "D1_development_only_equal_input_strata",
        "salt": SALT, "source_sha256": EXPECTED,
        "adapter_selection_sha256": digest(args.adapter_selection.read_bytes()),
        "selector_sha256": digest(Path(__file__).read_bytes()),
        "excluded_source_groups": len(excluded), "previous_eligible_endpoint_pairs": len(previous_endpoints),
        "previous_unsupported": dict(rejected_previous), "row_accounting": dict(counts),
        "eligible_groups_before_endpoint_deduplication": len(candidates),
        "groups_after_endpoint_deduplication": len(distinct),
        "available_by_stratum": dict(Counter(x["stratum"] for x in distinct)),
        "selected_by_stratum": dict(selected_counts), "selected": len(chosen),
        "per_stratum_requested": args.per_stratum,
        "source_line_semantics": "FlowER composite r_id, not historical Synister source_line",
        "dataset_sha256": digest((args.output / "inputs.csv.gz").read_bytes()),
        "selection_sha256": digest((args.output / "selection.json").read_bytes()),
        "accounting_sha256": digest((args.output / "accounting.json").read_bytes()),
        "reference_sha256": digest((args.output / "references.json").read_bytes()),
        "interpretation": "Input-stratified development sample, not population representative or independent confirmation; unknown pretrained training overlap",
    }
    save(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synister", type=Path, required=True)
    parser.add_argument("--composite", type=Path, required=True)
    parser.add_argument("--adapter-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=20)
    select(parser.parse_args())
