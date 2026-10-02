"""Outcome-blind C1 frame with independent row and group inclusion ranks."""
import argparse
from collections import Counter
import csv
import gzip
import io
import json
from pathlib import Path

from rdkit import rdBase
from Experiment.Synister.development import digest, save, snapshot, source_reaction
from Experiment.Synister.select_development import EXPECTED, endpoint_key, group, stratum
from synkit.Chem.Mapper.identifiability import parse_reaction
from synkit.Chem.Mapper.numerical_scope import validate_study_domain
from synkit.Chem.Mapper.prediction_adapter import unmapped_input


def rank(salt, identifier):
    return digest(f"{salt}\0{identifier}".encode())


def choose(rows, count):
    representatives = {}
    for row in rows:
        source = row["source_sequence"]
        key = (rank("synister-c1-row-v1", row["r_id"]), row["r_id"])
        if source not in representatives or key < representatives[source][0]:
            representatives[source] = (key, row)
    frame, seen = [], set()
    for source in sorted(representatives):
        row = representatives[source][1]
        if row["endpoint_sha256"] not in seen:
            frame.append(dict(row, group_rank=rank("synister-c1-group-v1", source)))
            seen.add(row["endpoint_sha256"])
    frame.sort(key=lambda x: (x["group_rank"], x["source_sequence"]))
    if len(frame) < count:
        raise ValueError(f"Only {len(frame)} groups for requested {count}")
    return frame[:count], frame, representatives


def rows(path):
    with gzip.open(path, "rt") as stream:
        return list(csv.DictReader(stream))


def normalized(identifier, mapped):
    return unmapped_input(source_reaction(dict(reaction_id=identifier, mapped_reaction=mapped)))


def select(args):
    for name, expected in EXPECTED.items():
        assert digest(getattr(args, name).read_bytes()) == expected, name
    adapter = json.loads(args.adapter_selection.read_text())
    assert digest(args.elementary.read_bytes()) == adapter["elementary_dataset_sha256"]
    d1 = json.loads(args.d1_selection.read_text())
    previous, composite = rows(args.synister), rows(args.composite)
    assert len({x["r_id"] for x in composite}) == len(composite)
    excluded = {group(x["reaction_id"]) for x in previous}
    excluded.update(x["source_sequence"] for x in adapter["records"] + d1)
    old = [(x["reaction_id"], x["mapped_reaction"]) for x in previous]
    old += [(x["original_id"], x["ground_truth"]) for x in composite if group(x["original_id"]) in excluded]
    adapter_ids = {x["original_id"] for x in adapter["records"]}
    elementary = rows(args.elementary)
    assert adapter_ids <= {x["original_id"] for x in elementary}
    old += [(x["original_id"], x["ground_truth"]) for x in elementary if x["original_id"] in adapter_ids]
    old_pairs, rejected_old = set(), Counter()
    candidates, accounting, references = [], [], {}
    with rdBase.BlockLogs():
        for identifier, mapped in old:
            try:
                old_pairs.add(endpoint_key(normalized(identifier, mapped)))
            except ValueError as exc:
                rejected_old[str(exc)] += 1
        for row in composite:
            source = group(row["original_id"])
            record = dict(r_id=row["r_id"], original_id=row["original_id"], source_sequence=source)
            if source in excluded:
                record["status"] = "previous_source_group"
            else:
                try:
                    reaction = normalized(row["original_id"], row["ground_truth"])
                    r, p = parse_reaction(reaction)
                    validate_study_domain(r, p)
                    key = endpoint_key(reaction)
                    record.update(endpoint_sha256=key, heavy_atoms=len(r.atomic_numbers), stratum=stratum(len(r.atomic_numbers)))
                    record["status"] = "previous_endpoint_pair" if key in old_pairs else "eligible"
                    if record["status"] == "eligible":
                        candidates.append(dict(record, reaction=reaction))
                        references[row["r_id"]] = source_reaction(dict(reaction_id=row["original_id"], mapped_reaction=row["ground_truth"]))
                except ValueError as exc:
                    record.update(status="unsupported_input", reason=str(exc))
            accounting.append(record)
    chosen, frame, representatives = choose(candidates, 1000)
    frame_ids, selected_ids = ({x["r_id"] for x in collection} for collection in (frame, chosen))
    for record in accounting:
        if record["status"] == "eligible":
            identifier = record["r_id"]
            record["status"] = ("selected" if identifier in selected_ids else
                                "frame_not_selected" if identifier in frame_ids else
                                "nonrepresentative_row" if representatives[record["source_sequence"]][1]["r_id"] != identifier else
                                "duplicate_representative_endpoint")
    assert len({x["source_sequence"] for x in chosen}) == 1000
    assert not ({x["source_sequence"] for x in chosen} & excluded)
    assert not ({x["endpoint_sha256"] for x in chosen} & old_pairs)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("selection", chosen), ("frame", frame), ("accounting", accounting),
                        ("references", [dict(reaction_id=x["original_id"], mapped_reaction=references[x["r_id"]]) for x in chosen])):
        save(args.output / f"{name}.json", value)
    with (args.output / "inputs.csv.gz").open("xb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=("source_line", "reaction_id", "mapped_reaction"))
                writer.writeheader()
                for x in chosen:
                    writer.writerow(dict(source_line=x["r_id"], reaction_id=x["original_id"], mapped_reaction=x["reaction"]))
    manifest = dict(scope="C1_outcome_blind_confirmation_selection", selected=1000, frame_groups=len(frame),
                    excluded_source_groups=len(excluded), excluded_endpoint_pairs=len(old_pairs),
                    old_normalization_failures=dict(rejected_old),
                    row_accounting=dict(Counter(x["status"] for x in accounting)),
                    selected_by_stratum=dict(Counter(x["stratum"] for x in chosen)),
                    source_sha256=EXPECTED, source_snapshot_sha256=snapshot(args.output),
                    protocol_sha256=digest(args.protocol.read_bytes()),
                    exclusion_inputs_sha256={str(p): digest(p.read_bytes()) for p in (args.adapter_selection, args.d1_selection, args.elementary)},
                    dataset_sha256=digest((args.output / "inputs.csv.gz").read_bytes()))
    for name in ("selection", "frame", "accounting", "references"):
        manifest[f"{name}_sha256"] = digest((args.output / f"{name}.json").read_bytes())
    save(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("synister", "composite", "elementary", "adapter-selection", "d1-selection", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    select(parser.parse_args())
