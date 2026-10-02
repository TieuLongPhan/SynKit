"""Select a source-disjoint low-complexity FlowER adapter-development cohort."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path


SALT = "synister-template-feasibility-v1"


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("rt", encoding="utf-8", newline="")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group(identifier: str) -> str:
    if ":" not in identifier:
        raise ValueError(f"invalid FlowER identifier {identifier!r}")
    return identifier.split(":", 1)[0]


def _mapped_atom_count(reaction: str) -> int:
    return len(re.findall(r":\d+\]", reaction)) // 2


def select(synister: Path, elementary: Path, *, maximum_atoms: int = 12) -> dict[str, object]:
    """Select one hash-ranked eligible elementary step per source sequence."""
    if maximum_atoms < 1:
        raise ValueError("maximum_atoms must be positive")
    with _open(synister) as stream:
        source_groups = {_group(row["reaction_id"]) for row in csv.DictReader(stream)}
    candidates = {}
    eligible_rows = 0
    with _open(elementary) as stream:
        for row in csv.DictReader(stream):
            original_id = str(row["original_id"])
            group = _group(original_id)
            reaction = str(row["ground_truth"])
            atom_count = _mapped_atom_count(reaction)
            if group in source_groups or atom_count > maximum_atoms:
                continue
            eligible_rows += 1
            rank = hashlib.sha256(f"{SALT}\0{original_id}".encode("utf-8")).hexdigest()
            record = {
                "r_id": int(row["r_id"]),
                "original_id": original_id,
                "source_sequence": group,
                "mapped_atom_count": atom_count,
                "ground_truth_sha256": hashlib.sha256(reaction.encode("utf-8")).hexdigest(),
                "selection_rank_sha256": rank,
            }
            current = candidates.get(group)
            if current is None or rank < current["selection_rank_sha256"]:
                candidates[group] = record
    records = sorted(candidates.values(), key=lambda record: record["selection_rank_sha256"])
    return {
        "schema_version": 1,
        "kind": "synister_template_adapter_feasibility_selection",
        "synister_dataset_sha256": _sha256(synister),
        "elementary_dataset_sha256": _sha256(elementary),
        "selection": {
            "source_sequence_exclusion": True,
            "maximum_mapped_atom_count": maximum_atoms,
            "one_row_per_source_sequence": True,
            "rank_salt": SALT,
            "eligible_rows": eligible_rows,
            "eligible_source_sequences": len(candidates),
        },
        "records": records,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synister", type=Path, required=True)
    parser.add_argument("--elementary", type=Path, required=True)
    parser.add_argument("--maximum-atoms", type=int, default=12)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    result = select(args.synister, args.elementary, maximum_atoms=args.maximum_atoms)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
