"""Audit source-sequence overlap before using FlowER data as held-out input."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def _open_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("rt", encoding="utf-8", newline="")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_group(identifier: str) -> str:
    """Return the FlowER source-sequence prefix of a step/path identifier."""
    if not identifier or ":" not in identifier:
        raise ValueError(f"invalid FlowER source identifier: {identifier!r}")
    return identifier.split(":", 1)[0]


def _read_column(path: Path, column: str) -> list[str]:
    with _open_csv(path) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{path} lacks required {column!r} column")
        values = [str(row[column]) for row in reader]
    if any(not value for value in values):
        raise ValueError(f"{path} contains an empty {column!r} value")
    return values


def _candidate_summary(
    name: str,
    path: Path,
    identifiers: list[str],
    source_identifiers: set[str],
    source_groups: set[str],
):
    groups = [_sequence_group(identifier) for identifier in identifiers]
    direct_overlap = sum(identifier in source_identifiers for identifier in identifiers)
    group_overlap = sum(group in source_groups for group in groups)
    return {
        "name": name,
        "path": str(path),
        "sha256": _sha256(path),
        "rows": len(identifiers),
        "unique_identifiers": len(set(identifiers)),
        "unique_source_sequences": len(set(groups)),
        "direct_identifier_overlap": direct_overlap,
        "source_sequence_overlap": group_overlap,
        "rows_remaining_after_source_sequence_exclusion": len(identifiers) - group_overlap,
        "source_sequences_remaining_after_exclusion": len(set(groups) - source_groups),
    }


def audit(synister: Path, elementary: Path, composite: Path) -> dict[str, object]:
    """Return a deterministic source-sequence overlap audit."""
    synister_ids = _read_column(synister, "reaction_id")
    source_groups = {_sequence_group(identifier) for identifier in synister_ids}
    return {
        "schema_version": 1,
        "kind": "synister_flower_source_sequence_overlap_audit",
        "synister": {
            "path": str(synister),
            "sha256": _sha256(synister),
            "rows": len(synister_ids),
            "unique_reaction_ids": len(set(synister_ids)),
            "unique_source_sequences": len(source_groups),
        },
        "candidates": [
            _candidate_summary(
                "flower_elementary",
                elementary,
                _read_column(elementary, "original_id"),
                set(synister_ids),
                source_groups,
            ),
            _candidate_summary(
                "flower_composite",
                composite,
                _read_column(composite, "original_id"),
                set(synister_ids),
                source_groups,
            ),
        ],
        "interpretation": (
            "A candidate row sharing a source-sequence prefix with the Synister "
            "cohort is excluded from a held-out study. Raw row or path IDs alone "
            "do not establish independence."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synister", required=True, type=Path)
    parser.add_argument("--elementary", required=True, type=Path)
    parser.add_argument("--composite", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    result = audit(args.synister, args.elementary, args.composite)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
