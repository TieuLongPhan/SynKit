"""Verify and summarize a Synister global-shell campaign directory."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _verify_embedded_digest(
    payload: dict[str, object],
    digest_field: str,
    *,
    context: str,
) -> str:
    """Verify a digest computed before its own field was inserted."""
    claimed = payload.get(digest_field)
    unsigned = dict(payload)
    unsigned.pop(digest_field, None)
    actual = _payload_sha256(unsigned)
    if claimed != actual:
        raise ValueError(f"{context} digest mismatch")
    return actual


def _verified_records(campaign: Path):
    manifest = json.loads((campaign / "manifest.json").read_text(encoding="utf-8"))
    manifest_sha256 = _verify_embedded_digest(
        manifest,
        "manifest_sha256",
        context="campaign manifest",
    )
    records = []
    for path in sorted((campaign / "cases").glob("line_*.json.gz")):
        with gzip.open(path, "rt", encoding="ascii") as stream:
            record = json.load(stream)
        _verify_embedded_digest(
            record,
            "record_sha256",
            context=f"case payload {path}",
        )
        if record.get("campaign_manifest_sha256") != manifest_sha256:
            raise ValueError(f"case manifest binding mismatch: {path}")
        records.append(record)
    return manifest, records


def _mode_summary(records, mode):
    selected = [
        record
        for record in records
        if record.get("status") != "error" and record["shells"][mode]["complete"]
    ]
    shells = [record["shells"][mode] for record in selected]
    if not shells:
        return {"cases": 0}
    alternative_counts = []
    for shell in shells:
        structure = shell["structure"]
        if not structure["complete"]:
            continue
        reference_observed = bool(structure["reference_its_class_observed"])
        alternative_counts.append(
            int(structure["observed_its_class_count"]) - int(reference_observed)
        )
    alternative_histogram = Counter(alternative_counts)
    nonminimal_references = []
    if mode == "minimal":
        for record, shell in zip(selected, shells):
            if (
                shell["minimum_cost"] is not None
                and shell["reference_cd"] > shell["minimum_cost"]
            ):
                nonminimal_references.append(
                    {
                        "source_line": record["source_line"],
                        "reaction_id": record["reaction_id"],
                        "atom_count": record["atom_count"],
                        "reference_cd": shell["reference_cd"],
                        "minimum_cd": shell["minimum_cost"],
                    }
                )
    return {
        "cases": len(shells),
        "atom_count_median": statistics.median(
            record["atom_count"] for record in selected
        ),
        "atom_count_range": [
            min(record["atom_count"] for record in selected),
            max(record["atom_count"] for record in selected),
        ],
        "elapsed_seconds_median": statistics.median(
            shell["elapsed_seconds"] for shell in shells
        ),
        "elapsed_seconds_maximum": max(shell["elapsed_seconds"] for shell in shells),
        "labeled_mapping_count_median": statistics.median(
            int(shell["labeled_solution_count"]) for shell in shells
        ),
        "labeled_mapping_count_maximum": max(
            int(shell["labeled_solution_count"]) for shell in shells
        ),
        "multiple_exact_its_classes": sum(
            shell["structure"]["observed_its_class_count"] > 1 for shell in shells
        ),
        "multiple_exact_template_classes": sum(
            shell["structure"]["observed_template_class_count"] > 1 for shell in shells
        ),
        "alternative_its_application_complete_cases": len(alternative_counts),
        "cases_with_alternative_its": sum(value > 0 for value in alternative_counts),
        "alternative_its_classes_relative_to_reference": sum(alternative_counts),
        "alternative_its_classes_per_case_histogram": {
            str(value): alternative_histogram[value]
            for value in sorted(alternative_histogram)
        },
        "reference_its_class_observed": sum(
            shell["structure"]["reference_its_class_observed"] for shell in shells
        ),
        "reference_not_global_minimum": nonminimal_references,
        "structure_complete": sum(shell["structure"]["complete"] for shell in shells),
        "symmetry_quotient_complete": sum(
            shell["symmetry_quotient_complete"] for shell in shells
        ),
        "unstable_reaction_centres": sum(
            shell["reaction_center"]["bond_union"]
            != shell["reaction_center"]["bond_intersection"]
            or shell["reaction_center"]["atom_union"]
            != shell["reaction_center"]["atom_intersection"]
            for shell in shells
        ),
    }


def summarize(campaign: Path) -> dict[str, object]:
    manifest, records = _verified_records(campaign)
    modes = (
        ("reference_cd", "minimal")
        if manifest["options"]["mode"] == "both"
        else (manifest["options"]["mode"],)
    )
    return {
        "schema_version": 1,
        "campaign_manifest_sha256": manifest["manifest_sha256"],
        "case_records": len(records),
        "modes": {mode: _mode_summary(records, mode) for mode in modes},
        "selection_warning": (
            "All fractions use complete-shell denominators only; this "
            "censored campaign is not a population estimate."
        ),
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    result = summarize(args.campaign)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
