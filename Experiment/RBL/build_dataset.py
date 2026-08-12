#!/usr/bin/env python3
"""Build the paired USPTO-50K dataset used by RBL reconstruction tests.

The raw CSV is joined to the existing atom-mapped, completed reactions by
zero-based CSV row index.  Output records have exactly four fields:

``R_id``
    Stable identifier of the form ``R_<zero-based CSV row index>``.
``raw``
    Canonical, stereo-free form of the potentially incomplete CSV reaction.
``complete``
    Canonical, stereo-free completed reaction with atom-map labels removed.
``aam``
    Completed atom-mapped reaction used to extract the replay rule.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, TextIO

from rdkit import Chem

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synkit.Chem.Reaction.standardize import Standardize  # noqa: E402
from synkit.Graph.Hyrogen.hcomplete import HComplete  # noqa: E402
from synkit.IO import its_to_rsmi, rsmi_to_its  # noqa: E402

DEFAULT_RAW = HERE / "USPTO_50K.csv"
DEFAULT_AAM = ROOT / "Data" / "smart.json.gz"
DEFAULT_OUTPUT = HERE / "uspto_50k_rbl.json.gz"
DEFAULT_REPORT = HERE / "uspto_50k_rbl_build_report.json"


def _file_manifest(path: Path) -> dict[str, Any]:
    """Return stable input/output provenance for one benchmark artifact."""
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        display_path = resolved.as_posix()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": display_path,
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _reaction_balance(rsmi: str) -> tuple[bool, bool]:
    """Return element and net-formal-charge conservation flags."""
    if rsmi.count(">>") != 1:
        return False, False
    inventories = []
    charges = []
    for side in rsmi.split(">>"):
        molecule = Chem.MolFromSmiles(side)
        if molecule is None:
            return False, False
        expanded = Chem.AddHs(molecule)
        inventories.append(Counter(atom.GetSymbol() for atom in expanded.GetAtoms()))
        charges.append(Chem.GetFormalCharge(molecule))
    return inventories[0] == inventories[1], charges[0] == charges[1]


def _mapped_reaction_issue(rsmi: str) -> str | None:
    """Validate complete, side-symmetric atom-map identity with explicit H kept."""
    if rsmi.count(">>") != 1:
        return "invalid_reaction_separator"
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    side_maps = []
    for side_name, side in zip(
        ("reactants", "products"),
        rsmi.split(">>"),
        strict=True,
    ):
        molecule = Chem.MolFromSmiles(side, parameters)
        if molecule is None:
            return f"{side_name}_parse_failed"
        maps = [int(atom.GetAtomMapNum()) for atom in molecule.GetAtoms()]
        if any(atom_map <= 0 for atom_map in maps):
            return f"{side_name}_unmapped_atom"
        if len(maps) != len(set(maps)):
            return f"{side_name}_duplicate_atom_map"
        side_maps.append(
            {
                int(atom.GetAtomMapNum()): atom.GetSymbol()
                for atom in molecule.GetAtoms()
            }
        )
    if side_maps[0] != side_maps[1]:
        return "side_asymmetric_atom_maps"
    return None


def _build_one(
    source_index: int,
    raw_reaction: str,
    source_aam: str,
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """Build one safe record or a structured exclusion."""
    record_id = f"R_{source_index}"
    standardize = Standardize()
    try:
        raw = standardize.fit(
            raw_reaction,
            remove_aam=True,
            ignore_stereo=True,
            remove_invalid=False,
        )
        mapped = standardize.fit(
            source_aam,
            remove_aam=False,
            ignore_stereo=True,
            remove_invalid=False,
        )
        complete = standardize.fit(
            source_aam,
            remove_aam=True,
            ignore_stereo=True,
            remove_invalid=False,
        )
        invalid_fields = [
            name
            for name, value in (
                ("raw", raw),
                ("mapped", mapped),
                ("complete", complete),
            )
            if value is None
        ]
        if invalid_fields:
            return None, {
                "R_id": record_id,
                "reason": "standardization_failed",
                "fields": invalid_fields,
            }

        mapped_issue = _mapped_reaction_issue(mapped)
        if mapped_issue is not None:
            return None, {
                "R_id": record_id,
                "reason": "invalid_source_aam",
                "detail": mapped_issue,
            }

        element_balanced, charge_balanced = _reaction_balance(complete)
        if not element_balanced or not charge_balanced:
            return None, {
                "R_id": record_id,
                "reason": "invalid_ground_truth_balance",
                "element_balanced": element_balanced,
                "charge_balanced": charge_balanced,
            }

        full_its = rsmi_to_its(mapped, core=False, format="tuple")
        completion = HComplete.complete_its(full_its, format="tuple")
        if not completion.ok or not completion.exhaustive:
            return None, {
                "R_id": record_id,
                "reason": "hydrogen_completion_rejected",
                "detail": completion.reason or "no_unambiguous_completion",
                "candidates": completion.candidates,
                "exhaustive": completion.exhaustive,
            }
        completed_aam = its_to_rsmi(
            completion.its,
            format="tuple",
            explicit_hydrogen=True,
        )
        completed_aam = standardize.fit(
            completed_aam,
            remove_aam=False,
            ignore_stereo=True,
            remove_invalid=False,
        )
        if completed_aam is None:
            return None, {
                "R_id": record_id,
                "reason": "completed_aam_standardization_failed",
            }
        completed_issue = _mapped_reaction_issue(completed_aam)
        if completed_issue is not None:
            return None, {
                "R_id": record_id,
                "reason": "invalid_completed_aam",
                "detail": completed_issue,
            }
        completed_ground_truth = standardize.fit(
            completed_aam,
            remove_aam=True,
            ignore_stereo=True,
            remove_invalid=False,
        )
        if completed_ground_truth != complete:
            return None, {
                "R_id": record_id,
                "reason": "hydrogen_completion_changed_ground_truth",
                "before": complete,
                "after": completed_ground_truth,
            }
        return {
            "R_id": record_id,
            "raw": raw,
            "complete": complete,
            "aam": completed_aam,
        }, {
            "R_id": record_id,
            "reason": "accepted",
            "hydrogen_candidates": completion.candidates,
            "hydrogen_exhaustive": completion.exhaustive,
        }
    except Exception as exc:
        return None, {
            "R_id": record_id,
            "reason": "technical_error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _open_json_text(path: Path) -> TextIO:
    """Open JSON text, detecting gzip by content rather than filename."""
    with path.open("rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    if is_gzip:
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _load_raw(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "reactions" not in rows[0]:
        raise ValueError(f"{path} must contain a 'reactions' column.")
    return rows


def _load_aam(path: Path) -> list[dict[str, Any]]:
    with _open_json_text(path) as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"{path} must contain a JSON list.")
    return records


def build_records(
    raw_rows: list[dict[str, str]],
    aam_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Join and validate the paired reactions in stable source-index order."""
    records, _report = build_records_with_report(raw_rows, aam_rows)
    return records


def build_records_with_report(
    raw_rows: list[dict[str, str]],
    aam_rows: list[dict[str, Any]],
    *,
    workers: int = 1,
    chunksize: int = 8,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build safe records and return a complete acceptance/exclusion audit."""
    mapped_by_index: dict[int, str] = {}
    for record in aam_rows:
        if "R-id" not in record or "smart" not in record:
            raise ValueError("Every mapped record requires 'R-id' and 'smart'.")
        source_index = int(record["R-id"])
        if source_index in mapped_by_index:
            raise ValueError(f"Duplicate mapped source index: {source_index}")
        if not 0 <= source_index < len(raw_rows):
            raise IndexError(
                f"Mapped source index {source_index} is outside the CSV row range."
            )
        mapped_by_index[source_index] = str(record["smart"])

    arguments = [
        (
            source_index,
            raw_rows[source_index]["reactions"],
            mapped_by_index[source_index],
        )
        for source_index in sorted(mapped_by_index)
    ]
    if workers == 1:
        results = [_build_one(*argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(
                    _build_one_from_tuple,
                    arguments,
                    chunksize=chunksize,
                )
            )
    records = [record for record, _audit in results if record is not None]
    audits = [audit for _record, audit in results]
    reason_counts = Counter(audit["reason"] for audit in audits)
    report = {
        "source_records": len(arguments),
        "accepted_records": len(records),
        "excluded_records": len(arguments) - len(records),
        "reason_counts": dict(sorted(reason_counts.items())),
        "hydrogen_policy": (
            "full mapped tuple ITS; exhaustive HComplete; exactly one "
            "equivariant completion class"
        ),
        "ground_truth_policy": "element and net-formal-charge balanced",
        "exclusions": [audit for audit in audits if audit["reason"] != "accepted"],
    }
    return records, report


def _build_one_from_tuple(
    arguments: tuple[int, str, str],
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    return _build_one(*arguments)


def write_json_gz(records: list[dict[str, str]], path: Path) -> None:
    """Write deterministic, genuinely gzip-compressed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8") as text_handle:
                json.dump(
                    records,
                    text_handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                text_handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--aam", type=Path, default=DEFAULT_AAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )
    parser.add_argument("--chunksize", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be at least 1")
    records, report = build_records_with_report(
        _load_raw(args.raw),
        _load_aam(args.aam),
        workers=args.workers,
        chunksize=args.chunksize,
    )
    report["inputs"] = {
        "raw": _file_manifest(args.raw),
        "aam": _file_manifest(args.aam),
    }
    write_json_gz(records, args.output)
    report["output"] = _file_manifest(args.output)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} records to {args.output}")
    print(f"Wrote build report to {args.report}")


if __name__ == "__main__":
    main()
