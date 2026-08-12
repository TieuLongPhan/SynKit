#!/usr/bin/env python3
"""Retry raw fast-track failures using the complete reaction as input.

The same normalized SynRule extracted from ``aam`` is replayed, but the RBL
input is ``complete`` rather than ``raw``.  The script also audits elemental
and formal-charge conservation for every prior non-exact candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import io
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.RBL.benchmark import _evaluate_unbounded  # noqa: E402

DEFAULT_INPUT = HERE / "fast_track_all.json.gz"
DEFAULT_OUTPUT = HERE / "complete_retry_hcomplete.json.gz"
DEFAULT_SUMMARY = HERE / "complete_retry_hcomplete_summary.json"


def reaction_balance(rsmi: str) -> dict[str, Any]:
    """Audit element conservation, charge conservation, and neutrality."""
    if rsmi.count(">>") != 1:
        return {"parsed": False, "error": "invalid_reaction_separator"}
    sides = rsmi.split(">>", 1)
    inventories = []
    for side_name, side in zip(("reactants", "products"), sides, strict=True):
        molecule = Chem.MolFromSmiles(side)
        if molecule is None:
            return {"parsed": False, "error": f"{side_name}_parse_failed"}
        with_hydrogen = Chem.AddHs(molecule)
        elements = Counter(atom.GetSymbol() for atom in with_hydrogen.GetAtoms())
        inventories.append(
            {
                "formula": CalcMolFormula(molecule),
                "elements": dict(sorted(elements.items())),
                "formal_charge": sum(
                    atom.GetFormalCharge() for atom in molecule.GetAtoms()
                ),
            }
        )
    reactants, products = inventories
    element_delta = Counter(products["elements"])
    element_delta.subtract(reactants["elements"])
    element_delta = Counter(
        {element: count for element, count in element_delta.items() if count}
    )
    charge_delta = products["formal_charge"] - reactants["formal_charge"]
    return {
        "parsed": True,
        "reactants": reactants,
        "products": products,
        "element_delta_products_minus_reactants": dict(sorted(element_delta.items())),
        "charge_delta_products_minus_reactants": charge_delta,
        "element_balanced": not element_delta,
        "charge_balanced": charge_delta == 0,
        "element_and_charge_balanced": not element_delta and charge_delta == 0,
        "both_sides_neutral": (
            reactants["formal_charge"] == products["formal_charge"] == 0
        ),
    }


def _quiet_worker() -> None:
    logging.disable(logging.CRITICAL)


def _retry_one(record: dict[str, Any]) -> dict[str, Any]:
    retry = _evaluate_unbounded(record, "fast_track", input_field="complete")
    return {
        "R_id": record["R_id"],
        "prior_status": record["status"],
        "retry_status": retry["status"],
        "seconds": retry["seconds"],
        "n_candidates": retry.get("n_candidates", 0),
        "candidates": retry.get("candidates", []),
        "stop_mode": retry.get("stop_mode"),
        "stop_reason": retry.get("stop_reason"),
        "error": retry.get("error"),
        "hydrogen_completion": retry.get("hydrogen_completion")
        or retry.get("rule_audit", {}).get("hydrogen_completion"),
        "normalization_violations": retry.get("rule_audit", {}).get(
            "normalization_violations"
        ),
    }


def _sample_evenly(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return list(records)
    if limit == 1:
        return [records[0]]
    indices = [
        round(index * (len(records) - 1) / (limit - 1)) for index in range(limit)
    ]
    return [records[index] for index in indices]


def _write_gzip_json(payload: dict[str, Any], path: Path) -> None:
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
                    payload,
                    text_handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                text_handle.write("\n")


def _balance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("raw", "complete", "candidate")
    counts = {field: Counter() for field in fields}
    audited = []
    for record in records:
        candidate = record["candidates"][0]
        reports = {
            "raw": reaction_balance(record["raw"]),
            "complete": reaction_balance(record["complete"]),
            "candidate": reaction_balance(candidate),
        }
        for field, report in reports.items():
            counts[field]["parsed"] += bool(report.get("parsed"))
            for key in (
                "element_balanced",
                "charge_balanced",
                "element_and_charge_balanced",
                "both_sides_neutral",
            ):
                counts[field][key] += bool(report.get(key))
        audited.append({"R_id": record["R_id"], **reports})
    return {
        "records": len(records),
        "counts": {
            field: dict(sorted(field_counts.items()))
            for field, field_counts in counts.items()
        },
        "audits": audited,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be at least 1")
    if args.sample_size < 0:
        raise ValueError("--sample-size cannot be negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit cannot be negative")
    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        source = json.load(handle)
    records = source["unsolved_records"]
    if args.limit is not None:
        records = records[: args.limit]
    nonexact = [
        record for record in records if record["status"] == "nonexact_candidate"
    ]
    balance = _balance_summary(nonexact)

    started = perf_counter()
    retries = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_quiet_worker,
    ) as executor:
        iterator = executor.map(_retry_one, records, chunksize=args.chunksize)
        for index, retry in enumerate(iterator, start=1):
            retries.append(retry)
            if index % args.progress_every == 0 or index == len(records):
                counts = Counter(item["retry_status"] for item in retries)
                print(
                    f"[{index}/{len(records)}] {dict(sorted(counts.items()))}",
                    flush=True,
                )

    by_id = {retry["R_id"]: retry for retry in retries}
    combined = [
        {**record, "complete_retry": by_id[record["R_id"]]} for record in records
    ]
    retry_counts = Counter(retry["retry_status"] for retry in retries)
    retry_by_prior_status = {}
    for prior_status in sorted({record["status"] for record in combined}):
        subset = [
            record["complete_retry"]["retry_status"]
            for record in combined
            if record["status"] == prior_status
        ]
        retry_by_prior_status[prior_status] = {
            "records": len(subset),
            "retry_statuses": dict(sorted(Counter(subset).items())),
        }
    payload = {
        "source": str(args.input.resolve()),
        "input_field": "complete",
        "method": "fast_track",
        "fusion_used": False,
        "records_total": len(records),
        "retry_statuses": dict(sorted(retry_counts.items())),
        "retry_by_prior_status": retry_by_prior_status,
        "wall_seconds": perf_counter() - started,
        "normalization_violations": sum(
            retry["normalization_violations"] or 0
            for retry in retries
            if retry["normalization_violations"] is not None
        ),
        "prior_nonexact_balance": balance,
        "records": combined,
    }
    _write_gzip_json(payload, args.output)

    solved = [
        record
        for record in combined
        if record["complete_retry"]["retry_status"] == "solved"
    ]
    still_unsolved = [
        record
        for record in combined
        if record["complete_retry"]["retry_status"] != "solved"
    ]
    summary = {
        key: payload[key]
        for key in (
            "source",
            "input_field",
            "method",
            "fusion_used",
            "records_total",
            "retry_statuses",
            "retry_by_prior_status",
            "wall_seconds",
            "normalization_violations",
        )
    }
    summary["prior_nonexact_balance_counts"] = balance["counts"]
    summary["solved_count"] = len(solved)
    summary["still_unsolved_count"] = len(still_unsolved)
    summary["solved_sample"] = _sample_evenly(solved, args.sample_size)
    summary["still_unsolved_sample"] = _sample_evenly(
        still_unsolved,
        args.sample_size,
    )
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
