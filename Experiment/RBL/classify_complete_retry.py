#!/usr/bin/env python3
"""Classify complete-input retry outputs by exactness and reaction balance."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import io
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.RBL.retry_complete import reaction_balance  # noqa: E402

DEFAULT_INPUT = HERE / "complete_retry_hcomplete.json.gz"
DEFAULT_OUTPUT = HERE / "complete_retry_hcomplete_balance_audit.json.gz"
DEFAULT_SUMMARY = HERE / "complete_retry_hcomplete_balance_summary.json"


def balance_status(report: dict[str, Any]) -> str:
    """Return a mutually exclusive balance status for one reaction."""
    if not report.get("parsed"):
        return "unparseable"
    element_balanced = report["element_balanced"]
    charge_balanced = report["charge_balanced"]
    if element_balanced and charge_balanced:
        return "balanced"
    if not element_balanced and not charge_balanced:
        return "element_and_charge_unbalanced"
    if not element_balanced:
        return "element_only_unbalanced"
    return "charge_only_unbalanced"


def classify_record(record: dict[str, Any]) -> dict[str, Any]:
    """Classify exactness and balance for a single retry record."""
    retry = record["complete_retry"]
    target_report = reaction_balance(record["complete"])
    target_status = balance_status(target_report)
    candidates = retry.get("candidates", [])
    candidate_reports = [reaction_balance(candidate) for candidate in candidates]
    candidate_statuses = [balance_status(report) for report in candidate_reports]

    if retry["retry_status"] == "solved":
        exact_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate == record["complete"]
        ]
        if not exact_indices:
            raise ValueError(
                f"{record['R_id']} is marked solved without an exact candidate"
            )
        exact_balanced = any(
            candidate_statuses[index] == "balanced" for index in exact_indices
        )
        classification = "exact_balanced" if exact_balanced else "exact_unbalanced"
        output_balance_status = candidate_statuses[exact_indices[0]]
    elif retry["retry_status"] == "nonexact_candidate":
        if len(candidates) != 1:
            raise ValueError(
                f"{record['R_id']} has {len(candidates)} non-exact candidates; "
                "this report expects the fast-track single-candidate contract"
            )
        output_balance_status = candidate_statuses[0]
        classification = (
            "nonexact_balanced"
            if output_balance_status == "balanced"
            else "nonexact_unbalanced"
        )
    elif retry["retry_status"] == "no_candidate":
        if candidates:
            raise ValueError(
                f"{record['R_id']} is marked no-candidate but stores candidates"
            )
        output_balance_status = None
        classification = "no_candidate"
    else:
        raise ValueError(
            f"Unsupported retry status for {record['R_id']}: "
            f"{retry['retry_status']}"
        )

    return {
        "R_id": record["R_id"],
        "prior_status": record["status"],
        "retry_status": retry["retry_status"],
        "classification": classification,
        "target_balance_status": target_status,
        "output_balance_status": output_balance_status,
        "target_balance": target_report,
        "candidate_balances": candidate_reports,
    }


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
                    payload, text_handle, ensure_ascii=False, separators=(",", ":")
                )
                text_handle.write("\n")


def _count_with_ids(values: list[str]) -> dict[str, Any]:
    return {"count": len(values), "R_ids": values}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        source = json.load(handle)
    audits = [classify_record(record) for record in source["records"]]

    category_ids: dict[str, list[str]] = defaultdict(list)
    target_statuses: Counter[str] = Counter()
    output_statuses: Counter[str] = Counter()
    by_prior_status: dict[str, Counter[str]] = defaultdict(Counter)
    for audit in audits:
        category_ids[audit["classification"]].append(audit["R_id"])
        target_statuses[audit["target_balance_status"]] += 1
        if audit["output_balance_status"] is not None:
            output_statuses[audit["output_balance_status"]] += 1
        by_prior_status[audit["prior_status"]][audit["classification"]] += 1

    category_order = (
        "exact_balanced",
        "exact_unbalanced",
        "nonexact_balanced",
        "nonexact_unbalanced",
        "no_candidate",
    )
    summary = {
        "source": str(args.input.resolve()),
        "records_total": len(audits),
        "acceptance_definition": (
            "exact_balanced requires equality to canonical complete and both "
            "element and net-formal-charge conservation"
        ),
        "categories": {
            category: _count_with_ids(category_ids[category])
            for category in category_order
        },
        "target_balance_statuses": dict(sorted(target_statuses.items())),
        "candidate_balance_statuses": dict(sorted(output_statuses.items())),
        "by_prior_status": {
            status: dict(sorted(counts.items()))
            for status, counts in sorted(by_prior_status.items())
        },
    }
    payload = {**summary, "records": audits}
    _write_gzip_json(payload, args.output)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value["count"] for key, value in summary["categories"].items()}
        )
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
