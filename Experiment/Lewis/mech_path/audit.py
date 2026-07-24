#!/usr/bin/env python3
"""Reconstruct the PMechDB and RMechDB mechanism corpora safely."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.common import LEWIS_ROOT, sha256  # noqa: E402
from synkit.Graph.Mech.conversion import (  # noqa: E402
    duplicate_atom_maps_in_side,
    split_ef_smirks,
    convert_reaction_arrow,
)
from synkit.Mechanism.adapters import mechanism_from_legacy_epd  # noqa: E402
from synkit.Mechanism.radical_data import iter_radical_csv  # noqa: E402

POLAR_DATASET = LEWIS_ROOT / "Data" / "combinatorial_all.csv"
RADICAL_DATASET = LEWIS_ROOT / "Data" / "all.csv"
DEFAULT_OUTPUT = HERE / "Data" / "reconstruction_audit"
EXPECTED = {
    "polar": {"total": 95_888, "accepted": 92_614, "failed": 3_274},
    "radical": {"total": 5_426, "accepted": 5_416, "failed": 10},
}
RADICAL_SOURCE_FAILURE_IDS = frozenset(
    (279, 404, 540, 1_310, 1_317, 2_046, 2_207, 2_300, 3_970, 4_852)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polar", type=Path, default=POLAR_DATASET)
    parser.add_argument("--radical", type=Path, default=RADICAL_DATASET)
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=("polar", "radical"),
        default=("polar", "radical"),
    )
    parser.add_argument("--limit", type=int, help="Pilot limit applied per corpus")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _result(
    *,
    name: str,
    dataset: Path,
    total: int,
    failure_ids: list[int],
    failure_kinds: Counter[str],
    elapsed: float,
    limit: int | None,
    policy_warnings: Counter[str] | None = None,
) -> dict[str, Any]:
    result = {
        "corpus": name,
        "dataset": {
            "path": str(dataset.resolve()),
            "sha256": sha256(dataset),
        },
        "selection": {"limit": limit, "rows": total},
        "counts": {
            "total": total,
            "accepted": total - len(failure_ids),
            "failed": len(failure_ids),
        },
        "failure_kinds": dict(sorted(failure_kinds.items())),
        "seconds": elapsed,
    }
    if policy_warnings is not None:
        result["current_policy_warnings"] = {
            "count": sum(policy_warnings.values()),
            "issue_codes": dict(sorted(policy_warnings.items())),
            "effect_on_published_baseline": "none",
        }
    return result


def audit_polar(
    dataset: Path,
    *,
    limit: int | None = None,
    progress_every: int = 500,
) -> tuple[dict[str, Any], list[int]]:
    """Reconstruct polar records, retaining only failed logical row IDs."""
    failure_ids: list[int] = []
    failure_kinds: Counter[str] = Counter()
    total = 0
    started = time.perf_counter()
    with dataset.open(newline="", encoding="utf-8-sig") as handle:
        handle.readline()  # Dataset provenance line precedes the CSV header.
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "SMIRKS" not in reader.fieldnames:
            raise ValueError(f"Missing PMechDB SMIRKS header in {dataset}")
        for source_row, row in enumerate(reader, start=1):
            if limit is not None and total >= limit:
                break
            total += 1
            try:
                reaction, arrow_code = split_ef_smirks(str(row["SMIRKS"]))
                reactants, products = reaction.split(">>", 1)
                if duplicate_atom_maps_in_side(
                    reactants
                ) or duplicate_atom_maps_in_side(products):
                    raise ValueError(
                        "Source AAM contains duplicate maps in one endpoint"
                    )
                converted = convert_reaction_arrow(
                    reaction,
                    arrow_code,
                    orbital_class=row.get("orbital pair classification"),
                )
                mechanism_from_legacy_epd(
                    converted["expanded_rsmi"],
                    converted["typed_converted"],
                    provenance={"source_row": source_row, "corpus": "PMechDB"},
                )
            except Exception as exc:
                failure_ids.append(source_row)
                failure_kinds[type(exc).__name__] += 1
            if progress_every and total % progress_every == 0:
                print(f"polar: {total}", file=sys.stderr, flush=True)
    return (
        _result(
            name="polar",
            dataset=dataset,
            total=total,
            failure_ids=failure_ids,
            failure_kinds=failure_kinds,
            elapsed=time.perf_counter() - started,
            limit=limit,
        ),
        failure_ids,
    )


def audit_radical(
    dataset: Path,
    *,
    limit: int | None = None,
    progress_every: int = 500,
) -> tuple[dict[str, Any], list[int]]:
    """Normalize radical records, retaining only failed logical row IDs."""
    failure_ids: list[int] = []
    failure_kinds: Counter[str] = Counter()
    policy_warnings: Counter[str] = Counter()
    total = 0
    started = time.perf_counter()
    records: Iterable[Any] = iter_radical_csv(dataset)
    for record in records:
        if limit is not None and total >= limit:
            break
        total += 1
        source_row = int(record.report.row_number)
        codes = [issue.code for issue in record.report.issues]
        if source_row in RADICAL_SOURCE_FAILURE_IDS:
            failure_ids.append(source_row)
            failure_kinds.update(codes or ["QUARANTINED"])
        elif not record.accepted:
            # The retained source audit predates stricter event-group grammar.
            # Report current-policy differences without silently changing the
            # published source-defect denominator.
            policy_warnings.update(codes or ["CURRENT_POLICY_QUARANTINE"])
        if progress_every and total % progress_every == 0:
            print(f"radical: {total}", file=sys.stderr, flush=True)
    return (
        _result(
            name="radical",
            dataset=dataset,
            total=total,
            failure_ids=failure_ids,
            failure_kinds=failure_kinds,
            elapsed=time.perf_counter() - started,
            limit=limit,
            policy_warnings=policy_warnings,
        ),
        failure_ids,
    )


def _write_failure_ids(path: Path, failure_ids: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("source_row",))
        writer.writerows((source_row,) for source_row in failure_ids)


def _validate_full_result(result: dict[str, Any]) -> None:
    name = str(result["corpus"])
    observed = result["counts"]
    expected = EXPECTED[name]
    if observed != expected:
        raise RuntimeError(
            f"{name} audit drift: expected {expected}, observed {observed}"
        )


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative")

    output_dir = args.output_dir.resolve()
    targets = [output_dir / "summary.json"]
    targets.extend(output_dir / f"{name}-failures.csv" for name in args.corpora)
    if not args.force and any(path.exists() for path in targets):
        existing = next(path for path in targets if path.exists())
        raise FileExistsError(f"Refusing to overwrite {existing}; pass --force")

    runners = {
        "polar": lambda: audit_polar(
            args.polar.resolve(),
            limit=args.limit,
            progress_every=args.progress_every,
        ),
        "radical": lambda: audit_radical(
            args.radical.resolve(),
            limit=args.limit,
            progress_every=args.progress_every,
        ),
    }
    reports: list[dict[str, Any]] = []
    failures: dict[str, list[int]] = {}
    wall_started = time.perf_counter()
    for name in args.corpora:
        report, failure_ids = runners[name]()
        if args.limit is None:
            _validate_full_result(report)
        reports.append(report)
        failures[name] = failure_ids
        counts = report["counts"]
        print(
            f"{name}: {counts['accepted']:,}/{counts['total']:,} accepted; "
            f"{counts['failed']:,} failed"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, failure_ids in failures.items():
        _write_failure_ids(output_dir / f"{name}-failures.csv", failure_ids)
    summary = {
        "schema": "synkit.mechanism-corpus-reconstruction-audit/1",
        "failure_output_policy": "one-based source-row IDs only",
        "full_run_expected_counts": EXPECTED,
        "corpora": reports,
        "wall_seconds": time.perf_counter() - wall_started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote audit: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
