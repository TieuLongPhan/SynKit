#!/usr/bin/env python3
"""Run streaming forward/backward LWG rule replay over one FLOWER batch."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Iterator

from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.common import (  # noqa: E402
    canonical_unmapped_reaction,
    canonical_unmapped_side,
    open_text,
    sha256,
    timing_summary,
    write_json,
)
from Experiment.Lewis.rule_replay.benchmark import (  # noqa: E402
    DIRECTIONS,
    REPRESENTATIONS,
    extract_rule,
    replay_direction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default="tuple",
        help="Graph rule representation (default: tuple/LWG).",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=DIRECTIONS,
        default=list(DIRECTIONS),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--rows-file",
        type=Path,
        help=(
            "Replay the positive one-based batch row numbers listed in this "
            "file; mutually exclusive with --offset/--limit."
        ),
    )
    parser.add_argument("--case-timeout", type=float)
    parser.add_argument("--embedding-threshold", type=int)
    parser.add_argument(
        "--failure-sample-limit",
        type=int,
        default=0,
        help=(
            "Store up to this many generated reactions for each failed " "direction."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "replay-debug",
    )
    return parser.parse_args()


def iter_rows(
    path: Path,
    offset: int = 0,
    limit: int | None = None,
    row_numbers: set[int] | None = None,
) -> Iterator[tuple[int, str, str]]:
    selected = 0
    final_row = max(row_numbers) if row_numbers else None
    with open_text(path) as handle:
        for row_number, line in enumerate(handle, start=1):
            if final_row is not None and row_number > final_row:
                break
            if row_numbers is not None and row_number not in row_numbers:
                continue
            if row_number <= offset:
                continue
            text = line.rstrip("\n")
            reaction, separator, source_label = text.rpartition("|")
            if not separator or reaction.count(">>") != 1:
                raise ValueError(f"Malformed FLOWER row {row_number}")
            yield row_number, source_label, reaction
            selected += 1
            if limit is not None and selected >= limit:
                break


def read_row_numbers(path: Path) -> set[int]:
    """Read a non-empty set of positive one-based batch row numbers."""
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: set[int] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        text = line.strip()
        if not text:
            continue
        try:
            row = int(text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid row number on line {line_number}: {text!r}"
            ) from exc
        if row < 1:
            raise ValueError("Selected row numbers must be positive")
        rows.add(row)
    if not rows:
        raise ValueError("Rows file must select at least one row")
    return rows


def case_identity(batch: Path, row_number: int) -> dict[str, Any]:
    """Return portable provenance for one row, including aggregate bug logs."""
    batch_file = batch.name
    return {
        "case_id": f"{batch_file}:{row_number}",
        "batch_file": batch_file,
        "batch_row": row_number,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901
    batch = args.batch.resolve()
    if not batch.is_file():
        raise FileNotFoundError(batch)
    if args.offset < 0:
        raise ValueError("Offset cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("Limit must be positive")
    if args.rows_file is not None and (args.offset or args.limit is not None):
        raise ValueError("--rows-file is mutually exclusive with offset/limit")
    if args.case_timeout is not None and args.case_timeout <= 0:
        raise ValueError("Timeout must be positive")
    if args.embedding_threshold is not None and args.embedding_threshold < 1:
        raise ValueError("Embedding threshold must be positive")
    if args.failure_sample_limit < 0:
        raise ValueError("Failure sample limit cannot be negative")
    if args.progress_every < 0:
        raise ValueError("Progress interval cannot be negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / "cases.jsonl.gz"
    bugs_path = args.output_dir / "bugs.jsonl"
    counts: Counter[str] = Counter()
    durations: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    selected_rows = (
        read_row_numbers(args.rows_file.resolve())
        if args.rows_file is not None
        else None
    )

    with (
        gzip.open(cases_path, "wt", encoding="utf-8") as output,
        bugs_path.open("w", encoding="utf-8") as bugs,
    ):

        def write_case(case: dict[str, Any]) -> None:
            line = json.dumps(case, sort_keys=True) + "\n"
            output.write(line)
            if case["status"] != "PASS":
                bugs.write(line)
                bugs.flush()

        for selected_index, (row_number, source_label, reaction) in enumerate(
            iter_rows(
                batch,
                args.offset,
                args.limit,
                selected_rows,
            ),
            start=1,
        ):
            case: dict[str, Any] = {
                **case_identity(batch, row_number),
                "source_label": source_label,
                "directions": {},
            }
            counts["rows_attempted"] += 1
            try:
                reactants, products = reaction.split(">>", 1)
                expected = canonical_unmapped_reaction(reaction)
                hosts = {
                    "forward": canonical_unmapped_side(reactants),
                    "backward": canonical_unmapped_side(products),
                }
            except Exception as exc:
                counts["normalization_error"] += 1
                case.update(
                    status="NORMALIZATION_ERROR",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                write_case(case)
                continue

            extraction_started = time.perf_counter()
            try:
                rule = extract_rule(reaction, args.representation)
                case["extraction_seconds"] = time.perf_counter() - extraction_started
                counts["rule_extracted"] += 1
            except Exception as exc:
                counts["rule_extraction_error"] += 1
                case.update(
                    status="RULE_EXTRACTION_ERROR",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    extraction_seconds=(time.perf_counter() - extraction_started),
                )
                write_case(case)
                continue

            for direction in args.directions:
                result = replay_direction(
                    host=hosts[direction],
                    expected=expected,
                    rule=rule,
                    representation=args.representation,
                    direction=direction,
                    embedding_threshold=args.embedding_threshold,
                    case_timeout=args.case_timeout,
                    failure_sample_limit=args.failure_sample_limit,
                )
                case["directions"][direction] = result
                durations[direction].append(float(result["seconds"]))
                counts[f"{direction}:attempted"] += 1
                counts[f"{direction}:{result['status'].lower()}"] += 1
                if result.get("error_type") == "CaseTimeout":
                    counts[f"{direction}:timeout"] += 1
            case["status"] = (
                "PASS"
                if all(
                    result["status"] == "PASS" for result in case["directions"].values()
                )
                else "INCOMPLETE"
            )
            counts[f"rows:{case['status'].lower()}"] += 1
            write_case(case)
            if args.progress_every and selected_index % args.progress_every == 0:
                output.flush()
                bugs.flush()
                print(f"replayed {selected_index} rows", flush=True)

    report = {
        "schema": "synkit.flower-bidirectional-rule-replay/1",
        "dataset": {
            "path": str(batch),
            "sha256": sha256(batch),
        },
        "selection": {
            "offset": args.offset,
            "limit": args.limit,
            "rows_file": (
                {
                    "path": str(args.rows_file.resolve()),
                    "sha256": sha256(args.rows_file.resolve()),
                    "selected_rows": len(selected_rows),
                }
                if args.rows_file is not None
                else None
            ),
            "rows": counts["rows_attempted"],
        },
        "representation": args.representation,
        "directions": args.directions,
        "policy": {
            "case_timeout_seconds": args.case_timeout,
            "timeout_scope": "each expansion stage",
            "embedding_threshold": args.embedding_threshold,
            "failure_sample_limit": args.failure_sample_limit,
            "reaction_center_edge_policy": "changed",
            "recovery": "canonical full reaction without AAM or stereo",
        },
        "counts": dict(sorted(counts.items())),
        "timing_seconds": {
            "wall": time.perf_counter() - started,
            "directions": {
                name: timing_summary(values)
                for name, values in sorted(durations.items())
            },
        },
        "case_file": str(cases_path.resolve()),
        "bug_file": str(bugs_path.resolve()),
    }
    write_json(args.output_dir / "summary.json", report)
    return report


def main() -> int:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    logging.disable(logging.INFO)
    report = run_replay(args)
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
