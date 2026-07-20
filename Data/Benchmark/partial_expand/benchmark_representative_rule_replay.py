#!/usr/bin/env python3
"""Compare tuple and legacy typesGH rules on the fixed 353-rule cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import signal
import sys
import time

from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (
    POLAR_DATASET,
    canonical_unmapped_reaction,
    canonical_unmapped_side,
    read_json,
    sha256,
    split_reaction,
    timing_summary,
    write_json,
    write_jsonl,
)  # noqa: E402
from benchmark_rule_replay import (  # noqa: E402
    CaseTimeout,
    _raise_case_timeout,
    extract_rule,
    make_reactor,
    polar_rows,
)

DEFAULT_MANIFEST = HERE.parents[2] / "sprint" / "sprint_25_rc_sample.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=POLAR_DATASET)
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--implementation", required=True)
    parser.add_argument(
        "--representation",
        choices=("tuple", "typesGH"),
        required=True,
        help="Rule/host representation; tuple carries Lewis-state electrons.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--embedding-threshold", type=int, default=10_000)
    parser.add_argument("--case-timeout", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    logging.disable(logging.INFO)

    dataset = args.dataset.resolve()
    manifest_path = args.sample_manifest.resolve()
    manifest = read_json(manifest_path)
    metadata = {
        int(item["record_id"]): item
        for item in manifest["selection"]["representatives"]
    }
    rows_by_id = {row["record_id"]: row for row in polar_rows(dataset)}
    selected = [rows_by_id[record_id] for record_id in metadata]

    counts: Counter[str] = Counter()
    timings: dict[str, list[float]] = defaultdict(list)
    solutions: dict[str, list[float]] = defaultdict(list)
    cases: list[dict[str, object]] = []
    started = time.perf_counter()

    for index, source in enumerate(selected, start=1):
        record_id = int(source["record_id"])
        reaction = str(source["reaction"])
        case: dict[str, object] = {
            "record_id": record_id,
            "cluster_id": metadata[record_id]["cluster_id"],
            "cluster_size": metadata[record_id]["cluster_size"],
            "selection_reasons": metadata[record_id]["selection_reasons"],
            "directions": {},
        }
        try:
            reactants, products = split_reaction(reaction)
            prep_started = time.perf_counter()
            expected = canonical_unmapped_reaction(reaction)
            hosts = {
                "forward": canonical_unmapped_side(reactants),
                "inverse": canonical_unmapped_side(products),
            }
            prep_seconds = time.perf_counter() - prep_started
            timings["endpoint_preparation"].append(prep_seconds)
            extraction_started = time.perf_counter()
            rule = extract_rule(reaction, "polar", args.representation)
            extraction_seconds = time.perf_counter() - extraction_started
            timings["rule_extraction"].append(extraction_seconds)
            case["endpoint_preparation_seconds"] = prep_seconds
            case["rule_extraction_seconds"] = extraction_seconds
            counts["rule_extracted"] += 1
        except Exception as exc:
            counts["rule_extraction_error"] += 1
            case["rule_extraction_error"] = f"{type(exc).__name__}: {exc}"
            cases.append(case)
            continue

        direction_records = case["directions"]
        assert isinstance(direction_records, dict)
        for direction, host in hosts.items():
            record: dict[str, object] = {}
            direction_records[direction] = record
            stage = "initialization"
            previous_handler = signal.signal(signal.SIGALRM, _raise_case_timeout)
            signal.setitimer(signal.ITIMER_REAL, args.case_timeout)
            try:
                stage_started = time.perf_counter()
                reactor = make_reactor(
                    host,
                    rule,
                    direction,
                    "polar",
                    args.embedding_threshold,
                    args.representation,
                )
                record["initialization_seconds"] = time.perf_counter() - stage_started

                stage = "matching"
                stage_started = time.perf_counter()
                mappings = reactor.mappings
                record["matching_seconds"] = time.perf_counter() - stage_started
                record["mapping_count"] = len(mappings)

                stage = "rewrite"
                stage_started = time.perf_counter()
                rewrites = reactor.its_list
                record["rewrite_seconds"] = time.perf_counter() - stage_started
                record["rewrite_count"] = len(rewrites)

                stage = "serialization"
                stage_started = time.perf_counter()
                outputs = reactor.smarts_list
                record["serialization_seconds"] = time.perf_counter() - stage_started
                record["serialized_count"] = len(outputs)

                stage = "recovery_check"
                stage_started = time.perf_counter()
                generated = {canonical_unmapped_reaction(output) for output in outputs}
                record["recovery_check_seconds"] = time.perf_counter() - stage_started
                record["expected_standardized_reaction"] = expected
                record["unique_reaction_count"] = len(generated)
                record["reference_recovered"] = expected in generated
                record["status"] = "PASS" if expected in generated else "FAIL"
                counts[
                    f"{direction}:recovered:{str(expected in generated).lower()}"
                ] += 1

                for name in (
                    "initialization",
                    "matching",
                    "rewrite",
                    "serialization",
                    "recovery_check",
                ):
                    timings[f"{direction}:{name}"].append(
                        float(record[f"{name}_seconds"])
                    )
                for name in ("mapping", "rewrite", "serialized", "unique_reaction"):
                    solutions[f"{direction}:{name}_count"].append(
                        float(record[f"{name}_count"])
                    )
            except Exception as exc:
                record.update(
                    status="ERROR",
                    stage=stage,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                counts[f"{direction}:error"] += 1
                if isinstance(exc, CaseTimeout):
                    counts[f"{direction}:timeout"] += 1
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)
            counts[f"{direction}:attempted"] += 1

        cases.append(case)
        if args.progress_every and index % args.progress_every == 0:
            print(
                f"representative replay {args.implementation}: {index}/{len(selected)}",
                file=sys.stderr,
                flush=True,
            )

    wall = time.perf_counter() - started
    report = {
        "schema": "synkit.representative-rule-replay/2",
        "implementation": args.implementation,
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "sample_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "rows": len(selected),
        "configuration": {
            "representation": args.representation,
            "electron_annotations": args.representation == "tuple",
            "directions": ["forward", "inverse"],
            "explicit_h": False,
            "rule_implicit_h": True,
            "reacting_hydrogen_policy": "reaction-centre hcount transition",
            "implicit_temp": False,
            "automorphism": True,
            "recovery_comparison": "Standardize.fit(remove_aam=True, ignore_stereo=True) on the complete reaction",
            "embedding_pre_filter": True,
            "embedding_threshold": args.embedding_threshold,
            "case_timeout_seconds": args.case_timeout,
        },
        "counts": dict(sorted(counts.items())),
        "solution_counts": {
            key: timing_summary(values) for key, values in sorted(solutions.items())
        },
        "timing_seconds": {
            "wall": wall,
            "stages": {
                key: timing_summary(values) for key, values in sorted(timings.items())
            },
        },
        "case_file": str(args.cases.resolve()),
    }
    write_json(args.output, report)
    write_jsonl(args.cases, cases)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
