#!/usr/bin/env python3
"""Benchmark reaction-centre extraction and forward/inverse rule replay.

The general suite supports both the electron-aware ``tuple`` representation
and the electron-free legacy ``typesGH`` representation.  Radical replay is
deliberately tuple-only because ``typesGH`` cannot encode side-specific
radical state.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import inspect
import json
import logging
from pathlib import Path
import platform
import signal
import sys
import time
from typing import Any

from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import synkit  # noqa: F401
except ModuleNotFoundError:
    # Preserve an explicit v1 PYTHONPATH; use the repository root only when
    # the selected environment has no importable SynKit package.
    sys.path.insert(0, str(HERE.parents[2]))

from common import (  # noqa: E402
    POLAR_DATASET,
    RADICAL_DATASET,
    canonical_unmapped_reaction,
    canonical_unmapped_side,
    read_json,
    sha256,
    split_reaction,
    timing_summary,
    write_json,
    write_jsonl,
)

from synkit.IO.chem_converter import rsmi_to_its  # noqa: E402
from synkit.Rule import SynRule  # noqa: E402
from synkit.Synthesis.Reactor.syn_reactor import SynReactor  # noqa: E402

LEGACY_NODE_ATTRS = [
    "element",
    "aromatic",
    "hcount",
    "charge",
    "neighbors",
    "atom_map",
]
LEGACY_EDGE_ATTRS = ["order"]
HAS_FORMAT = "format" in inspect.signature(SynRule.__init__).parameters


class CaseTimeout(TimeoutError):
    """Raised when one direction exceeds the disclosed benchmark ceiling."""


def _raise_case_timeout(_signum, _frame) -> None:
    raise CaseTimeout("Rule-replay direction exceeded the per-case time ceiling")


def polar_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {"record_id": int(row["R-id"]), "reaction": str(row["smart"])}
        for row in read_json(path)
    ]


def radical_rows(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            reaction = row[0].strip().split(None, 1)[0]
            records.append({"record_id": row_number, "reaction": reaction})
    return records


def extract_rule(
    reaction: str,
    suite: str,
    representation: str | None = None,
) -> SynRule:
    resolved_format = representation or ("tuple" if suite == "radical" else "typesGH")
    if resolved_format == "typesGH":
        kwargs: dict[str, Any] = {
            "core": True,
            "drop_non_aam": False,
            "use_index_as_atom_map": True,
            "node_attrs": LEGACY_NODE_ATTRS,
            "edge_attrs": LEGACY_EDGE_ATTRS,
        }
        if HAS_FORMAT:
            kwargs["format"] = resolved_format
        graph = rsmi_to_its(reaction, **kwargs)
        rule_kwargs: dict[str, Any] = {"canon": False, "implicit_h": True}
        if HAS_FORMAT:
            rule_kwargs["format"] = resolved_format
        return SynRule(graph, **rule_kwargs)

    graph = rsmi_to_its(
        reaction,
        core=True,
        drop_non_aam=False,
        use_index_as_atom_map=True,
        format=resolved_format,
    )
    return SynRule(graph, canon=False, implicit_h=True, format=resolved_format)


def make_reactor(
    host: str,
    rule: SynRule,
    direction: str,
    suite: str,
    embedding_threshold: int,
    representation: str | None = None,
) -> SynReactor:
    resolved_format = representation or ("tuple" if suite == "radical" else "typesGH")
    kwargs: dict[str, Any] = {
        "invert": direction == "inverse",
        "explicit_h": False,
        "implicit_temp": False,
        "automorphism": True,
        "embed_threshold": embedding_threshold,
        "embed_pre_filter": True,
    }
    if HAS_FORMAT:
        kwargs.update(
            template_format=resolved_format,
            radical_policy="strict" if resolved_format == "tuple" else "ignore",
            stereo_mode="ignore",
        )
    return SynReactor(host, rule, **kwargs)


def main() -> int:  # noqa: C901 - one-pass benchmark stage accounting
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("polar", "radical"), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--implementation", required=True)
    parser.add_argument(
        "--representation",
        choices=("tuple", "typesGH"),
        help=(
            "Rule and host representation. Defaults to tuple for radical data "
            "and must be selected explicitly for the general comparison."
        ),
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=("forward", "inverse"),
        default=["forward", "inverse"],
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path)
    parser.add_argument(
        "--cases",
        type=Path,
        help="Optional gzip/JSONL file with one record per selected reaction",
    )
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--embedding-threshold",
        type=int,
        default=10_000,
        help="Abort and record a case when raw embedding enumeration exceeds this bound",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=5.0,
        help="Abort and record one replay direction after this many wall seconds",
    )
    args = parser.parse_args()

    if args.suite == "radical" and not HAS_FORMAT:
        parser.error(
            "The radical tuple benchmark requires current SynKit; v1 is inapplicable"
        )
    if args.suite == "radical" and args.representation not in (None, "tuple"):
        parser.error(
            "Radical replay is tuple-only: typesGH has no side-specific "
            "radical-state representation"
        )
    if args.suite == "polar" and args.representation is None:
        parser.error("--representation is required for the general/polar suite")
    representation = args.representation or "tuple"
    RDLogger.DisableLog("rdApp.*")
    logging.disable(logging.INFO)
    dataset = (
        args.dataset or (RADICAL_DATASET if args.suite == "radical" else POLAR_DATASET)
    ).resolve()
    rows = radical_rows(dataset) if args.suite == "radical" else polar_rows(dataset)
    selected = rows[
        args.offset : None if args.limit is None else args.offset + args.limit
    ]

    counts: Counter[str] = Counter()
    solution_counts: dict[str, list[int]] = defaultdict(list)
    timings: dict[str, list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, source in enumerate(selected, start=1):
        record_id = int(source["record_id"])
        reaction = str(source["reaction"])
        case: dict[str, Any] = {"record_id": record_id, "directions": {}}
        if args.suite == "radical":
            completion_started = time.perf_counter()
            try:
                from synkit.Mechanism.radical_data import complete_radical_aam

                completion = complete_radical_aam(reaction)
                timings["aam_completion"].append(
                    time.perf_counter() - completion_started
                )
                if not completion.usable or completion.mapped_reaction is None:
                    counts["aam_completion_error"] += 1
                    failures.append(
                        {
                            "record_id": record_id,
                            "stage": "aam_completion",
                            "message": completion.failure_reason,
                        }
                    )
                    case["status"] = "AAM_COMPLETION_ERROR"
                    case["message"] = completion.failure_reason
                    cases.append(case)
                    continue
                reaction = completion.mapped_reaction
                counts[f"aam:{completion.status.lower()}"] += 1
                case["aam_completion_status"] = completion.status
                case["aam_completion_seconds"] = timings["aam_completion"][-1]
            except Exception as exc:
                timings["aam_completion"].append(
                    time.perf_counter() - completion_started
                )
                counts["aam_completion_error"] += 1
                failures.append(
                    {
                        "record_id": record_id,
                        "stage": "aam_completion",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                case["status"] = "AAM_COMPLETION_ERROR"
                case["error_type"] = type(exc).__name__
                case["message"] = str(exc)
                cases.append(case)
                continue

        try:
            reactants, products = split_reaction(reaction)
            preparation_started = time.perf_counter()
            expected = canonical_unmapped_reaction(reaction)
            hosts = {
                "forward": canonical_unmapped_side(reactants),
                "inverse": canonical_unmapped_side(products),
            }
            timings["endpoint_preparation"].append(
                time.perf_counter() - preparation_started
            )
            extraction_started = time.perf_counter()
            rule = extract_rule(reaction, args.suite, representation)
            extraction_seconds = time.perf_counter() - extraction_started
            timings["rule_extraction"].append(extraction_seconds)
            counts["rule_extracted"] += 1
            case["endpoint_preparation_seconds"] = timings["endpoint_preparation"][-1]
            case["rule_extraction_seconds"] = extraction_seconds
        except Exception as exc:
            counts["rule_extraction_error"] += 1
            failures.append(
                {
                    "record_id": record_id,
                    "stage": "rule_extraction",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            case["status"] = "RULE_EXTRACTION_ERROR"
            case["error_type"] = type(exc).__name__
            case["message"] = str(exc)
            cases.append(case)
            continue

        for direction in args.directions:
            host = hosts[direction]
            stage = "initialization"
            direction_record: dict[str, Any] = {}
            case["directions"][direction] = direction_record
            try:
                previous_handler = signal.signal(signal.SIGALRM, _raise_case_timeout)
                signal.setitimer(signal.ITIMER_REAL, args.case_timeout)
                stage_started = time.perf_counter()
                reactor = make_reactor(
                    host,
                    rule,
                    direction,
                    args.suite,
                    args.embedding_threshold,
                    representation,
                )
                initialization_seconds = time.perf_counter() - stage_started
                timings[f"{direction}:initialization"].append(initialization_seconds)
                direction_record["initialization_seconds"] = initialization_seconds

                stage = "matching"
                stage_started = time.perf_counter()
                mappings = reactor.mappings
                matching_seconds = time.perf_counter() - stage_started
                timings[f"{direction}:matching"].append(matching_seconds)
                solution_counts[f"{direction}:mappings"].append(len(mappings))
                direction_record["matching_seconds"] = matching_seconds
                direction_record["mapping_count"] = len(mappings)

                stage = "rewrite"
                stage_started = time.perf_counter()
                its_values = reactor.its_list
                rewrite_seconds = time.perf_counter() - stage_started
                timings[f"{direction}:rewrite"].append(rewrite_seconds)
                solution_counts[f"{direction}:rewrites"].append(len(its_values))
                direction_record["rewrite_seconds"] = rewrite_seconds
                direction_record["rewrite_count"] = len(its_values)

                stage = "serialization"
                stage_started = time.perf_counter()
                reactions = reactor.smarts_list
                serialization_seconds = time.perf_counter() - stage_started
                timings[f"{direction}:serialization"].append(serialization_seconds)
                solution_counts[f"{direction}:serialized"].append(len(reactions))
                direction_record["serialization_seconds"] = serialization_seconds
                direction_record["serialized_count"] = len(reactions)

                stage = "recovery_check"
                stage_started = time.perf_counter()
                generated = set()
                for candidate in reactions:
                    generated.add(canonical_unmapped_reaction(candidate))
                solution_counts[f"{direction}:unique_reactions"].append(len(generated))
                recovered = expected in generated
                recovery_seconds = time.perf_counter() - stage_started
                timings[f"{direction}:recovery_check"].append(recovery_seconds)
                counts[f"{direction}:attempted"] += 1
                counts[f"{direction}:recovered:{str(recovered).lower()}"] += 1
                direction_record.update(
                    recovery_check_seconds=recovery_seconds,
                    unique_reaction_count=len(generated),
                    reference_recovered=recovered,
                    status="PASS" if recovered else "FAIL",
                )
                if not recovered:
                    failures.append(
                        {
                            "record_id": record_id,
                            "direction": direction,
                            "stage": "recovery_check",
                            "expected": expected,
                            "mapping_count": len(mappings),
                            "rewrite_count": len(its_values),
                            "serialized_count": len(reactions),
                            "generated": sorted(generated)[:20],
                        }
                    )
            except Exception as exc:
                counts[f"{direction}:attempted"] += 1
                counts[f"{direction}:error"] += 1
                if isinstance(exc, CaseTimeout):
                    counts[f"{direction}:timeout"] += 1
                failures.append(
                    {
                        "record_id": record_id,
                        "direction": direction,
                        "stage": stage,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                direction_record.update(
                    status="ERROR",
                    stage=stage,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)

        case["status"] = (
            "PASS"
            if all(
                record.get("status") == "PASS" for record in case["directions"].values()
            )
            else "INCOMPLETE"
        )
        cases.append(case)

        if args.progress_every and index % args.progress_every == 0:
            print(
                f"rule replay {args.suite}/{args.implementation}: {index}/{len(selected)}",
                file=sys.stderr,
                flush=True,
            )

    wall = time.perf_counter() - started
    report = {
        "schema": "synkit.rule-replay-benchmark/3",
        "suite": args.suite,
        "implementation": args.implementation,
        "representation": representation,
        "representation_label": (
            "electron-aware Lewis-labelled graph (tuple)"
            if representation == "tuple"
            else "legacy atom-bond graph (typesGH)"
        ),
        "automorphism": True,
        "recovery_comparison": "Standardize.fit(remove_aam=True, ignore_stereo=True) on the complete reaction",
        "embedding_threshold": args.embedding_threshold,
        "embedding_pre_filter": True,
        "case_timeout_seconds": args.case_timeout,
        "directions": args.directions,
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "selection": {
            "offset": args.offset,
            "limit": args.limit,
            "rows": len(selected),
        },
        "counts": dict(sorted(counts.items())),
        "solution_counts": {
            key: timing_summary([float(value) for value in values])
            for key, values in sorted(solution_counts.items())
        },
        "timing_seconds": {
            "wall": wall,
            "stages": {
                key: timing_summary(values) for key, values in sorted(timings.items())
            },
        },
        "failure_count": len(failures),
        "case_file": str(args.cases.resolve()) if args.cases else None,
        "environment": {
            "python": platform.python_version(),
            "has_tuple_format": HAS_FORMAT,
        },
    }
    write_json(args.output, report)
    if args.failures:
        write_jsonl(args.failures, failures)
    if args.cases:
        write_jsonl(args.cases, cases)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
