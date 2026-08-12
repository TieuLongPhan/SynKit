#!/usr/bin/env python3
"""Compare RBL search profiles on paired incomplete/complete reactions.

For every selected record, the atom-mapped ``aam`` reaction is used as the
rule/template and applied to ``raw``.  A method solves the record only when at
least one output is exactly equal to ``complete`` after the same stereo-free,
atom-map-free SynKit standardization used by ``build_dataset.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import rdkit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import synkit  # noqa: E402
from synkit.Chem.Reaction.standardize import Standardize  # noqa: E402
from synkit.Graph.Hyrogen.hcomplete import HComplete  # noqa: E402
from synkit.IO import rsmi_to_its  # noqa: E402
from synkit.Rule import SynRule  # noqa: E402
from synkit.Synthesis.RBL import RBLEngine  # noqa: E402

DEFAULT_DATA = HERE / "uspto_50k_rbl.json.gz"
DEFAULT_OUTPUT = HERE / "benchmark_results.json"
RBL_BENCHMARK_SCHEMA = "synkit.rbl-benchmark/1"
RBL_BENCHMARK_RECORD_SCHEMA = "synkit.rbl-benchmark-record/1"
RBL_EVALUATION_SCHEMA = "synkit.rbl-evaluation/1"
METHODS: dict[str, dict[str, Any]] = {
    "fast_track": {"mode": "fast_track"},
    "fast_fusion": {"mode": "fast_fusion"},
    "early_stop": {"mode": "early_stop"},
    "full": {"mode": "full"},
    "verified": {"mode": "verified"},
}
RULE_ADAPTER = "SynRule(implicit_h=False, format='tuple')"
RULE_EXTRACTION_DESCRIPTION = (
    "rsmi_to_its(aam, core=False, format='tuple') -> "
    "HComplete unique exhaustive completion -> "
    f"{RULE_ADAPTER}"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_manifest(path: Path) -> dict[str, Any]:
    """Return a stable content manifest for one benchmark input."""
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    try:
        display = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        display = resolved.as_posix()
    return {
        "path": display,
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def record_digest(record: dict[str, Any]) -> str:
    """Bind a benchmark row to its exact semantic input fields."""
    payload = {
        key: record.get(key)
        for key in ("R_id", "raw", "complete", "aam")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def runtime_provenance(data: Path) -> dict[str, Any]:
    """Capture code, environment, and dataset identity for reproducibility."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "dataset": file_manifest(data),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "synkit": synkit.__version__,
        "rdkit": rdkit.__version__,
        "process_start_method": "fork",
        "thread_environment": {
            key: os.environ[key] for key in thread_variables if key in os.environ
        },
        "method_configuration": METHODS,
    }


class HydrogenCompletionRejected(ValueError):
    """The mapped reaction has no exhaustive, unambiguous H completion."""

    def __init__(self, record_id: str, reason: str, candidates: int) -> None:
        self.record_id = record_id
        self.reason = reason
        self.candidates = candidates
        super().__init__(
            f"{record_id}: hydrogen completion rejected: {reason} "
            f"({candidates} candidate classes observed)"
        )


def load_records(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"{path} must contain a JSON list.")
    return records


def select_records(
    records: list[dict[str, str]],
    limit: int | None,
    selection: str,
) -> list[dict[str, str]]:
    if limit is None or limit >= len(records):
        return list(records)
    if limit <= 0:
        raise ValueError("limit must be positive or omitted.")
    if selection == "first":
        return records[:limit]
    if selection == "shortest":
        return sorted(records, key=lambda row: (len(row["aam"]), row["R_id"]))[:limit]
    if limit == 1:
        return [records[0]]
    indices = [
        round(index * (len(records) - 1) / (limit - 1)) for index in range(limit)
    ]
    return [records[index] for index in indices]


def canonical_reaction(value: str, standardizer: Standardize) -> str | None:
    return standardizer.fit(
        value,
        remove_aam=True,
        ignore_stereo=True,
        remove_invalid=False,
    )


def extract_normalized_rule(record: dict[str, str]) -> tuple[SynRule, dict[str, Any]]:
    """Complete H on the full mapped ITS, then normalize its unique RC."""
    full_its = rsmi_to_its(record["aam"], core=False, format="tuple")
    completion = HComplete.complete_its(full_its, format="tuple")
    if not completion.ok or not completion.exhaustive:
        raise HydrogenCompletionRejected(
            record["R_id"],
            completion.reason or "no_unambiguous_completion",
            completion.candidates,
        )
    source = completion.rc
    if source is None:  # narrowed by completion.ok; keeps type checkers honest
        raise HydrogenCompletionRejected(record["R_id"], "missing_rc", 0)
    rule = SynRule(
        source,
        name=record["R_id"],
        format="tuple",
        implicit_h=False,
    )
    normalized = rule.rc.raw
    # With an explicit-H rule, shared hcount is local matching context (for
    # example, the second H of water), not a transferable resource delta.
    # Hydrogen changes themselves are represented by mapped H vertices/edges.
    resources = ("lone_pairs",)
    violations: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for node, attributes in normalized.nodes(data=True):
        for resource in resources:
            value = attributes.get(resource)
            if not (
                isinstance(value, tuple)
                and len(value) == 2
                and all(isinstance(endpoint, (int, float)) for endpoint in value)
            ):
                continue
            if min(value) != 0:
                violations.append(
                    {"node": node, "attribute": resource, "value": list(value)}
                )
            if node in source:
                original = source.nodes[node].get(resource)
                if original != value:
                    changes.append(
                        {
                            "node": node,
                            "attribute": resource,
                            "before": (
                                list(original)
                                if isinstance(original, tuple)
                                else original
                            ),
                            "after": list(value),
                        }
                    )
    if violations:
        raise ValueError(
            f"SynRule left non-relative resource pairs in {record['R_id']}: "
            f"{violations[:3]}"
        )
    return rule, {
        "source": (
            "rsmi_to_its(aam, core=False, format='tuple') -> "
            "HComplete.complete_its(format='tuple') -> unique rc"
        ),
        "adapter": RULE_ADAPTER,
        "full_its_nodes": full_its.number_of_nodes(),
        "source_nodes": source.number_of_nodes(),
        "normalized_nodes": normalized.number_of_nodes(),
        "hydrogen_completion": {
            "status": "unambiguous",
            "candidates": completion.candidates,
            "exhaustive": completion.exhaustive,
            "signature": completion.signature,
        },
        "resource_changes": changes[:20],
        "resource_changes_truncated": max(0, len(changes) - 20),
        "normalization_violations": 0,
    }


def diagnostic_codes(result: dict[str, Any]) -> list[str]:
    return sorted(
        {
            issue["code"]
            for reports in result.get("diagnostics", {}).values()
            for report in reports
            for issue in report.get("issues", [])
            if "code" in issue
        }
    )


def _evaluate_unbounded(
    record: dict[str, str],
    method: str,
    input_field: str = "raw",
) -> dict[str, Any]:
    started = perf_counter()
    try:
        standardizer = Standardize()
        rule, rule_audit = extract_normalized_rule(record)
        engine = RBLEngine(**METHODS[method]).process(
            record[input_field],
            rule,
            replace_wc=True,
        )
        canonical_candidates = []
        for candidate in engine.fused_rsmis:
            canonical = canonical_reaction(candidate, standardizer)
            if canonical is not None and canonical not in canonical_candidates:
                canonical_candidates.append(canonical)
        solved = record["complete"] in canonical_candidates
        result = engine.result
        return {
            "schema": RBL_EVALUATION_SCHEMA,
            "status": (
                "solved"
                if solved
                else ("nonexact_candidate" if canonical_candidates else "no_candidate")
            ),
            "seconds": perf_counter() - started,
            "n_candidates": len(canonical_candidates),
            "candidates": canonical_candidates[:20],
            "candidates_truncated": max(0, len(canonical_candidates) - 20),
            "stop_mode": result["mode"],
            "stop_reason": result["reason"],
            "n_forward_its": result["n_forward_its"],
            "n_backward_its": result["n_backward_its"],
            "fusion_search": result["fusion_search"],
            "search_status": result["search_status"],
            "search_complete": result["complete"],
            "reason_incomplete": result["reason_incomplete"],
            "search_policy": result["search_policy"],
            "acceptance_policy": result["acceptance_policy"],
            "diagnostic_codes": diagnostic_codes(result),
            "rule_audit": rule_audit,
            "input_field": input_field,
        }
    except HydrogenCompletionRejected as exc:
        return {
            "schema": RBL_EVALUATION_SCHEMA,
            "status": "hydrogen_completion_rejected",
            "seconds": perf_counter() - started,
            "error": str(exc),
            "hydrogen_completion": {
                "reason": exc.reason,
                "candidates": exc.candidates,
            },
        }
    except Exception as exc:  # keep per-record technical failures visible
        return {
            "schema": RBL_EVALUATION_SCHEMA,
            "status": "error",
            "seconds": perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _evaluation_worker(
    connection: Any,
    record: dict[str, str],
    method: str,
    input_field: str,
) -> None:
    """Run one case in a killable child process."""
    try:
        connection.send(_evaluate_unbounded(record, method, input_field=input_field))
    finally:
        connection.close()


def evaluate(
    record: dict[str, str],
    method: str,
    timeout: float,
    input_field: str = "raw",
) -> dict[str, Any]:
    """Evaluate one case with a hard wall-time bound, including C extensions."""
    started = perf_counter()
    context = mp.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_evaluation_worker,
        args=(child, record, method, input_field),
    )
    process.start()
    child.close()
    try:
        if parent.poll(None if timeout <= 0 else timeout):
            result = parent.recv()
            process.join(timeout=1)
            return result
        process.terminate()
        process.join()
        return {
            "schema": RBL_EVALUATION_SCHEMA,
            "status": "timeout",
            "seconds": perf_counter() - started,
            "error": f"case exceeded {timeout:g} seconds",
        }
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join()


def summarize(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in methods:
        results = [row["methods"][method] for row in rows]
        statuses = Counter(result["status"] for result in results)
        solved_ids = [
            row["R_id"]
            for row, result in zip(rows, results, strict=True)
            if result["status"] == "solved"
        ]
        summary[method] = {
            "statuses": dict(sorted(statuses.items())),
            "solved_R_ids": solved_ids,
            "total_seconds": sum(result["seconds"] for result in results),
            "median_seconds": sorted(result["seconds"] for result in results)[
                len(results) // 2
            ],
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--selection",
        choices=("stratified", "first", "shortest"),
        default="stratified",
    )
    parser.add_argument(
        "--methods",
        default="fast_track,fast_fusion,early_stop,full,verified",
        help=f"Comma-separated subset of: {','.join(METHODS)}",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    logging.getLogger().setLevel(logging.ERROR)
    selected = select_records(load_records(args.data), args.limit, args.selection)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        method_results = {
            method: evaluate(record, method, args.timeout) for method in methods
        }
        solved_by = [
            method for method in methods if method_results[method]["status"] == "solved"
        ]
        rows.append(
            {
                "schema": RBL_BENCHMARK_RECORD_SCHEMA,
                "R_id": record["R_id"],
                "raw": record["raw"],
                "complete": record["complete"],
                "input_digest": record_digest(record),
                "solved_by": solved_by,
                "methods": method_results,
            }
        )
        statuses = " ".join(
            f"{method}={method_results[method]['status']}" for method in methods
        )
        print(f"[{index}/{len(selected)}] {record['R_id']} {statuses}", flush=True)

    payload = {
        "schema": RBL_BENCHMARK_SCHEMA,
        "record_schema": RBL_BENCHMARK_RECORD_SCHEMA,
        "evaluation_schema": RBL_EVALUATION_SCHEMA,
        "dataset": str(args.data.resolve()),
        "provenance": runtime_provenance(args.data),
        "rule_adapter": RULE_ADAPTER,
        "rule_extraction": RULE_EXTRACTION_DESCRIPTION,
        "selection": args.selection,
        "limit": len(selected),
        "timeout_seconds_per_method": args.timeout,
        "success_definition": (
            "candidate equals complete after SynKit Standardize with "
            "remove_aam=True and ignore_stereo=True"
        ),
        "methods": methods,
        "summary": summarize(rows, methods),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    for method in methods:
        statuses = payload["summary"][method]["statuses"]
        print(f"{method}: {statuses}")


if __name__ == "__main__":
    main()
