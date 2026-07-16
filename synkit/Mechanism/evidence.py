"""Reproducible evidence collection for executable MechanismBench records."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median
import time
import tracemalloc
from typing import Any, Iterable, Mapping

from .benchmark import corrupt_record
from .model import MechanismRecord
from .replay import MechanismReplayer


@dataclass(frozen=True)
class EvidenceCase:
    """One executable typed record extracted from a benchmark manifest."""

    case_id: str
    partition: str
    record: MechanismRecord


def load_evidence_cases(benchmark_dir: str | Path) -> tuple[EvidenceCase, ...]:
    """Load every typed ``MechanismRecord`` from the public benchmark files.

    Reaction-SMILES and graph-rewrite stereo fixtures remain executable through
    their dedicated reactor tests, but do not yet share the typed replay
    representation required for the runtime and corruption measurements.
    """
    directory = Path(benchmark_dir)
    cases: list[EvidenceCase] = []
    for partition in ("polar", "radical", "stereo"):
        payload = json.loads((directory / f"{partition}.json").read_text())
        for serialized in payload.get("cases", ()):
            if "record" not in serialized:
                continue
            cases.append(
                EvidenceCase(
                    case_id=serialized["case_id"],
                    partition=partition,
                    record=MechanismRecord.from_dict(serialized["record"]),
                )
            )
    return tuple(cases)


def _measure_replay(record: MechanismRecord, mode: str) -> dict[str, Any]:
    tracemalloc.start()
    started = time.perf_counter_ns()
    result = MechanismReplayer(verify_stereo=mode).replay(record)
    elapsed_ns = time.perf_counter_ns() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    certificate = result.certificate
    return {
        "status": certificate.status,
        "final_matches": bool(certificate.final_match.get("matches")),
        "issue_codes": [issue.code for issue in certificate.issues],
        "elapsed_ns": elapsed_ns,
        "python_peak_bytes": peak_bytes,
    }


def _summary(values: Iterable[int]) -> dict[str, int]:
    observed = sorted(values)
    if not observed:
        return {"count": 0, "median": 0, "min": 0, "max": 0}
    return {
        "count": len(observed),
        "median": int(median(observed)),
        "min": observed[0],
        "max": observed[-1],
    }


def _manifest_fingerprints(benchmark_dir: Path) -> dict[str, str]:
    return {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(benchmark_dir.glob("*.json"))
    }


def collect_evidence(
    benchmark_dir: str | Path,
    *,
    repetitions: int = 3,
) -> dict[str, Any]:
    """Collect replay, corruption, and Python-allocation evidence.

    ``repetitions`` controls repeated strict stepwise replay timing.  The
    report intentionally identifies measurements that are not currently
    supportable: first-failing-step localization is not scored because the
    generated corruptions do not declare a ground-truth step, and non-record
    stereo fixtures are not coerced into a typed replay representation.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    directory = Path(benchmark_dir)
    cases = load_evidence_cases(directory)
    if not cases:
        raise ValueError("No typed MechanismRecord cases were found.")

    per_partition: dict[str, Counter[str]] = defaultdict(Counter)
    runtimes: list[int] = []
    peaks: list[int] = []
    endpoint_disagreements = 0
    valid_replays = 0
    final_matches = 0
    clean_valid = 0
    clean_invalid = 0
    corruption_totals: Counter[str] = Counter()
    corruption_detected: Counter[str] = Counter()
    corruption_issue_matched: Counter[str] = Counter()
    issue_mismatches: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in cases:
        per_partition[case.partition]["typed_records"] += 1
        endpoint = _measure_replay(case.record, "endpoint")
        stepwise = _measure_replay(case.record, "stepwise")
        if endpoint["status"] != stepwise["status"]:
            endpoint_disagreements += 1
        if stepwise["status"] == "VALID":
            valid_replays += 1
            clean_valid += 1
        else:
            clean_invalid += 1
        final_matches += int(stepwise["final_matches"])

        timing_samples = [
            _measure_replay(case.record, "stepwise") for _ in range(repetitions)
        ]
        runtimes.extend(sample["elapsed_ns"] for sample in timing_samples)
        peaks.extend(sample["python_peak_bytes"] for sample in timing_samples)

        for corruption in corrupt_record(case.record):
            corruption_totals[corruption.corruption] += 1
            observed = corruption.observed_issue_codes()
            detected = bool(observed)
            if detected:
                corruption_detected[corruption.corruption] += 1
            if corruption.expected_issue_code in observed:
                corruption_issue_matched[corruption.corruption] += 1
            elif len(issue_mismatches[corruption.corruption]) < 5:
                issue_mismatches[corruption.corruption].append(
                    {
                        "case_id": case.case_id,
                        "partition": case.partition,
                        "expected_issue_code": corruption.expected_issue_code,
                        "observed_issue_codes": list(observed),
                    }
                )

    clean_total = clean_valid + clean_invalid
    corruption_total = sum(corruption_totals.values())
    true_positive = sum(corruption_detected.values())
    false_negative = corruption_total - true_positive
    false_positive = clean_invalid
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if corruption_total else 0.0
    )
    negative_recall = clean_valid / clean_total if clean_total else 0.0
    macro_f1 = (
        (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    )
    macro_f1 = (macro_f1 + negative_recall) / 2

    per_corruption = {
        name: {
            "count": count,
            "detected": corruption_detected[name],
            "expected_issue_matched": corruption_issue_matched[name],
            "detection_recall": corruption_detected[name] / count,
            "issue_code_accuracy": corruption_issue_matched[name] / count,
            "issue_code_mismatch_examples": issue_mismatches[name],
        }
        for name, count in sorted(corruption_totals.items())
    }
    return {
        "schema": "SynKit-MechanismBench-evidence-v1",
        "benchmark_fingerprints": _manifest_fingerprints(directory),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "scope": {
            "typed_replay_cases": len(cases),
            "partition_counts": {
                partition: counters["typed_records"]
                for partition, counters in sorted(per_partition.items())
            },
            "excluded_fixture_representations": [
                "reaction_smiles",
                "non_tetrahedral_rewrite",
            ],
        },
        "replay": {
            "valid": valid_replays,
            "total": len(cases),
            "final_product_matches": final_matches,
            "endpoint_stepwise_status_disagreements": endpoint_disagreements,
        },
        "corruption_detection": {
            "clean_valid": clean_valid,
            "clean_invalid": clean_invalid,
            "corrupted_total": corruption_total,
            "detected": true_positive,
            "expected_issue_matched": sum(corruption_issue_matched.values()),
            "precision": precision,
            "recall": recall,
            "macro_f1": macro_f1,
            "issue_code_accuracy": (
                sum(corruption_issue_matched.values()) / corruption_total
            ),
            "per_corruption": per_corruption,
        },
        "runtime": {
            "repetitions_per_case": repetitions,
            "stepwise_replay_ns": _summary(runtimes),
            "python_tracemalloc_peak_bytes": _summary(peaks),
        },
        "limitations": {
            "memory": "Python allocations measured by tracemalloc; this is not process RSS.",
            "first_failing_step_localization": (
                "Not scored: the current corruption generator does not encode a "
                "ground-truth failing step."
            ),
            "unsupported_ablations": [
                "no_grouping",
                "no_radical_state",
                "no_stereo_references",
            ],
        },
    }


def write_evidence_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    """Write an evidence report as deterministic, human-readable JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
