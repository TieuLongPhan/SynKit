#!/usr/bin/env python3
"""Evaluate exact configured-stereograph mirror identity on the ACS corpus."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import statistics
import signal
import sys
import time
from typing import Any

from rdkit import Chem
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.Chirality.published import (  # noqa: E402
    EXPECTED_SHA256,
    load_dataset,
)
from synkit.Graph.Stereo import (  # noqa: E402
    MirrorIdentityProfile,
    classify_rdkit_stereograph_mirror,
)


def _parse_supplied_configured_smiles(smiles: str) -> Any:
    """Parse a configured SMILES without losing source-declared atom stereo.

    RDKit's default ``cleanIt=True`` stereo assignment removes some explicit
    tetrahedral tags.  Retain the normal sanitized molecule and its bond-stereo
    handling, then restore every explicit CW/CCW atom tag from the audited input
    because this task evaluates the supplied configured representation.
    """
    molecule = Chem.MolFromSmiles(smiles)
    source = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is None or source is None:
        return None
    for source_atom in source.GetAtoms():
        source_tag = source_atom.GetChiralTag()
        if source_tag in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }:
            molecule.GetAtomWithIdx(source_atom.GetIdx()).SetChiralTag(source_tag)
    return molecule


class _CaseTimeout(TimeoutError):
    pass


class _case_time_limit:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.previous_handler: Any = None

    def __enter__(self) -> None:
        if not hasattr(signal, "setitimer"):
            return
        self.previous_handler = signal.signal(signal.SIGALRM, self._expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *_error: Any) -> None:
        if not hasattr(signal, "setitimer"):
            return
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self.previous_handler)

    @staticmethod
    def _expired(_signum: int, _frame: Any) -> None:
        raise _CaseTimeout


def benchmark_exact_acs_chirality(
    path: Path,
    *,
    case_timeout_seconds: float = 5.0,
    identity_profile: MirrorIdentityProfile = "chemical",
) -> dict[str, Any]:
    rows = load_dataset(path)
    records = []
    durations = []
    started_all = time.perf_counter()
    for row in rows:
        molecule = _parse_supplied_configured_smiles(row["Input SMILES"])
        if molecule is None:
            records.append(
                {
                    "id": row["ID"],
                    "manual": row["manual"].lower(),
                    "status": "parse_failure",
                }
            )
            continue
        started = time.perf_counter_ns()
        try:
            with _case_time_limit(case_timeout_seconds):
                result = classify_rdkit_stereograph_mirror(
                    molecule,
                    require_complete=False,
                    identity_profile=identity_profile,
                )
        except _CaseTimeout:
            duration = time.perf_counter_ns() - started
            durations.append(duration)
            records.append(
                {
                    "id": row["ID"],
                    "manual": row["manual"].lower(),
                    "status": "case_timeout",
                    "duration_ms": duration / 1_000_000,
                }
            )
            continue
        except Exception as error:
            duration = time.perf_counter_ns() - started
            durations.append(duration)
            records.append(
                {
                    "id": row["ID"],
                    "manual": row["manual"].lower(),
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "duration_ms": duration / 1_000_000,
                }
            )
            continue
        duration = time.perf_counter_ns() - started
        durations.append(duration)
        records.append(
            {
                "id": row["ID"],
                "manual": row["manual"].lower(),
                "status": result.status.value,
                "descriptor_count": result.descriptor_count,
                "incomplete_loci": list(result.incomplete_loci),
                "unsupported_loci": list(result.unsupported_loci),
                "unsupported_families": list(result.unsupported_families),
                "method": result.method,
                "original_digest": (
                    result.original.canonical_digest if result.original else None
                ),
                "mirror_digest": (
                    result.mirror.canonical_digest if result.mirror else None
                ),
                "duration_ms": duration / 1_000_000,
            }
        )
    definitive = [
        record for record in records if record["status"] in {"chiral", "achiral"}
    ]
    correct = sum(record["status"] == record["manual"] for record in definitive)
    disagreements = [
        record["id"] for record in definitive if record["status"] != record["manual"]
    ]
    definitive_with_unresolved = [
        record for record in definitive if record.get("incomplete_loci")
    ]
    durations_ms = [duration / 1_000_000 for duration in durations]
    return {
        "schema": "synkit.exact-acs-mirror-benchmark/3",
        "dataset": {
            "records": len(rows),
            "audited_sha256": EXPECTED_SHA256,
            "manual_labels": dict(
                sorted(Counter(row["manual"].lower() for row in rows).items())
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "task": (
            "global mirror identity of the source-declared configured " "stereograph"
        ),
        "identity_profile": identity_profile,
        "case_timeout_seconds": case_timeout_seconds,
        "outcomes": dict(
            sorted(Counter(record["status"] for record in records).items())
        ),
        "definitive_records": len(definitive),
        "definitive_coverage": len(definitive) / len(rows),
        "correct_definitive": correct,
        "definitive_with_unresolved_loci": len(definitive_with_unresolved),
        "definitive_with_unresolved_locus_ids": [
            record["id"] for record in definitive_with_unresolved
        ],
        "accuracy_among_definitive": (
            correct / len(definitive) if definitive else None
        ),
        "correct_over_all_records": correct,
        "accuracy_over_all_records": correct / len(rows),
        "disagreement_ids": disagreements,
        "nondefinitive_ids": [
            record["id"] for record in records if record not in definitive
        ],
        "timing": {
            "total_seconds": time.perf_counter() - started_all,
            "mean_ms": statistics.mean(durations_ms) if durations_ms else None,
            "median_ms": (statistics.median(durations_ms) if durations_ms else None),
            "p95_ms": (
                sorted(durations_ms)[int(0.95 * (len(durations_ms) - 1))]
                if durations_ms
                else None
            ),
        },
        "records": records,
        "claim_boundary": (
            "This is the exact configured-stereograph path under the recorded "
            "identity profile, not the separate ACS-specialized "
            "topology-completion classifier. The chemical profile compares "
            "the enumerated resonance family while preserving genuine "
            "bond-order distinctions. The method does "
            "not invent configurations absent from the input: every explicit "
            "source atom configuration is preserved, and undeclared supported "
            "loci remain unconstrained instead of becoming completeness "
            "blockers. This is therefore a supplied-configuration result, not "
            "a claim that every possible stereo locus was configured. "
            "Enhanced stereo populations remain nondefinitive."
        ),
    }
