#!/usr/bin/env python3
"""Run the separately named exact-plus-global-certificate ACS protocol."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.Chirality.exact_mirror import (  # noqa: E402
    _case_time_limit,
    _parse_supplied_configured_smiles,
)
from Experiment.Stereo.Chirality.published import (  # noqa: E402
    EXPECTED_SHA256,
    load_dataset,
)
from Experiment.Stereo.provenance import capture_source, finish_source  # noqa: E402
from synkit.Chem.Molecule.global_stereo import (  # noqa: E402
    analyze_global_stereo_support,
)
from synkit.Graph.Stereo import (  # noqa: E402
    StereographMirrorStatus,
    classify_rdkit_stereograph_mirror,
)

SCHEMA = "synkit.exact-plus-global-certificate-acs/2"


def _classify_record(row: dict[str, str]) -> dict[str, Any]:
    """Classify one record without consulting its expected label."""
    started = time.perf_counter()
    sanitized = Chem.MolFromSmiles(row["Input SMILES"])
    restored = _parse_supplied_configured_smiles(row["Input SMILES"])
    if sanitized is None or restored is None:
        return {
            "id": row["ID"],
            "manual": row["manual"].lower(),
            "status": "unsupported",
            "correct": False,
            "reason": "parse_failure",
        }
    exact = classify_rdkit_stereograph_mirror(
        restored,
        require_complete=False,
        identity_profile="chemical",
    )
    certificate = analyze_global_stereo_support(sanitized)
    status = (
        "chiral"
        if (
            exact.status is StereographMirrorStatus.CHIRAL
            or certificate.necessarily_chiral
        )
        else exact.status.value
    )
    return {
        "id": row["ID"],
        "manual": row["manual"].lower(),
        "status": status,
        "correct": status == row["manual"].lower(),
        "source_declared_status": exact.status.value,
        "global_information_state": certificate.state.value,
        "global_necessarily_chiral": certificate.necessarily_chiral,
        "global_original_digest": certificate.original_digest,
        "global_mirror_digest": certificate.mirror_digest,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "method": (
            "exact_source_declared_chemical_or_" "necessary_global_topology_certificate"
        ),
    }


def classify_record(row: dict[str, str], case_timeout_seconds: float = 30) -> dict[str, Any]:
    """Bound both evidence computations and retain every refused input record."""
    started = time.perf_counter()
    try:
        with _case_time_limit(case_timeout_seconds):
            return _classify_record(row)
    except Exception as error:
        return {
            "id": row["ID"], "manual": row["manual"].lower(), "correct": False,
            "status": "case_timeout" if isinstance(error, TimeoutError) else "error",
            "reason": type(error).__name__, "detail": str(error),
            "duration_ms": (time.perf_counter() - started) * 1000,
        }


def run_benchmark(case_timeout_seconds: float = 30) -> dict[str, Any]:
    """Return the complete deterministic scientific result."""
    source = capture_source()
    started = time.perf_counter()
    records = [classify_record(row, case_timeout_seconds) for row in load_dataset()]
    definitive = [
        record for record in records if record["status"] in {"achiral", "chiral"}
    ]
    correct = sum(bool(record["correct"]) for record in records)
    disagreements = [record["id"] for record in records if not bool(record["correct"])]
    return {
        "schema": SCHEMA,
        "provenance": finish_source(source),
        "case_timeout_seconds": case_timeout_seconds,
        "identity_profile": "chemical source identity; separately declared global topology evidence",
        "input_contract": {
            "structural_input": (
                "same pinned SMILES; restored source tags for identity; normally sanitized input for global support"
            ),
            "external_orientation": False, "topology_completion": False,
            "global_necessary_chirality_evidence": True,
            "manual_labels_used_for_prediction": False,
            "interpretation": (
                "necessary global chirality may be proved while the absolute configured orientation remains unspecified"
            ),
        },
        "task": (
            "source-declared exact chemical stereograph augmented by an "
            "orientation-unspecified necessary-global-chirality certificate"
        ),
        "dataset": {
            "records": len(records),
            "audited_sha256": EXPECTED_SHA256,
        },
        "records": records,
        "definitive_records": len(definitive),
        "correct_over_all_records": correct,
        "accuracy_over_all_records": correct / len(records),
        "disagreement_ids": disagreements,
        "nondefinitive_ids": [record["id"] for record in records if record not in definitive],
        "definitive_disagreement_ids": [record["id"] for record in definitive if not record["correct"]],
        "outcomes": dict(sorted(Counter(record["status"] for record in records).items())),
        "timing": {
            "wall_seconds": round(time.perf_counter() - started, 6),
        },
        "claim_boundary": (
            "This protocol does not reinterpret source @/@@ tags and does not "
            "select a global enantiomer from topology. It may prove necessary "
            "global chirality while configured orientation remains unspecified."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-timeout", type=float, default=30)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.case_timeout <= 0:
        raise ValueError("Case timeout must be positive.")
    report = run_benchmark(arguments.case_timeout)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "correct": report["correct_over_all_records"],
                "disagreement_ids": report["disagreement_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
