#!/usr/bin/env python3
"""Run the separately named exact-plus-global-certificate ACS protocol."""

from __future__ import annotations

import argparse
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
    _parse_supplied_configured_smiles,
)
from Experiment.Stereo.Chirality.published import (  # noqa: E402
    EXPECTED_SHA256,
    load_dataset,
)
from synkit.Chem.Molecule.global_stereo import (  # noqa: E402
    analyze_global_stereo_support,
)
from synkit.Graph.Stereo import (  # noqa: E402
    StereographMirrorStatus,
    classify_rdkit_stereograph_mirror,
)

SCHEMA = "synkit.exact-plus-global-certificate-acs/1"


def classify_record(row: dict[str, str]) -> dict[str, Any]:
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
            "exact_source_declared_chemical_or_"
            "necessary_global_topology_certificate"
        ),
    }


def run_benchmark() -> dict[str, Any]:
    """Return the complete deterministic scientific result."""
    started = time.perf_counter()
    records = [classify_record(row) for row in load_dataset()]
    definitive = [
        record for record in records if record["status"] in {"achiral", "chiral"}
    ]
    correct = sum(bool(record["correct"]) for record in records)
    disagreements = [
        record["id"] for record in records if not bool(record["correct"])
    ]
    return {
        "schema": SCHEMA,
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
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = run_benchmark()
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
