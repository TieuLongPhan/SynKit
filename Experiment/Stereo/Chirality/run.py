#!/usr/bin/env python3
"""Run whole-molecule chirality or stereoisomer-relation benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.Chirality.exact_mirror import (  # noqa: E402
    benchmark_exact_acs_chirality,
)
from Experiment.Stereo.Chirality.relations import (  # noqa: E402
    benchmark_stereoisomer_relations,
)
from Experiment.Stereo.Chirality.published import (  # noqa: E402
    DATASET,
)

CHIRALITY_DATA_ROOT = ROOT / "Experiment" / "Stereo" / "Data" / "Chirality"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("acs", "relations"))
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--case-timeout", type=float, default=5.0)
    parser.add_argument(
        "--identity-profile",
        choices=("chemical", "lewis_state", "acs_topology"),
        default="chemical",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.case_timeout <= 0:
        parser.error("--case-timeout must be positive.")
    RDLogger.DisableLog("rdApp.*")

    if arguments.task == "acs":
        report = benchmark_exact_acs_chirality(
            arguments.dataset,
            case_timeout_seconds=arguments.case_timeout,
            identity_profile=arguments.identity_profile,
        )
        default_output = CHIRALITY_DATA_ROOT / "exact_acs_mirror_report.json"
        summary = {
            "records": report["dataset"]["records"],
            "definitive_records": report["definitive_records"],
            "definitive_coverage": report["definitive_coverage"],
            "accuracy_among_definitive": report["accuracy_among_definitive"],
            "accuracy_over_all_records": report["accuracy_over_all_records"],
            "seconds": report["timing"]["total_seconds"],
        }
    else:
        report = benchmark_stereoisomer_relations()
        default_output = CHIRALITY_DATA_ROOT / "stereoisomer_relation_report.json"
        summary = {
            **report["totals"],
            "accuracy": (report["totals"]["correct"] / report["totals"]["cases"]),
            "seconds": report["seconds"],
        }

    output = arguments.output or default_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "task": arguments.task,
                "output": str(output),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
