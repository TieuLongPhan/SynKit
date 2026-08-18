#!/usr/bin/env python
"""Render deterministic LaTeX and tabular assets from MTG evidence JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

EXPECTED_SCHEMA = "synkit.mtg-validation/1"
STAGE_LABELS = (
    ("rule_construction", "Rule construction"),
    ("extended_match_matrix", "Component matrix"),
    ("overlap_enumeration", "Overlap enumeration"),
    ("composite_construction_family", "Composite construction"),
    ("exact_quotient", "Exact quotient"),
    ("certificate_replay_family", "Certificate replay"),
    ("process_construction", "Process construction"),
    ("mtg_derivation", "MTG derivation"),
)


def _validated(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"Expected evidence schema {EXPECTED_SCHEMA!r}.")
    if report.get("status") != "PASS":
        raise ValueError("Paper assets require passing frozen evidence.")
    missing = [name for name, _ in STAGE_LABELS if name not in report["stage_timings"]]
    if missing:
        raise ValueError(f"Evidence is missing timing stages: {missing!r}.")
    return report


def latex_macros(report: Mapping[str, Any]) -> str:
    """Return stable result macros; empirical values remain visibly named."""
    report = _validated(report)
    observed = report["observed"]
    cases = report["case_studies"]
    timings = report["stage_timings"]
    values = {
        "MTGRawOverlaps": observed["raw_overlaps"],
        "MTGAcceptedWitnesses": observed["accepted_witnesses"],
        "MTGExactClasses": observed["exact_classes"],
        "MTGExploredStates": observed["explored_states"],
        "MTGPeakPythonMiB": observed["peak_python_mib"],
        "MTGRuleConstructionMs": timings["rule_construction"]["median_ms_per_unit"],
        "MTGMatrixMs": timings["extended_match_matrix"]["median_ms_per_unit"],
        "MTGEnumerationMs": timings["overlap_enumeration"]["median_ms_per_unit"],
        "MTGCompositeMs": timings["composite_construction_family"][
            "median_ms_per_unit"
        ],
        "MTGQuotientMs": timings["exact_quotient"]["median_ms_per_unit"],
        "MTGReplayMs": timings["certificate_replay_family"]["median_ms_per_unit"],
        "MTGProcessMs": timings["process_construction"]["median_ms_per_unit"],
        "MTGDerivationMs": timings["mtg_derivation"]["median_ms_per_unit"],
        "MTGAldolMechanisms": cases["aldol"]["mechanism_count"],
        "MTGAldolSteps": sum(cases["aldol"]["step_counts"]),
        "MTGSynthesisSteps": cases["multistep_synthesis"]["step_count"],
        "MTGGAThreePAlternatives": cases["glycolysis_ga3p"]["alternative_count"],
    }
    lines = ["% Generated from passing synkit.mtg-validation/1 evidence."]
    lines.extend(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()
    )
    lines.append(
        "\\newcommand{\\MTGInputDigest}{\\detokenize{%s}}" % report["input_sha256"]
    )
    return "\n".join(lines) + "\n"


def timing_table(report: Mapping[str, Any]) -> str:
    """Return TSV source data for the empirical timing table/figure."""
    report = _validated(report)
    rows = ["stage\tlabel\tmedian_ms\tmedian_ms_per_unit\twork_units"]
    for stage, label in STAGE_LABELS:
        timing = report["stage_timings"][stage]
        rows.append(
            "\t".join(
                (
                    stage,
                    label,
                    str(timing["median_ms"]),
                    str(timing["median_ms_per_unit"]),
                    str(timing["work_units_per_iteration"]),
                )
            )
        )
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--latex-output", required=True, type=Path)
    parser.add_argument("--timing-output", required=True, type=Path)
    arguments = parser.parse_args()
    report = json.loads(arguments.evidence.read_text(encoding="utf-8"))
    arguments.latex_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.timing_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.latex_output.write_text(latex_macros(report), encoding="utf-8")
    arguments.timing_output.write_text(timing_table(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
