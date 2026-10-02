"""Verify the frozen Synister evidence campaigns and their promoted counts.

The verifier intentionally checks a small, reviewable set of manuscript-facing
counts after ``summarize_synister_evidence`` has verified every case digest and
manifest binding.  It does not replace either immutable campaign record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_synister_evidence import summarize

# These values are restricted to claims C03 and C04 in the claim ledger.  They
# deliberately omit timings and other machine-dependent descriptive fields.
FROZEN_CAMPAIGNS = (
    {
        "name": "pilot100_v4",
        "path": ROOT / "paper/synister/evidence/pilot100_v4",
        "manifest_sha256": "2a10b5d9b46c6d585a319f7940d808ace3338c6ef22137d3fb23bb7615e5bbb7",
        "case_records": 100,
        "modes": {
            "minimal": {
                "cases": 46,
                "structure_complete": 46,
                "multiple_exact_its_classes": 16,
                "alternative_its_classes_relative_to_reference": 39,
            },
            "reference_cd": {
                "cases": 52,
                "structure_complete": 52,
                "multiple_exact_its_classes": 18,
                "alternative_its_classes_relative_to_reference": 54,
            },
        },
    },
    {
        "name": "flower10k_v4_30s",
        "path": ROOT / "benchmark_results/synister_global_shells_flower10k_v4_30s",
        "manifest_sha256": "be0e4968af7ddfc6c6f6d90aff5de441de99cc95cd1aec7ccbf9800e58f0969e",
        "case_records": 392,
        "modes": {
            "minimal": {
                "cases": 244,
                "structure_complete": 244,
                "multiple_exact_its_classes": 81,
                "reference_its_class_observed": 231,
            },
            "reference_cd": {
                "cases": 280,
                "structure_complete": 279,
                "multiple_exact_its_classes": 103,
                "reference_its_class_observed": 279,
            },
        },
    },
)


def audit_campaign(spec: dict[str, object]) -> dict[str, object]:
    """Summarize one campaign and reject drift in its promoted counts."""
    path = Path(spec["path"])
    summary = summarize(path)
    for key in ("campaign_manifest_sha256", "case_records"):
        expected = spec["manifest_sha256"] if key == "campaign_manifest_sha256" else spec[key]
        if summary[key] != expected:
            raise ValueError(
                f"{spec['name']}: {key} is {summary[key]!r}, expected {expected!r}"
            )
    for mode, expected_values in spec["modes"].items():
        observed = summary["modes"].get(mode)
        if observed is None:
            raise ValueError(f"{spec['name']}: missing {mode!r} mode")
        for key, expected in expected_values.items():
            if observed.get(key) != expected:
                raise ValueError(
                    f"{spec['name']} {mode}: {key} is {observed.get(key)!r}, "
                    f"expected {expected!r}"
                )
    return {
        "name": spec["name"],
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "campaign_manifest_sha256": summary["campaign_manifest_sha256"],
        "case_records": summary["case_records"],
        "modes": {
            mode: {key: summary["modes"][mode][key] for key in values}
            for mode, values in spec["modes"].items()
        },
    }


def verify_frozen_campaigns() -> dict[str, object]:
    """Return a deterministic compact audit for all promoted frozen campaigns."""
    return {
        "schema_version": 1,
        "kind": "synister_frozen_evidence_audit",
        "campaigns": [audit_campaign(spec) for spec in FROZEN_CAMPAIGNS],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the JSON audit to this path")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    payload = json.dumps(verify_frozen_campaigns(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
