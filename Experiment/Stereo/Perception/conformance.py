#!/usr/bin/env python3
"""Run the designed stereo-perception conformance dataset.

Every family is scored by its declared executor: connectivity perception,
configured RDKit adapters, or formal sidecar contracts. Genuine missing
capabilities are reported as failures rather than hidden as not applicable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

from rdkit import Chem
import rdkit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synkit.Chem.Molecule.stereo_perception import (  # noqa: E402
    PotentialStereoElement,
    StereoCarrierStatus,
    detect_potential_stereo_elements,
)
from synkit.Graph.Stereo.supports import (  # noqa: E402
    AtomStereoSupport,
    AxisStereoSupport,
    BondStereoSupport,
)

from Experiment.Stereo.Perception.family_executors import (  # noqa: E402
    evaluate_family_scope,
    neutralize_rdkit_configuration,
)

DEFAULT_DATASET = (
    ROOT
    / "Experiment"
    / "Stereo"
    / "Data"
    / "Perception"
    / "perception_conformance_cases.json"
)
DEFAULT_REPORT = (
    ROOT
    / "Experiment"
    / "Stereo"
    / "Data"
    / "Perception"
    / "perception_conformance_report.json"
)
def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _normalize_frame(frame: list[Any] | tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(sorted((repr(reference) for reference in frame)))


def _normalize_axis(
    path: list[int] | tuple[int, ...],
    frames: list[list[Any]] | tuple[tuple[Any, ...], ...],
) -> tuple[Any, ...]:
    forward = (
        tuple(path),
        _normalize_frame(frames[0]),
        _normalize_frame(frames[1]),
    )
    reverse = (
        tuple(reversed(path)),
        _normalize_frame(frames[1]),
        _normalize_frame(frames[0]),
    )
    return min(forward, reverse, key=repr)


def _expected_support(value: dict[str, Any]) -> tuple[Any, ...]:
    kind = value["kind"]
    if kind == "atom":
        return kind, int(value["center"])
    if kind == "bond":
        return kind, tuple(sorted(value["bond"]))
    if kind in {"axis", "path"} and "terminal_frames" in value:
        return kind, _normalize_axis(value["path"], value["terminal_frames"])
    raise ValueError(f"Unsupported in-scope expected support: {value!r}")


def _observed_support(element: PotentialStereoElement) -> tuple[Any, ...]:
    support = element.support
    if isinstance(support, AtomStereoSupport):
        return "atom", support.center
    if isinstance(support, BondStereoSupport):
        return "bond", tuple(sorted(support.endpoints))
    if isinstance(support, AxisStereoSupport):
        kind = (
            "path"
            if element.element_type.value == "extended_cis_trans"
            else "axis"
        )
        return kind, _normalize_axis(support.path, support.terminal_frames)
    raise TypeError(f"Unsupported observed support: {type(support).__name__}")


def _serialize_element(element: PotentialStereoElement) -> dict[str, Any]:
    support = element.support
    if isinstance(support, AtomStereoSupport):
        serialized_support: dict[str, Any] = {
            "kind": "atom",
            "center": support.center,
        }
    elif isinstance(support, BondStereoSupport):
        serialized_support = {
            "kind": "bond",
            "bond": list(support.endpoints),
        }
    elif isinstance(support, AxisStereoSupport):
        serialized_support = {
            "kind": (
                "path"
                if element.element_type.value == "extended_cis_trans"
                else "axis"
            ),
            "path": list(support.path),
            "terminal_frames": [
                list(frame) for frame in support.terminal_frames
            ],
        }
    else:  # pragma: no cover - guarded by the current perception type set
        serialized_support = {"kind": type(support).__name__}
    return {
        "family": element.element_type.value,
        "carrier_status": element.carrier_status.value,
        "carrier_reason": element.carrier_reason,
        "support": serialized_support,
        "source_identifier": element.source_identifier,
    }


def _evaluate_case(
    case: dict[str, Any],
    *,
    family_scope: str,
    structure: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "id": case["id"],
        "family": case["family"],
        "case_type": case["case_type"],
        "perception_scope": family_scope,
        "executor": family_scope,
        "expected_outcome": case["expected"]["outcome"],
        "expected_reason_code": case["expected"]["reason_code"],
    }
    if family_scope != "current_topology_detector":
        return {
            **base,
            **evaluate_family_scope(case, structure, family_scope),
        }
    if structure["format"] != "smiles":
        return {
            **base,
            "status": "error",
            "status_reason": "current_detector_requires_smiles",
            "carrier_check": "failed",
            "configuration_check": "not_applicable",
            "reason_check": "not_evaluated",
            "observed": [],
        }

    molecule = Chem.MolFromSmiles(structure["value"])
    if molecule is None:
        return {
            **base,
            "status": "error",
            "status_reason": "smiles_parse_failure",
            "carrier_check": "failed",
            "configuration_check": "not_applicable",
            "reason_check": "not_evaluated",
            "observed": [],
        }
    molecule = neutralize_rdkit_configuration(molecule)
    observed = detect_potential_stereo_elements(molecule)
    family_elements = tuple(
        element for element in observed if element.element_type.value == case["family"]
    )
    confirmed_elements = tuple(
        element
        for element in family_elements
        if element.carrier_status is StereoCarrierStatus.CONFIRMED
    )
    expected = case["expected"]
    expected_present = expected["outcome"] == "carrier_present"
    if expected_present:
        expected_support = _expected_support(expected["support"])
        matching = tuple(
            element
            for element in confirmed_elements
            if _observed_support(element) == expected_support
        )
        carrier_passed = bool(matching)
    else:
        matching = ()
        carrier_passed = not confirmed_elements

    expected_reason = expected["reason_code"]
    reason_check = "not_evaluated"
    if expected_reason is not None and any(
        element.carrier_reason == expected_reason for element in family_elements
    ):
        reason_check = "passed"

    passed = carrier_passed
    return {
        **base,
        "status": "passed" if passed else "failed",
        "status_reason": None if passed else "perception_mismatch",
        "carrier_check": "passed" if carrier_passed else "failed",
        "configuration_check": "ignored",
        "reason_check": reason_check,
        "observed": [_serialize_element(element) for element in family_elements],
    }


def benchmark_perception_conformance(path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    """Evaluate every record with its declared family executor."""
    started = time.perf_counter()
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        display_path = str(path.resolve().relative_to(ROOT))
    except ValueError:
        display_path = str(path.resolve())
    structures = payload["structures"]
    families = payload["families"]
    records = [
        _evaluate_case(
            case,
            family_scope=families[case["family"]]["perception_scope"],
            structure=structures[case["structure"]],
        )
        for case in payload["cases"]
    ]
    status_counts = Counter(record["status"] for record in records)
    evaluated = [record for record in records if record["status"] != "not_applicable"]
    scored = [record for record in evaluated if record["status"] != "error"]
    passed = [record for record in scored if record["status"] == "passed"]
    by_family: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["family"]].append(record)
    for family in families:
        family_records = grouped[family]
        family_scored = [
            record
            for record in family_records
            if record["status"] not in {"not_applicable", "error"}
        ]
        family_passed = [
            record for record in family_scored if record["status"] == "passed"
        ]
        by_family[family] = {
            "scope": families[family]["perception_scope"],
            "records": len(family_records),
            "scored": len(family_scored),
            "passed": len(family_passed),
            "failed": len(family_scored) - len(family_passed),
            "conformance": _ratio(len(family_passed), len(family_scored)),
        }

    configuration_evaluated = [
        record
        for record in records
        if record["configuration_check"] in {"passed", "failed"}
    ]
    configuration_passed = [
        record
        for record in configuration_evaluated
        if record["configuration_check"] == "passed"
    ]
    reason_evaluated = [
        record
        for record in records
        if record["reason_check"] in {"passed", "failed"}
    ]
    reason_passed = [
        record for record in reason_evaluated if record["reason_check"] == "passed"
    ]
    return {
        "schema": "synkit.stereo-perception-conformance-report/1",
        "dataset": {
            "path": display_path,
            "schema": payload["schema"],
            "records": len(payload["cases"]),
            "families": len(payload["families"]),
        },
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "protocol": {
            "scored_scopes": [
                "current_topology_detector",
                "configured_adapter_only",
                "sidecar_only",
            ],
            "not_applicable_scopes": [],
            "configuration_evidence": "removed_before_perception",
            "reason_codes_scored": "when_structured_evidence_is_emitted",
        },
        "summary": {
            "records": len(records),
            "scored_records": len(scored),
            "passed": len(passed),
            "failed": len(scored) - len(passed),
            "not_applicable": status_counts["not_applicable"],
            "errors": status_counts["error"],
            "strict_conformance": _ratio(len(passed), len(scored)),
            "configuration_checks": len(configuration_evaluated),
            "configuration_checks_passed": len(configuration_passed),
            "configuration_conformance": _ratio(
                len(configuration_passed),
                len(configuration_evaluated),
            ),
            "reason_checks": len(reason_evaluated),
            "reason_checks_passed": len(reason_passed),
            "reason_conformance": _ratio(
                len(reason_passed),
                len(reason_evaluated),
            ),
        },
        "failure_ids": [
            record["id"] for record in records if record["status"] == "failed"
        ],
        "by_family": by_family,
        "records": records,
        "seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This is designed API conformance, not empirical chemical "
            "accuracy. Connectivity, configured-adapter, and formal-sidecar "
            "results are separate executor contracts. Configuration evidence "
            "is removed or ignored before every carrier decision. Formal "
            "graphs marked "
            "chemical_validation=false do not establish chemical truth. "
            "Unconfigured non-tetrahedral geometry is not inferred from "
            "connectivity alone. Atrop candidates carry no stability claim."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = benchmark_perception_conformance(arguments.dataset)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "summary": report["summary"],
                "failure_ids": report["failure_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
