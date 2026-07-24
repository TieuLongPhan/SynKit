#!/usr/bin/env python3
"""Run SynKit's independent local-label evaluator on the CIP suite."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import rdCIPLabeler
import rdkit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.datasets import (  # noqa: E402
    CIP_SHA256,
    load_cip,
)
from synkit.Chem.Molecule.cip_assignment import (  # noqa: E402
    CIPAssignment,
    CIPAssignmentStatus,
    assign_cip_labels,
)
from synkit.Graph.Stereo import (  # noqa: E402
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    PlanarBondStereo,
    TetrahedralStereo,
    descriptors_from_rdkit,
)

_LABEL_PATTERN = re.compile(r"^(\d+)([A-Za-z]+)$")
_EXTENDED_UNSUPPORTED = frozenset({"TH3", "TH5"})
_REVIEWED_MISMATCH_CAUSES = {
    "VS014": "label_projection_defect",
    "VS032": "ranking_defect",
    "VS033": "ranking_defect",
    "VS038": "ranking_defect",
    "VS039": "ranking_defect",
    "VS115": "ranking_defect",
    "VS122": "ranking_defect",
    "VS130": "label_projection_defect",
    "VS172": "ranking_defect",
    "VS174": "ranking_defect",
    "VS176": "label_projection_defect",
    "VS177": "ranking_defect",
}
_LIMITATION_PRIORITY = (
    "unsupported_class",
    "ranking_defect",
    "missing_orientation_evidence",
    "label_projection_defect",
    "disputed_reference",
)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _unit_categories(record: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(
        item.strip() for item in str(record["stereo_units"]).split(",") if item.strip()
    )
    return values or ("none",)


def _label_parts(label: str) -> tuple[int, str]:
    match = _LABEL_PATTERN.fullmatch(label)
    if match is None:
        raise ValueError(f"Unsupported CIP reference label: {label!r}.")
    position, descriptor = match.groups()
    return int(position), descriptor


def _descriptor_positions(descriptor: Any) -> tuple[int, ...]:
    if isinstance(descriptor, TetrahedralStereo):
        return (int(descriptor.center),)
    if isinstance(descriptor, (PlanarBondStereo, AtropBondStereo)):
        return tuple(int(value) for value in descriptor.atoms[2:4])
    if isinstance(descriptor, CumuleneAxisStereo):
        return descriptor.axis_path
    if isinstance(descriptor, ExtendedCisTransStereo):
        return descriptor.path[0], descriptor.path[-1]
    if isinstance(descriptor, HelicalStereo):
        return descriptor.reported_positions or descriptor.path
    return ()


def _predicted_labels(
    descriptors: Iterable[Any],
    assignments: Iterable[CIPAssignment],
) -> set[str]:
    labels = set()
    for descriptor, assignment in zip(descriptors, assignments):
        if not assignment.assigned or assignment.label is None:
            continue
        labels.update(
            f"{position}{assignment.label}"
            for position in _descriptor_positions(descriptor)
        )
    return labels


def _record_limitations(
    record: dict[str, Any],
    expected: set[str],
    predicted: set[str],
    descriptors: tuple[Any, ...],
    assignments: tuple[CIPAssignment, ...],
    extraction_error: str | None,
) -> tuple[str, ...]:
    if expected == predicted:
        return ()
    limitations = []
    units = set(_unit_categories(record))
    expected_parts = tuple(_label_parts(label) for label in expected)
    if units & _EXTENDED_UNSUPPORTED:
        limitations.append("unsupported_class")
    if "HE" in units:
        limitations.append("missing_orientation_evidence")
    if any(label.islower() for _position, label in expected_parts):
        limitations.append("ranking_defect")
    statuses = {assignment.status for assignment in assignments}
    if CIPAssignmentStatus.UNSUPPORTED_DESCRIPTOR in statuses:
        limitations.append("unsupported_class")
    if statuses & {
        CIPAssignmentStatus.UNRESOLVED_PRIORITY,
        CIPAssignmentStatus.UNSUPPORTED_SEQUENCE_RULE,
    }:
        limitations.append("ranking_defect")
    if CIPAssignmentStatus.UNSPECIFIED_CONFIGURATION in statuses:
        limitations.append("missing_orientation_evidence")
    covered_positions = {
        position
        for descriptor in descriptors
        for position in _descriptor_positions(descriptor)
    }
    expected_positions = {position for position, _label in expected_parts}
    if extraction_error is not None or expected_positions - covered_positions:
        limitations.append("missing_orientation_evidence")
    if not limitations:
        limitations.append(
            _REVIEWED_MISMATCH_CAUSES.get(
                str(record["ID"]),
                "label_projection_defect",
            )
        )
    return tuple(dict.fromkeys(limitations))


def _record_result(
    record: dict[str, Any],
    molecule: Chem.Mol,
) -> dict[str, Any]:
    expected = set(record["recommended_labels"])
    extraction_error = None
    try:
        descriptors = tuple(
            descriptors_from_rdkit(
                molecule,
                require_atom_maps=False,
            ).values()
        )
        reference_to_index = {
            index + 1: index for index in range(molecule.GetNumAtoms())
        }
        assignments = assign_cip_labels(
            molecule,
            descriptors,
            reference_to_index=reference_to_index,
        )
    except (TypeError, ValueError, NotImplementedError) as error:
        descriptors = ()
        assignments = ()
        extraction_error = type(error).__name__
    predicted = _predicted_labels(descriptors, assignments)
    limitations = _record_limitations(
        record,
        expected,
        predicted,
        descriptors,
        assignments,
        extraction_error,
    )
    primary_limitation = next(
        (item for item in _LIMITATION_PRIORITY if item in limitations),
        None,
    )
    return {
        "id": str(record["ID"]),
        "stereo_units": list(_unit_categories(record)),
        "expected": sorted(expected),
        "predicted": sorted(predicted),
        "exact": expected == predicted,
        "limitations": list(limitations),
        "primary_limitation": primary_limitation,
        "assignment_statuses": dict(
            sorted(Counter(item.status.value for item in assignments).items())
        ),
        "extraction_error": extraction_error,
    }


def _rdkit_labels(molecule: Chem.Mol) -> set[str]:
    working = Chem.Mol(molecule)
    rdCIPLabeler.AssignCIPLabels(working)
    labels = set()
    for atom in working.GetAtoms():
        if atom.HasProp("_CIPCode"):
            labels.add(f"{atom.GetIdx() + 1}{atom.GetProp('_CIPCode')}")
    for bond in working.GetBonds():
        if not bond.HasProp("_CIPCode"):
            continue
        label = bond.GetProp("_CIPCode")
        labels.add(f"{bond.GetBeginAtomIdx() + 1}{label}")
        labels.add(f"{bond.GetEndAtomIdx() + 1}{label}")
    return labels


def _score_records(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = tuple(results)
    expected = sum(len(item["expected"]) for item in values)
    predicted = sum(len(item["predicted"]) for item in values)
    true_positive = sum(
        len(set(item["expected"]) & set(item["predicted"])) for item in values
    )
    exact = sum(bool(item["exact"]) for item in values)
    return {
        "records": len(values),
        "exact_records": exact,
        "exact_record_accuracy": _safe_ratio(exact, len(values)),
        "expected_labels": expected,
        "predicted_labels": predicted,
        "true_positive_labels": true_positive,
        "micro_label_recall": _safe_ratio(true_positive, expected),
        "micro_label_precision": _safe_ratio(true_positive, predicted),
    }


def _category_scores(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = tuple(results)
    categories = sorted(
        {category for item in values for category in item["stereo_units"]}
    )
    return {
        category: _score_records(
            item for item in values if category in item["stereo_units"]
        )
        for category in categories
    }


def benchmark_cip_native(path: Path) -> dict[str, Any]:
    """Evaluate exact local label sets and type every non-exact record."""
    records = load_cip(path)
    results = []
    rdkit_results = []
    parse_failures = []
    started = time.perf_counter()
    for record in records:
        molecule = Chem.MolFromSmiles(str(record["SMILES"]))
        if molecule is None:
            parse_failures.append(str(record["ID"]))
            continue
        results.append(_record_result(record, molecule))
        expected = sorted(set(record["recommended_labels"]))
        rdkit_predicted = sorted(_rdkit_labels(molecule))
        rdkit_results.append(
            {
                "id": str(record["ID"]),
                "expected": expected,
                "predicted": rdkit_predicted,
                "exact": expected == rdkit_predicted,
            }
        )
    seconds = time.perf_counter() - started
    nonexact = [item for item in results if not item["exact"]]
    limitations = Counter(
        limitation for item in nonexact for limitation in item["limitations"]
    )
    primary_limitations = Counter(item["primary_limitation"] for item in nonexact)
    status_counts = Counter(
        status
        for item in results
        for status, count in item["assignment_statuses"].items()
        for _ in range(count)
    )
    return {
        "schema": "synkit.cip-native-validation/2",
        "dataset": {
            "records": len(records),
            "audited_sha256": CIP_SHA256,
            "structures_vendored": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "method": (
            "RDKit supplies parsed relative stereo descriptors only; SynKit "
            "independently ranks ligands with exact nuclide masses and "
            "projects local label sets"
        ),
        "parse_failures": parse_failures,
        "overall": _score_records(results),
        "by_stereo_unit": _category_scores(results),
        "comparators": {
            "rdkit": {
                **_score_records(rdkit_results),
                "method": "rdCIPLabeler.AssignCIPLabels",
                "disagreement_ids": [
                    item["id"] for item in rdkit_results if not item["exact"]
                ],
            },
            "stereomolgraph": {
                "applicable": False,
                "reason": "No local CIP-label assignment API is exposed.",
            },
        },
        "assignment_statuses": dict(sorted(status_counts.items())),
        "limitation_counts": dict(sorted(limitations.items())),
        "primary_limitation_counts": {
            item: primary_limitations.get(item, 0) for item in _LIMITATION_PRIORITY
        },
        "exact_ids": [item["id"] for item in results if item["exact"]],
        "nonexact_records": nonexact,
        "seconds": seconds,
        "mean_ms_per_input": 1000.0 * seconds / len(records),
        "claim_boundary": (
            "This is local CIP-label validation, not whole-molecule chirality "
            "accuracy. Non-exact records remain typed limitations."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cip-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    report = benchmark_cip_native(arguments.cip_path)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
