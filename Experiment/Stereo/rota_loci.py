#!/usr/bin/env python3
"""Score SynKit's typed axis-locus perception on the 650-record RotA corpus."""

from __future__ import annotations

import argparse
from ast import literal_eval
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

from rdkit import Chem, RDLogger
import rdkit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.datasets import (  # noqa: E402
    ROTA_SHA256,
    load_rota,
)
from synkit.Chem.Molecule.stereo_perception import (  # noqa: E402
    StereoElementType,
    detect_potential_stereo_elements,
)

_AXIS_TYPES = frozenset(
    {
        StereoElementType.ATROP_AXIS,
        StereoElementType.CUMULENE_AXIS,
    }
)


def _undirected_pair(path: Iterable[int]) -> tuple[int, int]:
    values = tuple(path)
    if len(values) < 2:
        raise ValueError("An axis path requires at least two atoms.")
    return tuple(sorted((values[0], values[-1])))


def _normalized_path(path: Iterable[int]) -> tuple[int, ...]:
    values = tuple(path)
    return min(values, tuple(reversed(values)))


def _axis_predictions(molecule: Chem.Mol) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "type": element.element_type.value,
            "pair": _undirected_pair(element.support.path),
            "path": _normalized_path(element.support.path),
        }
        for element in detect_potential_stereo_elements(molecule)
        if element.element_type in _AXIS_TYPES
    )


def _renumbering_issues(
    molecule: Chem.Mol,
    predictions: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    size = molecule.GetNumAtoms()
    order = tuple(reversed(range(size)))
    old_to_new = {old: new for new, old in enumerate(order)}
    renumbered = Chem.RenumberAtoms(molecule, order)
    transported = {
        (
            item["type"],
            _normalized_path(old_to_new[index] for index in item["path"]),
        )
        for item in predictions
    }
    observed = {(item["type"], item["path"]) for item in _axis_predictions(renumbered)}
    return tuple(
        [
            *(f"missing:{item}" for item in sorted(transported - observed)),
            *(f"extra:{item}" for item in sorted(observed - transported)),
        ]
    )


def _record_result(
    index: int,
    row: dict[str, str],
    *,
    check_renumbering: bool,
) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(row["SMILES"])
    if molecule is None:
        return {
            "id": f"RotA-{index:04d}",
            "chiral_type": row["chiral_type"],
            "parse_failure": True,
        }
    reference_pairs = {tuple(sorted(pair)) for pair in literal_eval(row["label"])}
    reference_paths = {
        _normalized_path(path) for path in literal_eval(row["label_expanded"])
    }
    predictions = _axis_predictions(molecule)
    predicted_pairs = {item["pair"] for item in predictions}
    predicted_paths = {
        item["path"]
        for item in predictions
        if item["type"] == StereoElementType.CUMULENE_AXIS.value
    }
    issues = _renumbering_issues(molecule, predictions) if check_renumbering else ()
    true_pairs = reference_pairs & predicted_pairs
    return {
        "id": f"RotA-{index:04d}",
        "chiral_type": row["chiral_type"],
        "parse_failure": False,
        "reference_pairs": sorted(reference_pairs),
        "predicted_axes": list(predictions),
        "true_positive_pairs": sorted(true_pairs),
        "false_negative_pairs": sorted(reference_pairs - predicted_pairs),
        "false_positive_pairs": sorted(predicted_pairs - reference_pairs),
        "all_reference_pairs_recovered": reference_pairs <= predicted_pairs,
        "exact_pair_set": reference_pairs == predicted_pairs,
        "reference_expanded_paths": sorted(reference_paths),
        "true_positive_expanded_paths": sorted(reference_paths & predicted_paths),
        "renumbering_invariant": not issues if check_renumbering else None,
        "renumbering_issues": list(issues),
    }


def _sum_lengths(records: Iterable[dict[str, Any]], field: str) -> int:
    return sum(len(record.get(field, ())) for record in records)


def benchmark_rota_loci(
    path: Path,
    *,
    check_renumbering: bool = True,
) -> dict[str, Any]:
    rows = load_rota(path)
    started = time.perf_counter()
    records = [
        _record_result(index, row, check_renumbering=check_renumbering)
        for index, row in enumerate(rows)
    ]
    seconds = time.perf_counter() - started
    valid = [record for record in records if not record["parse_failure"]]
    reference = _sum_lengths(valid, "reference_pairs")
    predicted = sum(len(record["predicted_axes"]) for record in valid)
    true_positive = _sum_lengths(valid, "true_positive_pairs")
    expanded_reference = _sum_lengths(valid, "reference_expanded_paths")
    expanded_true_positive = _sum_lengths(valid, "true_positive_expanded_paths")
    precision = true_positive / predicted if predicted else None
    recall = true_positive / reference if reference else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    by_type: dict[str, Any] = {}
    for chiral_type in sorted({row["chiral_type"] for row in rows}):
        subset = [record for record in valid if record["chiral_type"] == chiral_type]
        subset_reference = _sum_lengths(subset, "reference_pairs")
        subset_predicted = sum(len(record["predicted_axes"]) for record in subset)
        subset_true_positive = _sum_lengths(subset, "true_positive_pairs")
        by_type[chiral_type] = {
            "records": len(subset),
            "reference_loci": subset_reference,
            "predicted_loci": subset_predicted,
            "true_positive_loci": subset_true_positive,
            "recall": (
                subset_true_positive / subset_reference if subset_reference else None
            ),
        }
    prediction_types = Counter(
        prediction["type"]
        for record in valid
        for prediction in record["predicted_axes"]
    )
    return {
        "schema": "synkit.rota-axis-locus-benchmark/1",
        "dataset": {
            "records": len(rows),
            "audited_sha256": ROTA_SHA256,
            "positive_only": True,
        },
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
        },
        "task": "typed axial carrier endpoint/path detection from supplied 2D input",
        "totals": {
            "parsed_records": len(valid),
            "parse_failures": len(rows) - len(valid),
            "reference_loci": reference,
            "predicted_loci": predicted,
            "true_positive_loci": true_positive,
            "records_with_all_reference_loci": sum(
                record["all_reference_pairs_recovered"] for record in valid
            ),
            "records_with_exact_locus_set": sum(
                record["exact_pair_set"] for record in valid
            ),
            "expanded_reference_paths": expanded_reference,
            "expanded_true_positive_paths": expanded_true_positive,
            "renumbering_checked_records": sum(
                record["renumbering_invariant"] is not None for record in valid
            ),
            "renumbering_invariant_records": sum(
                record["renumbering_invariant"] is True for record in valid
            ),
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "expanded_path_recall": (
            expanded_true_positive / expanded_reference if expanded_reference else None
        ),
        "prediction_types": dict(sorted(prediction_types.items())),
        "by_chiral_type": by_type,
        "parse_failure_ids": [
            record["id"] for record in records if record["parse_failure"]
        ],
        "renumbering_failures": [
            {
                "id": record["id"],
                "issues": record["renumbering_issues"],
            }
            for record in valid
            if record["renumbering_invariant"] is False
        ],
        "records": records,
        "seconds": seconds,
        "mean_ms_per_input": 1000.0 * seconds / len(rows),
        "claim_boundary": (
            "RotA is positive-only. This benchmark scores supplied axial "
            "locus endpoints and cumulene paths; it does not score negatives, "
            "axis handedness, configurational stability, or global chirality."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rota-path",
        type=Path,
        default=(
            ROOT / "Experiment" / "Stereo" / "Data" / "ChiralFinder-RotA" / "RotA.xlsx"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-renumbering", action="store_true")
    arguments = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    report = benchmark_rota_loci(
        arguments.rota_path,
        check_renumbering=not arguments.skip_renumbering,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
