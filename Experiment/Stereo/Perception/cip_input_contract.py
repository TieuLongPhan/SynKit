#!/usr/bin/env python3
"""Audit whether the pinned CIP labels are identifiable from supplied input."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from typing import Any

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.datasets import (  # noqa: E402
    CIP_3D_SHA256,
    CIP_SHA256,
    load_cip,
)

SCHEMA = "synkit.cip-input-contract-audit/2"


def _normalized_oriented_volume(
    molecule: Chem.Mol,
    atom_positions: tuple[int, int, int, int],
) -> float:
    conformer = molecule.GetConformer()
    a, b, c, d = (conformer.GetAtomPosition(index) for index in atom_positions)
    x = (b.x - a.x, b.y - a.y, b.z - a.z)
    y = (c.x - a.x, c.y - a.y, c.z - a.z)
    z = (d.x - a.x, d.y - a.y, d.z - a.z)
    triple_product = (
        x[0] * (y[1] * z[2] - y[2] * z[1])
        - x[1] * (y[0] * z[2] - y[2] * z[0])
        + x[2] * (y[0] * z[1] - y[1] * z[0])
    )
    denominator = math.sqrt(
        sum(value * value for value in x)
        * sum(value * value for value in y)
        * sum(value * value for value in z)
    )
    return triple_product / denominator if denominator else 0.0


def _orientation_witness(
    molecules: dict[str, Chem.Mol],
    record_ids: list[str],
) -> dict[str, Any]:
    first = molecules[record_ids[0]]
    atom_positions = next(
        (
            quartet
            for quartet in combinations(range(first.GetNumAtoms()), 4)
            if abs(_normalized_oriented_volume(first, quartet)) > 1.0e-5
        ),
        None,
    )
    if atom_positions is None:
        raise ValueError(f"No non-coplanar 3D witness exists for {record_ids[0]}.")
    orientations = []
    for record_id in record_ids:
        molecule = molecules[record_id]
        if molecule.GetNumAtoms() <= max(atom_positions):
            raise ValueError(f"3D record {record_id} lacks the witness atom positions.")
        value = _normalized_oriented_volume(molecule, atom_positions)
        if abs(value) <= 1.0e-5:
            raise ValueError(
                f"3D record {record_id} has a degenerate orientation witness."
            )
        orientations.append(
            {
                "record_id": record_id,
                "orientation": "positive" if value > 0 else "negative",
            }
        )
    return {
        "record_ids": record_ids,
        "atom_positions_one_based": [position + 1 for position in atom_positions],
        "orientations": orientations,
        "distinguished": len({item["orientation"] for item in orientations})
        == len(orientations),
        "invariance": "translation_rotation_and_positive_scale",
        "reflection_sensitive": True,
    }


def _audit_coordinate_extension(
    path: Path,
    conflicts: list[dict[str, Any]],
    expected_ids: set[str],
) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CIP_3D_SHA256:
        raise ValueError(f"Unexpected CIP 3D Validation Suite SHA-256: {digest}")
    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=False,
        sanitize=False,
    )
    molecules: dict[str, Chem.Mol] = {}
    for position, molecule in enumerate(supplier, start=1):
        if molecule is None:
            raise ValueError(f"Cannot parse CIP 3D record {position}.")
        if not molecule.HasProp("STRUCTURE_ID"):
            raise ValueError(f"CIP 3D record {position} has no STRUCTURE_ID.")
        record_id = molecule.GetProp("STRUCTURE_ID")
        if record_id in molecules:
            raise ValueError(f"Duplicate CIP 3D record {record_id}.")
        molecules[record_id] = molecule
    if set(molecules) != expected_ids:
        missing = sorted(expected_ids - set(molecules))
        additional = sorted(set(molecules) - expected_ids)
        raise ValueError(
            "CIP 3D IDs do not match compounds.smi: "
            f"missing={missing}, additional={additional}."
        )
    witnesses = [
        _orientation_witness(molecules, conflict["record_ids"])
        for conflict in conflicts
    ]
    return {
        "provided": True,
        "audited_sha256": digest,
        "records": len(molecules),
        "structures_vendored": False,
        "conflict_orientation_witnesses": witnesses,
        "all_smiles_conflicts_distinguished": all(
            witness["distinguished"] for witness in witnesses
        ),
        "claim_boundary": (
            "Signed coordinate volumes prove that the external 3D input "
            "distinguishes the conflicting configured structures. They do not "
            "perform helical perception or CIP ranking."
        ),
    }


def audit_cip_input_contract(
    path: Path,
    *,
    coordinate_path: Path | None = None,
) -> dict[str, Any]:
    """Prove the deterministic ceiling imposed by the supplied SMILES."""
    rows = load_cip(path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["SMILES"])].append(row)

    conflicts = []
    maximum_exact = 0
    for records in groups.values():
        label_sets = Counter(tuple(record["recommended_labels"]) for record in records)
        maximum_exact += max(label_sets.values())
        if len(label_sets) < 2:
            continue
        conflicts.append(
            {
                "record_ids": sorted(str(record["ID"]) for record in records),
                "stereo_units": sorted(
                    {
                        unit.strip()
                        for record in records
                        for unit in str(record["stereo_units"]).split(",")
                        if unit.strip()
                    }
                ),
                "expected_label_sets": [
                    {
                        "labels": list(labels),
                        "record_ids": sorted(
                            str(record["ID"])
                            for record in records
                            if tuple(record["recommended_labels"]) == labels
                        ),
                    }
                    for labels in sorted(label_sets)
                ],
                "reason": (
                    "Byte-identical molecular input has mutually exclusive "
                    "configured label sets."
                ),
            }
        )
    coordinate_extension = (
        {
            "provided": False,
            "required_sha256": CIP_3D_SHA256,
            "structures_vendored": False,
        }
        if coordinate_path is None
        else _audit_coordinate_extension(
            coordinate_path,
            conflicts,
            {str(row["ID"]) for row in rows},
        )
    )
    return {
        "schema": SCHEMA,
        "dataset": {
            "records": len(rows),
            "audited_sha256": CIP_SHA256,
            "structures_vendored": False,
        },
        "deterministic_input_contract": "supplied_smiles_only",
        "unique_input_strings": len(groups),
        "conflicting_input_groups": conflicts,
        "conflicting_records": sum(
            len(conflict["record_ids"]) for conflict in conflicts
        ),
        "maximum_exact_records_from_supplied_smiles": maximum_exact,
        "complete_300_record_assignment_identifiable": maximum_exact == len(rows),
        "coordinate_contract_extension": coordinate_extension,
        "complete_300_record_input_identifiable_with_coordinates": (
            maximum_exact == len(rows)
            or coordinate_extension.get(
                "all_smiles_conflicts_distinguished",
                False,
            )
        ),
        "required_contract_extension": (
            "A declared helical/axial orientation sidecar or coordinates that "
            "distinguish every configured input with identical constitution."
        ),
        "claim_boundary": (
            "This is an information-theoretic input audit. It does not assess "
            "the correctness or completeness of the CIP ranking algorithm."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cip-path", required=True, type=Path)
    parser.add_argument("--cip-3d-path", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit_cip_input_contract(
        arguments.cip_path,
        coordinate_path=arguments.cip_3d_path,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
