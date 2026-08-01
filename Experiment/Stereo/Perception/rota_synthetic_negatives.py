#!/usr/bin/env python3
"""Build exact-symmetry negative controls for RotA-style axis perception.

These fixtures are synthetic constitutional controls. A negative label is
accepted only when either no supported axis topology exists or an explicit
graph automorphism fixes the ordered axis and exchanges a terminal ligand
pair. The suite does not claim a rotational barrier or experimental stability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synkit.Chem.Molecule._stereo_axis_evidence import (  # noqa: E402
    StereoCarrierStatus,
    axis_terminal_symmetry_witnesses,
)
from synkit.Chem.Molecule.stereo_perception import (  # noqa: E402
    StereoElementType,
    _constitutional_graph,
    detect_potential_stereo_elements,
)

SCHEMA = "synkit.rota-synthetic-negative-controls/1"
_AXIS_TYPES = frozenset(
    {
        StereoElementType.ATROP_AXIS,
        StereoElementType.CUMULENE_AXIS,
        StereoElementType.EXTENDED_CIS_TRANS,
    }
)

CASES = (
    ("sym-biaryl-01", "c1ccccc1-c2ccccc2", "axis_fixed_automorphism"),
    ("sym-biaryl-02", "Fc1ccc(-c2ccc(F)cc2)cc1", "axis_fixed_automorphism"),
    ("sym-biaryl-03", "Clc1ccc(-c2ccc(Cl)cc2)cc1", "axis_fixed_automorphism"),
    ("sym-biaryl-04", "Cc1ccc(-c2ccc(C)cc2)cc1", "axis_fixed_automorphism"),
    ("sym-biaryl-05", "COc1ccc(-c2ccc(OC)cc2)cc1", "axis_fixed_automorphism"),
    ("sym-biaryl-06", "c1ccncc1-c2ccncc2", "axis_fixed_automorphism"),
    (
        "sym-biaryl-07",
        "Cc1c(C)cccc1-c2c(C)cccc2C",
        "axis_fixed_automorphism",
    ),
    ("sym-cumulene-01", "FC(F)=C=C(Cl)Br", "axis_fixed_automorphism"),
    ("sym-cumulene-02", "ClC(Cl)=C=C(F)Br", "axis_fixed_automorphism"),
    ("sym-cumulene-03", "BrC(Br)=C=C(F)Cl", "axis_fixed_automorphism"),
    ("sym-extended-01", "FC(F)=C=C=C(Cl)Br", "axis_fixed_automorphism"),
    ("sym-extended-02", "ClC(Cl)=C=C=C(F)Br", "axis_fixed_automorphism"),
    ("sym-extended-03", "BrC(Br)=C=C=C(F)Cl", "axis_fixed_automorphism"),
    ("no-axis-01", "C=C=C", "no_supported_axis"),
    ("no-axis-02", "F/C=C/F", "no_supported_axis"),
    ("no-axis-03", "CCCC", "no_supported_axis"),
    ("no-axis-04", "c1ccccc1", "no_supported_axis"),
    ("no-axis-05", "c1ccc2ccccc2c1", "no_supported_axis"),
    ("no-axis-06", "C1CCC2(CC1)CCCCC2", "no_supported_axis"),
    ("no-axis-07", "C1CCC2CCCCC2C1", "no_supported_axis"),
    ("no-axis-08", "FC#CCl", "no_supported_axis"),
    ("no-axis-09", "C1CCCCC1", "no_supported_axis"),
    ("no-axis-10", "c1ccccc1CCc2ccccc2", "no_supported_axis"),
    ("no-axis-11", "CC#CC", "no_supported_axis"),
)


def _record(
    fixture_id: str,
    smiles: str,
    proof_kind: str,
) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Cannot parse synthetic RotA control {fixture_id}.")
    axes = tuple(
        element
        for element in detect_potential_stereo_elements(molecule)
        if element.element_type in _AXIS_TYPES
    )
    axis_records = []
    for element in axes:
        support = element.support
        graph = _constitutional_graph(molecule, support.path[0])
        witnesses = axis_terminal_symmetry_witnesses(graph, support)
        axis_records.append(
            {
                "identifier": element.identifier,
                "element_type": element.element_type.value,
                "path": list(support.path),
                "terminal_frames": [list(frame) for frame in support.terminal_frames],
                "carrier_status": element.carrier_status.value,
                "carrier_reason": element.carrier_reason,
                "symmetry_witnesses": [
                    (None if witness is None else [list(pair) for pair in witness])
                    for witness in witnesses
                ],
            }
        )
    if proof_kind == "axis_fixed_automorphism":
        if not axes:
            raise AssertionError(f"{fixture_id} has no broad axis candidate.")
        if any(
            element.carrier_status is not StereoCarrierStatus.SYMMETRY_RELATED
            for element in axes
        ):
            raise AssertionError(f"{fixture_id} contains a confirmed axis.")
        if any(
            not any(witness is not None for witness in item["symmetry_witnesses"])
            for item in axis_records
        ):
            raise AssertionError(f"{fixture_id} lacks an exact symmetry witness.")
    elif proof_kind == "no_supported_axis":
        if axes:
            raise AssertionError(f"{fixture_id} unexpectedly contains an axis.")
    else:
        raise ValueError(f"Unknown synthetic negative proof: {proof_kind}.")
    return {
        "fixture_id": fixture_id,
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        ),
        "expected": "constitutionally_nonstereogenic_axis",
        "proof_kind": proof_kind,
        "axes": axis_records,
    }


def build_report() -> dict[str, Any]:
    """Return deterministic synthetic negatives and their exact proofs."""
    records = [_record(*case) for case in CASES]
    automorphism = sum(
        record["proof_kind"] == "axis_fixed_automorphism" for record in records
    )
    return {
        "schema": SCHEMA,
        "records": records,
        "summary": {
            "records": len(records),
            "axis_fixed_automorphism_negatives": automorphism,
            "no_supported_axis_negatives": len(records) - automorphism,
            "confirmed_axis_false_positives": sum(
                any(
                    axis["carrier_status"] == StereoCarrierStatus.CONFIRMED.value
                    for axis in record["axes"]
                )
                for record in records
            ),
        },
        "claim_boundary": (
            "Synthetic controls validate constitutional axis specificity and "
            "invariance. They do not validate rotational barriers, isolation "
            "timescales, experimental stability, or configured handedness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = build_report()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
