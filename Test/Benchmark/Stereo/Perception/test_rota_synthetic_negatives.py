"""Specificity gates for symmetry-certified synthetic RotA negatives."""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem

from Experiment.Stereo.Perception.rota_synthetic_negatives import (
    CASES,
    SCHEMA,
    build_report,
)
from synkit.Chem.Molecule.stereo_perception import (
    StereoElementType,
    detect_potential_stereo_elements,
)

ROOT = Path(__file__).resolve().parents[4]
FROZEN = (
    ROOT / "Experiment" / "Stereo" / "Data" / "RotA-Synthetic" / "negative_axes.json"
)
_AXIS_TYPES = {
    StereoElementType.ATROP_AXIS,
    StereoElementType.CUMULENE_AXIS,
    StereoElementType.EXTENDED_CIS_TRANS,
}


def _normalized_path(path: tuple[int, ...]) -> tuple[int, ...]:
    return min(path, tuple(reversed(path)))


def _axes(molecule: Chem.Mol) -> set[tuple[str, tuple[int, ...], str]]:
    return {
        (
            element.element_type.value,
            _normalized_path(tuple(element.support.path)),
            element.carrier_status.value,
        )
        for element in detect_potential_stereo_elements(molecule)
        if element.element_type in _AXIS_TYPES
    }


def test_frozen_synthetic_negative_report_is_exactly_reproducible() -> None:
    observed = build_report()
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))

    assert observed == frozen
    assert observed["schema"] == SCHEMA
    assert observed["summary"] == {
        "records": 24,
        "axis_fixed_automorphism_negatives": 13,
        "no_supported_axis_negatives": 11,
        "confirmed_axis_false_positives": 0,
    }


def test_every_candidate_negative_contains_an_axis_fixed_witness() -> None:
    report = build_report()

    for record in report["records"]:
        if record["proof_kind"] != "axis_fixed_automorphism":
            assert record["axes"] == []
            continue
        for axis in record["axes"]:
            path = set(axis["path"])
            witnessed_frames = [
                (frame, witness)
                for frame, witness in zip(
                    axis["terminal_frames"],
                    axis["symmetry_witnesses"],
                )
                if witness is not None
            ]
            witnesses = [witness for _frame, witness in witnessed_frames]
            assert witnesses
            for frame, witness in witnessed_frames:
                mapping = dict(witness)
                assert all(mapping[atom] == atom for atom in path)
                assert mapping[frame[0]] == frame[1]
                assert len(set(mapping.values())) == len(mapping)


def test_synthetic_negative_verdict_is_atom_renumbering_invariant() -> None:
    for fixture_id, smiles, _proof_kind in CASES:
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None, fixture_id
        order = tuple(reversed(range(molecule.GetNumAtoms())))
        old_to_new = {old: new for new, old in enumerate(order)}
        renumbered = Chem.RenumberAtoms(molecule, order)
        transported = {
            (
                element_type,
                _normalized_path(tuple(old_to_new[atom] for atom in path)),
                status,
            )
            for element_type, path, status in _axes(molecule)
        }

        assert transported == _axes(renumbered), fixture_id
