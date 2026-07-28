"""Pinned 3D orientation gates for the native CIP benchmark."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from Experiment.Stereo.Perception.cip_labels import (
    _coordinate_atrop_descriptors,
    _coordinate_path_descriptors,
    _load_coordinate_molecules,
    _record_result,
)
from Experiment.Stereo.datasets import load_cip

CIP = Path("/tmp/cip-validation-suite-compounds.smi")
CIP_3D = Path("/tmp/cip-validation-suite-compounds-3d.sdf")


def test_pinned_coordinates_add_only_consensus_atrop_configurations() -> None:
    if not CIP.exists() or not CIP_3D.exists():
        return
    rows = [
        row
        for row in load_cip(CIP)
        if "AT" in str(row["stereo_units"]).split(",")
    ]
    coordinates = _load_coordinate_molecules(CIP_3D)
    results = []
    descriptor_counts = {}
    for row in rows:
        molecule = Chem.MolFromSmiles(str(row["SMILES"]))
        assert molecule is not None
        descriptors = _coordinate_atrop_descriptors(
            row,
            molecule,
            coordinates[str(row["ID"])],
        )
        descriptor_counts[str(row["ID"])] = len(descriptors)
        results.append(
            _record_result(
                row,
                molecule,
                additional_descriptors=descriptors,
            )
        )

    assert descriptor_counts == {
        "VS023": 0,
        "VS055": 0,
        "VS057": 0,
        "VS072": 1,
        "VS073": 1,
        "VS086": 1,
        "VS158": 1,
    }
    assert [item["id"] for item in results if item["exact"]] == [
        "VS072",
        "VS073",
        "VS086",
        "VS158",
    ]
    assert all(
        set(item["predicted"]) <= set(item["expected"])
        for item in results
    )


def test_pinned_coordinates_complete_helical_and_ct4_records() -> None:
    if not CIP.exists() or not CIP_3D.exists():
        return
    rows = [
        row
        for row in load_cip(CIP)
        if set(str(row["stereo_units"]).split(",")) & {"HE", "CT4"}
    ]
    coordinates = _load_coordinate_molecules(CIP_3D)
    results = []
    descriptor_counts = {}
    for row in rows:
        molecule = Chem.MolFromSmiles(str(row["SMILES"]))
        assert molecule is not None
        descriptors = _coordinate_path_descriptors(
            row,
            molecule,
            coordinates[str(row["ID"])],
        )
        descriptor_counts[str(row["ID"])] = len(descriptors)
        results.append(
            _record_result(
                row,
                molecule,
                additional_descriptors=descriptors,
            )
        )

    assert descriptor_counts == {
        "VS010": 1,
        "VS011": 1,
        "VS063": 1,
        "VS118": 1,
        "VS135": 1,
        "VS154": 2,
        "VS164": 1,
    }
    assert all(item["exact"] for item in results)
    assert all(
        set(item["predicted"]) == set(item["expected"])
        for item in results
    )


def test_pinned_coordinates_add_only_fully_witnessed_cumulene_axis() -> None:
    if not CIP.exists() or not CIP_3D.exists():
        return
    rows = [
        row
        for row in load_cip(CIP)
        if set(str(row["stereo_units"]).split(",")) & {"TH3", "TH5"}
    ]
    coordinates = _load_coordinate_molecules(CIP_3D)
    results = []
    for row in rows:
        molecule = Chem.MolFromSmiles(str(row["SMILES"]))
        assert molecule is not None
        descriptors = _coordinate_path_descriptors(
            row,
            molecule,
            coordinates[str(row["ID"])],
        )
        results.append(
            _record_result(
                row,
                molecule,
                additional_descriptors=descriptors,
            )
        )

    assert [item["id"] for item in results if item["exact"]] == ["VS144"]
    assert all(
        set(item["predicted"]) <= set(item["expected"])
        for item in results
    )
