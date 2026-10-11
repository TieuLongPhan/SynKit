"""Integrity and source boundaries for the provided licensed CIP benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from Experiment.Stereo import cip_sources
from Experiment.Stereo.datasets import load_cip
from Experiment.Stereo.Perception.cip_input_contract import audit_cip_input_contract
from Experiment.Stereo.Perception.cip_labels import _load_coordinate_molecules


def test_publisher_derivation_preserves_all_scientific_fields() -> None:
    report = cip_sources.verify_published_inputs()
    with ZipFile(cip_sources.ARCHIVE_PATH) as archive:
        original = archive.read("ValidationSuite-master/compounds.smi").splitlines()
    prepared = cip_sources.SMILES_PATH.read_bytes().splitlines()
    changed = []
    for before, after in zip(original, prepared, strict=True):
        old, new = before.split(b"\t"), after.split(b"\t")
        assert old[:3] + old[4:] == new[:3] + new[4:]
        if before != after:
            changed.append(old[1].decode())
    assert sorted(changed) == ["VS297", "VS298"]
    assert len(load_cip(cip_sources.SMILES_PATH)) == 300
    assert report["scientific_fields_changed"] == 0
    assert report["bibliographic_fields_changed"] == 2


@pytest.mark.parametrize("component", ["archive", "metadata"])
def test_changed_publisher_evidence_is_rejected(component: str) -> None:
    archive = cip_sources.ARCHIVE_PATH.read_bytes()
    metadata = cip_sources.METADATA_PATH.read_bytes()
    if component == "archive":
        archive += b"changed"
    else:
        decoded = json.loads(metadata)
        decoded["license"]["name"] = "MIT"
        metadata = json.dumps(decoded).encode()
    with pytest.raises(ValueError, match="checksum mismatch"):
        cip_sources.publisher_inputs(archive, metadata)


def test_original_coordinates_cover_all_300_inputs_and_resolve_helical_conflict() -> None:
    molecules = _load_coordinate_molecules(cip_sources.COORDINATE_PATH)
    rows = load_cip(cip_sources.SMILES_PATH)
    assert set(molecules) == {row["ID"] for row in rows}
    report = audit_cip_input_contract(
        cip_sources.SMILES_PATH, coordinate_path=cip_sources.COORDINATE_PATH,
    )
    assert report["dataset"]["structures_vendored"]
    assert report["maximum_exact_records_from_supplied_smiles"] == 299
    coordinates = report["coordinate_contract_extension"]
    assert coordinates["audited_sha256"] == cip_sources.ORIGINAL_3D_SHA256
    assert coordinates["structures_vendored"]
    assert coordinates["source"]["license"] == "CC BY-NC 4.0"
    assert coordinates["all_smiles_conflicts_distinguished"]


def test_modified_coordinate_input_gets_no_license_and_is_not_evaluated(tmp_path: Path) -> None:
    path = tmp_path / "compounds_3d.sdf"
    path.write_bytes(cip_sources.COORDINATE_PATH.read_bytes() + b"modified\n")
    metadata = cip_sources.source_metadata(path, coordinates=True)
    assert not metadata["structures_vendored"]
    assert metadata["source"]["license"] is None
    assert metadata["source"]["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Unexpected CIP 3D"):
        _load_coordinate_molecules(path)


def test_unprovided_coordinates_are_distinct_from_provided_smiles() -> None:
    report = audit_cip_input_contract(cip_sources.SMILES_PATH)
    assert report["dataset"]["structures_vendored"]
    assert not report["coordinate_contract_extension"]["structures_vendored"]
    assert not report["coordinate_contract_extension"]["provided"]
    assert not report["complete_300_record_input_identifiable_with_coordinates"]


def test_publisher_evidence_is_included_in_input_capture() -> None:
    paths = cip_sources.provenance_inputs(cip_sources.SMILES_PATH, cip_sources.COORDINATE_PATH)
    assert set(paths) == {
        cip_sources.SMILES_PATH, cip_sources.COORDINATE_PATH,
        cip_sources.ARCHIVE_PATH, cip_sources.METADATA_PATH,
    }
