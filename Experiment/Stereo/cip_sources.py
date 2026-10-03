"""Verify publisher-licensed CIP inputs without changing scientific fields."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

PUBLISHED = Path(__file__).resolve().parent / "Data/Revision/CIPPublished"
SMILES_PATH = PUBLISHED / "compounds.smi"
COORDINATE_PATH = PUBLISHED / "compounds_3d.sdf"
ARCHIVE_PATH = PUBLISHED / "ci8b00324_si_001.zip"
METADATA_PATH = PUBLISHED / "publisher_metadata.json"
SMILES_SHA256 = "df178635c00b6c41fad820d2609fc4ff18403c63dec4e5c3e1756a6db5858059"
ORIGINAL_SMILES_SHA256 = "4cc0c3e9b07b4503e50c44e5f7bb7d5cb79ec9dcf09145965b8da0c41345208d"
ORIGINAL_3D_SHA256 = "afc874e693a64211d4278952009f5b982fe66adcf712a65c252a86ae297e1ba6"
AMENDED_3D_SHA256 = "28a000b36506dabe5a45f6d7672451c61c6e34f3e0f027f2599000d850273325"
ACCEPTED_3D_SHA256 = frozenset((ORIGINAL_3D_SHA256, AMENDED_3D_SHA256))
ARCHIVE_SHA256 = "7b45eb4c3cb4a6a4138eeaa4264eca2781ed7dc62936ef78e5b9e062f141085b"
METADATA_SHA256 = "e4e12dc360439c9873c9fab01327f432d40c823c820c6b7c3feee51f436d26bc"
SUPPORT_DOI = "10.1021/acs.jcim.8b00324.s001"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"


def _digest(data: bytes) -> str:
    """Hash scientific inputs and the publisher evidence without text conversion."""
    return hashlib.sha256(data).hexdigest()


def publisher_inputs(archive: bytes, metadata: bytes) -> tuple[bytes, bytes]:
    """Bind the license to the exact archive and verify the two citation repairs."""
    if _digest(archive) != ARCHIVE_SHA256 or _digest(metadata) != METADATA_SHA256:
        raise ValueError("CIP publisher archive or metadata checksum mismatch.")
    manifest = json.loads(metadata)
    if manifest["doi"] != SUPPORT_DOI or manifest["license"]["url"] != LICENSE_URL:
        raise ValueError("CIP publisher source or license mismatch.")
    entry = manifest["files"][0]
    if entry["id"] != 12802589 or hashlib.md5(archive).hexdigest() != entry["supplied_md5"]:
        raise ValueError("CIP publisher file identity mismatch.")
    with ZipFile(io.BytesIO(archive)) as source:
        original = source.read("ValidationSuite-master/compounds.smi")
        coordinates = source.read("ValidationSuite-master/compounds_3d.sdf")
    if _digest(original) != ORIGINAL_SMILES_SHA256 or _digest(coordinates) != ORIGINAL_3D_SHA256:
        raise ValueError("CIP original scientific input checksum mismatch.")
    normalized = original.replace(b"P-93 5.3.5", b"P-93.5.3.5")
    before, after = original.splitlines(), normalized.splitlines()
    changes = [(old.split(b"\t"), new.split(b"\t")) for old, new in zip(before, after) if old != new]
    if len(before) != len(after) or len(changes) != 2:
        raise ValueError("CIP preparation must change exactly two citation fields.")
    for old, new in changes:
        if len(old) != len(new) or old[:3] + old[4:] != new[:3] + new[4:]:
            raise ValueError("CIP preparation changed a scientific input field.")
    if _digest(normalized) != SMILES_SHA256:
        raise ValueError("CIP prepared SMILES checksum mismatch.")
    return normalized, coordinates


def verify_published_inputs(*, prepare: bool = False) -> dict:
    """Check the provided bytes or reconstruct them from the unchanged archive."""
    smiles, coordinates = publisher_inputs(ARCHIVE_PATH.read_bytes(), METADATA_PATH.read_bytes())
    for path, expected in ((SMILES_PATH, smiles), (COORDINATE_PATH, coordinates)):
        if prepare:
            path.write_bytes(expected)
        elif path.read_bytes() != expected:
            raise ValueError("CIP prepared input differs from publisher derivation: " + str(path))
    return {"archive_sha256": ARCHIVE_SHA256, "metadata_sha256": METADATA_SHA256,
            "smiles_sha256": SMILES_SHA256, "coordinate_sha256": ORIGINAL_3D_SHA256,
            "bibliographic_fields_changed": 2, "scientific_fields_changed": 0,
            "source_doi": SUPPORT_DOI, "license": "CC BY-NC 4.0", "license_url": LICENSE_URL}


def source_metadata(path: Path | None, *, coordinates: bool = False) -> dict:
    """Describe actual input content; do not extend the original grant to amended coordinates."""
    digest = _digest(path.read_bytes()) if path is not None and path.is_file() else None
    licensed = digest == (ORIGINAL_3D_SHA256 if coordinates else SMILES_SHA256)
    if licensed:
        verify_published_inputs()
    return {"structures_vendored": licensed,
            "source": {"input_sha256": digest,
                       "identity": ("publisher_original_coordinates" if coordinates else "publisher_citation_repair")
                       if licensed else "external_or_unverified",
                       "source_doi": SUPPORT_DOI if licensed else None,
                       "license": "CC BY-NC 4.0" if licensed else None,
                       "license_url": LICENSE_URL if licensed else None,
                       "bibliographic_fields_changed": 0 if coordinates or not licensed else 2,
                       "scientific_fields_changed": 0 if licensed else None}}


def provenance_inputs(*paths: Path | None) -> list[Path]:
    """Include supplied inputs and the publisher evidence used by source metadata."""
    return [path for path in paths if path is not None] + [
        ARCHIVE_PATH, METADATA_PATH, SMILES_PATH, COORDINATE_PATH,
    ]


def main() -> int:
    """Verify the release data, optionally recreating its two extracted inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(verify_published_inputs(prepare=arguments.prepare), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
