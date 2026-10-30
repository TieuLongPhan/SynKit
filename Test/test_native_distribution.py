"""Integrity of the native libraries in installed wheel and Conda packages."""

import hashlib
import json
from pathlib import Path

import pytest

from synkit import _native
from synkit.version import __version__


@pytest.fixture
def installed_library(tmp_path, native_wheel_record):
    package = tmp_path / "synkit"
    path = package / "CRN/lib/libsynkit_crn.so"
    path.parent.mkdir(parents=True)
    # RECORD describes the final bytes, including changes made by wheel repair.
    path.write_bytes(b"native library after wheel repair")
    entry, installed = native_wheel_record(path, package)
    return path, entry, installed


def test_repaired_library_matches_installed_record(installed_library):
    path, _, _ = installed_library
    assert (
        _native.verify_wheel_library(path)
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_modified_library_is_rejected(installed_library):
    path, _, _ = installed_library
    path.write_bytes(path.read_bytes() + b"unexpected bytes")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _native.verify_wheel_library(path)


def test_missing_record_hash_is_rejected(installed_library):
    path, entry, _ = installed_library
    entry.hash = None
    with pytest.raises(RuntimeError, match="no installed wheel SHA-256 record"):
        _native.verify_wheel_library(path)


def test_another_installation_cannot_supply_the_record(installed_library):
    path, _, installed = installed_library
    installed.locate_file = lambda name: Path("/other-installation") / str(name)
    with pytest.raises(RuntimeError, match="no installed wheel SHA-256 record"):
        _native.verify_wheel_library(path)


@pytest.fixture
def installed_conda_library(installed_library):
    path, _, _ = installed_library
    prefix = Path(_native.__file__).resolve().parent.parent
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b"native library after Conda prefix relocation")
    entry = {
        "_path": path.relative_to(prefix).as_posix(),
        "sha256": original,
        "sha256_in_prefix": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    record = {"name": "synkit", "version": __version__, "paths_data": {"paths": [entry]}}
    metadata = prefix / "conda-meta" / f"synkit-{__version__}-0.json"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps(record))
    return path, metadata, record, entry


def test_conda_relocated_library_uses_final_installed_hash(installed_conda_library):
    path, _, _, entry = installed_conda_library
    assert entry["sha256"] != entry["sha256_in_prefix"]
    assert _native.verify_wheel_library(path) == entry["sha256_in_prefix"]


def test_conda_unchanged_library_can_use_source_hash(installed_conda_library):
    path, metadata, record, entry = installed_conda_library
    entry["sha256"] = entry.pop("sha256_in_prefix")
    metadata.write_text(json.dumps(record))
    assert _native.verify_wheel_library(path) == entry["sha256"]


def test_modified_conda_library_is_rejected(installed_conda_library):
    path, _, _, _ = installed_conda_library
    path.write_bytes(path.read_bytes() + b"unexpected bytes")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _native.verify_wheel_library(path)


def test_conda_record_requires_exact_owned_path(installed_conda_library):
    path, metadata, record, entry = installed_conda_library
    entry["_path"] += ".another-library"
    metadata.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _native.verify_wheel_library(path)


def test_conda_record_for_another_package_is_not_used(installed_conda_library):
    path, metadata, record, _ = installed_conda_library
    record["name"] = "synkit-another-package"
    metadata.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _native.verify_wheel_library(path)


@pytest.mark.parametrize("digest", [None, "g" * 64, "0" * 63, 42])
def test_conda_record_requires_valid_sha256(installed_conda_library, digest):
    path, metadata, record, entry = installed_conda_library
    entry.pop("sha256")
    entry["sha256_in_prefix"] = digest
    metadata.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="no installed Conda SHA-256 record"):
        _native.verify_wheel_library(path)


def test_malformed_conda_metadata_is_rejected(installed_conda_library):
    path, metadata, _, _ = installed_conda_library
    metadata.write_text("not valid JSON")
    with pytest.raises(RuntimeError, match="Cannot read installed SynKit Conda metadata"):
        _native.verify_wheel_library(path)
