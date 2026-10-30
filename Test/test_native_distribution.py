"""Integrity of the native libraries selected from an installed wheel."""

import hashlib
from pathlib import Path

import pytest

from synkit import _native


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
