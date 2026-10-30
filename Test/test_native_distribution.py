"""Integrity of the native libraries selected from an installed wheel."""

import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from synkit import _native


@pytest.fixture
def installed_library(tmp_path, monkeypatch):
    package = tmp_path / "synkit"
    path = package / "CRN/lib/libsynkit_crn.so"
    path.parent.mkdir(parents=True)
    # RECORD describes the final bytes, including changes made by wheel repair.
    path.write_bytes(b"native library after wheel repair")
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
        .decode()
        .rstrip("=")
    )

    class RecordPath(str):
        hash = SimpleNamespace(mode="sha256", value=digest)

    entry = RecordPath("synkit/CRN/lib/libsynkit_crn.so")
    installed = SimpleNamespace(
        files=[entry], locate_file=lambda name: tmp_path / str(name)
    )
    monkeypatch.setattr(_native, "__file__", str(package / "_native.py"))
    monkeypatch.setattr(_native, "distribution", lambda name: installed)
    _native._wheel_record_digest.cache_clear()
    yield path, entry, installed
    _native._wheel_record_digest.cache_clear()


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
