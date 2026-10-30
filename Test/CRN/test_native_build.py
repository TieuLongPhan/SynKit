"""Explicit CRN source builds keep immutable, verified cache artifacts."""

import hashlib
import json

import pytest

from synkit.CRN import _build, build_native


def test_conda_windows_boost_header_location_is_discovered(tmp_path, monkeypatch):
    include = tmp_path / "Library/include"
    header = include / "boost/multiprecision/cpp_int.hpp"
    header.parent.mkdir(parents=True)
    header.write_text("// Discovery fixture: no compiler is invoked.\n")
    monkeypatch.delenv("BOOST_INCLUDE_DIR", raising=False)
    monkeypatch.setattr(_build.sys, "prefix", str(tmp_path))
    assert _build.boost_include() == include


def test_structural_build_reuses_verified_cache_and_rejects_changed_bytes(tmp_path):
    library = build_native(tmp_path, with_semiflows=False)
    manifest = json.loads(library.with_suffix(".json").read_text())
    assert manifest["semiflows_abi"] is None
    assert "structural_sets" in manifest["capabilities"]
    assert "exact_semiflows" not in manifest["capabilities"]
    assert manifest["configuration"]["sources"]
    assert manifest["configuration"]["boost_headers_sha256"] is None
    before = library.stat().st_mtime_ns
    assert build_native(tmp_path, with_semiflows=False) == library
    assert library.stat().st_mtime_ns == before
    library.write_bytes(library.read_bytes() + b"changed cache bytes")
    with pytest.raises(RuntimeError, match="checksum or configuration mismatch"):
        build_native(tmp_path, with_semiflows=False)


def test_cached_library_requires_loadable_abi_even_with_matching_checksum(tmp_path):
    library = build_native(tmp_path, with_semiflows=False)
    manifest_path = library.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    library.write_bytes(b"This is not a native library")
    manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="incompatible ABI"):
        build_native(tmp_path, with_semiflows=False)
