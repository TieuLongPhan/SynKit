"""Native build artifacts must be loadable, verified and suitable for their host."""

import ctypes
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from synkit.Chem.Mapper.exact.native import build


def test_compiled_native_artifact_exports_supported_abi(native_library):
    suffix = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    assert native_library.suffix == suffix
    assert ctypes.CDLL(str(native_library)).synkit_distance_abi() == 3


@pytest.mark.parametrize("record_matching_hash", [False, True])
def test_broken_cached_native_artifact_is_rejected(native_library, tmp_path, record_matching_hash):
    library = tmp_path / native_library.name
    manifest = library.with_suffix(".json")
    shutil.copyfile(native_library.with_suffix(".json"), manifest)
    library.write_bytes(b"invalid native library")
    if record_matching_hash:
        recorded = json.loads(manifest.read_text())
        recorded["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(recorded))
    message = "cannot be loaded" if record_matching_hash else "integrity verification"
    with pytest.raises(RuntimeError, match=message):
        build.build_native(tmp_path)


@pytest.mark.parametrize("host,suffix", [("win32", ".dll"), ("darwin", ".dylib"), ("linux", ".so")])
def test_build_uses_host_library_format_and_verifies_before_publication(monkeypatch, tmp_path, host, suffix):
    verified = []
    commands = []
    monkeypatch.setattr(build, "sys", SimpleNamespace(platform=host, executable=sys.executable))
    monkeypatch.setattr(build.shutil, "which", lambda _compiler: "/compiler/g++")

    def run(command, **kwargs):
        commands.append(command)
        if "--version" in command:
            return SimpleNamespace(stdout="test C++17 compiler", stderr="", returncode=0)
        candidate = Path(command[command.index("-o") + 1])
        assert candidate.suffix == suffix
        if host == "win32":
            assert "-static" in command and "-Wl,--export-all-symbols" in command
            assert "-fPIC" not in command
        elif host == "darwin":
            assert "-dynamiclib" in command
        candidate.write_bytes(b"test artifact")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    def verify(candidate):
        assert candidate.exists()
        assert not list(tmp_path.glob(f"libsynkit_distance_*{suffix}"))
        verified.append(candidate)

    monkeypatch.setattr(build.subprocess, "run", run)
    monkeypatch.setattr(build, "_verify_library", verify)
    library = build.build_native(tmp_path)
    assert library.suffix == suffix and library.read_bytes() == b"test artifact"
    assert len(verified) == 1 and len(commands) == 2
    assert json.loads(library.with_suffix(".json").read_text())["platform"] == host


def test_unloadable_new_artifact_is_not_published(monkeypatch, tmp_path):
    monkeypatch.setattr(build.shutil, "which", lambda _compiler: "/compiler/g++")

    def run(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(stdout="test C++17 compiler", stderr="", returncode=0)
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"broken artifact")
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="dependent DLL missing", returncode=1)

    monkeypatch.setattr(build.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="dependent DLL missing"):
        build.build_native(tmp_path)
    assert list(tmp_path.iterdir()) == []
