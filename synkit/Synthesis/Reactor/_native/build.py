"""Content-addressed, explicit builds of the dependency-free C++ kernel."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _build_lock(directory, key):
    """Serialize publication so a second builder cannot truncate a loaded DLL."""
    with (directory / (".partial-" + key[:20] + ".lock")).open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            if not stream.read(1):
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def build_native_reactor(output_dir, *, compiler=None):
    """Build C ABI 1 with CMake. Importing the package never builds code."""
    cmake = shutil.which("cmake")
    compiler = shutil.which(compiler or os.environ.get("CXX", "c++"))
    if cmake is None or compiler is None:
        raise RuntimeError("An explicit reactor build requires CMake and a C++17 compiler.")
    source = Path(__file__).resolve().parents[1] / "cpp"
    files = sorted(p for p in source.rglob("*") if p.is_file())
    configuration = {
        "abi": 1,
        "sources": {str(p.relative_to(source)): _digest(p) for p in files},
        "builder": _digest(__file__),
        "compiler": compiler,
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True),
        "cmake_version": subprocess.check_output([cmake, "--version"], text=True),
        "platform": sys.platform,
        "machine": platform.machine(),
        "environment": {k: os.environ.get(k) for k in ("CXXFLAGS", "LDFLAGS", "CMAKE_GENERATOR")},
    }
    key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    library = directory / ("libsynkit_partial_" + key[:20] + suffix)
    manifest = library.with_suffix(".json")
    with _build_lock(directory, key):
        if not library.exists():
            with tempfile.TemporaryDirectory(prefix="partial-build-", dir=directory) as folder:
                snapshot = Path(folder) / "source"
                shutil.copytree(source, snapshot)
                copied = {str(p.relative_to(snapshot)): _digest(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
                if copied != configuration["sources"]:
                    raise RuntimeError("Native sources changed while preparing the build; retry.")
                configure = [
                    cmake,
                    "-S",
                    str(snapshot),
                    "-B",
                    str(Path(folder) / "cmake"),
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_CXX_COMPILER=" + compiler,
                ]
                for command in (
                    configure,
                    [cmake, "--build", str(Path(folder) / "cmake"), "--config", "Release", "--parallel", "2"],
                ):
                    result = subprocess.run(command, capture_output=True, text=True)
                    if result.returncode:
                        raise RuntimeError("Native reactor build failed:\n" + (result.stdout + result.stderr)[-8000:])
                built = list(Path(folder).rglob("*synkit_partial" + suffix))
                if len(built) != 1:
                    raise RuntimeError(f"Expected one reactor library, found {built}")
                staged_library = Path(folder) / library.name
                shutil.copy2(built[0], staged_library)
                staged_manifest = Path(folder) / manifest.name
                staged_manifest.write_text(
                    json.dumps({"configuration": configuration, "library_sha256": _digest(staged_library)}, indent=2)
                    + "\n"
                )
                os.replace(staged_library, library)
                os.replace(staged_manifest, manifest)
        saved = json.loads(manifest.read_text())
        if saved.get("configuration") != configuration or saved.get("library_sha256") != _digest(library):
            raise RuntimeError("Native reactor build integrity verification failed.")
        if ctypes.CDLL(str(library)).skp_abi() != 1:
            raise RuntimeError("Unsupported native reactor ABI.")
    return library
