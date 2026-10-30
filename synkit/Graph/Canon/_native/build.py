"""Explicit immutable CMake builds of the dependency-free C++17 graph core."""

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


def _verify(path):
    probe = subprocess.run(
        [sys.executable, "-c", "import ctypes,sys; assert ctypes.CDLL(sys.argv[1]).synkit_canon_abi()==1", str(path)],
        capture_output=True, text=True,
    )
    if probe.returncode:
        raise RuntimeError("Built graph core advertises an incompatible ABI")


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Native graph build failed:\n" + (result.stdout + result.stderr)[-6000:])


def build_native_canon(output_dir, *, compiler=None):
    """Build explicitly and return a content-addressed library with provenance.

    No native build or dependency download occurs during package import.
    CMake and a C++17 compiler are developer requirements, not runtime SDKs.
    """
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("CMake is required for an explicit native graph build")
    if compiler is not None:
        compiler = shutil.which(compiler)
        if compiler is None:
            raise RuntimeError("Requested C++ compiler is unavailable")
    source = Path(__file__).resolve().parents[1] / "cpp"
    inputs = [p for p in source.rglob("*") if p.is_file()
              and (p.name in {"CMakeLists.txt", "README.md"}
                   or p.relative_to(source).parts[0] in {"include", "src", "examples", "tests"})]
    detected_compiler = compiler or shutil.which(os.environ.get("CXX", "cl" if os.name == "nt" else "c++"))
    compiler_info = subprocess.run([detected_compiler, "--version"], capture_output=True, text=True) \
        if detected_compiler else None
    config = {
        "abi": 1, "builder_sha256": _digest(__file__),
        "sources": {str(p.relative_to(source)): _digest(p) for p in sorted(inputs)},
        "cmake": subprocess.check_output([cmake, "--version"], text=True),
        "compiler": compiler, "platform": sys.platform, "machine": platform.machine(),
        "detected_compiler": detected_compiler,
        "compiler_version": (compiler_info.stdout + compiler_info.stderr) if compiler_info else None,
        "environment": {k: os.environ.get(k) for k in ("CXX", "CXXFLAGS", "CMAKE_GENERATOR")},
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    library = output / ("libsynkit_canon_" + key[:20] + suffix)
    manifest = library.with_suffix(".json")
    if library.exists():
        if not manifest.exists():
            raise RuntimeError("Existing graph library has no build manifest")
        saved = json.loads(manifest.read_text())
        if saved.get("configuration") != config or saved.get("library_sha256") != _digest(library):
            raise RuntimeError("Native graph build integrity verification failed")
        _verify(library)
        return library
    with tempfile.TemporaryDirectory(prefix="synkit-canon-build-", dir=output) as name:
        work = Path(name)
        configure = [cmake, "-S", str(source), "-B", str(work), "-DCMAKE_BUILD_TYPE=Release"]
        if compiler:
            configure.append("-DCMAKE_CXX_COMPILER=" + compiler)
        _run(configure)
        _run([cmake, "--build", str(work), "--config", "Release", "--parallel", "2"])
        built = list(work.rglob("*synkit_canon" + suffix))
        if len(built) != 1:
            raise RuntimeError(f"Expected one native graph library, found {built}")
        _verify(built[0])
        cache = (work / "CMakeCache.txt").read_text()
        shutil.copy2(built[0], library)
        manifest.write_text(json.dumps({
            "configuration": config, "library_sha256": _digest(library),
            "cmake_cache": cache,
        }, indent=2) + "\n")
    return library
