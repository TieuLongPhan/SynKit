"""Explicit immutable CMake builds of the optional CRN backend."""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from ._build import CAPABILITIES, boost_include


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _boost_digest(include):
    if include is None:
        return None
    digest = hashlib.sha256()
    for path in sorted((include / "boost").rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(include)).replace(os.sep, "/").encode())
            digest.update(b"\0")
            digest.update(bytes.fromhex(_digest(path)))
    return digest.hexdigest()


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Native CRN build failed:\n" + (result.stdout + result.stderr)[-6000:])
    return result.stdout


def _verify(library, exact):
    script = """
import ctypes, sys
library = ctypes.CDLL(sys.argv[1])
if library.synkit_crn_abi() != 1 or library.synkit_crn_structural_abi() != 2:
    raise RuntimeError('Unsupported CRN structural ABI')
for name in ('semiflows', 'algebra', 'pathway', 'symmetry'):
    function = getattr(library, 'synkit_crn_' + name + '_abi', None)
    if sys.argv[2] == '1':
        if function is None or function() != 1:
            raise RuntimeError('Unsupported CRN exact ABI: ' + name)
    elif function is not None:
        raise RuntimeError('Unexpected exact ABI in structural build: ' + name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(library), str(int(exact))],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError("Native CRN library has an incompatible ABI:\n" + result.stderr[-3000:])


def _sources(source):
    return {
        str(path.relative_to(source)): path.read_bytes()
        for path in sorted(source.rglob("*"))
        if path.is_file() and (path.suffix in {".cpp", ".h", ".hpp", ".cmake"} or path.name == "CMakeLists.txt")
    }


def _configure(work, source, compiler, boost):
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("CMake is required for an explicit native CRN build")
    command = [cmake, "-S", str(source), "-B", str(work), "-DCMAKE_BUILD_TYPE=Release",
               "-DSYNKIT_CRN_EXACT=" + ("ON" if boost is not None else "OFF")]
    if compiler is not None:
        resolved = shutil.which(compiler)
        if resolved is None:
            raise RuntimeError(f"C++17 compiler unavailable: {compiler}")
        command.append("-DCMAKE_CXX_COMPILER=" + resolved)
    if boost is not None:
        command.append("-DSYNKIT_CRN_BOOST_INCLUDE_DIR=" + str(boost))
    _run(command)
    prefixes = ("CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_SHARED_LINKER_FLAGS", "CMAKE_GENERATOR",
                "CMAKE_BUILD_TYPE", "CMAKE_OSX_", "CMAKE_VS_PLATFORM_TOOLSET")
    cache = {}
    for line in (work / "CMakeCache.txt").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key.startswith(prefixes):
            cache[key] = value
    return {"cmake": _run([cmake, "--version"]), "cmake_path": cmake, "cache": cache,
            "compiler": json.loads((work / "crn-build.json").read_text())}


def _reuse(library, manifest, configuration, exact):
    try:
        saved = json.loads(manifest.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Existing immutable CRN library has no valid manifest") from exc
    expected_capabilities = list(CAPABILITIES if exact else CAPABILITIES[:4])
    if (saved.get("configuration") != configuration or saved.get("library_sha256") != _digest(library)
            or saved.get("abi") != 1 or saved.get("semiflows_abi") != (1 if exact else None)
            or saved.get("capabilities") != expected_capabilities):
        raise RuntimeError("Existing immutable CRN library checksum or configuration mismatch")
    _verify(library, exact)
    return library


def _publish(work, output, configuration, boost):
    key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    library = output / ("libsynkit_crn_" + key[:16] + suffix)
    manifest = library.with_suffix(".json")
    if library.exists():
        return _reuse(library, manifest, configuration, boost is not None)
    _run([configuration["toolchain"]["cmake_path"], "--build", str(work), "--config", "Release", "--parallel", "2"])
    candidates = list(work.rglob("*synkit_crn" + suffix))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one built native CRN library, found {candidates}")
    candidate = candidates[0]
    _verify(candidate, boost is not None)
    if _boost_digest(boost) != configuration["boost_headers_sha256"]:
        raise RuntimeError("Boost headers changed during the native CRN build")
    metadata = json.loads(candidate.with_suffix(".json").read_text())
    metadata.update(configuration=configuration, build_sha256=key, library_sha256=_digest(candidate))
    try:
        os.link(candidate, library)
    except FileExistsError:
        raise RuntimeError("Concurrent immutable CRN build installed the same artifact; rerun to verify") from None
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return library


def build_native(output_dir, compiler=None, *, boost_include_dir=None, with_semiflows=None):
    """Build and verify the CRN C ABI with the platform's CMake toolchain.

    Exact analysis uses installed Boost headers; ``with_semiflows=False`` builds
    only structural operations. Importing this module never invokes a compiler.
    """
    boost = boost_include(boost_include_dir)
    if with_semiflows is True and boost is None:
        raise RuntimeError(
            "Exact native semiflows require Boost headers; supply boost_include_dir or BOOST_INCLUDE_DIR"
        )
    if with_semiflows is False:
        boost = None
    if boost is not None:
        boost = boost.resolve()
    source = Path(__file__).parent / "cpp"
    payloads = _sources(source)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crn-build-", dir=output) as folder:
        work = Path(folder)
        snapshot = work / "sources"
        for name, payload in payloads.items():
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        build = work / "build"
        configuration = {
            "abi": 1, "builder_sha256": _digest(__file__),
            "support_sha256": _digest(Path(__file__).with_name("_build.py")),
            "sources": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
            "boost_headers_sha256": _boost_digest(boost), "platform": sys.platform, "machine": platform.machine(),
            "environment": {
                name: os.environ.get(name) for name in (
                    "CXX", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CL", "_CL_", "INCLUDE", "LIB", "LIBPATH",
                    "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "CMAKE_GENERATOR", "CMAKE_GENERATOR_PLATFORM",
                    "CMAKE_GENERATOR_TOOLSET", "VCToolsInstallDir", "VCToolsVersion", "WindowsSdkDir",
                    "WindowsSDKVersion", "UniversalCRTSdkDir", "UCRTVersion",
                )
            },
            "toolchain": _configure(build, snapshot, compiler, boost),
        }
        return _publish(build, output, configuration, boost)
