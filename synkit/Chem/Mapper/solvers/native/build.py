"""Explicit, immutable builds of the C++ mapper core and optional solver SDKs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ortools_libraries(prefix):
    """Locate the supplied C++ SDK's libraries without assuming a host OS."""
    prefix = Path(prefix).resolve()
    libraries = {
        path.resolve()
        for pattern in (
            "lib*/libortools.so*",
            "lib*/libortools*.dylib",
            "lib*/libortools*.a",
            "lib*/ortools*.lib",
            "lib*/libortools*.lib",
            "bin/*ortools*.dll",
        )
        for path in prefix.glob(pattern)
        if path.is_file()
    }
    if not libraries:
        raise RuntimeError(f"Cannot find OR-Tools C++ libraries under {prefix}")
    return {str(path): _digest(path) for path in sorted(libraries)}


def _runtime_dependencies(paths):
    result = {}
    if not sys.platform.startswith("linux"):
        return result
    for path in paths:
        listing = subprocess.run(
            ["ldd", str(path)], capture_output=True, text=True, check=False
        )
        text = listing.stdout + listing.stderr
        if listing.returncode and not any(
            value in text for value in ("not a dynamic executable", "statically linked")
        ):
            raise RuntimeError(f"Cannot inspect native dependencies: {text}")
        if "not found" in text:
            raise RuntimeError(f"Unresolved native dependencies: {text}")
        for line in text.splitlines():
            name = line.split("=>", 1)[-1].strip().split(" ", 1)[0]
            if name.startswith("/") and Path(name).is_file():
                name = str(Path(name).resolve())
                result[name] = _digest(name)
    return result


def _verify(path):
    script = (
        "import ctypes,sys; library=ctypes.CDLL(sys.argv[1]); "
        "assert library.synkit_mapper_abi()==1, 'unsupported mapper ABI'; "
        "assert library.synkit_mapper_core_abi()==1, 'unsupported core ABI'; "
        "assert library.synkit_distance_abi()==3, 'unsupported distance ABI'; "
        "library.synkit_mapper_capabilities.restype=ctypes.c_uint64; "
        "assert library.synkit_mapper_capabilities() & 2, 'PABS is unavailable'"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Cannot load native mapper library {path}:\n{result.stderr}"
        )


def build_mapper_native(
    output_dir,
    compiler="g++",
    *,
    ortools_dir=None,
    openwbo_executable=None,
    evalmaxsat_executable=None,
    test_hooks=False,
):
    """Build the C++17 core explicitly; return a verified immutable library.

    The default build needs no Python, SciPy, OR-Tools or RDKit headers.
    ``ortools_dir`` enables certified LP bounds and the optional solver adapters.
    ``test_hooks`` explicitly includes internal native oracle exports; production
    builds leave them disabled. Existing ABI3-only builds remain separate.
    """
    if not isinstance(test_hooks, bool):
        raise TypeError("test_hooks must be boolean")
    source_root = Path(__file__).resolve().parent / "cpp"
    source_names = [
        "src/slap.cpp",
        "src/core/dispatch.cpp",
        "src/pabs/bridge.cpp",
        "src/pabs/sparse_embedding.cpp",
        "src/assignment/abb.cpp",
        "src/supporting/dfs.cpp",
        "src/supporting/edit_support.cpp",
        "src/sepdp/sepdp.cpp",
        "src/mces/mces.cpp",
        "src/hydrogen/hydrogen.cpp",
        "src/astar/astar.cpp",
        "src/astar/bound.cpp",
        "src/symmetry/classification.cpp",
        "src/symmetry/classification_pipeline.cpp",
        "src/symmetry/cpabs.cpp",
        "src/symmetry/edge_deck.cpp",
        "src/core/pmcd_witness.cpp",
    ]
    nauty_enabled = os.name != "nt"
    if nauty_enabled:
        source_names += [
            "src/symmetry/vendor/nauty/nauty.c",
            "src/symmetry/vendor/nauty/nautil.c",
            "src/symmetry/vendor/nauty/nausparse.c",
            "src/symmetry/vendor/nauty/naugraph.c",
            "src/symmetry/vendor/nauty/schreier.c",
            "src/symmetry/vendor/nauty/naurng.c",
        ]
    sources = [source_root / name for name in source_names]
    legacy_source = source_root.parents[1] / "exact/native/native_distance.cpp"
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing shipped native mapper sources: {missing}")
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise RuntimeError(f"C++17 compiler unavailable: {compiler}")
    flags = ["-std=c++17", "-O3", "-Wall", "-Wextra"]
    flags += [
        "-DSYNKIT_MAPPER_WITH_ASTAR",
        "-DSYNKIT_MAPPER_WITH_CPABS",
        "-DSYNKIT_MAPPER_WITH_ASSIGNMENT",
        "-DSYNKIT_MAPPER_WITH_SUPPORTING",
        "-DSYNKIT_MAPPER_WITH_MCES",
    ]
    if test_hooks:
        flags.append("-DSYNKIT_MAPPER_TEST_HOOKS")
    if nauty_enabled:
        flags.append("-DSYNKIT_MAPPER_WITH_NAUTY")
    if sys.platform == "win32":
        suffix = ".dll"
        flags += ["-shared", "-static", "-Wl,--export-all-symbols"]
    elif sys.platform == "darwin":
        suffix = ".dylib"
        flags += ["-fPIC", "-dynamiclib"]
    else:
        suffix = ".so"
        flags += ["-fPIC", "-shared"]
    source_files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and (
            path.name == "CMakeLists.txt"
            or path.relative_to(source_root).parts[0]
            in {"include", "src", "tests", "examples"}
        )
    )
    payloads = {
        str(path.relative_to(source_root)): path.read_bytes() for path in source_files
    }
    legacy_payload = legacy_source.read_bytes()
    ortools_libraries = _ortools_libraries(ortools_dir) if ortools_dir else {}
    configuration = {
        "abi": 1,
        "builder_sha256": _digest(__file__),
        "sources": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
        "compiled_sources": [str(path.relative_to(source_root)) for path in sources],
        "legacy_pabs_source_sha256": hashlib.sha256(legacy_payload).hexdigest(),
        "ortools_prefix": str(Path(ortools_dir).resolve()) if ortools_dir else None,
        "ortools_library_sha256": next(iter(ortools_libraries.values()), None),
        "ortools_libraries": ortools_libraries,
        "test_hooks": test_hooks,
        "nauty_enabled": nauty_enabled,
        "openwbo_executable": str(Path(openwbo_executable).resolve())
        if openwbo_executable
        else None,
        "openwbo_sha256": _digest(openwbo_executable) if openwbo_executable else None,
        "evalmaxsat_executable": str(Path(evalmaxsat_executable).resolve())
        if evalmaxsat_executable
        else None,
        "evalmaxsat_sha256": _digest(evalmaxsat_executable)
        if evalmaxsat_executable
        else None,
        "compiler": compiler_path,
        "compiler_version": subprocess.check_output(
            [compiler_path, "--version"], text=True
        ),
        "flags": flags,
        "platform": sys.platform,
        "machine": platform.machine(),
    }
    key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / f"libsynkit_mapper_{key[:20]}{suffix}"
    manifest = library.with_suffix(".json")
    if library.exists():
        if not manifest.exists():
            raise RuntimeError(
                f"Existing native mapper library lacks provenance: {library}"
            )
        saved = json.loads(manifest.read_text())
        if saved["configuration"] != configuration or saved[
            "library_sha256"
        ] != _digest(library):
            raise RuntimeError(
                f"Native mapper build failed integrity verification: {library}"
            )
        for section in ("dependencies", "external_headers"):
            if any(
                not Path(path).is_file() or _digest(path) != digest
                for path, digest in saved.get(section, {}).items()
            ):
                raise RuntimeError(
                    f"Native mapper {section} failed integrity verification: {library}"
                )
        _verify(library)
        return library
    with tempfile.TemporaryDirectory(
        prefix="synkit-mapper-build-", dir=destination
    ) as temporary:
        candidate = Path(temporary) / f"mapper{suffix}"
        snapshot = Path(temporary) / "snapshot/Mapper/solvers/native/cpp"
        for name, payload in payloads.items():
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        legacy_snapshot = snapshot.parents[1] / "exact/native/native_distance.cpp"
        legacy_snapshot.parent.mkdir(parents=True, exist_ok=True)
        legacy_snapshot.write_bytes(legacy_payload)
        compiled = [snapshot / name for name in source_names]
        if ortools_dir or openwbo_executable or evalmaxsat_executable:
            cmake_dir = Path(temporary) / "cmake"
            command = [
                "cmake",
                "-S",
                str(snapshot),
                "-B",
                str(cmake_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-DCMAKE_CXX_COMPILER={compiler_path}",
                "-DSYNKIT_MAPPER_TEST_HOOKS=" + ("ON" if test_hooks else "OFF"),
                "-DSYNKIT_MAPPER_WITH_NAUTY=" + ("ON" if nauty_enabled else "OFF"),
            ]
            if ortools_dir:
                command += [
                    "-DSYNKIT_MAPPER_WITH_ORTOOLS=ON",
                    f"-DCMAKE_PREFIX_PATH={Path(ortools_dir).resolve()}",
                ]
            if openwbo_executable:
                command += [
                    f"-DSYNKIT_MAPPER_OPENWBO_EXECUTABLE={Path(openwbo_executable).resolve()}"
                ]
            if evalmaxsat_executable:
                command += [
                    f"-DSYNKIT_MAPPER_EVALMAXSAT_EXECUTABLE={Path(evalmaxsat_executable).resolve()}"
                ]
            configure = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if configure.returncode:
                raise RuntimeError(configure.stdout + configure.stderr)
            result = subprocess.run(
                ["cmake", "--build", str(cmake_dir), "-j", "2"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            shutil.copy2(cmake_dir / f"libsynkit_mapper{suffix}", candidate)
        else:
            command = [
                compiler_path,
                *flags,
                "-I",
                str(snapshot / "include"),
                "-I",
                str(snapshot / "src/symmetry/vendor/nauty"),
                *map(str, compiled),
                "-o",
                str(candidate),
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
        if result.returncode:
            raise RuntimeError(f"Native mapper compilation failed:\n{result.stderr}")
        _verify(candidate)
        record = {
            "configuration": configuration,
            "build_key": key,
            "library_sha256": _digest(candidate),
            "library": str(library),
            "command": command,
            "compiler_stderr": result.stderr,
            "dependencies": _runtime_dependencies(
                [candidate]
                + [
                    Path(value)
                    for value in (openwbo_executable, evalmaxsat_executable)
                    if value
                ]
            ),
        }
        if ortools_dir or openwbo_executable or evalmaxsat_executable:
            record["compile_commands"] = json.loads(
                (cmake_dir / "compile_commands.json").read_text()
            )
            external_headers = {}
            for depfile in cmake_dir.rglob("*.o.d"):
                text = depfile.read_text().replace("\\\n", " ")
                for name in shlex.split(text.split(":", 1)[-1]):
                    path = Path(name)
                    if path.is_file() and not path.is_relative_to(snapshot.parents[2]):
                        external_headers[str(path.resolve())] = _digest(path)
            record["external_headers"] = external_headers
        archive = library.with_suffix(".sources.tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(snapshot.parents[2], arcname="Mapper")
        record.update(
            source_archive=str(archive), source_archive_sha256=_digest(archive)
        )
        try:
            os.link(candidate, library)
        except FileExistsError:
            raise RuntimeError(
                f"Concurrent native mapper build installed {library}; rerun to verify"
            ) from None
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    return library


def main():
    """Run the explicit native-core build command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "synkit-mapper"
    )
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--ortools-dir", type=Path)
    parser.add_argument("--openwbo-executable", type=Path)
    parser.add_argument("--evalmaxsat-executable", type=Path)
    parser.add_argument(
        "--test-hooks",
        action="store_true",
        help="Enable internal native oracle exports",
    )
    options = parser.parse_args()
    print(
        build_mapper_native(
            options.output_dir,
            options.compiler,
            ortools_dir=options.ortools_dir,
            openwbo_executable=options.openwbo_executable,
            evalmaxsat_executable=options.evalmaxsat_executable,
            test_hooks=options.test_hooks,
        )
    )


if __name__ == "__main__":
    main()
