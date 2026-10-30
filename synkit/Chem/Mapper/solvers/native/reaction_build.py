"""Explicit immutable bundles for the optional RDKit C++ reaction pipeline."""

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
from dataclasses import dataclass
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class NativeReactionBuild:
    library: Path
    core: Path
    executable: Path
    manifest: Path


def build_reaction_native(
    output_dir,
    *,
    rdkit_dir,
    compiler="g++",
    ortools_dir=None,
    openwbo_executable=None,
    evalmaxsat_executable=None,
):
    """Compile an isolated native bundle; never install or compile on import.

    ``rdkit_dir`` is an existing C++ SDK prefix, including compatible Boost and
    Eigen development packages. The optional frontend uses C++20 for RDKit
    2026 public headers; the prepared-graph core remains C++17.
    """
    root = Path(__file__).resolve().parent
    cpp = root / "cpp"
    payloads = {
        str(path.relative_to(cpp)): path.read_bytes()
        for path in sorted(cpp.rglob("*"))
        if path.is_file()
    }
    legacy = (root.parent / "exact/native/native_distance.cpp").read_bytes()
    compiler_path = shutil.which(compiler)
    if not compiler_path:
        raise RuntimeError(f"C++ compiler unavailable: {compiler}")
    sdk = Path(rdkit_dir).resolve()
    if not (sdk / "include/rdkit/GraphMol/ROMol.h").is_file():
        raise ValueError("rdkit_dir lacks C++ development headers")
    sdk_records = {
        str(p.relative_to(sdk)): _sha(p)
        for p in sorted((sdk / "conda-meta").glob("*.json"))
    }
    # Also cover non-Conda SDKs and detect binary/header replacement.
    sdk_inputs = {
        str(p.relative_to(sdk)): _sha(p)
        for directory in (sdk / "include/rdkit", sdk / "lib/cmake/rdkit")
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }
    sdk_inputs.update(
        {
            str(p.relative_to(sdk)): _sha(p)
            for p in sorted((sdk / "lib").glob("*RDKit*.so*"))
            if p.is_file()
        }
    )
    configuration = {
        "builder_sha256": _sha(__file__),
        "platform": sys.platform,
        "machine": platform.machine(),
        "compiler": compiler_path,
        "compiler_version": subprocess.check_output(
            [compiler_path, "--version"], text=True
        ),
        "sources": {
            name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()
        },
        "legacy_source_sha256": hashlib.sha256(legacy).hexdigest(),
        "rdkit_dir": str(sdk),
        "sdk_records": sdk_records,
        "sdk_inputs": sdk_inputs,
        "ortools_dir": str(Path(ortools_dir).resolve()) if ortools_dir else None,
        "ortools_libraries": {
            str(p): _sha(p)
            for p in sorted(Path(ortools_dir).glob("lib*/libortools.so*"))
        }
        if ortools_dir
        else {},
        "openwbo": {
            "path": str(Path(openwbo_executable).resolve()),
            "sha256": _sha(openwbo_executable),
        }
        if openwbo_executable
        else None,
        "evalmaxsat": {
            "path": str(Path(evalmaxsat_executable).resolve()),
            "sha256": _sha(evalmaxsat_executable),
        }
        if evalmaxsat_executable
        else None,
    }
    key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / key[:20]
    suffix = (
        ".dylib"
        if sys.platform == "darwin"
        else ".dll"
        if sys.platform == "win32"
        else ".so"
    )

    def artifacts():
        return NativeReactionBuild(
            bundle / f"libsynkit_mapper_reaction{suffix}",
            bundle / f"libsynkit_mapper{suffix}",
            bundle / "mapper_reaction",
            bundle / "manifest.json",
        )

    if bundle.exists():
        record = json.loads((bundle / "manifest.json").read_text())
        if record["configuration"] != configuration or any(
            _sha(bundle / name) != digest
            for name, digest in record["artifacts"].items()
        ):
            raise RuntimeError("Native reaction bundle integrity verification failed")
        for section in ("dependencies", "external_headers"):
            if any(
                not Path(name).is_file() or _sha(name) != digest
                for name, digest in record.get(section, {}).items()
            ):
                raise RuntimeError(
                    f"Native reaction bundle {section} integrity verification failed"
                )
        return artifacts()
    with tempfile.TemporaryDirectory(prefix="reaction-build-", dir=output) as temporary:
        working = Path(temporary)
        source = working / "snapshot/Mapper/solvers/native/cpp"
        for name, data in payloads.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        legacy_path = source.parents[1] / "exact/native/native_distance.cpp"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_bytes(legacy)
        build = working / "build"
        command = [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
            f"-DCMAKE_CXX_COMPILER={compiler_path}",
            "-DSYNKIT_MAPPER_WITH_RDKIT=ON",
            "-DCMAKE_PREFIX_PATH="
            + ";".join(
                [str(sdk)] + ([str(Path(ortools_dir).resolve())] if ortools_dir else [])
            ),
        ]
        if ortools_dir:
            command.append("-DSYNKIT_MAPPER_WITH_ORTOOLS=ON")
        for option, value in (
            ("OPENWBO", openwbo_executable),
            ("EVALMAXSAT", evalmaxsat_executable),
        ):
            if value:
                command.append(
                    f"-DSYNKIT_MAPPER_{option}_EXECUTABLE={Path(value).resolve()}"
                )
        configure = subprocess.run(command, capture_output=True, text=True, check=False)
        if configure.returncode:
            raise RuntimeError(configure.stdout + configure.stderr)
        compile_command = ["cmake", "--build", str(build), "-j", "2"]
        compiled = subprocess.run(
            compile_command, capture_output=True, text=True, check=False
        )
        if compiled.returncode:
            raise RuntimeError(compiled.stdout + compiled.stderr)
        staged = working / "bundle"
        staged.mkdir()
        names = [
            f"libsynkit_mapper{suffix}",
            f"libsynkit_mapper_reaction{suffix}",
            "mapper_reaction",
        ]
        for name in names:
            shutil.copy2(build / name, staged / name)
        dependencies = {}
        if sys.platform.startswith("linux"):
            native_inputs = [staged / name for name in names] + [
                Path(value).resolve()
                for value in (openwbo_executable, evalmaxsat_executable)
                if value
            ]
            for native_input in native_inputs:
                dependency_listing = subprocess.run(
                    ["ldd", str(native_input)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                listing = dependency_listing.stdout + dependency_listing.stderr
                if dependency_listing.returncode and not any(
                    message in listing
                    for message in ("not a dynamic executable", "statically linked")
                ):
                    raise RuntimeError(
                        f"Cannot inspect native dependencies:\n{listing}"
                    )
                if "not found" in listing:
                    raise RuntimeError(f"Unresolved native dependency:\n{listing}")
                for line in listing.splitlines():
                    candidate = line.split("=>", 1)[-1].strip().split(" ", 1)[0]
                    if candidate.startswith("/") and Path(candidate).is_file():
                        resolved = Path(candidate).resolve()
                        if resolved.parent != staged:
                            dependencies[str(resolved)] = _sha(resolved)
        external_headers = {}
        for depfile in build.rglob("*.o.d"):
            text = depfile.read_text().replace("\\\n", " ")
            for name in shlex.split(text.split(":", 1)[-1]):
                path = Path(name)
                if path.is_file() and not path.is_relative_to(source.parents[2]):
                    external_headers[str(path.resolve())] = _sha(path)
        with tarfile.open(staged / "sources.tar.gz", "w:gz") as archive:
            archive.add(source.parents[2], arcname="Mapper")
        record = {
            "configuration": configuration,
            "build_key": key,
            "commands": [command, compile_command],
            "compile_commands": json.loads(
                (build / "compile_commands.json").read_text()
            ),
            "dependencies": dependencies,
            "external_headers": external_headers,
            "artifacts": {
                name: _sha(staged / name) for name in [*names, "sources.tar.gz"]
            },
            "compiler_log": compiled.stdout + compiled.stderr,
        }
        (staged / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        os.rename(staged, bundle)
    return artifacts()


def main():
    """Run the explicit native-reaction build command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rdkit-dir", required=True)
    parser.add_argument("--ortools-dir")
    parser.add_argument("--openwbo-executable")
    parser.add_argument("--evalmaxsat-executable")
    options = parser.parse_args()
    print(build_reaction_native(**vars(options)).library)


if __name__ == "__main__":
    main()
