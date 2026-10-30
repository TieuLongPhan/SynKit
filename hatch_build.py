"""Build all bundled C ABI libraries without importing SynKit dependencies."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

NATIVE_PACKAGES = {
    "mapper": "synkit/Chem/Mapper/solvers/native/lib",
    "crn": "synkit/CRN/lib",
    "canon": "synkit/Graph/Canon/_native/lib",
}


class CustomBuildHook(BuildHookInterface):
    """Configure, build and install all wheel libraries in one CMake invocation."""

    def initialize(self, version, build_data):
        if self.target_name != "wheel" or version == "editable":
            return
        root = Path(self.root)
        spec = importlib.util.spec_from_file_location(
            "synkit_crn_build", root / "synkit/CRN/_build.py"
        )
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        boost = support.boost_include()
        if boost is None:
            raise RuntimeError(
                "Building SynKit wheels requires Boost headers for exact CRN analysis; "
                "install them or set BOOST_INCLUDE_DIR to the directory containing boost/"
            )
        self._temporary = tempfile.TemporaryDirectory(prefix="synkit-wheel-")
        staging = Path(self._temporary.name)
        build = staging / "build"
        configure = [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSYNKIT_NATIVE_WHEEL=ON",
            "-DSYNKIT_MAPPER_TEST_HOOKS=OFF",
            "-DSYNKIT_MAPPER_BUILD_EXAMPLES=OFF",
            "-DSYNKIT_MAPPER_WITH_RDKIT=OFF",
            "-DSYNKIT_CANON_BUILD_TESTS=OFF",
            "-DSYNKIT_CANON_BUILD_EXAMPLES=OFF",
            f"-DSYNKIT_CRN_BOOST_INCLUDE_DIR={boost}",
        ]
        sdk = os.environ.get("SYNKIT_MAPPER_ORTOOLS_DIR")
        configure += [f"-DSYNKIT_MAPPER_WITH_ORTOOLS={'ON' if sdk else 'OFF'}"]
        if sdk:
            configure.append(f"-DCMAKE_PREFIX_PATH={Path(sdk).resolve()}")
        if sys.platform == "darwin":
            architecture = platform.machine()
            deployment = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")
            configure += [
                f"-DCMAKE_OSX_ARCHITECTURES={architecture}",
                f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment}",
            ]
        subprocess.run(configure, check=True)
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "--config",
                "Release",
                "--parallel",
                os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "2"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--install",
                str(build),
                "--config",
                "Release",
                "--prefix",
                str(staging / "install"),
            ],
            check=True,
        )
        libraries = {}
        for component, package_path in NATIVE_PACKAGES.items():
            candidates = [
                p
                for p in (staging / "install" / package_path).iterdir()
                if p.suffix in {".so", ".dylib", ".dll"}
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected one bundled {component} library, found {candidates}"
                )
            library = libraries[component] = candidates[0]
            build_data["force_include"][str(library)] = f"{package_path}/{library.name}"
        manifest = libraries["crn"].with_suffix(".json")
        metadata = json.loads(manifest.read_text())
        metadata.pop("library_sha256")
        metadata.pop("build_sha256")
        metadata = {
            **metadata,
            "root_cmake_sha256": hashlib.sha256(
                (root / "CMakeLists.txt").read_bytes()
            ).hexdigest(),
            "platform": sysconfig.get_platform(),
            # Repair tools can rewrite libraries. RECORD covers the final bytes.
            "integrity": "wheel-record",
        }
        metadata["build_sha256"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()
        manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        build_data["force_include"][
            str(manifest)
        ] = f"{NATIVE_PACKAGES['crn']}/{manifest.name}"
        platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        if sys.platform == "darwin":
            platform_tag = f"macosx_{deployment.replace('.', '_')}_{architecture}"
        build_data["tag"] = f"py3-none-{platform_tag}"
        build_data["pure_python"] = False

    def finalize(self, version, build_data, artifact_path):
        if hasattr(self, "_temporary"):
            self._temporary.cleanup()
