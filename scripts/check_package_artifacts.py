#!/usr/bin/env python3
"""Validate built SynKit distributions and smoke-test the wheel."""

from __future__ import annotations

import argparse
import base64
import csv
from email.parser import Parser
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import venv
import zipfile
from pathlib import Path

DATA_FILES = ("aldol.json.gz", "paracetamol.json.gz")

#: Package data outside ``synkit/Data`` that must ship, as
#: ``(package path, filename)`` pairs relative to ``synkit``.
PACKAGE_DATA = tuple(
    ("CRN/cpp", filename)
    for filename in (
        "closure.cpp",
        "structural.cpp",
        "semiflows.cpp",
        "exact.h",
        "algebra.cpp",
        "pathway.cpp",
        "symmetry.cpp",
        "CMakeLists.txt",
        "write_manifest.cmake",
    )
)

GUIDE_DOCS = (
    "doc/chem.rst",
    "doc/chem/reaction.rst",
    "doc/chem/molecule.rst",
    "doc/mapper.rst",
    "doc/mapper/advanced.rst",
    "doc/mapper/implementation.rst",
)


def _assert_source_documentation(members: set[str], artifact: Path) -> None:
    """Require the Chem guides and mapper references in a source distribution."""
    missing = [
        name
        for name in GUIDE_DOCS
        if not any(member.endswith("/" + name) for member in members)
    ]
    if missing:
        raise RuntimeError(
            f"{artifact.name} is missing documentation: {', '.join(missing)}"
        )


def _one_artifact(directory: Path, pattern: str) -> Path:
    """Return the only artifact matching a glob.

    :param directory: Distribution directory.
    :type directory: Path
    :param pattern: Glob used to select an artifact.
    :type pattern: str
    :return: Matching artifact.
    :rtype: Path
    :raises RuntimeError: If the match count is not one.
    """
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {pattern!r} artifact in {directory}, found {matches}."
        )
    return matches[0]


def _assert_member_suffixes(members: set[str], artifact: Path) -> None:
    """Require package data and metadata paths in an archive.

    :param members: Archive member names.
    :type members: set[str]
    :param artifact: Artifact used in error messages.
    :type artifact: Path
    :raises RuntimeError: If a required member is absent.
    """
    required = (
        ["synkit/__init__.py"]
        + [
            "synkit/CRN/_build.py",
            "synkit/CRN/_native_build.py",
            "synkit/Chem/Mapper/mapper.py",
            "synkit/Chem/Mapper/solvers/exact/native_build.py",
            "synkit/Chem/Mapper/solvers/exact/native/build.py",
            "synkit/Chem/Mapper/solvers/exact/native/native_distance.cpp",
            "synkit/Chem/Mapper/solvers/native/build.py",
            "synkit/Chem/Mapper/solvers/native/implicit.py",
            "synkit/Chem/Mapper/solvers/native/cpp/CMakeLists.txt",
            "synkit/Chem/Mapper/solvers/native/cpp/include/synkit_mapper/mapper.h",
        ]
        + [f"synkit/Data/{filename}" for filename in DATA_FILES]
        + [f"synkit/{package}/{filename}" for package, filename in PACKAGE_DATA]
    )
    project = Path(__file__).resolve().parents[1]
    required += [
        path.relative_to(project).as_posix()
        for directory in (
            project / "synkit/Chem/Mapper/solvers/native",
            project / "synkit/Graph/Canon/cpp",
        )
        for path in directory.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".c", ".cpp", ".h", ".hpp", ".inc", ".py", ".rst"}
            or path.name in {"CMakeLists.txt", "COPYRIGHT", "LICENSE-2.0.txt"}
        )
    ]
    missing = [
        suffix
        for suffix in required
        if not any(member.endswith(suffix) for member in members)
    ]
    if missing:
        raise RuntimeError(f"{artifact.name} is missing: {', '.join(missing)}")


def inspect_archives(wheel: Path, sdist: Path) -> None:
    """Inspect wheel and source archives for required files.

    :param wheel: Built wheel path.
    :type wheel: Path
    :param sdist: Built source distribution path.
    :type sdist: Path
    """
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        _assert_member_suffixes(members, wheel)
        libraries = [
            name
            for name in members
            if name.startswith("synkit/Chem/Mapper/solvers/native/lib/")
            and name.endswith((".so", ".dylib", ".dll"))
        ]
        if len(libraries) != 1:
            raise RuntimeError(f"{wheel.name} must contain one bundled mapper library")
        canon_libraries = [
            name
            for name in members
            if name.startswith("synkit/Graph/Canon/_native/lib/")
            and name.endswith((".so", ".dylib", ".dll"))
        ]
        if len(canon_libraries) != 1:
            raise RuntimeError(
                f"{wheel.name} must contain one bundled graph/stereograph library"
            )
        crn_libraries = [
            name
            for name in members
            if name.startswith("synkit/CRN/lib/")
            and name.endswith((".so", ".dylib", ".dll"))
        ]
        if len(crn_libraries) != 1:
            raise RuntimeError(f"{wheel.name} must contain one bundled CRN library")
        crn_library = crn_libraries[0]
        manifest_name = str(Path(crn_library).with_suffix(".json"))
        manifest = json.loads(archive.read(manifest_name))
        required_capabilities = {
            "closure",
            "structural_sets",
            "structural_unlimited",
            "exhaustive_siphons",
            "exact_semiflows",
            "exact_algebra",
            "crnt",
            "pathway",
            "symmetry",
        }
        if not required_capabilities <= set(manifest.get("capabilities", [])):
            raise RuntimeError(f"{wheel.name} has an incomplete CRN backend")
        if manifest.get("integrity") != "wheel-record" or manifest.get("abi") != 1:
            raise RuntimeError(f"{wheel.name} has invalid CRN build metadata")
        for source, expected in manifest["source_sha256"].items():
            if (
                hashlib.sha256(archive.read("synkit/CRN/cpp/" + source)).hexdigest()
                != expected
            ):
                raise RuntimeError(
                    f"{wheel.name}: CRN source digest mismatch for {source}"
                )
        record_name = next(
            name for name in members if name.endswith(".dist-info/RECORD")
        )
        records = {
            row[0]: row[1]
            for row in csv.reader(io.StringIO(archive.read(record_name).decode()))
        }
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(archive.read(crn_library)).digest())
            .decode()
            .rstrip("=")
        )
        if records.get(crn_library) != "sha256=" + digest:
            raise RuntimeError(
                f"{wheel.name}: CRN library differs from its wheel record"
            )
        metadata_name = next(
            name for name in members if name.endswith(".dist-info/WHEEL")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        if metadata["Root-Is-Purelib"] != "false" or any(
            tag.endswith("-any") for tag in metadata.get_all("Tag", [])
        ):
            raise RuntimeError(
                f"{wheel.name} incorrectly labels a binary wheel as portable"
            )
        if any(
            name.startswith("Experiment/CRN/")
            or name.endswith("Benchmark/data/kegg_modules.json")
            for name in members
        ):
            raise RuntimeError(
                "Research experiments or fixtures leaked into the runtime wheel"
            )
    with tarfile.open(sdist, "r:gz") as archive:
        members = set(archive.getnames())
        if any(
            "/Experiment/CRN/results/" in name
            or "/Experiment/CRN/data/biomodels/" in name
            for name in members
        ):
            raise RuntimeError(
                "Generated CRN outputs or the BioModels XML cache leaked into the source distribution"
            )
        _assert_member_suffixes(members, sdist)
        _assert_source_documentation(members, sdist)
        if not any(name.endswith("/hatch_build.py") for name in members):
            raise RuntimeError(
                "Source distribution is missing the native wheel build hook"
            )
        if not any(
            name.endswith("/CMakeLists.txt") and name.count("/") == 1
            for name in members
        ):
            raise RuntimeError(
                "Source distribution is missing the unified native build"
            )
        for suffix in (
            "scripts/check_package_artifacts.py",
            "scripts/smoke_native_wheel.py",
            "Experiment/CRN/benchmarks/networks.py",
            "Experiment/CRN/benchmarks/scaling.py",
            "Experiment/CRN/benchmarks/glycolysis.py",
            "Experiment/CRN/data/kegg_modules.json",
            "Experiment/CRN/data/glycolysis_routes.json",
            "Experiment/CRN/data/biomodels_manifest.json",
            "Experiment/CRN/environments/crnt4sbml-requirements.txt",
        ):
            if not any(name.endswith(suffix) for name in members):
                raise RuntimeError(
                    f"Source distribution is missing research source: {suffix}"
                )


def smoke_test_wheel(
    wheel: Path, project_root: Path, *, native_smoke: bool = False
) -> None:
    """Install a wheel without dependencies and import it outside the checkout.

    :param wheel: Built wheel path.
    :type wheel: Path
    :param project_root: Repository root containing ``pyproject.toml``.
    :type project_root: Path
    :param native_smoke: Exercise the installed mapper and CRN libraries using host dependencies.
    :type native_smoke: bool
    """
    with (project_root / "pyproject.toml").open("rb") as handle:
        expected_version = tomllib.load(handle)["project"]["version"]

    with tempfile.TemporaryDirectory(prefix="synkit-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=native_smoke).create(
            environment
        )
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                str(wheel),
            ],
            check=True,
            cwd=root,
        )
        code = f"""
from importlib.resources import files
import synkit

expected = {expected_version!r}
if synkit.__version__ != expected:
    raise SystemExit("version mismatch: %s != %s" % (synkit.__version__, expected))
for filename in {DATA_FILES!r}:
    resource = files("synkit").joinpath("Data", filename)
    if not resource.is_file():
        raise SystemExit("missing installed data file: %s" % filename)
for package, filename in {PACKAGE_DATA!r}:
    resource = files("synkit").joinpath(package, filename)
    if not resource.is_file():
        raise SystemExit("missing installed package data: %s/%s" % (package, filename))
print("synkit %s: wheel import and data files verified" % synkit.__version__)
"""
        subprocess.run([str(python), "-c", code], check=True, cwd=root)
        if native_smoke:
            subprocess.run(
                [str(python), str(project_root / "scripts/smoke_native_wheel.py")],
                check=True,
                cwd=root,
                env={**os.environ, "PATH": ""},
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    :return: Parsed command-line namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("dist"),
        help="directory containing wheels and one source archive",
    )
    parser.add_argument(
        "--native-smoke",
        "--mapper-smoke",
        dest="native_smoke",
        action="store_true",
        help="Check installed mapper and CRN backends using host dependencies; needs no compiler",
    )
    parser.add_argument(
        "--archives-only",
        action="store_true",
        help="Inspect all platform wheels without installing them",
    )
    return parser.parse_args()


def main() -> None:
    """Run archive inspection and the isolated wheel smoke test."""
    args = parse_args()
    directory = args.directory.resolve()
    wheels = sorted(directory.glob("*.whl"))
    if not wheels:
        raise RuntimeError("No wheels found")
    sdist = _one_artifact(directory, "*.tar.gz")
    project_root = Path(__file__).resolve().parents[1]
    for wheel in wheels:
        inspect_archives(wheel, sdist)
    if not args.archives_only:
        if len(wheels) != 1:
            raise RuntimeError("Use --archives-only for multiple platform wheels")
        smoke_test_wheel(wheels[0], project_root, native_smoke=args.native_smoke)


if __name__ == "__main__":
    main()
