#!/usr/bin/env python3
"""Validate built SynKit distributions and smoke-test the wheel."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile

DATA_FILES = ("aldol.json.gz", "paracetamol.json.gz")

#: Package data outside ``synkit/Data`` that must ship, as
#: ``(package path, filename)`` pairs relative to ``synkit``.
PACKAGE_DATA = (("CRN/Benchmark/data", "kegg_modules.json"),)


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
            "synkit/Chem/Mapper/exact/native_build.py",
            "synkit/Chem/Mapper/exact/native_distance.cpp",
        ]
        + [f"synkit/Data/{filename}" for filename in DATA_FILES]
        + [f"synkit/{package}/{filename}" for package, filename in PACKAGE_DATA]
    )
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
        _assert_member_suffixes(set(archive.namelist()), wheel)
    with tarfile.open(sdist, "r:gz") as archive:
        _assert_member_suffixes(set(archive.getnames()), sdist)


def smoke_test_wheel(
    wheel: Path, project_root: Path, *, mapper_smoke: bool = False
) -> None:
    """Install a wheel without dependencies and import it outside the checkout.

    :param wheel: Built wheel path.
    :type wheel: Path
    :param project_root: Repository root containing ``pyproject.toml``.
    :type project_root: Path
    :param mapper_smoke: Exercise the installed mapper and native build using host dependencies.
    :type mapper_smoke: bool
    """
    with (project_root / "pyproject.toml").open("rb") as handle:
        expected_version = tomllib.load(handle)["project"]["version"]

    with tempfile.TemporaryDirectory(prefix="synkit-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=mapper_smoke).create(
            environment
        )
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)],
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
        if mapper_smoke:
            mapper_code = f"""
from pathlib import Path
import synkit.Chem.Mapper as mapper
from synkit.Chem.Mapper.exact import native_build
assert Path(mapper.__file__).resolve().is_relative_to(Path({str(environment)!r}).resolve())
assert callable(mapper.class_correspondence_from_export)
assert Path(native_build.__file__).with_name('native_distance.cpp').is_file()
library = native_build.build_native(Path({str(root)!r}) / 'native')
assert library.is_file() and library.with_suffix('.json').is_file()
print('installed mapper and packaged native builder verified')
"""
            subprocess.run([str(python), "-c", mapper_code], check=True, cwd=root)


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
        help="directory containing exactly one wheel and one source archive",
    )
    parser.add_argument(
        "--mapper-smoke",
        action="store_true",
        help="Check the installed mapper and optional kernel; requires declared dependencies and g++",
    )
    return parser.parse_args()


def main() -> None:
    """Run archive inspection and the isolated wheel smoke test."""
    args = parse_args()
    directory = args.directory.resolve()
    wheel = _one_artifact(directory, "*.whl")
    sdist = _one_artifact(directory, "*.tar.gz")
    project_root = Path(__file__).resolve().parents[1]
    inspect_archives(wheel, sdist)
    smoke_test_wheel(wheel, project_root, mapper_smoke=args.mapper_smoke)


if __name__ == "__main__":
    main()
