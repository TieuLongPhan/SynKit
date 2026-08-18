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


def smoke_test_wheel(wheel: Path, project_root: Path) -> None:
    """Install a wheel without dependencies and import it outside the checkout.

    :param wheel: Built wheel path.
    :type wheel: Path
    :param project_root: Repository root containing ``pyproject.toml``.
    :type project_root: Path
    """
    with (project_root / "pyproject.toml").open("rb") as handle:
        expected_version = tomllib.load(handle)["project"]["version"]

    with tempfile.TemporaryDirectory(prefix="synkit-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
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
    return parser.parse_args()


def main() -> None:
    """Run archive inspection and the isolated wheel smoke test."""
    args = parse_args()
    directory = args.directory.resolve()
    wheel = _one_artifact(directory, "*.whl")
    sdist = _one_artifact(directory, "*.tar.gz")
    project_root = Path(__file__).resolve().parents[1]
    inspect_archives(wheel, sdist)
    smoke_test_wheel(wheel, project_root)


if __name__ == "__main__":
    main()
