"""Installed and source-checkout version discovery."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

try:
    # A source checkout can coexist with an older installed distribution.
    # Prefer its adjacent project metadata; installed wheels do not include
    # the repository-level pyproject and therefore fall through naturally.
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    __version__ = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "version"
    ]
except (KeyError, OSError, tomllib.TOMLDecodeError):
    try:
        __version__ = version("synkit")
    except PackageNotFoundError:
        __version__ = "2.0.0b1"
