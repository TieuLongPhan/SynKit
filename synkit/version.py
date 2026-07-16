"""Installed and source-checkout version discovery."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

try:
    __version__ = version("synkit")
except PackageNotFoundError:
    try:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        __version__ = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
            "version"
        ]
    except (KeyError, OSError, tomllib.TOMLDecodeError):
        __version__ = "2.0.0.dev1"
