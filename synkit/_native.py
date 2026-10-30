"""Shared discovery and wheel-record verification for bundled C ABI libraries."""

import base64
from functools import lru_cache
import hashlib
from importlib.metadata import distribution
from pathlib import Path
import sys


def bundled_library(package_dir, stem):
    """Locate one packaged library without compiling or downloading anything."""
    name = (
        stem + ".dll"
        if sys.platform == "win32"
        else (
            "lib" + stem + ".dylib"
            if sys.platform == "darwin"
            else "lib" + stem + ".so"
        )
    )
    path = Path(package_dir).resolve() / "lib" / name
    if not path.is_file():
        raise FileNotFoundError(f"No bundled {stem} library is installed")
    return path


@lru_cache(maxsize=8)
def _wheel_record_digest(library_file):
    installed = distribution("synkit")
    relative = (
        "synkit/" + library_file.relative_to(Path(__file__).resolve().parent).as_posix()
    )
    for entry in installed.files or ():
        if (
            str(entry) == relative
            and Path(installed.locate_file(entry)).resolve() == library_file
        ):
            if entry.hash is not None and entry.hash.mode == "sha256":
                return base64.urlsafe_b64decode(entry.hash.value + "==").hex()
    raise RuntimeError(
        f"The bundled library has no installed wheel SHA-256 record: {relative}"
    )


def verify_wheel_library(path):
    """Check installed bytes against the final wheel's RECORD after repair."""
    path = Path(path).resolve()
    expected = _wheel_record_digest(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Bundled native library checksum mismatch: {path.name}")
    return actual
