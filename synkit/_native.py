"""Shared discovery and installed-file verification for bundled C ABI libraries."""

import base64
from functools import lru_cache
import hashlib
from importlib.metadata import distribution
import json
import os
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
def _conda_record_digest(library_file):
    """Read the owning Conda package's checksum after prefix relocation."""
    package = Path(__file__).resolve().parent
    library_file.relative_to(package)
    for prefix in package.parents:
        metadata = prefix / "conda-meta"
        if metadata.is_dir():
            return _conda_prefix_digest(prefix, library_file)
    return None


def _conda_prefix_digest(prefix, library_file):
    relative = os.path.normcase(library_file.relative_to(prefix).as_posix())
    for filename in sorted((prefix / "conda-meta").glob("synkit-*.json")):
        try:
            record = json.loads(filename.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("Cannot read installed SynKit Conda metadata") from exc
        if record.get("name") != "synkit":
            continue
        for entry in record.get("paths_data", {}).get("paths", ()):
            if os.path.normcase(entry.get("_path", "")) != relative:
                continue
            # Conda and Mamba record bytes after relocation/code signing here.
            digest = entry.get("sha256_in_prefix") or entry.get("sha256")
            if (
                isinstance(digest, str)
                and len(digest) == 64
                and all(c in "0123456789abcdefABCDEF" for c in digest)
            ):
                return digest.lower()
            raise RuntimeError(f"The bundled library has no installed Conda SHA-256 record: {relative}")
    return None


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
    """Check final installed bytes against their Conda or wheel SHA-256 record."""
    path = Path(path).resolve()
    expected = _conda_record_digest(path) or _wheel_record_digest(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Bundled native library checksum mismatch: {path.name}")
    return actual
