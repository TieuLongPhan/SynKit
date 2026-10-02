#!/usr/bin/env python3
"""Explicitly build the optional native solver into an immutable, hashed artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def build_native(output_dir, compiler="g++", *, native_arch=False):
    """Build and verify a source/compiler-bound optional kernel artifact."""
    source = Path(__file__).with_name("native_distance.cpp")
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise RuntimeError(f"C++17 compiler unavailable: {compiler}")
    version = subprocess.run(
        [compiler_path, "--version"], check=True, capture_output=True, text=True
    ).stdout
    flags = ["-std=c++17", "-O3", "-Wall", "-Wextra", "-fPIC", "-shared"]
    target_options = None
    if native_arch:
        flags += ["-march=native", "-mtune=native"]
        # Include resolved CPU options in the cache key: the same source and
        # literal -march=native flags can generate different code on another host.
        target_options = subprocess.run(
            [
                compiler_path,
                *flags,
                "-Q",
                "--help=target",
                "-x",
                "c++",
                "-c",
                "/dev/null",
                "-o",
                "/dev/null",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    provenance = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": compiler_path,
        "compiler_version": version,
        "flags": flags,
    }
    if native_arch:
        provenance["resolved_target_options"] = target_options
        provenance["portability"] = (
            "requires the recorded host CPU instruction features"
        )
    build_key = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()
    ).hexdigest()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    library = output_dir / f"libsynkit_distance_{build_key[:20]}.so"
    manifest = library.with_suffix(".json")
    if library.exists():
        if not manifest.exists():
            raise RuntimeError(f"Existing library lacks build provenance: {library}")
        recorded = json.loads(manifest.read_text())
        if (
            recorded["build_key"] != build_key
            or recorded["library_sha256"]
            != hashlib.sha256(library.read_bytes()).hexdigest()
        ):
            raise RuntimeError(
                f"Existing native artifact failed integrity verification: {library}"
            )
        return library
    # A distinct path avoids overwriting code already mapped into another process.
    with tempfile.TemporaryDirectory(
        prefix="synkit-build-", dir=output_dir
    ) as temporary:
        candidate = Path(temporary) / "kernel.so"
        result = subprocess.run(
            [compiler_path, *flags, str(source), "-o", str(candidate)],
            check=True,
            capture_output=True,
            text=True,
        )
        provenance.update(
            build_key=build_key,
            library=str(library),
            library_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            compiler_stderr=result.stderr,
        )
        # Hard-link installation refuses to replace any concurrently installed binary.
        try:
            os.link(candidate, library)
        except FileExistsError:
            raise RuntimeError(
                f"Concurrent build installed {library}; rerun to verify it"
            )
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    return library


def main():
    """Build the optional kernel from an installed SynKit package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/synkit-native"))
    parser.add_argument("--compiler", default="g++")
    parser.add_argument(
        "--native-arch",
        action="store_true",
        help="Opt in to host CPU instructions; resulting library is host-specific.",
    )
    args = parser.parse_args()
    print(build_native(args.output_dir, args.compiler, native_arch=args.native_arch))


if __name__ == "__main__":
    main()
