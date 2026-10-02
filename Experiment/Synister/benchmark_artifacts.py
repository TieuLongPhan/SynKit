"""Source snapshots and metadata shared by focused search benchmarks."""

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def save(path, value):
    """Write a consistently formatted JSON benchmark artifact."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def freeze_source(root, destination):
    """Copy project Python sources, pruning environments and archived runs."""
    hashes = {}
    for base in ("synkit", "Experiment/Synister"):
        for directory, subdirectories, files in os.walk(root / base):
            subdirectories[:] = sorted(
                name
                for name in subdirectories
                if not name.startswith(".")
                and name not in {"runs", "venv", "__pycache__"}
            )
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                source = Path(directory) / name
                relative = source.relative_to(root)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                hashes[str(relative)] = sha256(target.read_bytes()).hexdigest()
    return hashes


def verify_source(destination, hashes):
    """Reject modifications to a frozen benchmark source file."""
    for name, expected in hashes.items():
        if sha256((destination / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen benchmark source changed: {name}")


def environment():
    """Record the interpreter, numerical dependencies, platform and CPU."""
    dependencies = {}
    for package in ("numpy", "scipy", "rdkit", "networkx"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    cpuinfo = Path("/proc/cpuinfo")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": dependencies,
        "cpuinfo": cpuinfo.read_text() if cpuinfo.is_file() else None,
    }
