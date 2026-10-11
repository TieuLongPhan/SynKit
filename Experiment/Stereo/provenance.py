"""Capture immutable source and input evidence for research reports."""

from pathlib import Path
import hashlib
import platform
import subprocess

import networkx
import rdkit

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    """Return the content checksum of an existing artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_source(input_paths=()) -> dict:
    """Record executed Python and explicitly supplied immutable input files."""
    paths = sorted((ROOT / "synkit").rglob("*.py"))
    paths += sorted((ROOT / "Experiment/Stereo").rglob("*.py"))
    changes = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    captured = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "working_tree_dirty": bool(changes),
        "working_tree_changes": changes,
        "dirty_state_policy": "content hashes identify the executed source; dirty state is disclosed",
        "python": platform.python_version(), "rdkit": rdkit.__version__,
        "networkx": networkx.__version__, "platform": platform.platform(),
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths},
    }
    if input_paths:
        inputs = sorted({Path(path).resolve() for path in input_paths})
        captured["input_sha256"] = {str(path): sha256(path) for path in inputs}
    return captured


def finish_source(start: dict) -> dict:
    """Refuse a report if executable sources or captured inputs changed."""
    end = capture_source()
    if start["source_sha256"] != end["source_sha256"]:
        raise RuntimeError("Research sources changed during report generation.")
    completed = {**start, "source_unchanged_during_run": True}
    if "input_sha256" in start:
        for name, digest in start["input_sha256"].items():
            if sha256(Path(name)) != digest:
                raise RuntimeError("Research input changed during report generation: " + name)
        completed["inputs_unchanged_during_run"] = True
    return completed
