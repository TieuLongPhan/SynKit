"""Source snapshots and bounded independent typed-map replay."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import platform
import subprocess

import networkx as nx
import numpy as np
import rdkit

from Experiment.Stereo.Canonicalization.runtime import wall_time_limit

ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    paths = sorted(path for directory in (ROOT / "synkit", ROOT / "Experiment/Stereo")
                   for path in directory.rglob("*.py"))
    return {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
            "python": platform.python_version(), "rdkit": rdkit.__version__, "networkx": nx.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths}}


@contextmanager
def wall_limit(seconds):
    """Bound chemical operations with deadlines and available POSIX alarms."""
    with wall_time_limit(seconds, TimeoutError("chemical_validation_wall_limit")):
        yield


def replay_typed_map(source, target, mapping):
    """Check original complete vertex and edge colors, not refined signatures."""
    from Experiment.Stereo.Canonicalization.independent_orbits import color_signature

    if set(mapping) != set(source) or set(mapping.values()) != set(target) or len(mapping) != len(target):
        raise ValueError("chemical_witness_not_a_full_bijection")
    if source.number_of_edges() != target.number_of_edges():
        raise ValueError("chemical_witness_edge_count_mismatch")
    for node, image in mapping.items():
        if color_signature(source.nodes[node]["color"]) != color_signature(target.nodes[image]["color"]):
            raise ValueError("chemical_witness_atom_color_mismatch")
    for left, right, data in source.edges(data=True):
        if (not target.has_edge(mapping[left], mapping[right]) or
                color_signature(data["color"]) !=
                color_signature(target.edges[mapping[left], mapping[right]]["color"])):
            raise ValueError("chemical_witness_bond_color_mismatch")
    return True
