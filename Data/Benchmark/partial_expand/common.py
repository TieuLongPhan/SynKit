"""Shared, dependency-light helpers for the expansion/replay benchmarks."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Iterator

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
try:
    from synkit.Chem.Reaction.standardize import Standardize
except ModuleNotFoundError:
    # Standalone execution from this nested benchmark directory does not put
    # the checkout root on sys.path. Preserve an explicit versioned
    # PYTHONPATH, otherwise fall back to the current source tree.
    sys.path.insert(0, str(ROOT))
    from synkit.Chem.Reaction.standardize import Standardize

POLAR_DATASET = ROOT / "Data" / "Benchmark" / "benchmark.json.gz"
RADICAL_DATASET = ROOT / "Data" / "Benchmark" / "radical" / "all.csv"
STANDARDIZER = Standardize()


def open_text(path: Path, mode: str = "rt"):
    """Open gzip by magic/suffix and ordinary text otherwise."""
    if "r" in mode:
        compressed = path.read_bytes()[:2] == b"\x1f\x8b"
    else:
        compressed = path.suffix == ".gz"
    opener = gzip.open if compressed else open
    return opener(path, mode, encoding="utf-8")


def read_json(path: Path) -> Any:
    with open_text(path) as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timing_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "total": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "maximum": ordered[-1],
        "total": sum(ordered),
    }


def canonical_unmapped_side(side: str) -> str:
    """Canonicalize one possibly disconnected endpoint without AAM labels."""
    molecule = Chem.MolFromSmiles(side)
    if molecule is None:
        raise ValueError(f"RDKit rejected endpoint: {side!r}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def canonical_unmapped_reaction(reaction: str) -> str:
    """Return SynKit's canonical reaction SMILES with AAM and stereo removed."""
    standardized = STANDARDIZER.fit(
        reaction,
        remove_aam=True,
        ignore_stereo=True,
    )
    if standardized is None:
        raise ValueError(f"SynKit could not standardize reaction: {reaction!r}")
    return standardized


def split_reaction(reaction: str) -> tuple[str, str]:
    if reaction.count(">>") != 1:
        raise ValueError("Expected exactly one reaction separator")
    return tuple(reaction.split(">>", 1))  # type: ignore[return-value]
