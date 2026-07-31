"""Shared data and reporting helpers for Lewis-state experiments."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from rdkit import Chem

from synkit.Chem.Reaction.standardize import Standardize

ROOT = Path(__file__).resolve().parents[2]
LEWIS_ROOT = Path(__file__).resolve().parent
POLAR_DATASET = LEWIS_ROOT / "Data" / "benchmark.json.gz"
RADICAL_DATASET = LEWIS_ROOT / "Data" / "all.csv"
STANDARDIZER = Standardize()


def open_text(path: Path, mode: str = "rt"):
    """Open gzip by magic/suffix and ordinary text otherwise."""
    compressed = (
        path.read_bytes()[:2] == b"\x1f\x8b" if "r" in mode else path.suffix == ".gz"
    )
    opener = gzip.open if compressed else open
    return opener(path, mode, encoding="utf-8")


def read_json(path: Path) -> Any:
    with open_text(path) as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[round((len(ordered) - 1) * 0.95)],
        "maximum": ordered[-1],
        "total": sum(ordered),
    }


def canonical_unmapped_side(side: str) -> str:
    """Canonicalize an unordered molecular multiset without maps or stereo."""
    molecule = Chem.MolFromSmiles(side)
    if molecule is None:
        raise ValueError(f"RDKit rejected endpoint: {side!r}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(molecule)
    components = [
        Chem.MolToSmiles(
            fragment,
            canonical=True,
            isomericSmiles=False,
        )
        for fragment in Chem.GetMolFrags(
            molecule,
            asMols=True,
            sanitizeFrags=False,
        )
    ]
    return ".".join(sorted(components))


def canonical_unmapped_reaction(reaction: str) -> str:
    """Canonicalize both unordered endpoints without maps or stereo."""
    standardized = STANDARDIZER.fit(
        reaction,
        remove_aam=True,
        ignore_stereo=True,
        remove_invalid=False,
    )
    if standardized is None:
        raise ValueError(f"SynKit could not standardize reaction: {reaction!r}")
    reactants, separator, products = standardized.partition(">>")
    if not separator or ">>" in products:
        raise ValueError(f"Malformed standardized reaction: {standardized!r}")
    return (
        f"{canonical_unmapped_side(reactants)}" f">>{canonical_unmapped_side(products)}"
    )
