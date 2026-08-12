"""Shared data and reporting helpers for Lewis-state experiments."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
LEWIS_ROOT = Path(__file__).resolve().parent
POLAR_DATASET = LEWIS_ROOT / "Data" / "benchmark.json.gz"
RADICAL_DATASET = LEWIS_ROOT / "Data" / "all.csv"


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
    """Standardize both unordered endpoints without maps or stereo."""
    reactants, separator, products = reaction.partition(">>")
    if not separator or not reactants or not products or ">>" in products:
        raise ValueError(f"Malformed reaction: {reaction!r}")
    return (
        f"{_standardized_unmapped_side(reactants)}"
        f">>{_standardized_unmapped_side(products)}"
    )


@lru_cache(maxsize=8192)
def _standardized_unmapped_side(side: str) -> str:
    """Apply the established strict endpoint pipeline with a local cache.

    Keeping the mapped canonicalization pass is important for a small class
    of fused aromatic/Kekule products: it makes symmetry-related spellings
    converge to the same endpoint. Splitting the reaction pipeline by side
    lets repeated substrates share that exact work across all applications.
    """
    molecules = []
    for fragment in side.split("."):
        molecule = Chem.MolFromSmiles(fragment, sanitize=False)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f"RDKit rejected endpoint fragment: {fragment!r}")
        try:
            Chem.SanitizeMol(molecule)
        except Exception as exc:
            raise ValueError(f"RDKit rejected endpoint fragment: {fragment!r}") from exc
        molecules.append(molecule)
    if not molecules:
        raise ValueError(f"Empty endpoint: {side!r}")

    mapped = ".".join(
        sorted(
            Chem.MolToSmiles(molecule, isomericSmiles=False) for molecule in molecules
        )
    )
    molecule = Chem.MolFromSmiles(mapped)
    if molecule is None:
        raise ValueError(f"RDKit rejected standardized endpoint: {mapped!r}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    unmapped = Chem.MolToSmiles(molecule, canonical=True).replace("[HH]", "[H][H]")
    return canonical_unmapped_side(unmapped)


def unique_standardized_reactions(reactions: Iterable[str]) -> set[str]:
    """Standardize reaction SMILES and remove canonical duplicates.

    Atom maps and stereochemistry are removed by
    :func:`canonical_unmapped_reaction`; molecular components on both sides
    are canonicalized and sorted before set-based deduplication.
    """
    return {canonical_unmapped_reaction(reaction) for reaction in reactions}
