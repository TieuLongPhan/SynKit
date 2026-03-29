from __future__ import annotations

from typing import Any, Optional

try:
    from rdkit import Chem
except ImportError:
    Chem = None


def mol_from_smiles_safe(smiles: str) -> Optional[Any]:
    if Chem is None or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles, sanitize=False)
    except Exception:
        return None


def sanitize_safe(mol: Any) -> bool:
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def has_atom_maps(mol: Any) -> bool:
    try:
        return any(a.GetAtomMapNum() > 0 for a in mol.GetAtoms())
    except Exception:
        return False


def strip_maps_and_canonical_from_mol(mol: Any) -> Optional[str]:
    try:
        for a in mol.GetAtoms():
            a.SetAtomMapNum(0)
        mol_nomap = Chem.RemoveAllHs(mol)
        return Chem.MolToSmiles(mol_nomap, canonical=True)
    except Exception:
        return None


def canonical_from_mol(mol: Any) -> Optional[str]:
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def assign_deterministic_maps_and_canonical(nomap_smiles: str) -> Optional[str]:
    try:
        mol2 = Chem.MolFromSmiles(nomap_smiles, sanitize=False)
        if mol2 is None:
            return None
        if not sanitize_safe(mol2):
            return None
        for a in mol2.GetAtoms():
            a.SetAtomMapNum(a.GetIdx() + 1)
        return Chem.MolToSmiles(mol2, canonical=True)
    except Exception:
        return None


def standardize_smiles_rdkit(smiles: str, *, keep_aam: bool) -> Optional[str]:
    """
    Standardize one SMILES string.

    Behavior:
      - If RDKit is unavailable: passthrough.
      - If keep_aam=False: strip atom maps, remove Hs, return canonical SMILES.
      - If keep_aam=True:
          * keep mapped canonical SMILES if maps already exist
          * otherwise assign deterministic maps and return canonical mapped SMILES
    """
    if not smiles:
        return None

    if Chem is None:
        return smiles

    mol = mol_from_smiles_safe(smiles)
    if mol is None:
        return None

    if not sanitize_safe(mol):
        return None

    if not keep_aam:
        return strip_maps_and_canonical_from_mol(mol)

    if has_atom_maps(mol):
        return canonical_from_mol(mol)

    nomap = strip_maps_and_canonical_from_mol(mol)
    if nomap is None:
        return None
    return assign_deterministic_maps_and_canonical(nomap)
