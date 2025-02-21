from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover


def normalize_molecule(mol: Chem.Mol) -> Chem.Mol:
    """
    Normalize a molecule using RDKit's Normalizer.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object to be normalized.

    Returns:
    - Chem.Mol: Normalized RDKit Mol object.
    """
    normalizer = rdMolStandardize.Normalizer()
    return normalizer.normalize(mol)


def canonicalize_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """
    Canonicalize the tautomer of a molecule using RDKit's TautomerCanonicalizer.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with canonicalized tautomer.
    """
    tautomer_canonicalizer = rdMolStandardize.TautomerEnumerator()
    return tautomer_canonicalizer.Canonicalize(mol)


def salts_remover(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove salt fragments from a molecule using RDKit's SaltRemover.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with salts removed.
    """
    remover = SaltRemover()
    return remover.StripMol(mol)


def uncharge_molecule(mol: Chem.Mol) -> Chem.Mol:
    """
    Neutralize a molecule by removing counter-ions using RDKit's Uncharger.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Neutralized Mol object.
    """
    uncharger = rdMolStandardize.Uncharger()
    return uncharger.uncharge(mol)


def fragments_remover(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove small fragments from a molecule, keeping only the largest one.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with small fragments removed.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(frags, default=None, key=lambda m: m.GetNumAtoms())


def remove_explicit_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove explicit hydrogens from a molecule to leave only the heavy atoms.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with explicit hydrogens removed.
    """
    return Chem.RemoveHs(mol)


def remove_radicals_and_add_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove radicals from a molecule by setting radical electrons to zero and adding hydrogens where needed.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with radicals removed and necessary hydrogens added.
    """
    mol = Chem.RemoveHs(mol)  # Remove explicit hydrogens first
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            atom.SetNumExplicitHs(
                atom.GetNumExplicitHs() + atom.GetNumRadicalElectrons()
            )
        atom.SetNumRadicalElectrons(0)
    mol = rdmolops.AddHs(mol)  # Add hydrogens back
    return remove_explicit_hydrogens(mol)


def remove_isotopes(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove isotopic information from a molecule.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with isotopes removed.
    """
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol


def clear_stereochemistry(mol: Chem.Mol) -> Chem.Mol:
    """
    Clear all stereochemical information from a molecule.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with stereochemistry cleared.
    """
    Chem.RemoveStereochemistry(mol)
    return mol
