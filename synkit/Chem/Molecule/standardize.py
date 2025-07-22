from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
from typing import Optional


def sanitize_and_canonicalize_smiles(smiles: str) -> Optional[str]:
    """Sanitize and canonicalize a SMILES string.

    :param smiles: Input SMILES string.
    :type smiles: str
    :returns: Canonical SMILES if valid, otherwise None.
    :rtype: Optional[str]
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def normalize_molecule(mol: Chem.Mol) -> Chem.Mol:
    """Normalize a molecule using RDKit's Normalizer.

    :param mol: RDKit Mol object to normalize.
    :type mol: Chem.Mol
    :returns: Normalized RDKit Mol object.
    :rtype: Chem.Mol
    """
    normalizer = rdMolStandardize.Normalizer()
    return normalizer.normalize(mol)


def canonicalize_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """Canonicalize the tautomeric form of a molecule.

    :param mol: RDKit Mol object to canonicalize.
    :type mol: Chem.Mol
    :returns: Mol object with a canonical tautomer.
    :rtype: Chem.Mol
    """
    tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
    return tautomer_enumerator.Canonicalize(mol)


def salts_remover(mol: Chem.Mol) -> Chem.Mol:
    """Remove salt fragments from a molecule.

    :param mol: RDKit Mol object to process.
    :type mol: Chem.Mol
    :returns: Mol object with salts removed.
    :rtype: Chem.Mol
    """
    remover = SaltRemover()
    return remover.StripMol(mol)


def uncharge_molecule(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize a molecule by removing charges.

    :param mol: RDKit Mol object to neutralize.
    :type mol: Chem.Mol
    :returns: Neutralized Mol object.
    :rtype: Chem.Mol
    """
    uncharger = rdMolStandardize.Uncharger()
    return uncharger.uncharge(mol)


def fragments_remover(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Keep only the largest fragment of a molecule.

    :param mol: RDKit Mol object to fragment.
    :type mol: Chem.Mol
    :returns: Mol object of the largest fragment, or None if input is
        empty.
    :rtype: Optional[Chem.Mol]
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(frags, default=None, key=lambda m: m.GetNumAtoms())


def remove_explicit_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Remove all explicit hydrogens from a molecule.

    :param mol: RDKit Mol object to process.
    :type mol: Chem.Mol
    :returns: Mol object without explicit hydrogens.
    :rtype: Chem.Mol
    """
    return Chem.RemoveHs(mol)


def remove_radicals_and_add_hydrogens(
    mol: Chem.Mol, removeH: bool = True
) -> Optional[Chem.Mol]:
    """Replace radical electrons by hydrogens and optionally remove explicit H.

    :param mol: RDKit Mol object with possible radicals.
    :type mol: Chem.Mol
    :param removeH: If True, remove explicit hydrogens after addition.
    :type removeH: bool
    :returns: Mol object with radicals neutralized and hydrogens
        adjusted.
    :rtype: Optional[Chem.Mol]
    """
    for atom in mol.GetAtoms():
        rad = atom.GetNumRadicalElectrons()
        if rad > 0:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + rad)
            atom.SetNumRadicalElectrons(0)
    mol = rdmolops.AddHs(mol)
    return remove_explicit_hydrogens(mol) if removeH else mol


def remove_isotopes(mol: Chem.Mol) -> Chem.Mol:
    """Remove all isotope labels from a molecule.

    :param mol: RDKit Mol object to process.
    :type mol: Chem.Mol
    :returns: Mol object with isotopes cleared.
    :rtype: Chem.Mol
    """
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol


def clear_stereochemistry(mol: Chem.Mol) -> Chem.Mol:
    """Remove stereochemical annotations from a molecule.

    :param mol: RDKit Mol object to process.
    :type mol: Chem.Mol
    :returns: Mol object with stereochemistry removed.
    :rtype: Chem.Mol
    """
    Chem.RemoveStereochemistry(mol)
    return mol


def fix_radical_rsmi(rsmi: str, removeH: bool = True) -> str:
    """Fix radicals in a reaction SMILES by converting them to hydrogens.

    :param rsmi: Reaction SMILES string, format 'reactant>>product'.
    :type rsmi: str
    :param removeH: If True, remove explicit hydrogens after addition.
    :type removeH: bool
    :returns: Corrected reaction SMILES with radicals replaced.
    :rtype: str
    """
    react_smiles, prod_smiles = rsmi.split(">>")
    r_mol = Chem.MolFromSmiles(react_smiles, sanitize=False)
    p_mol = Chem.MolFromSmiles(prod_smiles, sanitize=False)
    Chem.SanitizeMol(r_mol)
    Chem.SanitizeMol(p_mol)

    if r_mol and p_mol:
        r_fixed = remove_radicals_and_add_hydrogens(r_mol, removeH)
        p_fixed = remove_radicals_and_add_hydrogens(p_mol, removeH)
        r_out = Chem.MolToSmiles(r_fixed) if r_fixed else react_smiles
        p_out = Chem.MolToSmiles(p_fixed) if p_fixed else prod_smiles
        return f"{r_out}>>{p_out}"
    return rsmi
